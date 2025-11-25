#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype: A-graph GNN solver for pressure Poisson

Graph      : coefficient matrix A (from pEqn_*.dat)
Data loss  : L_data = mean( (x_pred - x_true)^2 )  (if x_true があれば / 無ければ 0)
PDE loss   : L_pde  = L_A + L_div
  - L_A   = mean( (A x_pred - b)^2 )
  - L_div = mean( w_i * ( (A x_pred - b)_i / V_i )^2 )
      * V_i は pEqn_*.dat のセル体積 (なければ cellSize^3 でフォールバック)
      * w_i は cellSize, aspectRatio から作る mesh-quality weight
total loss = L_data + lambda_pde * L_pde
"""

import os
import glob
import argparse
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

# ------------------------------------------------------------
# 0. 定数とハイパーパラメータ設定
# ------------------------------------------------------------

# 数値計算用の小さな値（ゼロ除算防止）
EPSILON_VOLUME = 1e-12
EPSILON_NORM = 1e-20


@dataclass
class TrainingConfig:
    """学習設定パラメータ"""
    # データ
    data_dir: str = "./gnn"
    batch_size: int = 1  # グラフのバッチサイズ

    # モデル
    hidden_dim: int = 64
    num_layers: int = 3

    # 学習
    num_epochs: int = 200
    learning_rate: float = 1e-3
    lambda_pde: float = 1.0

    # Early stopping
    early_stopping_patience: int = 30

    # スケジューラ
    scheduler_factor: float = 0.3
    scheduler_patience: int = 10

    # 再現性
    seed: int = 42

    # チェックポイント
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 50  # N epoch ごとに保存

    # 出力
    model_output: str = "pressure_gnn_prototype.pt"
    train_loss_file: str = "train_loss.dat"
    val_loss_file: str = "val_loss.dat"


def parse_args() -> TrainingConfig:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Pressure Poisson GNN Solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # データ
    parser.add_argument("--data-dir", type=str, default="./gnn",
                        help="Directory containing pEqn_*.dat files")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for training")

    # モデル
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension size")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of GNN layers")

    # 学習
    parser.add_argument("--num-epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-3,
                        dest="learning_rate", help="Learning rate")
    parser.add_argument("--lambda-pde", type=float, default=1.0,
                        help="Weight for PDE loss term")

    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=30,
                        help="Patience for early stopping (0 to disable)")

    # スケジューラ
    parser.add_argument("--scheduler-factor", type=float, default=0.3,
                        help="Factor for learning rate reduction")
    parser.add_argument("--scheduler-patience", type=int, default=10,
                        help="Patience for learning rate scheduler")

    # 再現性
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # チェックポイント
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N epochs")

    # 出力
    parser.add_argument("--model-output", type=str, default="pressure_gnn_prototype.pt",
                        help="Path to save final model")
    parser.add_argument("--train-loss-file", type=str, default="train_loss.dat",
                        help="File to save training loss")
    parser.add_argument("--val-loss-file", type=str, default="val_loss.dat",
                        help="File to save validation loss")

    args = parser.parse_args()

    return TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lambda_pde=args.lambda_pde,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        model_output=args.model_output,
        train_loss_file=args.train_loss_file,
        val_loss_file=args.val_loss_file,
    )


def set_random_seed(seed: int) -> None:
    """再現性のための乱数シード設定"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDNNの決定的動作を有効化（若干遅くなるが再現性が向上）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# 1. ファイル読み込み & グラフ構築
# ------------------------------------------------------------

def parse_cells_format_a(
    cell_lines: List[str],
    nCells: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    フォーマットA（拡張+体積あり）のセル情報を解析

    フォーマット: id x y z diag b skew nonOrtho aspect diagContrast V cellSize sizeJump

    Returns:
        coords, diag, bvec, volume, cell_size, aspect_ratio
    """
    coords = np.zeros((nCells, 3), dtype=np.float64)
    diag = np.zeros(nCells, dtype=np.float64)
    bvec = np.zeros(nCells, dtype=np.float64)
    volume = np.zeros(nCells, dtype=np.float64)
    cell_size = np.zeros(nCells, dtype=np.float64)
    aspect_ratio = np.ones(nCells, dtype=np.float64)

    for ln in cell_lines:
        parts = ln.split()
        if len(parts) < 13:
            raise RuntimeError(f"CELLS 行のフォーマットが揃っていません: {ln}")

        cid = int(parts[0])
        coords[cid] = [float(parts[1]), float(parts[2]), float(parts[3])]
        diag[cid] = float(parts[4])
        bvec[cid] = float(parts[5])
        aspect_ratio[cid] = float(parts[8])
        volume[cid] = float(parts[10])
        cell_size[cid] = float(parts[11])

    return coords, diag, bvec, volume, cell_size, aspect_ratio


def parse_cells_format_b(
    cell_lines: List[str],
    nCells: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    フォーマットB（拡張、体積なし）のセル情報を解析

    フォーマット: id x y z diag b skew nonOrtho aspect diagContrast cellSize sizeJump

    Returns:
        coords, diag, bvec, volume, cell_size, aspect_ratio
    """
    coords = np.zeros((nCells, 3), dtype=np.float64)
    diag = np.zeros(nCells, dtype=np.float64)
    bvec = np.zeros(nCells, dtype=np.float64)
    cell_size = np.zeros(nCells, dtype=np.float64)
    aspect_ratio = np.ones(nCells, dtype=np.float64)

    for ln in cell_lines:
        parts = ln.split()
        if len(parts) < 12:
            raise RuntimeError(f"CELLS 行のフォーマットが揃っていません: {ln}")

        cid = int(parts[0])
        coords[cid] = [float(parts[1]), float(parts[2]), float(parts[3])]
        diag[cid] = float(parts[4])
        bvec[cid] = float(parts[5])
        aspect_ratio[cid] = float(parts[8])
        h_i = float(parts[10])  # cellSize
        cell_size[cid] = h_i

    # 体積は cellSize^3 で近似
    volume = cell_size ** 3

    return coords, diag, bvec, volume, cell_size, aspect_ratio


def parse_cells_format_c(
    cell_lines: List[str],
    nCells: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    フォーマットC（最小版）のセル情報を解析

    フォーマット: id x y z diag b

    Returns:
        coords, diag, bvec
    """
    coords = np.zeros((nCells, 3), dtype=np.float64)
    diag = np.zeros(nCells, dtype=np.float64)
    bvec = np.zeros(nCells, dtype=np.float64)

    for ln in cell_lines:
        parts = ln.split()
        if len(parts) < 6:
            raise RuntimeError(f"CELLS 行のフォーマットが想定と違います: {ln}")

        cid = int(parts[0])
        coords[cid] = [float(parts[1]), float(parts[2]), float(parts[3])]
        diag[cid] = float(parts[4])
        bvec[cid] = float(parts[5])

    return coords, diag, bvec


def estimate_cell_properties(
    coords: np.ndarray,
    neighbors: List[List[int]],
    nCells: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    近傍情報からcellSizeとaspectRatioを推定

    Args:
        coords: セル座標 [N, 3]
        neighbors: 各セルの近傍セルリスト
        nCells: セル数

    Returns:
        cell_size, aspect_ratio, volume
    """
    cell_size = np.zeros(nCells, dtype=np.float64)
    aspect_ratio = np.ones(nCells, dtype=np.float64)

    for i in range(nCells):
        nbrs = neighbors[i]
        if len(nbrs) == 0:
            cell_size[i] = 1.0
            aspect_ratio[i] = 1.0
            continue

        diffs = coords[nbrs] - coords[i]
        dists = np.linalg.norm(diffs, axis=1)
        mean_dist = float(dists.mean())
        cell_size[i] = mean_dist if mean_dist > 0.0 else 1.0

        if len(dists) > 1:
            d_min = float(dists.min())
            d_max = float(dists.max())
            if d_min <= 0.0:
                aspect_ratio[i] = 1.0
            else:
                aspect_ratio[i] = d_max / d_min
        else:
            aspect_ratio[i] = 1.0

    # 体積は cellSize^3 から近似
    volume = cell_size ** 3

    return cell_size, aspect_ratio, volume


def parse_edges(
    edge_lines: List[str],
    nFaces: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    エッジ情報を解析

    Returns:
        lower_ids, upper_ids, lower_vals, upper_vals
    """
    lower_ids = np.zeros(nFaces, dtype=np.int64)
    upper_ids = np.zeros(nFaces, dtype=np.int64)
    lower_vals = np.zeros(nFaces, dtype=np.float64)
    upper_vals = np.zeros(nFaces, dtype=np.float64)

    for k, ln in enumerate(edge_lines):
        parts = ln.split()
        if len(parts) < 5:
            raise RuntimeError(f"EDGES 行のフォーマットが想定と違います: {ln}")
        lower_ids[k] = int(parts[1])
        upper_ids[k] = int(parts[2])
        lower_vals[k] = float(parts[3])
        upper_vals[k] = float(parts[4])

    return lower_ids, upper_ids, lower_vals, upper_vals


def load_pEqn_graph(pEqn_path: str) -> Data:
    """
    pEqn_<time>.dat を読み込んで torch_geometric.data.Data を作る。

    想定フォーマット（新しい順）:

    [A] 拡張 + 体積あり  (len(parts) == 13)
        id x y z diag b skew nonOrtho aspect diagContrast V cellSize sizeJump

    [B] 拡張 (体積なし) (len(parts) == 12)
        id x y z diag b skew nonOrtho aspect diagContrast cellSize sizeJump
      -> V ≒ cellSize^3 として近似

    [C] 最小版              (len(parts) >= 6)
        id x y z diag b
      -> cellSize, aspectRatio は近傍距離から推定し、V ≒ cellSize^3
    """
    # ファイル読み込みとセクション分割
    with open(pEqn_path, "r") as f:
        raw_lines = [ln.strip() for ln in f.readlines()]

    lines = [ln for ln in raw_lines if ln != ""]

    assert lines[0].startswith("nCells"), f"Invalid format: {pEqn_path}"
    nCells = int(lines[0].split()[1])
    assert lines[1].startswith("nFaces"), f"Invalid format: {pEqn_path}"
    nFaces = int(lines[1].split()[1])

    # CELLSとEDGESセクションの位置を特定
    cells_start = None
    edges_start = None
    for i, ln in enumerate(lines):
        if ln.startswith("CELLS"):
            cells_start = i
        elif ln.startswith("EDGES"):
            edges_start = i

    if cells_start is None or edges_start is None:
        raise RuntimeError(f"CELLS/EDGES セクションが見つかりません: {pEqn_path}")

    cell_lines = lines[cells_start + 1 : edges_start]
    edge_lines = lines[edges_start + 1 :]

    if len(cell_lines) != nCells:
        print(f"[WARN] nCells={nCells} だが CELLS 行数={len(cell_lines)}: {pEqn_path}")

    # --- CELLS 部分の解析 ---
    # 先頭行からフォーマットを判定
    first_parts = cell_lines[0].split()
    n_fields = len(first_parts)

    if n_fields >= 13:
        # フォーマット A: 拡張 + 体積あり
        coords, diag, bvec, volume, cell_size, aspect_ratio = parse_cells_format_a(
            cell_lines, nCells
        )
    elif n_fields == 12:
        # フォーマット B: 拡張（体積なし）
        coords, diag, bvec, volume, cell_size, aspect_ratio = parse_cells_format_b(
            cell_lines, nCells
        )
    else:
        # フォーマット C: 最小版
        coords, diag, bvec = parse_cells_format_c(cell_lines, nCells)
        # cellSize/aspectRatio/volumeは後で推定

    # --- EDGES 部分の解析 ---
    lower_ids, upper_ids, lower_vals, upper_vals = parse_edges(edge_lines, nFaces)

    # --- 近傍リストの構築（フォーマットCの場合に必要）---
    neighbors: List[List[int]] = [[] for _ in range(nCells)]
    for li, ui in zip(lower_ids, upper_ids):
        neighbors[li].append(ui)
        neighbors[ui].append(li)

    # フォーマットC の場合、近傍から cellSize/aspectRatio/volume を推定
    if n_fields < 12:
        cell_size, aspect_ratio, volume = estimate_cell_properties(
            coords, neighbors, nCells
        )

    # --- GNN用 edge_index (無向グラフ) ---
    ei_src = np.concatenate([lower_ids, upper_ids])
    ei_dst = np.concatenate([upper_ids, lower_ids])
    edge_index = torch.tensor(
        np.vstack([ei_src, ei_dst]), dtype=torch.long
    )  # [2, 2*nFaces]

    # --- ノード特徴量 x_feat = [b, diag, cellSize, aspectRatio, x, y, z]
    b_t = torch.from_numpy(bvec).float().view(-1, 1)
    diag_t = torch.from_numpy(diag).float().view(-1, 1)
    cell_size_t = torch.from_numpy(cell_size).float().view(-1, 1)
    aspect_t = torch.from_numpy(aspect_ratio).float().view(-1, 1)
    coords_t = torch.from_numpy(coords).float()  # [N,3]

    x_feat = torch.cat([b_t, diag_t, cell_size_t, aspect_t, coords_t], dim=1)

    # --- A, mesh-quality を Data に保持
    diag_t_flat = diag_t.view(-1)
    b_t_flat = b_t.view(-1)
    lower_idx_t = torch.from_numpy(lower_ids).long()
    upper_idx_t = torch.from_numpy(upper_ids).long()
    lower_val_t = torch.from_numpy(lower_vals).float()
    upper_val_t = torch.from_numpy(upper_vals).float()
    volume_t = torch.from_numpy(volume).float().view(-1)

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        num_nodes=nCells,
    )
    data.diag = diag_t_flat
    data.b = b_t_flat
    data.lower_index = lower_idx_t
    data.upper_index = upper_idx_t
    data.lower_val = lower_val_t
    data.upper_val = upper_val_t
    data.cell_size = cell_size_t.view(-1)
    data.aspect_ratio = aspect_t.view(-1)
    data.volume = volume_t  # ★ 厳密な V_i（あるいは近似）

    # --- (オプション) x_<time>.dat があれば ground-truth として読み込み
    base = os.path.basename(pEqn_path)
    time_str = base[len("pEqn_") : -4]  # "pEqn_10.dat" -> "10"
    x_path = os.path.join(os.path.dirname(pEqn_path), f"x_{time_str}.dat")
    if os.path.isfile(x_path):
        x_vals = np.zeros(nCells, dtype=np.float64)
        with open(x_path, "r") as fx:
            for ln in fx:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                cid = int(parts[0])
                val = float(parts[1])
                x_vals[cid] = val
        data.x_true = torch.from_numpy(x_vals).float()

    return data


class PoissonSystemDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, "pEqn_*.dat")))
        if not self.files:
            raise RuntimeError(f"pEqn_*.dat が見つかりません: {root_dir}")

    def len(self) -> int:
        return len(self.files)

    def get(self, idx: int) -> Data:
        path = self.files[idx]
        return load_pEqn_graph(path)


# ------------------------------------------------------------
# 2. Early Stopping とチェックポイント管理
# ------------------------------------------------------------

class EarlyStopping:
    """Early Stopping クラス"""

    def __init__(self, patience: int = 30, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: 改善が見られないエポック数の許容範囲
            min_delta: 改善とみなす最小の変化量
            verbose: メッセージを出力するかどうか
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        検証損失をチェックして early stopping すべきか判定

        Args:
            val_loss: 検証損失

        Returns:
            True なら学習を停止すべき
        """
        if self.patience <= 0:
            # patience が 0 以下なら early stopping を無効化
            return False

        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            # 改善があった
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"  [EarlyStopping] Validation loss improved to {val_loss:.3e}")
        else:
            # 改善がなかった
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] Stopping early after {self.patience} epochs without improvement")
                return True

        return False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_path: str,
) -> None:
    """
    学習状態をチェックポイントとして保存

    Args:
        model: モデル
        optimizer: オプティマイザ
        scheduler: 学習率スケジューラ（オプション）
        epoch: エポック数
        train_loss: 訓練損失
        val_loss: 検証損失
        checkpoint_path: 保存先パス
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> int:
    """
    チェックポイントから学習状態を復元

    Args:
        checkpoint_path: チェックポイントファイルのパス
        model: モデル
        optimizer: オプティマイザ（オプション）
        scheduler: 学習率スケジューラ（オプション）

    Returns:
        復元されたエポック数
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")

    return epoch


# ------------------------------------------------------------
# 3. GNN モデル定義
# ------------------------------------------------------------

class PressureGNN(nn.Module):
    """
    簡単な GCN ベースの GNN。
    入力: node features [b, diag, cellSize, aspectRatio, x, y, z]
    出力: 各セルの x_pred (圧力解ベクトル)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, 1))

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = torch.relu(x)

        return x.view(-1)  # [N]


# ------------------------------------------------------------
# 4. Ax, 残差 r, L_A, L_div, w_i の計算
# ------------------------------------------------------------

def apply_A_to_x(data: Data, x_vec: torch.Tensor) -> torch.Tensor:
    """
    OpenFOAM の lduMatrix::Amul に対応する Ax を計算する。
    Apsi[cell] = diag[cell]*x[cell]
    for each face:
        Apsi[u] += lower[face]*x[l]
        Apsi[l] += upper[face]*x[u]
    """
    diag = data.diag
    lower_idx = data.lower_index
    upper_idx = data.upper_index
    lower_val = data.lower_val
    upper_val = data.upper_val

    Ax = diag * x_vec  # diag 部分

    Ax.index_add_(0, upper_idx, lower_val * x_vec[lower_idx])
    Ax.index_add_(0, lower_idx, upper_val * x_vec[upper_idx])

    return Ax


def compute_losses(
    data: Data,
    x_pred: torch.Tensor,
    lambda_pde: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    data   : 1つのグラフ（1つの Poisson システム）
    x_pred : GNN が出力した解ベクトル [N]

    戻り値:
      total_loss, data_loss, pde_loss
        - data_loss : (x_pred - x_true)^2 の平均（x_true が無ければ 0）
        - pde_loss  : L_A + L_div
    """
    b = data.b  # [N]

    # Ax_pred
    Ax = apply_A_to_x(data, x_pred)

    # 離散残差 r = A x_pred - b
    r = Ax - b

    # --- data loss: x_true があれば教師あり, 無ければ 0
    if hasattr(data, "x_true"):
        x_true = data.x_true.to(x_pred.device)
        data_loss = torch.mean((x_pred - x_true) ** 2)
    else:
        data_loss = torch.tensor(0.0, device=x_pred.device)

    # --- PDE loss: L_A + L_div

    # L_A: 行列表現の PDE 残差
    L_A = torch.mean(r ** 2)

    # L_div: div ≒ r / V （Thompson 系の「フラックス発散」的な項）
    volume = data.volume  # ★ 厳密な V_i（ない場合は読み込み時に cellSize^3 で近似）
    div_hat = r / (volume + EPSILON_VOLUME)

    # w_i: cellSize と aspectRatio から作る mesh-quality weight
    cell_size = data.cell_size
    aspect = data.aspect_ratio

    cs_norm = cell_size / (cell_size.mean() + EPSILON_VOLUME)
    ar_norm = aspect / (aspect.mean() + EPSILON_VOLUME)
    w = 0.5 * (cs_norm + ar_norm)

    L_div = torch.mean(w * (div_hat ** 2))

    pde_loss = L_A + L_div

    total_loss = data_loss + lambda_pde * pde_loss
    return total_loss, data_loss.detach(), pde_loss.detach()


# ------------------------------------------------------------
# 5. 学習ループ
# ------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_pde: float,
) -> Tuple[float, float, float]:
    """
    1エポック分の学習を実行

    Args:
        model: GNNモデル
        dataloader: データローダー（バッチ処理）
        optimizer: オプティマイザ
        device: デバイス（CPU/GPU）
        lambda_pde: PDE損失の重み

    Returns:
        (平均total_loss, 平均data_loss, 平均pde_loss)
    """
    model.train()
    total_loss = 0.0
    total_data = 0.0
    total_pde = 0.0
    n = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        x_pred = model(batch)
        loss, data_loss, pde_loss = compute_losses(
            batch, x_pred, lambda_pde=lambda_pde
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_data += data_loss.item()
        total_pde += pde_loss.item()
        n += 1

    return total_loss / n, total_data / n, total_pde / n


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    lambda_pde: float,
) -> Tuple[float, float, float]:
    """
    1エポック分の評価を実行

    Args:
        model: GNNモデル
        dataloader: データローダー（バッチ処理）
        device: デバイス（CPU/GPU）
        lambda_pde: PDE損失の重み

    Returns:
        (平均total_loss, 平均data_loss, 平均pde_loss)
    """
    model.eval()
    total_loss = 0.0
    total_data = 0.0
    total_pde = 0.0
    n = 0

    for batch in dataloader:
        batch = batch.to(device)
        x_pred = model(batch)
        loss, data_loss, pde_loss = compute_losses(
            batch, x_pred, lambda_pde=lambda_pde
        )
        total_loss += loss.item()
        total_data += data_loss.item()
        total_pde += pde_loss.item()
        n += 1

    return total_loss / n, total_data / n, total_pde / n


# ------------------------------------------------------------
# 6. 予測結果の保存
# ------------------------------------------------------------

def save_predictions(
    model: nn.Module,
    graphs: List[Data],
    device: torch.device,
    output_dir: str = "./predictions",
) -> None:
    """
    モデルの予測結果を保存

    Args:
        model: 学習済みモデル
        graphs: データのリスト
        device: デバイス
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    print(f"\nSaving predictions to {output_dir}...")

    with torch.no_grad():
        for idx, data in enumerate(graphs):
            data = data.to(device)
            x_pred = model(data).cpu().numpy()

            # ファイル名を生成（グラフのインデックスベース）
            output_path = os.path.join(output_dir, f"prediction_{idx:04d}.dat")

            # 予測結果を保存
            with open(output_path, "w") as f:
                f.write(f"# Prediction for graph {idx}\n")
                f.write(f"# nCells {len(x_pred)}\n")
                for cell_id, val in enumerate(x_pred):
                    f.write(f"{cell_id} {val:.12e}\n")

            # x_true がある場合は誤差も計算して保存
            if hasattr(data, "x_true"):
                x_true = data.x_true.cpu().numpy()
                error = x_pred - x_true
                error_path = os.path.join(output_dir, f"error_{idx:04d}.dat")

                with open(error_path, "w") as f:
                    f.write(f"# Prediction error for graph {idx}\n")
                    f.write(f"# nCells {len(error)}\n")
                    f.write(f"# L2_norm {np.linalg.norm(error):.12e}\n")
                    f.write(f"# Relative_L2 {np.linalg.norm(error) / (np.linalg.norm(x_true) + EPSILON_NORM):.12e}\n")
                    for cell_id, (err, pred, true) in enumerate(zip(error, x_pred, x_true)):
                        f.write(f"{cell_id} {err:.12e} {pred:.12e} {true:.12e}\n")

    print(f"Saved {len(graphs)} prediction files to {output_dir}")


# ------------------------------------------------------------
# 7. 診断用ダイアグノスティクス
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_diagnostics(
    model: nn.Module,
    graphs: List[Data],
    device: torch.device,
) -> None:
    """
    Validation グラフに対して:

      E_p      = ||x_pred - x_true|| / ||x_true||
      R_pred   = ||A x_pred - b|| / ||b||
      R_true   = ||A x_true - b|| / ||b||
      div_pred = sqrt(mean( ( (A x_pred - b)/V )^2 ))
      div_true = 同様

    を x_true を持つグラフについて平均して表示する。
    """
    model.eval()

    sum_rel_L2 = 0.0
    sum_res_pred = 0.0
    sum_res_true = 0.0
    sum_div_pred = 0.0
    sum_div_true = 0.0
    n = 0

    for data in graphs:
        if not hasattr(data, "x_true"):
            continue  # x_true がないグラフはスキップ

        data = data.to(device)
        x_true = data.x_true.to(device)
        x_pred = model(data)

        b = data.b
        volume = data.volume

        Ax_pred = apply_A_to_x(data, x_pred)
        r_pred = Ax_pred - b

        Ax_true = apply_A_to_x(data, x_true)
        r_true = Ax_true - b

        rel_L2 = torch.norm(x_pred - x_true) / (torch.norm(x_true) + EPSILON_NORM)
        rel_res_pred = torch.norm(r_pred) / (torch.norm(b) + EPSILON_NORM)
        rel_res_true = torch.norm(r_true) / (torch.norm(b) + EPSILON_NORM)

        div_pred = torch.sqrt(torch.mean((r_pred / (volume + EPSILON_VOLUME)) ** 2))
        div_true = torch.sqrt(torch.mean((r_true / (volume + EPSILON_VOLUME)) ** 2))

        sum_rel_L2 += rel_L2.item()
        sum_res_pred += rel_res_pred.item()
        sum_res_true += rel_res_true.item()
        sum_div_pred += div_pred.item()
        sum_div_true += div_true.item()
        n += 1

    if n == 0:
        print("Diagnostics: x_true を持つ validation グラフが無いためスキップしました。")
        return

    avg_rel_L2 = sum_rel_L2 / n
    avg_res_pred = sum_res_pred / n
    avg_res_true = sum_res_true / n
    avg_div_pred = sum_div_pred / n
    avg_div_true = sum_div_true / n

    ratio_res = avg_res_pred / (avg_res_true + EPSILON_NORM)
    ratio_div = avg_div_pred / (avg_div_true + EPSILON_NORM)

    print("\n=== Diagnostics on validation graphs (x_true があるものの平均) ===")
    print(f"  <E_p>                 = {avg_rel_L2:.3e}  # ||x_pred - x_true|| / ||x_true||")
    print(f"  <R_pred>              = {avg_res_pred:.3e}  # ||A x_pred - b|| / ||b||")
    print(f"  <R_true>              = {avg_res_true:.3e}  # ||A x_true - b|| / ||b||")
    print(f"  <div_pred>_rms        = {avg_div_pred:.3e}  # rms( (A x_pred - b)/V )")
    print(f"  <div_true>_rms        = {avg_div_true:.3e}  # rms( (A x_true - b)/V )")
    print(f"  ratio R_pred / R_true = {ratio_res:.3e}")
    print(f"  ratio div_pred/div_true = {ratio_div:.3e}")
    print("  ※ (A)/(B)/(C) の目安:")
    print("    (A) preconditioner 的:   E_p ~ 1e-1, R_pred ~ 1e-2〜1e-1, div_ratio ~ O(1〜10)")
    print("    (B) pEqn 短縮レベル:     E_p ~ 1e-2, R_pred ~ 1e-3,      div_ratio ~ O(1〜2)")
    print("    (C) 完全置き換えレベル: それよりさらに良いレンジ")


# ------------------------------------------------------------
# 8. メイン
# ------------------------------------------------------------
def main():
    # ==== パラメータ読み込み ====
    config = parse_args()
    print("=" * 60)
    print("Training Configuration:")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Model: hidden_dim={config.hidden_dim}, num_layers={config.num_layers}")
    print(f"  Training: epochs={config.num_epochs}, lr={config.learning_rate}, lambda_pde={config.lambda_pde}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")
    print(f"  Random seed: {config.seed}")
    print("=" * 60)

    # ==== 再現性のためのシード設定 ====
    set_random_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==== データセット読み込み ====
    dataset = PoissonSystemDataset(config.data_dir)
    print(f"Found {len(dataset)} pEqn systems.")

    # いったん全部メモリに読み込む（スナップショット数は多くない前提）
    graphs = [dataset[i] for i in range(len(dataset))]

    # ---- ★ train/val 用にランダムシャッフルして 80/20 分割 ----
    n_total = len(graphs)
    idx = np.arange(n_total)
    rng = np.random.default_rng(seed=config.seed)  # 再現性のため固定シード
    rng.shuffle(idx)
    graphs = [graphs[i] for i in idx]

    n_train = max(1, int(0.8 * n_total))
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:] if n_total > 1 else graphs

    print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")

    # ==== DataLoader の作成 ====
    train_loader = DataLoader(
        train_graphs,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # PyTorch Geometric ではマルチプロセスに注意
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"Batch size: {config.batch_size}")

    # ==== モデル定義 ====
    in_dim = graphs[0].x.shape[1]
    model = PressureGNN(in_dim=in_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.scheduler_factor,
        patience=config.scheduler_patience, verbose=True
    )

    # ==== Early Stopping とチェックポイント準備 ====
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)

    # チェックポイントディレクトリの作成
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")

    # ==== loss 出力ファイルを準備（ヘッダを書いておく）====
    with open(config.train_loss_file, "w") as f_tr:
        f_tr.write("# epoch train_loss train_data_loss train_pde_loss\n")
    with open(config.val_loss_file, "w") as f_va:
        f_va.write("# epoch val_loss val_data_loss val_pde_loss\n")

    # ==== 学習ループ ====
    best_val_loss = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_data, train_pde = train_epoch(
            model, train_loader, optimizer, device, config.lambda_pde
        )
        val_loss, val_data, val_pde = eval_epoch(
            model, val_loader, device, config.lambda_pde
        )

        # 画面表示
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.3e} "
            f"(data={train_data:.3e}, pde={train_pde:.3e}) "
            f"val_loss={val_loss:.3e} "
            f"(data={val_data:.3e}, pde={val_pde:.3e})",
            flush=True,
        )

        # loss を .dat に追記
        with open(config.train_loss_file, "a") as f_tr:
            f_tr.write(
                f"{epoch} {train_loss:.8e} {train_data:.8e} {train_pde:.8e}\n"
            )
            f_tr.flush()

        with open(config.val_loss_file, "a") as f_va:
            f_va.write(
                f"{epoch} {val_loss:.8e} {val_data:.8e} {val_pde:.8e}\n"
            )
            f_va.flush()

        # Best モデルの保存（validation loss が改善した場合）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss, best_model_path
            )
            print(f"  → Best model saved (val_loss={val_loss:.3e})")

        # 定期的なチェックポイント保存
        if config.save_every > 0 and epoch % config.save_every == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss, checkpoint_path
            )
            print(f"  → Checkpoint saved: {checkpoint_path}")

        # scheduler 用に val_loss を監視
        scheduler.step(val_loss)

        # Early stopping チェック
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Loss history is recorded in {config.train_loss_file} / {config.val_loss_file}")

    # ==== 学習済みモデルの保存（最終エポックのモデル）====
    torch.save(model.state_dict(), config.model_output)
    print(f"Final model saved to {config.model_output}")
    print(f"Best model saved to {best_model_path}")

    # ==== 評価には best model を使用 ====
    print("\n" + "=" * 60)
    print("Loading best model for final evaluation...")
    print("=" * 60)
    load_checkpoint(best_model_path, model)

    # ==== (1) テストスナップショットで x_true との相対誤差を確認 ====
    if hasattr(graphs[0], "x_true"):
        print("Checking relative error vs OpenFOAM solution (first graph)...")
        data0 = graphs[0].to(device)
        with torch.no_grad():
            x_pred = model(data0)
            x_true = data0.x_true.to(device)
            rel_err = torch.norm(x_pred - x_true) / (torch.norm(x_true) + EPSILON_NORM)
            print(f"  ||x_pred - x_true|| / ||x_true|| = {rel_err.item():.3e}")

    # ==== (2) Validation 全体で diagnostics を評価 ====
    evaluate_diagnostics(model, val_graphs, device)

    # ==== (3) 予測結果を保存 ====
    save_predictions(model, val_graphs, device, output_dir="./predictions")


if __name__ == "__main__":
    main()
