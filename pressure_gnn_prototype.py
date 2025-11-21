#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype: A-graph GNN solver for pressure Poisson
 - Graph = coefficient matrix A (from pEqn_*.dat)
 - Data loss      : L_data = mean( (A x_pred - b)^2 )
 - Physics loss   : L_phys = mean( w_i * (div_hat_i)^2 ),
                    where div_hat_i ≈ (A x_pred - b)_i / volume_i
 - Weights w_i    : from cellSize_i and aspectRatio_i
"""

import os
import glob
import math
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv

# ------------------------------------------------------------
# 1. ファイル読み込み & グラフ構築
# ------------------------------------------------------------

def load_pEqn_graph(pEqn_path: str) -> Data:
    """
    pEqn_<time>.dat を読み込んで torch_geometric.data.Data を作る。

    期待フォーマット（既存）:
        nCells <int>
        nFaces <int>
        CELLS
        <id> <x> <y> <z> <diag> <b> [その他があれば無視]
        ...
        EDGES
        <faceId> <lowerCell> <upperCell> <lowerCoeff> <upperCoeff>
        ...

    この関数内で:
      - cellSize, aspectRatio を座標と隣接関係から推定
      - x_true は x_<time>.dat があれば読み込む（なくてもOK）
    """
    with open(pEqn_path, "r") as f:
        raw_lines = [ln.strip() for ln in f.readlines()]

    # 空行を除外
    lines = [ln for ln in raw_lines if ln != ""]

    assert lines[0].startswith("nCells")
    nCells = int(lines[0].split()[1])
    assert lines[1].startswith("nFaces")
    nFaces = int(lines[1].split()[1])

    # "CELLS" と "EDGES" の位置を探す
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

    # --- CELLS 部分 ─ ノード情報
    coords = np.zeros((nCells, 3), dtype=np.float64)
    diag = np.zeros(nCells, dtype=np.float64)
    bvec = np.zeros(nCells, dtype=np.float64)

    for ln in cell_lines:
        parts = ln.split()
        if len(parts) < 6:
            raise RuntimeError(f"CELLS 行のフォーマットが想定と違います: {ln}")
        cid = int(parts[0])
        x, y, z = map(float, parts[1:4])
        d = float(parts[4])
        b = float(parts[5])
        coords[cid] = [x, y, z]
        diag[cid] = d
        bvec[cid] = b

    # --- EDGES 部分 ─ A の lower/upper と隣接情報
    lower_ids = np.zeros(nFaces, dtype=np.int64)
    upper_ids = np.zeros(nFaces, dtype=np.int64)
    lower_vals = np.zeros(nFaces, dtype=np.float64)
    upper_vals = np.zeros(nFaces, dtype=np.float64)

    for k, ln in enumerate(edge_lines):
        parts = ln.split()
        if len(parts) < 5:
            raise RuntimeError(f"EDGES 行のフォーマットが想定と違います: {ln}")
        # faceId は現在使わない
        # faceId = int(parts[0])
        lower_ids[k] = int(parts[1])
        upper_ids[k] = int(parts[2])
        lower_vals[k] = float(parts[3])
        upper_vals[k] = float(parts[4])

    # --- GNN用 edge_index (無向グラフ化)
    # faceごとに (lower -> upper) と (upper -> lower) の2本
    ei_src = np.concatenate([lower_ids, upper_ids])
    ei_dst = np.concatenate([upper_ids, lower_ids])
    edge_index = torch.tensor(
        np.vstack([ei_src, ei_dst]), dtype=torch.long
    )  # [2, 2*nFaces]

    # --- 近傍から cellSize, aspectRatio を推定
    #   neighbors[i] = セル i の隣接セル一覧
    neighbors: List[List[int]] = [[] for _ in range(nCells)]
    for li, ui in zip(lower_ids, upper_ids):
        neighbors[li].append(ui)
        neighbors[ui].append(li)

    cell_size = np.zeros(nCells, dtype=np.float64)
    aspect_ratio = np.ones(nCells, dtype=np.float64)

    for i in range(nCells):
        nbrs = neighbors[i]
        if len(nbrs) == 0:
            # 孤立セルは適当に1.0
            cell_size[i] = 1.0
            aspect_ratio[i] = 1.0
            continue
        diffs = coords[nbrs] - coords[i]  # shape (deg, 3)
        dists = np.linalg.norm(diffs, axis=1)  # 近傍までの距離
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

    # --- ノード特徴量 x_feat = [b, diag, cellSize, aspectRatio, x, y, z]
    b_t = torch.from_numpy(bvec).float().view(-1, 1)
    diag_t = torch.from_numpy(diag).float().view(-1, 1)
    cell_size_t = torch.from_numpy(cell_size).float().view(-1, 1)
    aspect_t = torch.from_numpy(aspect_ratio).float().view(-1, 1)
    coords_t = torch.from_numpy(coords).float()  # [N,3]

    x_feat = torch.cat([b_t, diag_t, cell_size_t, aspect_t, coords_t], dim=1)

    # --- A の情報も Data に保持しておく（Ax を計算するため）
    diag_t_flat = diag_t.view(-1)
    b_t_flat = b_t.view(-1)
    lower_idx_t = torch.from_numpy(lower_ids).long()
    upper_idx_t = torch.from_numpy(upper_ids).long()
    lower_val_t = torch.from_numpy(lower_vals).float()
    upper_val_t = torch.from_numpy(upper_vals).float()

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
# 2. GNN モデル定義
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

        # 1層目
        self.convs.append(GCNConv(in_dim, hidden_dim))
        # 中間層
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # 出力層（スカラー）
        self.convs.append(GCNConv(hidden_dim, 1))

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = torch.relu(x)

        # [N,1] -> [N]
        return x.view(-1)


# ------------------------------------------------------------
# 3. Ax, 残差 r, 「発散」 div_hat, w_i の計算
# ------------------------------------------------------------

def apply_A_to_x(data: Data, x_vec: torch.Tensor) -> torch.Tensor:
    """
    OpenFOAM の lduMatrix::Amul に対応する Ax を計算する。
    ApsiPtr[cell] = diag[cell]*x[cell]
    for each face:
        ApsiPtr[u] += lower[face]*x[l]
        ApsiPtr[l] += upper[face]*x[u]
    """
    diag = data.diag  # [N]
    lower_idx = data.lower_index  # [E]
    upper_idx = data.upper_index  # [E]
    lower_val = data.lower_val    # [E]
    upper_val = data.upper_val    # [E]

    Ax = diag * x_vec  # diag 部分

    # face ごとの off-diagonal 寄与
    # A[u,u] += lower * x[l]
    Ax.index_add_(0, upper_idx, lower_val * x_vec[lower_idx])
    # A[l,l] += upper * x[u]
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
    """
    b = data.b  # [N]

    # Ax_pred
    Ax = apply_A_to_x(data, x_pred)

    # 離散残差 r = A x_pred - b
    r = Ax - b

    # --- データ損失 L_data = mean(r^2)
    data_loss = torch.mean(r**2)

    # --- PDE 物理損失
    # cellSize から volume をざっくり推定 (volume ~ cellSize^3)
    cell_size = data.cell_size
    aspect = data.aspect_ratio

    volume = cell_size**3
    div_hat = r / (volume + 1.0e-12)  # 発散 ~ 残差 / 体積

    # w_i の作り方（例）:
    #   cellSize と aspectRatio をそれぞれ正規化して平均
    cs_norm = cell_size / (cell_size.mean() + 1.0e-12)
    ar_norm = aspect / (aspect.mean() + 1.0e-12)
    w = 0.5 * (cs_norm + ar_norm)

    pde_loss = torch.mean(w * (div_hat**2))

    total_loss = data_loss + lambda_pde * pde_loss
    return total_loss, data_loss.detach(), pde_loss.detach()


# ------------------------------------------------------------
# 4. 学習ループ
# ------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    graphs: List[Data],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_pde: float,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_data = 0.0
    total_pde = 0.0
    n = 0

    for data in graphs:
        data = data.to(device)
        optimizer.zero_grad()

        x_pred = model(data)
        loss, data_loss, pde_loss = compute_losses(
            data, x_pred, lambda_pde=lambda_pde
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
    graphs: List[Data],
    device: torch.device,
    lambda_pde: float,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_data = 0.0
    total_pde = 0.0
    n = 0

    for data in graphs:
        data = data.to(device)
        x_pred = model(data)
        loss, data_loss, pde_loss = compute_losses(
            data, x_pred, lambda_pde=lambda_pde
        )
        total_loss += loss.item()
        total_data += data_loss.item()
        total_pde += pde_loss.item()
        n += 1

    return total_loss / n, total_data / n, total_pde / n


# ------------------------------------------------------------
# 5. メイン
# ------------------------------------------------------------

def main():
    # ==== パラメータ ====
    data_dir = "./gnn"   # pEqn_*.dat があるディレクトリ
    num_epochs = 50
    hidden_dim = 64
    num_layers = 3
    lambda_pde = 1.0     # PDE 物理損失の重み（調整ポイント）
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==== データセット読み込み ====
    dataset = PoissonSystemDataset(data_dir)
    print(f"Found {len(dataset)} pEqn systems.")

    # いったん全部メモリに読み込む（スナップショット数は多くない前提）
    graphs = [dataset[i] for i in range(len(dataset))]

    # train / val に分割（単純に前 80% と後 20%）
    n_total = len(graphs)
    n_train = max(1, int(0.8 * n_total))
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:] if n_total > 1 else graphs

    print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")

    # ==== モデル定義 ====
    in_dim = graphs[0].x.shape[1]
    model = PressureGNN(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ==== 学習ループ ====
    for epoch in range(1, num_epochs + 1):
        train_loss, train_data, train_pde = train_epoch(
            model, train_graphs, optimizer, device, lambda_pde
        )
        val_loss, val_data, val_pde = eval_epoch(
            model, val_graphs, device, lambda_pde
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.3e} "
            f"(data={train_data:.3e}, pde={train_pde:.3e}) "
            f"val_loss={val_loss:.3e} "
            f"(data={val_data:.3e}, pde={val_pde:.3e})"
        )

    # ==== 学習済みモデルの保存 ====
    out_path = "pressure_gnn_prototype.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")

    # ==== (オプション) テストスナップショットで Ax-b と x_true を確認 ====
    if hasattr(graphs[0], "x_true"):
        print("Checking relative error vs OpenFOAM solution (first graph)...")
        data0 = graphs[0].to(device)
        with torch.no_grad():
            x_pred = model(data0)
            x_true = data0.x_true
            rel_err = torch.norm(x_pred - x_true) / torch.norm(x_true)
            print(f"  ||x_pred - x_true|| / ||x_true|| = {rel_err.item():.3e}")


if __name__ == "__main__":
    main()

