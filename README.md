# Pressure Poisson GNN Solver

Physics-Informed Graph Neural Network (GNN) for solving pressure Poisson equations from OpenFOAM simulations.

## 概要

このプロジェクトは、OpenFOAMの圧力ポアソン方程式をグラフニューラルネットワーク（GNN）を用いて解くプロトタイプです。物理法則（PDE損失）とデータ駆動学習を組み合わせたPhysics-Informed Neural Networksのアプローチを採用しています。

### 主な特徴

- **物理法則に基づいた学習**: PDE残差とメッシュ品質を考慮した損失関数
- **2つのGNNアーキテクチャ**:
  - `PressureGNN`: シンプルなGCNベース
  - `ImprovedPressureGNN`: Residual connections、Layer Normalization、Dropout付き
- **柔軟な設定**: コマンドライン引数で全てのハイパーパラメータを調整可能
- **Early Stopping**: 過学習を自動検出
- **チェックポイント機能**: ベストモデルの自動保存と学習再開
- **バッチ処理**: 複数グラフの効率的な並列学習
- **可視化ツール**: 学習曲線と予測結果の自動プロット
- **完全な再現性**: シード固定によるdeterministicな実行

## インストール

### 必要なライブラリ

```bash
pip install torch torch-geometric numpy matplotlib pytest
```

または、PyTorch Geometricの詳細なインストール方法については[公式ドキュメント](https://pytorch-geometric.readthedocs.io/)を参照してください。

## 使用方法

### 基本的な使い方

```bash
# デフォルト設定で学習
python pressure_gnn_prototype.py --data-dir ./gnn

# 改良版モデルで学習
python pressure_gnn_prototype.py --model-type improved --num-epochs 300

# バッチ処理を有効化
python pressure_gnn_prototype.py --batch-size 4 --hidden-dim 128
```

### 主要なコマンドライン引数

#### データ設定
- `--data-dir`: pEqn_*.datファイルのディレクトリ (デフォルト: `./gnn`)
- `--batch-size`: バッチサイズ (デフォルト: 1)

#### モデル設定
- `--model-type`: モデルタイプ (`basic` or `improved`, デフォルト: `basic`)
- `--hidden-dim`: 隠れ層の次元数 (デフォルト: 64)
- `--num-layers`: GNN層の数 (デフォルト: 3)
- `--dropout`: ドロップアウト率 (improved モデル用, デフォルト: 0.1)

#### 学習設定
- `--num-epochs`: エポック数 (デフォルト: 200)
- `--lr`: 学習率 (デフォルト: 0.001)
- `--lambda-pde`: PDE損失の重み (デフォルト: 1.0)

#### Early Stopping
- `--early-stopping-patience`: Early stoppingのpatience (デフォルト: 30)

#### その他
- `--seed`: ランダムシード (デフォルト: 42)
- `--checkpoint-dir`: チェックポイント保存ディレクトリ (デフォルト: `./checkpoints`)

### 使用例

```bash
# 基本的な学習（リアルタイムプロット有効）
python pressure_gnn_prototype.py --realtime-plot

# GAT (Graph Attention Networks) で学習
python pressure_gnn_prototype.py \
  --model-type gat \
  --hidden-dim 128 \
  --num-layers 4 \
  --num-heads 8 \
  --dropout 0.2 \
  --realtime-plot

# GraphSAGE で学習
python pressure_gnn_prototype.py \
  --model-type graphsage \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.1

# GIN (Graph Isomorphism Network) で高精度学習
python pressure_gnn_prototype.py \
  --model-type gin \
  --hidden-dim 256 \
  --num-layers 6 \
  --dropout 0.15 \
  --num-epochs 500 \
  --early-stopping-patience 50 \
  --realtime-plot

# TensorBoardを使用した高度なログ記録
python pressure_gnn_prototype.py \
  --use-tensorboard

# 物理損失の重みを調整
python pressure_gnn_prototype.py \
  --lambda-pde 2.0 \
  --lr 0.0005 \
  --realtime-plot \
  --plot-interval 5

# ヘルプを表示
python pressure_gnn_prototype.py --help
```

## 可視化

### リアルタイム可視化（学習中）

学習の進行状況をリアルタイムで監視できます：

#### matplotlibによる可視化

```bash
# リアルタイムプロットを有効化
python pressure_gnn_prototype.py --realtime-plot

# プロット更新間隔を調整（5エポックごと）
python pressure_gnn_prototype.py --realtime-plot --plot-interval 5
```

**表示される5つのグラフ**：
1. **Total Loss**: 訓練損失と検証損失の推移
2. **Data Loss**: データ損失（MSE）の推移
3. **PDE Loss**: PDE残差損失の推移（L_A + L_div）
4. **L_A Loss**: 行列残差損失の推移
5. **L_div Loss**: 発散項損失の推移

**注意事項**：
- GUIバックエンド（TkAgg）が必要です
- SSHやヘッドレス環境では動作しません（代わりにTensorBoardを使用）
- 学習中にウィンドウを閉じないでください

#### TensorBoardによる可視化

```bash
# TensorBoardログを有効化
python pressure_gnn_prototype.py --use-tensorboard

# 別のターミナルでTensorBoardを起動
tensorboard --logdir=runs

# ブラウザで http://localhost:6006 を開く
```

**TensorBoardの利点**：
- ブラウザベースで動作（SSH経由でも使用可能）
- より詳細なメトリクスとヒストグラム
- 複数の実験を比較可能
- スムージング機能付き

### 学習後の可視化

学習完了後、可視化スクリプトで詳細な分析ができます：

```bash
# 学習曲線と予測結果をプロット
python visualize_training.py

# カスタム設定
python visualize_training.py \
  --train-loss train_loss.dat \
  --val-loss val_loss.dat \
  --prediction-dir ./predictions \
  --output-dir ./plots
```

**生成されるプロット**：
- `total_loss.png`: 訓練・検証損失の推移
- `data_pde_loss.png`: データ損失とPDE損失の分離プロット
- `prediction_comparison_XXXX.png`: 予測 vs 真値の散布図と誤差分布

## テスト

```bash
# 全テストを実行
pytest test_pressure_gnn.py -v

# 特定のテストクラスのみ実行
pytest test_pressure_gnn.py::TestGNNModel -v
```

## ファイル構成

```
GNNtest1/
├── pressure_gnn_prototype.py  # メインスクリプト
├── visualize_training.py      # 可視化ツール
├── test_pressure_gnn.py        # ユニットテスト
├── README.md                   # このファイル
├── .gitignore                  # Git無視設定
├── gnn/                        # データディレクトリ
│   ├── pEqn_0.dat
│   ├── pEqn_1.dat
│   └── ...
├── checkpoints/                # チェックポイント（自動生成）
│   ├── best_model.pt
│   └── checkpoint_epoch_XXXX.pt
├── predictions/                # 予測結果（自動生成）
│   ├── prediction_0000.dat
│   └── error_0000.dat
└── plots/                      # プロット（自動生成）
    ├── total_loss.png
    └── ...
```

## データフォーマット

### 入力: pEqn_*.dat

3つのフォーマットに対応：

#### フォーマットA（拡張+体積あり）
```
nCells 1000
nFaces 2500
CELLS
id x y z diag b skew nonOrtho aspect diagContrast V cellSize sizeJump
0 0.0 0.0 0.0 1.5 0.3 0.1 0.05 1.2 0.9 0.001 0.1 0.05
...
EDGES
faceId lowerCell upperCell lowerCoeff upperCoeff
0 0 1 -0.5 -0.5
...
```

#### フォーマットB（拡張、体積なし）
```
id x y z diag b skew nonOrtho aspect diagContrast cellSize sizeJump
```
体積は `cellSize^3` で近似されます。

#### フォーマットC（最小版）
```
id x y z diag b
```
`cellSize` と `aspectRatio` は近傍セルから自動推定されます。

### 出力: prediction_*.dat

```
# Prediction for graph 0
# nCells 1000
0 1.234567890123e-01
1 2.345678901234e-01
...
```

## 損失関数

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{pde}} \mathcal{L}_{\text{PDE}}
$$

### データ損失
$$
\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} (x_{\text{pred},i} - x_{\text{true},i})^2
$$

### PDE損失
$$
\mathcal{L}_{\text{PDE}} = \mathcal{L}_A + \mathcal{L}_{\text{div}}
$$

$$
\mathcal{L}_A = \frac{1}{N} \sum_{i=1}^{N} (Ax_{\text{pred}} - b)_i^2
$$

$$
\mathcal{L}_{\text{div}} = \frac{1}{N} \sum_{i=1}^{N} w_i \left( \frac{(Ax_{\text{pred}} - b)_i}{V_i} \right)^2
$$

ここで：
- $A$: 係数行列
- $b$: 右辺ベクトル
- $V_i$: セルの体積
- $w_i$: メッシュ品質重み（cellSizeとaspectRatioから計算）

## モデルアーキテクチャ

本プロジェクトでは、5種類のGNNアーキテクチャをサポートしています：

### 1. PressureGNN (Basic) - `--model-type basic`
- **ベースライン**: シンプルなGCNスタック
- **特徴**: 高速で軽量
- **適用**: 小規模問題や初期検証に最適

### 2. ImprovedPressureGNN - `--model-type improved`
- **特徴**:
  - Residual (Skip) Connections
  - Layer Normalization
  - Dropout Regularization
- **適用**: より深いネットワークに対応、大規模問題や複雑な問題に最適

### 3. GATPressureGNN - `--model-type gat`
- **Graph Attention Networks (GAT)**
- **特徴**:
  - Multi-head attention mechanism
  - 各エッジの重要度を動的に学習
  - より表現力の高いグラフ畳み込み
- **適用**: 不均一なメッシュや複雑な境界条件を持つ問題

### 4. GraphSAGEPressureGNN - `--model-type graphsage`
- **GraphSAGE (Sample and Aggregate)**
- **特徴**:
  - 近傍ノードのサンプリングと集約
  - スケーラビリティに優れる
  - 大規模グラフでも効率的
- **適用**: 大規模メッシュや計算効率を重視する場合

### 5. GINPressureGNN - `--model-type gin`
- **Graph Isomorphism Network (GIN)**
- **特徴**:
  - Weisfeiler-Lehman testと同等の表現力
  - グラフ構造の識別能力が高い
  - 理論的に最も強力なGNNの一つ
- **適用**: 複雑なトポロジーや高精度が要求される問題

### モデル選択のガイドライン

```bash
# まずはベースラインで動作確認
python pressure_gnn_prototype.py --model-type basic

# 精度を上げたい場合は improved を試す
python pressure_gnn_prototype.py --model-type improved --num-layers 5

# 複雑なメッシュにはGATが有効
python pressure_gnn_prototype.py --model-type gat --num-heads 8 --hidden-dim 128

# 大規模問題にはGraphSAGE
python pressure_gnn_prototype.py --model-type graphsage --hidden-dim 256

# 最高精度を目指すならGIN
python pressure_gnn_prototype.py --model-type gin --hidden-dim 256 --num-layers 6
```

## トラブルシューティング

### データが見つからない
```
RuntimeError: pEqn_*.dat が見つかりません
```
→ `--data-dir` オプションでデータディレクトリを正しく指定してください。

### メモリ不足
```
RuntimeError: CUDA out of memory
```
→ `--batch-size` を小さくするか、`--hidden-dim` を減らしてください。

### 学習が収束しない
- `--lr` を小さくしてみてください（例: 0.0001）
- `--lambda-pde` を調整してください
- `--model-type improved` を試してください

## パフォーマンスチューニング

### GPUの活用
CUDAが利用可能な場合、自動的にGPUを使用します。

### バッチサイズの調整
```bash
# GPUメモリが十分にある場合
python pressure_gnn_prototype.py --batch-size 8
```

### モデルサイズの調整
```bash
# より大きなモデル（精度重視）
python pressure_gnn_prototype.py --hidden-dim 256 --num-layers 6

# より小さなモデル（速度重視）
python pressure_gnn_prototype.py --hidden-dim 32 --num-layers 2
```

## 引用

このコードを研究で使用する場合は、以下のように引用してください：

```bibtex
@software{pressure_gnn_solver,
  title = {Pressure Poisson GNN Solver},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/GNNtest1}
}
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

Issue や Pull Request を歓迎します！

## 今後の改善予定

- [ ] GAT (Graph Attention Networks) のサポート
- [ ] より高度な物理制約の追加
- [ ] マルチGPU対応
- [ ] ONNX形式でのモデルエクスポート
- [ ] リアルタイム推論モード
