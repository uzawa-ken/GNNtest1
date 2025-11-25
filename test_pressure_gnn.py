#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for pressure_gnn_prototype.py

Run with: pytest test_pressure_gnn.py -v
"""

import os
import tempfile
import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from pressure_gnn_prototype import (
    parse_cells_format_a,
    parse_cells_format_b,
    parse_cells_format_c,
    estimate_cell_properties,
    parse_edges,
    apply_A_to_x,
    compute_losses,
    PressureGNN,
    set_random_seed,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
)


class TestCellParsing:
    """セル情報解析のテスト"""

    def test_parse_cells_format_a(self):
        """フォーマットA（拡張+体積あり）の解析テスト"""
        cell_lines = [
            "0 0.0 0.0 0.0 1.0 0.5 0.1 0.2 1.5 0.8 0.001 0.1 0.05",
            "1 1.0 0.0 0.0 1.0 0.6 0.1 0.2 1.6 0.9 0.002 0.11 0.06",
        ]
        coords, diag, bvec, volume, cell_size, aspect_ratio = parse_cells_format_a(
            cell_lines, nCells=2
        )

        assert coords.shape == (2, 3)
        assert diag.shape == (2,)
        assert bvec.shape == (2,)
        assert volume.shape == (2,)
        assert cell_size.shape == (2,)
        assert aspect_ratio.shape == (2,)

        # 値のチェック
        np.testing.assert_array_almost_equal(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(coords[1], [1.0, 0.0, 0.0])
        assert diag[0] == 1.0
        assert bvec[0] == 0.5
        assert volume[0] == 0.001
        assert cell_size[0] == 0.1
        assert aspect_ratio[0] == 1.5

    def test_parse_cells_format_b(self):
        """フォーマットB（拡張、体積なし）の解析テスト"""
        cell_lines = [
            "0 0.0 0.0 0.0 1.0 0.5 0.1 0.2 1.5 0.8 0.1 0.05",
            "1 1.0 0.0 0.0 1.0 0.6 0.1 0.2 1.6 0.9 0.11 0.06",
        ]
        coords, diag, bvec, volume, cell_size, aspect_ratio = parse_cells_format_b(
            cell_lines, nCells=2
        )

        assert coords.shape == (2, 3)
        assert diag.shape == (2,)
        assert bvec.shape == (2,)
        assert volume.shape == (2,)
        assert cell_size.shape == (2,)

        # 体積はcellSize^3で近似されるべき
        np.testing.assert_array_almost_equal(volume[0], 0.1 ** 3)
        np.testing.assert_array_almost_equal(volume[1], 0.11 ** 3)

    def test_parse_cells_format_c(self):
        """フォーマットC（最小版）の解析テスト"""
        cell_lines = [
            "0 0.0 0.0 0.0 1.0 0.5",
            "1 1.0 0.0 0.0 1.0 0.6",
        ]
        coords, diag, bvec = parse_cells_format_c(cell_lines, nCells=2)

        assert coords.shape == (2, 3)
        assert diag.shape == (2,)
        assert bvec.shape == (2,)
        np.testing.assert_array_almost_equal(coords[0], [0.0, 0.0, 0.0])
        assert diag[0] == 1.0
        assert bvec[0] == 0.5


class TestCellPropertyEstimation:
    """セルプロパティ推定のテスト"""

    def test_estimate_cell_properties(self):
        """近傍からcellSizeとaspectRatioを推定"""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        neighbors = [
            [1, 2],  # セル0の近傍
            [0],     # セル1の近傍
            [0],     # セル2の近傍
        ]

        cell_size, aspect_ratio, volume = estimate_cell_properties(
            coords, neighbors, nCells=3
        )

        assert cell_size.shape == (3,)
        assert aspect_ratio.shape == (3,)
        assert volume.shape == (3,)

        # セル0の近傍距離は1.0と2.0なので平均1.5
        assert np.isclose(cell_size[0], 1.5)
        # アスペクト比は2.0/1.0=2.0
        assert np.isclose(aspect_ratio[0], 2.0)


class TestEdgeParsing:
    """エッジ情報解析のテスト"""

    def test_parse_edges(self):
        """エッジ情報の解析テスト"""
        edge_lines = [
            "0 0 1 -0.5 -0.5",
            "1 1 2 -0.6 -0.6",
        ]
        lower_ids, upper_ids, lower_vals, upper_vals = parse_edges(
            edge_lines, nFaces=2
        )

        assert lower_ids.shape == (2,)
        assert upper_ids.shape == (2,)
        assert lower_vals.shape == (2,)
        assert upper_vals.shape == (2,)

        assert lower_ids[0] == 0
        assert upper_ids[0] == 1
        assert lower_vals[0] == -0.5
        assert upper_vals[0] == -0.5


class TestMatrixOperations:
    """行列演算のテスト"""

    def test_apply_A_to_x(self):
        """Ax計算のテスト"""
        # 簡単な2x2システム
        diag = torch.tensor([2.0, 3.0])
        lower_idx = torch.tensor([0])
        upper_idx = torch.tensor([1])
        lower_val = torch.tensor([-1.0])
        upper_val = torch.tensor([-1.0])

        data = Data()
        data.diag = diag
        data.lower_index = lower_idx
        data.upper_index = upper_idx
        data.lower_val = lower_val
        data.upper_val = upper_val

        x = torch.tensor([1.0, 2.0])
        Ax = apply_A_to_x(data, x)

        # A = [[2, -1], [-1, 3]]
        # Ax = [2*1 + (-1)*2, (-1)*1 + 3*2] = [0, 5]
        expected = torch.tensor([0.0, 5.0])
        torch.testing.assert_close(Ax, expected)


class TestLossComputation:
    """損失計算のテスト"""

    def test_compute_losses_with_x_true(self):
        """x_trueがある場合の損失計算テスト"""
        # ダミーデータを作成
        data = Data()
        data.diag = torch.tensor([1.0, 1.0])
        data.b = torch.tensor([1.0, 1.0])
        data.lower_index = torch.tensor([0])
        data.upper_index = torch.tensor([1])
        data.lower_val = torch.tensor([0.0])
        data.upper_val = torch.tensor([0.0])
        data.volume = torch.tensor([1.0, 1.0])
        data.cell_size = torch.tensor([1.0, 1.0])
        data.aspect_ratio = torch.tensor([1.0, 1.0])
        data.x_true = torch.tensor([1.0, 1.0])

        x_pred = torch.tensor([1.0, 1.0])
        total_loss, data_loss, pde_loss = compute_losses(
            data, x_pred, lambda_pde=1.0
        )

        # x_pred == x_true なのでdata_lossは0に近いはず
        assert data_loss < 1e-6
        assert total_loss.item() >= 0  # 損失は非負


class TestGNNModel:
    """GNNモデルのテスト"""

    def test_pressure_gnn_forward(self):
        """GNNの順伝播テスト"""
        model = PressureGNN(in_dim=7, hidden_dim=16, num_layers=2)

        # ダミーグラフデータ
        x = torch.randn(5, 7)  # 5ノード、7次元特徴
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, num_nodes=5)
        output = model(data)

        assert output.shape == (5,)  # 各ノードに1つの出力


class TestRandomSeed:
    """ランダムシード設定のテスト"""

    def test_set_random_seed(self):
        """シード設定による再現性テスト"""
        set_random_seed(42)
        val1 = torch.rand(1).item()

        set_random_seed(42)
        val2 = torch.rand(1).item()

        assert val1 == val2  # 同じシードで同じ値が生成される


class TestEarlyStopping:
    """Early Stoppingのテスト"""

    def test_early_stopping_triggers(self):
        """Early stoppingがトリガーされるテスト"""
        early_stopping = EarlyStopping(patience=3, verbose=False)

        # 改善がない場合
        assert not early_stopping(1.0)
        assert not early_stopping(1.0)
        assert not early_stopping(1.0)
        assert early_stopping(1.0)  # patience=3で4回目にトリガー

    def test_early_stopping_reset(self):
        """Early stoppingが改善時にリセットされるテスト"""
        early_stopping = EarlyStopping(patience=3, verbose=False)

        assert not early_stopping(1.0)
        assert not early_stopping(0.9)  # 改善
        assert not early_stopping(0.9)
        assert not early_stopping(0.9)
        assert not early_stopping(0.9)
        assert early_stopping(0.9)  # patience=3で4回目にトリガー

    def test_early_stopping_disabled(self):
        """patience=0でEarly stoppingが無効化されるテスト"""
        early_stopping = EarlyStopping(patience=0, verbose=False)

        assert not early_stopping(1.0)
        assert not early_stopping(1.0)
        assert not early_stopping(1.0)  # 常にFalse


class TestCheckpoint:
    """チェックポイント保存・読み込みのテスト"""

    def test_save_and_load_checkpoint(self):
        """チェックポイントの保存と読み込みテスト"""
        model = PressureGNN(in_dim=7, hidden_dim=16, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")

            # 保存
            save_checkpoint(
                model, optimizer, None, epoch=10,
                train_loss=0.5, val_loss=0.3, checkpoint_path=checkpoint_path
            )

            assert os.path.exists(checkpoint_path)

            # 新しいモデルにロード
            new_model = PressureGNN(in_dim=7, hidden_dim=16, num_layers=2)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            epoch = load_checkpoint(checkpoint_path, new_model, new_optimizer)

            assert epoch == 10

            # パラメータが一致するか確認
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                torch.testing.assert_close(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
