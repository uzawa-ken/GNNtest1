#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習曲線と予測結果を可視化するスクリプト

Usage:
    python visualize_training.py
    python visualize_training.py --train-loss train_loss.dat --val-loss val_loss.dat
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_learning_curves(
    train_loss_file: str = "train_loss.dat",
    val_loss_file: str = "val_loss.dat",
    output_dir: str = "./plots",
):
    """
    学習曲線（損失の推移）を可視化

    Args:
        train_loss_file: 訓練損失ファイル
        val_loss_file: 検証損失ファイル
        output_dir: 出力ディレクトリ
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # データ読み込み
    try:
        train_data = np.loadtxt(train_loss_file, comments='#')
        val_data = np.loadtxt(val_loss_file, comments='#')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run training first to generate loss files.")
        return

    train_epochs = train_data[:, 0]
    train_loss = train_data[:, 1]
    train_data_loss = train_data[:, 2]
    train_pde_loss = train_data[:, 3]

    val_epochs = val_data[:, 0]
    val_loss = val_data[:, 1]
    val_data_loss = val_data[:, 2]
    val_pde_loss = val_data[:, 3]

    # プロットのスタイル設定
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    # 1. 全損失の推移
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_epochs, train_loss, label='Train Loss', linewidth=2, alpha=0.8)
    ax.plot(val_epochs, val_loss, label='Val Loss', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_loss.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/total_loss.png")
    plt.close()

    # 2. データ損失とPDE損失の分離プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # データ損失
    ax1.plot(train_epochs, train_data_loss, label='Train Data Loss', linewidth=2, alpha=0.8)
    ax1.plot(val_epochs, val_data_loss, label='Val Data Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Data Loss', fontsize=12)
    ax1.set_title('Data Loss (MSE)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # PDE損失
    ax2.plot(train_epochs, train_pde_loss, label='Train PDE Loss', linewidth=2, alpha=0.8)
    ax2.plot(val_epochs, val_pde_loss, label='Val PDE Loss', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('PDE Loss', fontsize=12)
    ax2.set_title('PDE Loss (Physics-Informed)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_pde_loss.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/data_pde_loss.png")
    plt.close()

    # 3. 損失の統計情報を表示
    print("\n=== Loss Statistics ===")
    print(f"Train Loss - Min: {train_loss.min():.3e}, Final: {train_loss[-1]:.3e}")
    print(f"Val Loss   - Min: {val_loss.min():.3e}, Final: {val_loss[-1]:.3e}")
    print(f"Best Val Loss at Epoch: {val_epochs[val_loss.argmin()].astype(int)}")


def plot_prediction_comparison(
    prediction_dir: str = "./predictions",
    output_dir: str = "./plots",
    max_plots: int = 5,
):
    """
    予測結果と真値の比較を可視化

    Args:
        prediction_dir: 予測結果ディレクトリ
        output_dir: 出力ディレクトリ
        max_plots: 最大プロット数
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pred_dir = Path(prediction_dir)

    if not pred_dir.exists():
        print(f"Prediction directory not found: {prediction_dir}")
        return

    # エラーファイルを探す
    error_files = sorted(pred_dir.glob("error_*.dat"))

    if not error_files:
        print("No error files found in prediction directory.")
        return

    print(f"\nFound {len(error_files)} error files.")

    for i, error_file in enumerate(error_files[:max_plots]):
        data = np.loadtxt(error_file, comments='#')

        if data.size == 0:
            continue

        cell_ids = data[:, 0].astype(int)
        errors = data[:, 1]
        predictions = data[:, 2]
        true_values = data[:, 3]

        # プロット
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 予測 vs 真値
        ax1.scatter(true_values, predictions, alpha=0.5, s=10)
        min_val = min(true_values.min(), predictions.min())
        max_val = max(true_values.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('True Values', fontsize=12)
        ax1.set_ylabel('Predicted Values', fontsize=12)
        ax1.set_title(f'Prediction vs True (Graph {i})', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 誤差の分布
        ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax2.set_xlabel('Prediction Error', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Error Distribution (Graph {i})', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # 統計情報を表示
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())
        rel_error = rmse / (np.abs(true_values).mean() + 1e-20)

        textstr = f'MAE: {mae:.3e}\nRMSE: {rmse:.3e}\nRel. Error: {rel_error:.3e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        output_path = f"{output_dir}/prediction_comparison_{i:04d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training results and predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--train-loss", type=str, default="train_loss.dat",
                        help="Path to training loss file")
    parser.add_argument("--val-loss", type=str, default="val_loss.dat",
                        help="Path to validation loss file")
    parser.add_argument("--prediction-dir", type=str, default="./predictions",
                        help="Directory containing prediction files")
    parser.add_argument("--output-dir", type=str, default="./plots",
                        help="Output directory for plots")
    parser.add_argument("--max-plots", type=int, default=5,
                        help="Maximum number of prediction comparison plots")

    args = parser.parse_args()

    print("=" * 60)
    print("Visualizing Training Results")
    print("=" * 60)

    # 学習曲線のプロット
    plot_learning_curves(args.train_loss, args.val_loss, args.output_dir)

    # 予測結果の比較プロット
    plot_prediction_comparison(args.prediction_dir, args.output_dir, args.max_plots)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
