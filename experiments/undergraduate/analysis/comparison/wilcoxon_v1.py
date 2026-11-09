import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def create_wilcoxon_plot(csv_file, output_file):
    # データの読み込み
    df = pd.read_csv(csv_file)

    df_subset = df.iloc[0:85].copy()

    # モデル名を取得
    model_x = df.columns[2]
    model_y = df.columns[1]

    x_data = df_subset[model_x]
    y_data = df_subset[model_y]

    stat, p_value = stats.wilcoxon(y_data, x_data)

    wins = (y_data > x_data).sum()
    losses = (y_data < x_data).sum()
    ties = (y_data == x_data).sum()

    differences = y_data - x_data
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_scatter = axes[0]
    colors = ['red' if y > x else 'blue' if y < x else 'green' 
              for x, y in zip(x_data, y_data)]
    ax_scatter.scatter(x_data, y_data, c=colors, alpha=0.7, edgecolors='black', s=50)
    ax_scatter.plot([0, 1.05], [0, 1.05], 'k-', zorder=0)

    stats_text = (f"{model_y} Wins: {wins}\n"
                  f"{model_x} Wins: {losses}\n"
                  f"Ties: {ties}\n\n"
                  f"Wilcoxon p-value:\n{p_value:.2e}")
    
    ax_scatter.text(0.05, 0.95, stats_text, transform=ax_scatter.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax_scatter.set_xlabel(f"{model_x}", fontsize=12)
    ax_scatter.set_ylabel(f"{model_y}", fontsize=12)
    ax_scatter.set_title(f"Pairwise Comparison\n({model_y} vs {model_x})")
    ax_scatter.set_xlim(0, 1.02)
    ax_scatter.set_ylim(0, 1.02)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    ax_scatter.set_aspect('equal')
    ax_hist = axes[1]
    
    # ヒストグラムの描画
    ax_hist.hist(differences, bins=15, color='gray', edgecolor='black', alpha=0.7)
    ax_hist.axvline(0, color='black', linestyle='-', linewidth=1)
    mean_diff = np.mean(differences)
    ax_hist.axvline(mean_diff, color='red', linestyle='--', linewidth=1.5, label=f'Mean Diff: {mean_diff:.3f}')
    ax_hist.set_xlabel(f"Difference ({model_y} - {model_x})", fontsize=12)
    ax_hist.set_ylabel("Frequency", fontsize=12)
    ax_hist.set_title("Distribution of Differences")
    ax_hist.legend()
    ax_hist.grid(True, linestyle=':', alpha=0.5)

    # 5. 画像の保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"検定結果付きグラフを {output_file} に保存しました。")
    print(f"p値: {p_value}")

# 実行
create_wilcoxon_plot('experiment/ieee_smc2026/performance/overall_macd/models_pf.csv', 'wilcoxon_plot.png')