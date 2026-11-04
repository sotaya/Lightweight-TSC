import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

def generate_mcm(csv_path, output_img="mcm_heatmap.png", output_csv="mcm_summary.csv"):
    # 1. データの読み込み
    df = pd.read_csv(csv_path)
    
    # 2. データのクレンジング
    # 0〜84行目が各データセットの個々のレコード（85、86行目の平均行などは除外）
    df_clean = df.iloc[0:85][["dataset", "MC-MHLF", "LITE", "MC-MHLITE", "PMC-MHLITE"]].copy()
    
    # 数値型に変換
    models = ["MC-MHLF", "LITE", "MC-MHLITE", "PMC-MHLITE"]
    for col in models:
        df_clean[col] = pd.to_numeric(df_clean[col])
        
    n_models = len(models)
    
    # マトリクス保存用の配列を初期化
    annot_labels = np.empty((n_models, n_models), dtype=object)
    val_matrix = np.zeros((n_models, n_models)) # 色付け用の勝率マトリクス
    
    # 3. ペアワイズ比較の計算（Win/Tie/Loss と ウィルコクソン検定）
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                annot_labels[i, j] = "-"
                val_matrix[i, j] = 0.5  # 対角成分
            else:
                wins = np.sum(df_clean[m1] > df_clean[m2])
                losses = np.sum(df_clean[m1] < df_clean[m2])
                ties = np.sum(df_clean[m1] == df_clean[m2])
                
                # ウィルコクソンの符号付き順位検定でp値を算出
                try:
                    _, p_val = wilcoxon(df_clean[m1], df_clean[m2], alternative='two-sided')
                    p_str = f"p={p_val:.2e}" if p_val < 0.01 else f"p={p_val:.3f}"
                except:
                    p_str = "p=N/A"
                
                # セル内のテキスト（勝/分/敗 と p値）
                annot_labels[i, j] = f"{wins}/{ties}/{losses}\n{p_str}"
                # 勝率をセルの背景色の基準にする
                val_matrix[i, j] = wins / (wins + losses + 1e-6)
                
    # 4. ヒートマップとしての可視化
    plt.tight_layout()
    sns.heatmap(
        val_matrix, 
        annot=annot_labels, 
        fmt="", 
        cmap="coolwarm",
        xticklabels=models, 
        yticklabels=models, 
        cbar_kws={'label': 'Win Ratio (Row vs Col)'},
        annot_kws={"size": 10}
    )
    plt.title("Multi-Comparison Matrix (MCM)\nCell: Wins/Ties/Losses & Wilcoxon p-value", fontsize=12, pad=10)
    plt.xlabel("Compared Model (Column)", fontsize=10)
    plt.ylabel("Base Model (Row)", fontsize=10)
    
    # 画像の保存
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. CSVファイルとして集計結果をエクスポート
    mcm_df = pd.DataFrame(annot_labels, index=models, columns=models)
    mcm_df.to_csv(output_csv)
    
    print(f"✓ MCM Heatmap saved to: {output_img}")
    print(f"✓ MCM Summary CSV saved to: {output_csv}")

if __name__ == "__main__":
    generate_mcm("experiment/iiai_aai2026/performance/valid/analysis/overall/performance.csv")