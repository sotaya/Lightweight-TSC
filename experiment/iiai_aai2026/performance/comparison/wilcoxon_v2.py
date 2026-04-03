import pandas as pd
from multi_comp_matrix import MCM

# 1. データの読み込み
df = pd.read_csv('your_results.csv') 

# 2. 比較の実行と保存
MCM.compare(
    df_results=df,
    output_dir="./output/",
    png_savename="model_comparison", # 画像として保存
    used_statistic="Accuracy",       # 指標の名前（表示用）
    pvalue_threshold=0.05            # 有意水準
)