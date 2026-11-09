import os
import pandas as pd
from multi_comp_matrix import MCM

# 1. データの読み込み
df = pd.read_csv('experiment/ieee_smc2026/performance/overall_macd/lite_models_pf.csv') 
df = df.set_index('Dataset')
os.makedirs("./output/", exist_ok=True)

# 2. 比較の実行と保存
MCM.compare(
    df_results=df,
    output_dir="./output/",
    png_savename="model_comparison", 
    used_statistic="Accuracy",       
    pvalue_threshold=0.05,
    font_size=20             
)