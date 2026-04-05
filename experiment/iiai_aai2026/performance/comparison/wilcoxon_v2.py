import pandas as pd
from multi_comp_matrix import MCM

df = pd.read_csv('experiment/iiai_aai2026/performance/valid/analysis/overall/lite_fam.csv') 

df_numeric = df.select_dtypes(include=['number'])

MCM.compare(
    df_results=df_numeric,          
    output_dir="./experiment/iiai_aai2026/performance/valid/analysis/wilcoxon/",
    png_savename="model_comparison", 
    used_statistic="Accuracy",       
    pvalue_threshold=0.05            
)
