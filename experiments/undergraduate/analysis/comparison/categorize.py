import pandas as pd

# ファイルの読み込み
category_df = pd.read_csv('category.csv')
pmc_df = pd.read_csv('PMC-MHLITE.csv')

# データセット名をキーにしたマッピング辞書を作成
pmc_map = pmc_df.set_index('dataset').to_dict('index')

# 1. 精度の転記 (PMC-MHLITE 列)
for i, ds in category_df['dataset'].items():
    if pd.notna(ds) and ds in pmc_map:
        category_df.at[i, 'PMC-MHLITE'] = round(pmc_map[ds]['best_accuracy'], 3)

# 2. パラメータ数の転記 (PMC-MHLITE.2 列)
for i, ds in category_df['dataset.2'].items():
    if pd.notna(ds) and ds in pmc_map:
        category_df.at[i, 'PMC-MHLITE.2'] = int(pmc_map[ds]['parameter_count'])

# 3. 実行時間の転記 (PMC-MHLITE.3 列)
for i, ds in category_df['dataset.3'].items():
    if pd.notna(ds) and ds in pmc_map:
        category_df.at[i, 'PMC-MHLITE.3'] = round(pmc_map[ds]['time_seconds'], 1)

# 結果を保存
category_df.to_csv('category_updated.csv', index=False)