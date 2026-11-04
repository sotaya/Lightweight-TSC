import pandas as pd
import numpy as np
import re

def update_category_from_performance(performance_path, category_path, output_path):
    """
    performance.csvを参照してcategory.csvのacc, parameter, timeを更新し、
    新しいCSVファイルとして保存する。

    Args:
        performance_path (str): performance.csvへのパス
        category_path (str): category.csvへのパス
        output_path (str): 更新結果を保存するCSVファイルへのパス
    """
    try:
        # --- 1. performance.csvの読み込みと整形 ---
        perf_df = pd.read_csv(performance_path, header=0)
        
        def extract_section_data(df, start_col_name, dataset_col_pattern):
            section_data = {}
            start_idx = df.columns.get_loc(start_col_name)
            dataset_col = None
            
            # Find the correct dataset column for this section
            for col in df.columns[start_idx:]:
                if re.match(dataset_col_pattern, col):
                    dataset_col = col
                    break
            
            if not dataset_col:
                return section_data

            # Find the end of the section (next empty column or end of df)
            end_idx = len(df.columns)
            for i in range(df.columns.get_loc(dataset_col) + 1, len(df.columns)):
                if df.columns[i].startswith('Unnamed:'):
                    end_idx = i
                    break
            
            section_cols = [dataset_col] + df.columns[df.columns.get_loc(dataset_col) + 1 : end_idx].tolist()
            section_df = df[section_cols].dropna(subset=[dataset_col])
            section_df = section_df.set_index(dataset_col)
            section_df.columns = [c.split('.')[0] for c in section_df.columns]
            return section_df.to_dict('index')

        acc_data = extract_section_data(perf_df, 'acc', r'^dataset$')
        param_data = extract_section_data(perf_df, 'parameter', r'^dataset\.')
        time_data = extract_section_data(perf_df, 'time', r'^dataset\.')

        models_to_update = ['MC-MHLF', 'LITE', 'MC-MHLITE', 'PMC-MHLITE', 'MC2-MHLITE', 'PMC2-MHLITE']

        # --- 2. category.csvの読み込み ---
        df_cat = pd.read_csv(category_path)

        # --- 列のマッピングを動的に作成 ---
        def get_col_map(df, section_keyword):
            col_map = {}
            try:
                start_idx = df.columns.get_loc(section_keyword)
                for col in df.columns[start_idx:]:
                    if col.startswith('Unnamed:'): break
                    model_name = col.split('.')[0]
                    if model_name in models_to_update:
                        col_map[model_name] = col
            except KeyError:
                pass # section not found
            return col_map

        acc_col_map = get_col_map(df_cat, 'dataset')
        param_col_map = get_col_map(df_cat, 'parameter_count')
        time_col_map = get_col_map(df_cat, 'time_seconds')

        # --- 3. category.csvのデータを更新 ---
        for index, row in df_cat.iterrows():
            dataset_name = row['dataset']
            if pd.notna(dataset_name) and "ave_" not in dataset_name:
                for model in models_to_update:
                    if dataset_name in acc_data and model in acc_data[dataset_name]:
                        df_cat.loc[index, acc_col_map.get(model, model)] = acc_data[dataset_name][model]
                    if dataset_name in param_data and model in param_data[dataset_name]:
                        df_cat.loc[index, param_col_map.get(model, f'{model}.2')] = param_data[dataset_name][model]
                    if dataset_name in time_data and model in time_data[dataset_name]:
                        df_cat.loc[index, time_col_map.get(model, f'{model}.3')] = time_data[dataset_name][model]

        # --- 4. 更新したDataFrameをCSVに出力 ---
        df_cat.to_csv(output_path, index=False)
        print(f"更新が完了し、'{output_path}' に保存しました。")

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。 {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


# --- プログラムの実行 ---
# ファイルパスを定義します。環境に合わせて適宜変更してください。
performance_file = '/home/sotaya/gitresearch/Lightweight-TSC/experiment/ieee_smc2026/performance/overall_macd/performance.csv'
category_file = '/home/sotaya/gitresearch/Lightweight-TSC/experiment/ieee_smc2026/performance/overall_macd/category.csv'
output_file = '/home/sotaya/gitresearch/Lightweight-TSC/experiment/ieee_smc2026/performance/overall_macd/category_updated.csv'

# 関数を実行してCSVを更新
update_category_from_performance(performance_file, category_file, output_file)
