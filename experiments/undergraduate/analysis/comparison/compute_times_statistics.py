#!/usr/bin/env python3
"""
trainのexecution_times.csvとtestのinference_times.csvから
1~5の結果の平均と分散を計算し、times_pf.csvに追加するプログラム
"""

import os
import pandas as pd
import numpy as np
import csv
import io

# パス設定
VALID_DIR = "/home/sotaya/gitresearch/Lightweight-TSC/experiment/ieee_smc2026/valid"
OUTPUT_FILE = "/home/sotaya/gitresearch/Lightweight-TSC/experiment/ieee_smc2026/performance/overall_macd/times_pf.csv"

# モデル名（出力列順）
MODELS = ["MC-MHLF", "LITE", "MC-MHLITE", "MC2-MHLITE", "PMC-MHLITE", "PMC2-MHLITE"]
MODEL_DIRS = {
    "MC-MHLF": "mc_mhlf",
    "LITE": "lite",
    "MC-MHLITE": "mc_mhlite",
    "MC2-MHLITE": "mc2_mhlite",
    "PMC-MHLITE": "pmc_mhlite",
    "PMC2-MHLITE": "pmc2_mhlite"
}

OUTPUT_DIRS = {
    "mc_mhlf": "mc_mhlf_train",
    "lite": "lite_train",
    "mc_mhlite": "mc_mhlite_train",
    "mc2_mhlite": "mc2_mhlite_train",
    "pmc_mhlite": "pmc_mhlite_train",
    "pmc2_mhlite": "pmc2_mhlite_train"
}

TEST_OUTPUT_DIRS = {
    "mc_mhlf": "mc_mhlf_test",
    "lite": "lite_test",
    "mc_mhlite": "mc_mhlite_test",
    "mc2_mhlite": "mc2_mhlite_test",
    "pmc_mhlite": "pmc_mhlite_test",
    "pmc2_mhlite": "pmc2_mhlite_test"
}

# trainとtestのoutput名パターン
TRAIN_OUTPUT_PATTERNS = {
    "mc_mhlf": "mcoutput",
    "lite": "loutput",
    "mc_mhlite": "mcloutput",
    "mc2_mhlite": "mc2loutput",
    "pmc_mhlite": "pmcloutput",
    "pmc2_mhlite": "pmc2loutput"
}

TEST_OUTPUT_PATTERNS = {
    "mc_mhlf": "test_mcoutput",
    "lite": "test_loutput",
    "mc_mhlite": "test_mcloutput",
    "mc2_mhlite": "test_mc2loutput",
    "pmc_mhlite": "test_pmcloutput",
    "pmc2_mhlite": "test_pmc2loutput"
}


def load_existing_dataset_metadata():
    """既存CSVからデータセット順とメタ情報を読み取る"""
    if not os.path.exists(OUTPUT_FILE):
        raise FileNotFoundError(f"{OUTPUT_FILE} not found")

    with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    dataset_order = []
    metadata = {}

    for row in rows:
        if row["Time"] not in ["average_learning_time", "average_inference_time"]:
            continue

        dataset = row["Dataset"].strip()
        if not dataset:
            continue

        if dataset not in metadata:
            metadata[dataset] = {
                "SampleCount": row.get("SampleCount", ""),
                "Length": row.get("Length", ""),
                "TotalComp": row.get("TotalComp", "")
            }
            dataset_order.append(dataset)

    return dataset_order, metadata


def collect_times(base_dir, output_dir, pattern, csv_name, value_column):
    """各モデルのrun1-5から時刻値を集計する"""
    times_data = {}
    model_base = os.path.join(VALID_DIR, base_dir, output_dir)

    for run_id in range(1, 6):
        output_path = os.path.join(model_base, f"{pattern}{run_id}")
        csv_file = os.path.join(output_path, csv_name)

        if not os.path.exists(csv_file):
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        for _, row in df.iterrows():
            dataset = str(row.get("dataset", "")).strip()
            if not dataset:
                continue

            value = row.get(value_column)
            if pd.isna(value):
                continue

            times_data.setdefault(dataset, []).append(float(value))

    return times_data


def compute_average(times_list):
    if not times_list:
        return None
    return float(np.mean(times_list))


def compute_variance(times_list):
    if not times_list:
        return None
    if len(times_list) == 1:
        return 0.0
    return float(np.var(np.array(times_list), ddof=1))


def format_value(value):
    if value is None:
        return ""
    return value


def build_output_rows(dataset_order, metadata, all_train_times, all_test_times):
    rows = []

    for dataset in dataset_order:
        meta = metadata.get(dataset, {})
        row_base = {
            "Dataset": dataset,
            "SampleCount": meta.get("SampleCount", ""),
            "Length": meta.get("Length", ""),
            "TotalComp": meta.get("TotalComp", "")
        }

        avg_train = {**row_base, "Time": "average_learning_time"}
        var_train = {**row_base, "Time": "learning_time_variance"}
        avg_test = {**row_base, "Time": "average_inference_time"}
        var_test = {**row_base, "Time": "inference_time_variance"}

        for model in MODELS:
            train_times = all_train_times.get(model, {}).get(dataset, [])
            test_times = all_test_times.get(model, {}).get(dataset, [])

            avg_train[model] = format_value(compute_average(train_times))
            var_train[model] = format_value(compute_variance(train_times))
            avg_test[model] = format_value(compute_average(test_times))
            var_test[model] = format_value(compute_variance(test_times))

        rows.extend([avg_train, var_train, avg_test, var_test])

    return rows


def main():
    try:
        dataset_order, metadata = load_existing_dataset_metadata()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("データを読み込んでいます...")
    all_train_times = {
        model: collect_times(
            MODEL_DIRS[model],
            OUTPUT_DIRS[MODEL_DIRS[model]],
            TRAIN_OUTPUT_PATTERNS[MODEL_DIRS[model]],
            "execution_times.csv",
            "time_seconds"
        )
        for model in MODELS
    }
    all_test_times = {
        model: collect_times(
            MODEL_DIRS[model],
            TEST_OUTPUT_DIRS[MODEL_DIRS[model]],
            TEST_OUTPUT_PATTERNS[MODEL_DIRS[model]],
            "inference_times.csv",
            "total_inference_time_seconds"
        )
        for model in MODELS
    }

    new_rows = build_output_rows(dataset_order, metadata, all_train_times, all_test_times)

    header = ["Dataset", "SampleCount", "Length", "TotalComp", "Time"] + MODELS

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in new_rows:
            writer.writerow(row)

    print(f"✓ {OUTPUT_FILE} を更新しました")
    print("✓ train/test の1~5 の平均と分散を再計算しました")


if __name__ == "__main__":
    main()
