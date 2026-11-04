# coding:utf-8
import time
import pathlib
import pickle
import random

import numpy as np
import sklearn.metrics

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

SEED = 42
rpy.set_seed(SEED)
rpy.verbosity(0)  # 学習中の進捗ログを抑制
np.random.seed(SEED)
random.seed(SEED)

RESERVOIR_UNITS = 500
SPECTRAL_RADIUS = 0.9
LEAKING_RATE = 0.3
INPUT_SCALING = 1.0
RIDGE_COEF = 1e-6


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------
def normalization(data):
    """サンプルごとに (値 - 平均) / 標準偏差 で正規化する"""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / std


def load_raw(path, dataset_name, sel_mode):
    """1ファイル分の生データ (特徴量 x, ラベル y) を読み込む"""
    data_path = pathlib.Path(path, dataset_name + sel_mode)
    data = np.loadtxt(data_path, delimiter=",")
    x = normalization(data[:, 1:])
    y = data[:, 0]
    return x, y


def one_hot(indices, n_classes):
    return np.eye(n_classes)[indices.astype(int)]


def load_dataset(path, dataset_name):
    
    x_train, y_train_raw = load_raw(path, dataset_name, "_TRAIN")
    x_test, y_test_raw = load_raw(path, dataset_name, "_TEST")

    classes = np.unique(y_train_raw)
    n_classes = len(classes)
    label_to_index = {label: i for i, label in enumerate(classes)}

    y_train_idx = np.array([label_to_index[v] for v in y_train_raw])
    # test に train 未出現のクラスがあれば、そのサンプルは学習不可能なので除外する
    known_mask = np.array([v in label_to_index for v in y_test_raw])
    if not known_mask.all():
        n_dropped = (~known_mask).sum()
        print(f"  警告: test に train 未出現のラベルが {n_dropped} 件あったため除外します", flush=True)
    x_test, y_test_raw = x_test[known_mask], y_test_raw[known_mask]
    y_test_idx = np.array([label_to_index[v] for v in y_test_raw])

    y_train = one_hot(y_train_idx, n_classes)
    y_test = one_hot(y_test_idx, n_classes)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    return x_train, y_train, x_test, y_test, n_classes


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------
def generate_model():
    reservoir = Reservoir(
        RESERVOIR_UNITS,
        sr=SPECTRAL_RADIUS,
        lr=LEAKING_RATE,
        input_scaling=INPUT_SCALING,
        seed=SEED,
    )
    readout = Ridge(ridge=RIDGE_COEF)
    return reservoir, readout


def run_reservoir(reservoir, x):
    """
    x 内の全サンプルを reservoir に流し、各サンプルの状態の時間平均を返す。
    Input shape:  (n_samples, timesteps, 1)
    Output shape: (n_samples, RESERVOIR_UNITS)

    reservoirpy の run() はサンプルのリストを渡すと、各系列を同じ初期状態から
    実行したうえで内部的に並列化 (workers) してくれるため、Python 側で
    1 サンプルずつ for ループを回して手動リセットする必要はない。
    """
    states = reservoir.run(list(x), reset=True, workers=-1)
    return np.array([s.mean(axis=0) for s in states])


def accuracy(y_pred, y_true):
    return sklearn.metrics.accuracy_score(
        np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)
    )


def mse_loss(y_pred, y_true):
    return sklearn.metrics.mean_squared_error(y_true, y_pred)


# 処理対象のデータセット名
TARGET_DATASETS = {
    "50words", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car",
    "CBF", "ChlorineConcentration", "CinC_ECG_torso", "Coffee", "Computers",
    "Cricket_X", "Cricket_Y", "Cricket_Z", "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays",
    "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FISH", "FordA",
    "FordB", "Gun_Point", "Ham", "HandOutlines", "Haptics", "Herring",
    "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand",
    "LargeKitchenAppliances", "Lighting2", "Lighting7", "MALLAT", "Meat",
    "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain",
    "NonInvasiveFatalECG_Thorax1", "NonInvasiveFatalECG_Thorax2", "OliveOil",
    "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "Plane",
    "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType", "ShapeletSim",
    "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface",
    "SonyAIBORobotSurfaceII", "StarLightCurves", "Strawberry", "SwedishLeaf",
    "Symbols", "synthetic_control", "ToeSegmentation1", "ToeSegmentation2",
    "Trace", "Two_Patterns", "TwoLeadECG", "uWaveGestureLibrary_X",
    "uWaveGestureLibrary_Y", "uWaveGestureLibrary_Z",
    "UWaveGestureLibraryAll", "wafer", "Wine", "WordsSynonyms", "Worms",
    "WormsTwoClass", "yoga",
}


def process_dataset(dataset_name, data_root, output_root):
    x_train, y_train, x_test, y_test, n_classes = load_dataset(
        pathlib.Path(data_root, dataset_name), dataset_name
    )

    reservoir, readout = generate_model()

    dataset_output_dir = pathlib.Path(output_root, dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # 貯水池に系列を流して特徴量(状態)を取り出す
    train_states = run_reservoir(reservoir, x_train)
    test_states = run_reservoir(reservoir, x_test)

    # fitting（Ridge回帰による読み出し層の学習）
    readout = readout.fit(train_states, y_train)

    # モデルの保存
    with open(dataset_output_dir / "model.pkl", "wb") as f:
        pickle.dump({"reservoir": reservoir, "readout": readout}, f)

    # 評価
    y_train_pred = readout.run(train_states)
    y_test_pred = readout.run(test_states)

    # パラメータ数の計算 (学習対象=読み出し層のみをカウント。reservoir はランダム固定で学習しない)
    total_params = readout.Wout.size
    if readout.fit_bias:
        total_params += readout.bias.size

    return {
        "params": total_params,
        "train_loss": mse_loss(y_train, y_train_pred),
        "train_acc": accuracy(y_train_pred, y_train),
        "test_loss": mse_loss(y_test, y_test_pred),
        "test_acc": accuracy(y_test_pred, y_test),
    }


def main():
    data_root = pathlib.Path("UCR_TS_Archive_2015")
    output_root = pathlib.Path("routput")
    output_root.mkdir(parents=True, exist_ok=True)

    flist = sorted(
        f.name for f in data_root.iterdir() if f.name in TARGET_DATASETS
    )
    total_datasets = len(flist)

    time_log_path = output_root / "execution_times.csv"
    with open(time_log_path, "w") as f:
        f.write("dataset,time_seconds\n")
    print(f"Time log file initialized at {time_log_path}", flush=True)

    results_log_path = output_root / "results.csv"
    with open(results_log_path, "w") as f:
        f.write("dataset,params,train_loss,train_acc,test_loss,test_acc\n")
    print(f"Results log file initialized at {results_log_path}", flush=True)

    for i, dataset_name in enumerate(flist):
        print(f"--- Processing {i + 1}/{total_datasets}: {dataset_name} ---", flush=True)
        start_time = time.time()

        try:
            result = process_dataset(dataset_name, data_root, output_root)
        except Exception as e:
            # 1件失敗しても全体を止めず、次のデータセットへ進む
            print(f"  失敗: {dataset_name} をスキップします ({e})", flush=True)
            with open(results_log_path, "a") as f:
                f.write(f"{dataset_name},ERROR,,,,\n")
            continue

        elapsed_time = time.time() - start_time
        print(f"Done. Time taken: {elapsed_time:.2f} seconds", flush=True)

        with open(time_log_path, "a") as f:
            f.write(f"{dataset_name},{elapsed_time}\n")

        with open(results_log_path, "a") as f:
            f.write(
                f"{dataset_name},{result['params']},{result['train_loss']},"
                f"{result['train_acc']},{result['test_loss']},{result['test_acc']}\n"
            )


if __name__ == "__main__":
    main()