# coding:utf-8

import argparse
import json
import time
import pathlib
import pickle
import random
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

SEED = 42
rpy.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

RESERVOIR_UNITS = 500
SPECTRAL_RADIUS = 0.9
LEAKING_RATE = 0.3
INPUT_SCALING = 1.0
RIDGE_COEF = 1e-6

def normalization(data):
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / std


def load_raw(path, dataset_name, sel_mode):
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

def generate_model(params=None):
    params = params or {
        "reservoir_units": RESERVOIR_UNITS,
        "spectral_radius": SPECTRAL_RADIUS,
        "leaking_rate": LEAKING_RATE,
        "input_scaling": INPUT_SCALING,
        "ridge_coef": RIDGE_COEF,
    }
    reservoir = Reservoir(
        params["reservoir_units"],
        sr=params["spectral_radius"],
        lr=params["leaking_rate"],
        input_scaling=params["input_scaling"],
        seed=SEED,
    )
    readout = Ridge(ridge=params["ridge_coef"])
    return reservoir, readout

def run_reservoir(reservoir, x):
    states = reservoir.run(list(x), workers=-1)
    return np.array([s.mean(axis=0) for s in states])

def accuracy(y_pred, y_true):
    return sklearn.metrics.accuracy_score(
        np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)
    )

def mse_loss(y_pred, y_true):
    return sklearn.metrics.mean_squared_error(y_true, y_pred)

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

def optimize_hyperparameters(x_train, y_train, n_trials):
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "Optuna が必要です。requirements.txt から依存関係をインストールしてください。"
        ) from error

    y_labels = np.argmax(y_train, axis=1)
    _, counts = np.unique(y_labels, return_counts=True)
    stratify = y_labels if np.min(counts) >= 2 else None

    train_indices, valid_indices = train_test_split(
        np.arange(len(x_train)),
        test_size=0.2,
        random_state=SEED,
        stratify=stratify,

    )

    def objective(trial):
        params = {
            "reservoir_units": trial.suggest_int("reservoir_units", 100, 1000, step=100),
            "spectral_radius": trial.suggest_float("spectral_radius", 0.1, 1.5),
            "leaking_rate": trial.suggest_float("leaking_rate", 0.1, 1.0),
            "input_scaling": trial.suggest_float("input_scaling", 0.1, 2.0),
            "ridge_coef": trial.suggest_float("ridge_coef", 1e-8, 1e-2, log=True),
        }
        reservoir, readout = generate_model(params)
        train_states = run_reservoir(reservoir, x_train[train_indices])
        valid_states = run_reservoir(reservoir, x_train[valid_indices])
        readout = readout.fit(train_states, y_train[train_indices])
        return accuracy(readout.run(valid_states), y_train[valid_indices])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def process_dataset(dataset_name, data_root, output_root, n_trials=0):
    x_train, y_train, x_test, y_test, n_classes = load_dataset(
        pathlib.Path(data_root, dataset_name), dataset_name
    )

    best_params = {}
    if n_trials > 0:
        print(f"  Optuna: {n_trials} 試行でハイパーパラメータを探索します", flush=True)
        best_params = optimize_hyperparameters(x_train, y_train, n_trials)
        print(f"  Optuna best: {best_params}", flush=True)

    reservoir, readout = generate_model(best_params)

    dataset_output_dir = pathlib.Path(output_root, dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    train_states = run_reservoir(reservoir, x_train)
    test_states = run_reservoir(reservoir, x_test)

    readout = readout.fit(train_states, y_train)

    with open(dataset_output_dir / "model.pkl", "wb") as f:
        pickle.dump(
            {"reservoir": reservoir, "readout": readout, "hyperparameters": best_params},
            f,
        )

    with open(dataset_output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    y_train_pred = readout.run(train_states)
    y_test_pred = readout.run(test_states)

    total_params = readout.Wout.size
    if readout.fit_bias:
        total_params += readout.bias.size

    return {
        "params": total_params,
        "train_loss": mse_loss(y_train, y_train_pred),
        "train_acc": accuracy(y_train_pred, y_train),
        "test_loss": mse_loss(y_test, y_test_pred),
        "test_acc": accuracy(y_test_pred, y_test),
        "hyperparameters": best_params,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-trials",
        type=int,
        default=0,
        help="データセットごとの Optuna 試行回数。0 なら既定値を使用",
    )
    args = parser.parse_args()
    if args.n_trials < 0:
        parser.error("--n-trials は 0 以上を指定してください")

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
            result = process_dataset(
                dataset_name, data_root, output_root, n_trials=args.n_trials
            )
        except Exception as e:
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