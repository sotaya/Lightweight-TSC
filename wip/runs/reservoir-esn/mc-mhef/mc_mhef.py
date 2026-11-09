# coding:utf-8

import argparse
import os
import pathlib
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv1D, Dense, GlobalAveragePooling1D, Input

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

SEED = 42
EPOCHS = 2000
RESERVOIR_UNITS = 64
SPECTRAL_RADIUS = 0.9
LEAKING_RATE = 0.3
INPUT_SCALING = 1.0
OUTPUT_ROOT = pathlib.Path("mcefoutput")
DATA_ROOT = pathlib.Path("UCR_TS_Archive_2015")

tf.random.set_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE_DIC = {
    "Adiac": 8, "Beef": 8, "CBF": 16, "Computers": 128, 
    "ECG200": 32, "Gun_Point": 8, "wafer": 8
}

def normalization(data):
    ans = []
    if data is None or not hasattr(data, "__len__") or len(data) == 0:
        return data
    for i in range(len(data)):
        row = data[i]
        if isinstance(row, (float, int)):
            continue
        m = np.average(row)
        s = np.std(row)
        if s == 0:
            s = 1.0
        ans.append((row - m) / s)
    return np.array(ans)


def load_data(path, dataset_name, sel_mode, n_classes=0):
    data_path = pathlib.Path(path, dataset_name + sel_mode)
    data = np.loadtxt(data_path, delimiter=',')
    x = normalization(data[:, 1:])
    y = data[:, 0].astype(float)

    def safe_load_extra(suffix):
        p = pathlib.Path(path, dataset_name + suffix + sel_mode)
        if p.exists():
            arr = np.loadtxt(p, delimiter=',')
            if arr.ndim > 1:
                return arr[:, 1:]
            else:
                return arr.reshape(-1, 1)
        return np.zeros((len(x), 1))

    hist_s = safe_load_extra("_histogram_s")
    hist_l = safe_load_extra("_histogram_l")
    md_s = safe_load_extra("_macd_s")
    md_l = safe_load_extra("_macd_l")

    if n_classes == 0:
        n_classes = len(np.unique(y))
        y = (y - y.min()) / (y.max() - y.min()) * (n_classes - 1)
        y = to_categorical(y, n_classes)
    else:
        y = (y - y.min()) / (y.max() - y.min()) * (n_classes - 1)
        y = to_categorical(y, n_classes)

    return x, y, hist_s, hist_l, n_classes, md_s, md_l

class ESNPrecomputer:
    def __init__(self, input_dim, units=RESERVOIR_UNITS, spectral_radius=SPECTRAL_RADIUS,
                 leaking_rate=LEAKING_RATE, input_scaling=INPUT_SCALING, seed=SEED):
        self.units = units
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        
        rng = np.random.default_rng(seed)
        self.w_in = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(input_dim, units))
        w_res = rng.normal(0.0, 1.0 / np.sqrt(units), size=(units, units))
        
        radius = np.max(np.abs(np.linalg.eigvals(w_res)))
        if radius == 0:
            radius = 1.0
        self.w_res = w_res * (spectral_radius / radius)
        self.bias = np.zeros(units)

    def transform(self, x_seq):
        num_samples, timesteps, _ = x_seq.shape
        states = np.zeros((num_samples, self.units), dtype=np.float32)

        input_projections = np.dot(x_seq, self.w_in) * self.input_scaling + self.bias

        for i in range(num_samples):
            state = np.zeros(self.units, dtype=np.float32)
            acc_state = np.zeros(self.units, dtype=np.float32)
            u_proj = input_projections[i] # (timesteps, units)
            
            for t in range(timesteps):
                pre_activation = u_proj[t] + np.dot(state, self.w_res)
                state = (1.0 - self.leaking_rate) * state + self.leaking_rate * np.tanh(pre_activation)
                acc_state += state
                
            states[i] = acc_state / timesteps
            
        return states

def generate_multi_input_mcmhef(max_sequence_length, nb_class, reservoir_units=RESERVOIR_UNITS):
    ip_esn = Input(shape=(reservoir_units,), name="esn_input")
    ip_fcn = Input(shape=(max_sequence_length, 3), name="fcn_input")

    x = Dense(32, activation='relu', kernel_initializer='he_uniform')(ip_esn)

    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip_fcn)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = GlobalAveragePooling1D()(y)

    combined = Concatenate()([x, y])

    out_x = Dense(nb_class, activation=None, kernel_initializer='he_uniform')(combined)
    out = Activation('softmax')(out_x)

    model = Model(inputs=[ip_esn, ip_fcn], outputs=out)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

class MyCSVLogger(keras.callbacks.Callback):
    def __init__(self, output_path, x_test_esn, x_test_fcn, y_test):
        self.output_path = output_path
        self.logs = {"acc": [], "loss": [], "val_acc": [], "val_loss": []}
        self.x_test_esn = x_test_esn
        self.x_test_fcn = x_test_fcn
        self.y_test = y_test

    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        self.logs["loss"].append(logs.get('loss'))
        self.logs["acc"].append(logs.get('accuracy'))
        
        scores = self.model.evaluate([self.x_test_esn, self.x_test_fcn], self.y_test, batch_size=128, verbose=0)
        self.logs["val_loss"].append(scores[0])
        self.logs["val_acc"].append(scores[1])
        
    def on_train_end(self, logs=None):
        with open(self.output_path, "w") as f:
            f.write("epoch,loss,acc,val_loss,val_acc,")
            for i in range(len(self.logs["loss"])):
                f.write(f"{i + 1},{self.logs['loss'][i]},{self.logs['acc'][i]},{self.logs['val_loss'][i]},{self.logs['val_acc'][i]},")

def run_single_dataset(dataset_name, data_root, output_root, reservoir_units):
    output_dir = pathlib.Path(output_root, dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, hist_s_train, hist_l_train, n_classes, _, _ = load_data(
        pathlib.Path(data_root, dataset_name), dataset_name, "_TRAIN"
    )
    x_test, y_test, hist_s_test, hist_l_test, _, _, _ = load_data(
        pathlib.Path(data_root, dataset_name), dataset_name, "_TEST", n_classes
    )

    fcn_train = np.dstack([x_train, hist_s_train, hist_l_train]).astype(np.float32)
    fcn_test = np.dstack([x_test, hist_s_test, hist_l_test]).astype(np.float32)

    print(f"[{dataset_name}] Precomputing ESN states (optimized)...", flush=True)
    esn_precomputer = ESNPrecomputer(input_dim=fcn_train.shape[2], units=reservoir_units)
    esn_train = esn_precomputer.transform(fcn_train)
    esn_test = esn_precomputer.transform(fcn_test)

    model = generate_multi_input_mcmhef(fcn_train.shape[1], n_classes, reservoir_units=reservoir_units)

    trainable_params = int(sum([np.prod(w.shape.as_list()) for w in model.trainable_weights]))
    
    logger = MyCSVLogger(str(output_dir / "training.log"), esn_test, fcn_test, y_test)
    checker = ModelCheckpoint(
        str(output_dir / "model.h5"),
        monitor='loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
    )

    history = model.fit(
        [esn_train, fcn_train],
        y_train,
        validation_data=([esn_test, fcn_test], y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE_DIC[dataset_name],
        callbacks=[logger, checker],
        verbose=1,
    )

    train_loss = history.history['loss'][-1]
    train_acc = history.history['accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    val_acc = history.history['val_accuracy'][-1]
    best_idx = int(np.argmin(history.history['loss']))

    return {
        "dataset": dataset_name,
        "best_epoch": best_idx + 1,
        "params": trainable_params,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


def _already_processed_datasets(results_log_path):
    done = set()
    if results_log_path.exists():
        with open(results_log_path) as f:
            next(f, None)
            for line in f:
                parts = line.split(",")
                name = parts[0].strip()
                if name and len(parts) > 3 and "ERROR" not in parts[3]:
                    done.add(name)
    return done

def run_benchmark(data_root=DATA_ROOT, output_root=OUTPUT_ROOT, resume=True):
    output_root = pathlib.Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    time_log_path = output_root / "execution_times.csv"
    results_log_path = output_root / "results.csv"

    done_datasets = set()
    if resume and results_log_path.exists():
        done_datasets = _already_processed_datasets(results_log_path)
        print(f"Resuming: {len(done_datasets)} datasets already completed, skipping them.", flush=True)

    if not time_log_path.exists():
        with open(time_log_path, "w") as f:
            f.write("dataset,time_seconds\n")
    if not results_log_path.exists():
        with open(results_log_path, "w") as f:
            f.write("dataset,best_epoch,params,train_loss,train_acc,val_loss,val_acc\n")

    dataset_names = sorted(
        [p.name for p in data_root.iterdir() if p.is_dir() and p.name in BATCH_SIZE_DIC]
    )
    dataset_names = [d for d in dataset_names if d not in done_datasets]
    total_datasets = len(dataset_names)
    print(f"Benchmark start: {total_datasets} datasets remaining", flush=True)

    for i, dataset_name in enumerate(dataset_names):
        print(f"=== Processing {i + 1}/{total_datasets}: {dataset_name} ===", flush=True)
        start_time = time.time()
        keras.backend.clear_session()

        try:
            result = run_single_dataset(
                dataset_name, data_root, output_root,
                reservoir_units=RESERVOIR_UNITS,
            )
            elapsed_time = time.time() - start_time
            print(f"Done. Time taken: {elapsed_time:.2f} seconds", flush=True)

            with open(time_log_path, "a") as f:
                f.write(f"{dataset_name},{elapsed_time}\n")

            with open(results_log_path, "a") as f:
                f.write(
                    f"{result['dataset']},{result['best_epoch']},{result['params']},"
                    f"{result['train_loss']},{result['train_acc']},"
                    f"{result['val_loss']},{result['val_acc']}\n"
                )
        except Exception as e:
            elapsed_time = time.time() - start_time
            err_msg = str(e).replace(",", " ")
            print(f"Failed: {dataset_name} skipped due to error: {repr(e)}", flush=True)
            with open(time_log_path, "a") as f:
                f.write(f"{dataset_name},{elapsed_time}\n")
            with open(results_log_path, "a") as f:
                f.write(f"{dataset_name},ERROR: {err_msg}\n")
        finally:
            keras.backend.clear_session()

    print(f"Benchmark finished. All logs saved to {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MC-MHEF Benchmark (Fully Optimized Precomputed ESN + FCN)")
    parser.add_argument("--data-root", type=str, default="UCR_TS_Archive_2015", help="path to UCR dataset directory")
    parser.add_argument("--output-root", type=str, default="mcefoutput", help="path to output directory")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing results.csv and start over")
    args = parser.parse_args()

    run_benchmark(
        data_root=pathlib.Path(args.data_root),
        output_root=pathlib.Path(args.output_root),
        resume=not args.no_resume,
    )