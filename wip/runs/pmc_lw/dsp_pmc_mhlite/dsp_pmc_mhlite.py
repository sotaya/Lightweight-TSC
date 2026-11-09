# coding:utf-8

import time, os, json
import pathlib
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Concatenate, Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from aeon.networks import LITENetwork

os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

LAMBDA = float(os.environ.get("DSP_LAMBDA", "1e-5"))
EPOCHS_A = int(os.environ.get("DSP_EPOCHS_A", "1500"))
EPOCHS_B = int(os.environ.get("DSP_EPOCHS_B", "2000"))

KERNEL_SIZE = 40
N_FILTERS = 32
INCEPTION_KERNELS = [KERNEL_SIZE // (2 ** i) for i in range(3)]
FULL_COUNTS = [N_FILTERS] * 5


def normalization(data):
    ans = []
    for i in range(len(data)):
        m = np.average(data[i])
        s = np.std(data[i])
        ans.append((data[i] - m) / s)
    return np.array(ans)


def load_data(path, dataset_name, sel_mode, n_classes=0):
    data_path = pathlib.Path(path, dataset_name + sel_mode)
    data = np.loadtxt(data_path, delimiter=',')
    x = normalization(data[:, 1:])
    y = data[:, 0].astype(float)

    histogram_path = pathlib.Path(path, dataset_name + "_histogram_s" + sel_mode)
    hist_s = np.loadtxt(histogram_path, delimiter=',')[:,1:]

    histogram_path = pathlib.Path(path, dataset_name + "_histogram_l" + sel_mode)
    hist_l = np.loadtxt(histogram_path, delimiter=',')[:,1:]

    if(n_classes == 0):
        n_classes = len(np.unique(y))
        y = (y - y.min())/(y.max()-y.min())*(n_classes-1)
        y = to_categorical(y, n_classes)
    else:
        y = (y - y.min())/(y.max()-y.min())*(n_classes-1)
        y = to_categorical(y, n_classes)

    return x, y, hist_s, hist_l, n_classes


class SparsityTap(Layer):

    def __init__(self, lam, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam

    def call(self, x, training=None):
        if training and self.lam > 0:
            norms = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) + 1e-12)
            self.add_loss(self.lam * tf.reduce_sum(norms))
        return x


def build_branch_model(name, time_series_length, counts, lam, tap):
    net = LITENetwork()
    inp = Input(shape=(time_series_length, 2))
    taps = []

    def maybe_tap(t):
        if not tap:
            return t
        y = SparsityTap(lam)(t)
        taps.append(y)
        return y

    conv_outs = []
    for k, n in zip(INCEPTION_KERNELS, counts[:3]):
        c = Conv1D(filters=n, kernel_size=k, padding='same', activation='linear', use_bias=False)(inp)
        conv_outs.append(maybe_tap(c))
    hybrid = net.hybrid_layer(input_tensor=inp, input_channels=2)
    x = Concatenate(axis=2)(conv_outs + [hybrid])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i, n in enumerate(counts[3:]):
        x = net._fcn_module(input_tensor=x, kernel_size=(KERNEL_SIZE // 2) // (2 ** i),
                            n_filters=n, dilation_rate=2 ** (i + 1))
        x = maybe_tap(x)

    gap = GlobalAveragePooling1D()(x)
    return Model(inputs=inp, outputs=[gap] + taps, name=name)


def build_pmc_dsp(time_series_length, n_classes, counts_long, counts_short, lam=0.0, tap=False):
    global_input_a = Input(shape=(time_series_length, 2), name="Input_Long_Pair")
    global_input_b = Input(shape=(time_series_length, 2), name="Input_Short_Pair")

    branch_a = build_branch_model("branch_long", time_series_length, counts_long, lam, tap)
    branch_b = build_branch_model("branch_short", time_series_length, counts_short, lam, tap)
    outs_a = branch_a(global_input_a)
    outs_b = branch_b(global_input_b)

    combined = Concatenate(name="concat_features")([outs_a[0], outs_b[0]])
    output = Dense(n_classes, activation='softmax', name="classification_output")(combined)
    model = Model(inputs=[global_input_a, global_input_b], outputs=output)

    feat_model = None
    if tap:
        feat_model = Model(inputs=[global_input_a, global_input_b], outputs=outs_a[1:] + outs_b[1:])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model, feat_model


def select_channels(feat_model, x_list):
    outs = feat_model.predict(x_list, batch_size=128, verbose=0)
    if not isinstance(outs, list):
        outs = [outs]

    counts, detail = [], []
    for F in outs:
        s = np.sqrt(np.sum(np.square(F), axis=1))          
        tau = s.mean(axis=1, keepdims=True)
        active = s >= tau
        keep = active.any(axis=0)
        if not keep.any():
            keep[np.argmax(s.max(axis=0))] = True
        counts.append(int(keep.sum()))
        detail.append({"channels_before": int(F.shape[-1]), "channels_after": int(keep.sum())})
    return counts, detail


class myCSVlogger(keras.callbacks.Callback):
    def __init__(self, output_path, x_test, y_test, fname):
        self.output_path = output_path
        self.logs = {"acc":[], "loss":[], "val_acc":[], "val_loss":[]}
        self.x_test = x_test
        self.y_test = y_test
        self.fname = fname

    def on_epoch_end(self, batch, logs):
        self.logs["loss"].append(logs.get('loss'))
        self.logs["acc"].append(logs.get('accuracy'))

        scores = self.model.evaluate(self.x_test, self.y_test, batch_size=128, verbose=0)
        self.logs["val_loss"].append(scores[0])
        self.logs["val_acc"].append(scores[1])

    def on_train_end(self, logs):
        with open(self.output_path, "w") as f:
            f.write("epoch,loss,acc,val_loss,val_acc,\n")
            for i in range(len(self.logs["loss"])):
                f.write(str(i+1)+",")
                f.write(str(self.logs["loss"][i])+",")
                f.write(str(self.logs["acc"][i])+",")
                f.write(str(self.logs["val_loss"][i])+",")
                f.write(str(self.logs["val_acc"][i])+",\n")


batch_size_dic = {
    "Adiac": 8, "Beef": 8, "CBF": 16, "Computers": 128,
    "ECG200": 32, "Gun_Point": 8, "wafer": 8
}

if __name__ == "__main__":
    flist = list(pathlib.Path("UCR_TS_Archive_2015").iterdir())
    flist.sort()

    basepath = pathlib.Path("dsp_pmcloutput")
    if not basepath.exists():
        basepath.mkdir(parents=True)

    output_time_path = pathlib.Path(basepath, "execution_times.csv")
    with open(output_time_path, "w") as f:
        f.write("dataset,time_seconds\n")
    print(f"Time log file initialized at {output_time_path}")
    print(f"LAMBDA={LAMBDA} EPOCHS_A={EPOCHS_A} EPOCHS_B={EPOCHS_B}")

    for fname in flist:
        if fname.name not in batch_size_dic:
            continue

        print(f"--- Processing: {fname.name} ---")
        start_time = time.time()

        data_archive_path = pathlib.Path("UCR_TS_Archive_2015", fname.name)
        x_train, y_train, hist_s_train, hist_l_train, n_classes = load_data(data_archive_path, fname.name, "_TRAIN")
        x_test, y_test, hist_s_test, hist_l_test, _ = load_data(data_archive_path, fname.name, "_TEST", n_classes)

        input_1_train = np.dstack([x_train, hist_l_train])
        input_1_test  = np.dstack([x_test,  hist_l_test])
        input_2_train = np.dstack([x_train, hist_s_train])
        input_2_test  = np.dstack([x_test,  hist_s_test])

        input_train = [input_1_train, input_2_train]
        input_test  = [input_1_test,  input_2_test]
        T = input_1_train.shape[1]
        batch_size = batch_size_dic[fname.name]

        dataset_output_path = pathlib.Path(basepath, fname.name)
        if(not dataset_output_path.exists()):
            dataset_output_path.mkdir(parents=True)

        model_sparse, feat_model = build_pmc_dsp(T, n_classes, FULL_COUNTS, FULL_COUNTS,
                                                 lam=LAMBDA, tap=True)
        stage_a_start = time.time()
        model_sparse.fit(input_train, y_train, epochs=EPOCHS_A, batch_size=batch_size, verbose=0)
        stage_a_time = time.time() - stage_a_start

        keep_counts, detail = select_channels(feat_model, input_train)
        counts_long, counts_short = keep_counts[:5], keep_counts[5:]
        print(f"  kept long={counts_long} short={counts_short}")

        model = build_pmc_dsp(T, n_classes, counts_long, counts_short, lam=0.0, tap=False)[0]
        n_params = model.count_params()

        logger = myCSVlogger( str(pathlib.Path(dataset_output_path, 'training.log')), input_test, y_test, fname.name )
        checker = ModelCheckpoint(str(pathlib.Path(dataset_output_path, 'model.h5')),
                                  monitor='loss', verbose=0, save_best_only=True,
                                  save_weights_only=False, mode='min')

        history = model.fit(input_train, y_train, validation_data = (input_test, y_test),
                                epochs=EPOCHS_B, batch_size=batch_size,
                                callbacks=[logger, checker])

        end_time = time.time()
        elapsed_time = end_time - start_time

        with open(pathlib.Path(dataset_output_path, 'prune_info.json'), "w") as f:
            json.dump({
                "lambda": LAMBDA, "epochs_a": EPOCHS_A, "epochs_b": EPOCHS_B,
                "stage_a_time_seconds": stage_a_time,
                "counts_long": counts_long, "counts_short": counts_short,
                "layers": detail, "parameter_count_after_prune": int(n_params),
            }, f, indent=2)

        print(f"Done. Time taken: {elapsed_time:.2f} seconds")

        with open(output_time_path, "a") as f:
            f.write(f"{fname.name},{elapsed_time}\n")

        keras.backend.clear_session()
