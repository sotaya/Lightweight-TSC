# coding:utf-8

import time, os
import pathlib
import numpy as np
import random
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from aeon.networks import LITENetwork

os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

SEED = 42
tensorflow.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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


def generate_pmc_mhlite_gap_shared(time_series_length, n_classes):

    lite_a_net = LITENetwork()
    input_tensor_a, gap_output_a = lite_a_net.build_network(input_shape=(time_series_length, 2))
    model_a = Model(inputs=input_tensor_a, outputs=gap_output_a, name="LITE_Branch_A")

    lite_b_net = LITENetwork()
    input_tensor_b, gap_output_b = lite_b_net.build_network(input_shape=(time_series_length, 2))
    model_b = Model(inputs=input_tensor_b, outputs=gap_output_b, name="LITE_Branch_B")

    global_input_a = Input(shape=(time_series_length, 2), name="Input_Long_Pair")
    global_input_b = Input(shape=(time_series_length, 2), name="Input_Short_Pair")

    gap_feat_a = model_a(global_input_a)
    gap_feat_b = model_b(global_input_b)

    gap_dim = gap_output_a.shape[-1]
    shared_post_gap_dense = Dense(gap_dim, activation='relu', name="shared_post_gap_dense")
    feat_a = shared_post_gap_dense(gap_feat_a)
    feat_b = shared_post_gap_dense(gap_feat_b)

    combined = Concatenate(name="concat_features")([feat_a, feat_b])

    output = Dense(n_classes, activation='softmax', name="classification_output")(combined)

    model = Model(inputs=[global_input_a, global_input_b], outputs=output)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

batch_size_dic = {
    "Adiac": 8, "Beef": 8, "CBF": 16, "Computers": 128, 
    "ECG200": 32, "Gun_Point": 8, "wafer": 8
 }

class myCSVlogger(keras.callbacks.Callback):
    def __init__(self, output_path, x_test, y_test, fname):
        self.output_path = output_path
        self.logs = {
                        "acc":[],
                        "loss":[],
                        "val_acc":[],
                        "val_loss":[]
                    }
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

if __name__ == "__main__":
    # reading dataset's name
    flist = list(pathlib.Path("UCR_TS_Archive_2015").iterdir())
    flist.sort()

    basepath = pathlib.Path("gs_pmcloutput")
    if not basepath.exists():
        basepath.mkdir(parents=True)

    output_time_path = pathlib.Path(basepath, "execution_times.csv")
    with open(output_time_path, "w") as f:
        f.write("dataset,time_seconds\n")
    print(f"Time log file initialized at {output_time_path}")

    for fname in flist:
        if fname.name not in batch_size_dic:
            continue

        print(f"--- Processing: {fname.name} ---")
        start_time = time.time()

        # load data
        data_archive_path = pathlib.Path("UCR_TS_Archive_2015", fname.name)
        x_train, y_train, hist_s_train, hist_l_train, n_classes = load_data(data_archive_path, fname.name, "_TRAIN")
        x_test, y_test, hist_s_test, hist_l_test, _ = load_data(data_archive_path, fname.name, "_TEST", n_classes)

        input_1_train = np.dstack([x_train, hist_l_train])
        input_1_test  = np.dstack([x_test,  hist_l_test])

        input_2_train = np.dstack([x_train, hist_s_train])
        input_2_test  = np.dstack([x_test,  hist_s_test])

        input_train = [input_1_train, input_2_train]
        input_test  = [input_1_test,  input_2_test]

        model = generate_pmc_mhlite_gap_shared(input_1_train.shape[1], n_classes)

        # make log file
        dataset_output_path = pathlib.Path(basepath, fname.name)
        if(not dataset_output_path.exists()):
            dataset_output_path.mkdir(parents=True)

        # fitting
        logger = myCSVlogger( str(pathlib.Path(dataset_output_path, 'training.log')), input_test, y_test, fname.name )
        # Using .h5 format for better compatibility
        checker = ModelCheckpoint(str(pathlib.Path(dataset_output_path, 'model.h5')),
                                  monitor='loss', verbose=0, save_best_only=True,
                                  save_weights_only=False, mode='min')

        history = model.fit(input_train, y_train, validation_data = (input_test, y_test),
                                epochs=2000, batch_size=batch_size_dic[fname.name],
                                callbacks=[logger, checker])

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Done. Time taken: {elapsed_time:.2f} seconds")

        with open(output_time_path, "a") as f:
            f.write(f"{fname.name},{elapsed_time}\n")

        keras.backend.clear_session()
