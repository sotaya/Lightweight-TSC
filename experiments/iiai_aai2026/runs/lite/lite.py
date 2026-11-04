# coding:utf-8
import time, os
import pathlib
import numpy as np
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow
from tensorflow import keras 

SEED = 42
tensorflow.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from aeon.networks._lite import LITENetwork

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

    if(n_classes == 0):
        n_classes = len(np.unique(y))
        y = (y - y.min())/(y.max()-y.min())*(n_classes-1)
        y = to_categorical(y, n_classes)
    else:
        y = (y - y.min())/(y.max()-y.min())*(n_classes-1)
        y = to_categorical(y, n_classes)
    
    return x, y, n_classes

def generate_lite(MAX_SEQUENCE_LENGTH, NB_CLASS):

    lite = LITENetwork()
    input_layer, gap = lite.build_network(input_shape=(MAX_SEQUENCE_LENGTH, 1))
    output_layer = tensorflow.keras.layers.Dense(NB_CLASS, activation='softmax')(gap)
    model = Model(inputs=input_layer, outputs=output_layer)
        
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=tensorflow.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    return model

batch_size_dic = {
    "50words":8,
    "Adiac":8,
    "ArrowHead":16,
    "Beef":8,
    "BeetleFly":8,
    "BirdChicken":16,
    "Car":32,
    "CBF":16,
    "ChlorineConcentration":128,
    "CinC_ECG_torso":16,
    "Coffee":8,
    "Computers":128,
    "Cricket_X":32,
    "Cricket_Y":32,
    "Cricket_Z":32,
    "DiatomSizeReduction":4,
    "DistalPhalanxOutlineAgeGroup":32,
    "DistalPhalanxOutlineCorrect":32,
    "DistalPhalanxTW":32,
    "Earthquakes":8,
    "ECG200":32,
    "ECG5000":32,
    "ECGFiveDays":16,
    "ElectricDevices":32,
    "FaceAll":8,
    "FaceFour":8,
    "FacesUCR":8,
    "FISH":8,
    "FordA":32,
    "FordB":32,
    "Gun_Point":8,
    "Ham":32,
    "HandOutlines":32,
    "Haptics":32,
    "Herring":8,
    "InlineSkate":32,
    "InsectWingbeatSound":32,
    "ItalyPowerDemand":8,
    "LargeKitchenAppliances":128,
    "Lighting2":32,
    "Lighting7":32,
    "MALLAT":16,
    "Meat":32,
    "MedicalImages":8,
    "MiddlePhalanxOutlineAgeGroup":8,
    "MiddlePhalanxOutlineCorrect":128,
    "MiddlePhalanxTW":32,
    "MoteStrain":32,
    "NonInvasiveFatalECG_Thorax1":32,
    "NonInvasiveFatalECG_Thorax2":32,
    "OliveOil":8,
    "OSULeaf":8,
    "PhalangesOutlinesCorrect":128,
    "Phoneme":8,
    "Plane":8,
    "ProximalPhalanxOutlineAgeGroup":8,
    "ProximalPhalanxOutlineCorrect":32,
    "ProximalPhalanxTW":8,
    "RefrigerationDevices":32,
    "ScreenType":32,
    "ShapeletSim":8,
    "ShapesAll":8,
    "SmallKitchenAppliances":128,
    "SonyAIBORobotSurface":8,
    "SonyAIBORobotSurfaceII":16,
    "StarLightCurves":128,
    "Strawberry":128,
    "SwedishLeaf":8,
    "Symbols":8,
    "synthetic_control":8,
    "ToeSegmentation1":16,
    "ToeSegmentation2":16,
    "Trace":8,
    "Two_Patterns":32,
    "TwoLeadECG":16,
    "uWaveGestureLibrary_X":8,
    "uWaveGestureLibrary_Y":8,
    "uWaveGestureLibrary_Z":8,
    "UWaveGestureLibraryAll":64,
    "wafer":8,
    "Wine":32,
    "WordsSynonyms":8,
    "Worms":32,
    "WormsTwoClass":32,
    "yoga":16
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
        
        # MODIFIED: Changed the batch_size to a fixed value (e.g., 128) 
        # to prevent GPU Out-of-Memory errors with large test sets.
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
    flist.sort()   # データセット辞書順並べ替え

    basepath = pathlib.Path("loutput")
    if not basepath.exists():
        basepath.mkdir(parents=True)

    output_time_path = pathlib.Path(basepath, "execution_times.csv")
    with open(output_time_path, "w") as f:
        f.write("dataset,time_seconds\n")
    print(f"Time log file initialized at {output_time_path}")

    output_results_path = pathlib.Path(basepath, "results.csv")
    with open(output_results_path, "w") as f:
        f.write("dataset,params,train_loss,train_acc,test_loss,test_acc\n")
    print(f"Results log file initialized at {output_results_path}")

    # 処理時間を記録するリストを初期化
    time_logs = []

    for fname in flist:
        if fname.name not in batch_size_dic:
            continue
        
        print(f"--- Processing: {fname.name} ---")
        # 計測開始
        start_time = time.time()

        # load data
        basepath = pathlib.Path("")
        x_train, y_train, n_classes = load_data( pathlib.Path(basepath, "UCR_TS_Archive_2015", fname.name), fname.name, "_TRAIN")
        x_test, y_test, _ = load_data( pathlib.Path(basepath, "UCR_TS_Archive_2015", fname.name), fname.name, "_TEST", n_classes)
        input_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        input_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # built model
        model = generate_lite(input_train.shape[1], n_classes)
        
        # make log file
        basepath = pathlib.Path("loutput")
        secondpath = pathlib.Path(fname.name)
        p = pathlib.Path(basepath, secondpath)
        if(not p.exists()):
            p.mkdir(parents=True)
        
        # fitting
        logger = myCSVlogger( str(pathlib.Path(basepath, secondpath, 'training.log')), input_test, y_test, fname.name )
        # Using .h5 format for better compatibility
        checker = ModelCheckpoint(str(pathlib.Path(basepath, secondpath, 'model.h5')), 
                                  monitor='loss', verbose=0, save_best_only=True, 
                                  save_weights_only=False, mode='min')
        
        history = model.fit(input_train, y_train, validation_data = (input_test, y_test), 
                                epochs=2000, batch_size=batch_size_dic[fname.name],
                                callbacks=[logger, checker])
        
        # 計測終了・時間計算
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Done. Time taken: {elapsed_time:.2f} seconds")

        with open(output_time_path, "a") as f:
            f.write(f"{fname.name},{elapsed_time}\n")

        # --- 結果サマリーの記録を追加 ---
        # 最終エポックのメトリクスを取得
        train_loss = history.history['loss'][-1]
        train_acc = history.history['accuracy'][-1]
        test_loss = history.history['val_loss'][-1]
        test_acc = history.history['val_acc'][-1]

        # パラメータ数を取得
        total_params = model.count_params()

        # results.csvに追記
        with open(output_results_path, "a") as f:
            f.write(f"{fname.name},{total_params},{train_loss},{train_acc},{test_loss},{test_acc}\n")

        # メモリリークを防ぐためにセッションをクリア（推奨）
        keras.backend.clear_session()
