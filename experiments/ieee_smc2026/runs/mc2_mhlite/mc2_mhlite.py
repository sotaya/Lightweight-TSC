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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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
    
    histogram_path = pathlib.Path(path, dataset_name + "_histogram_s" + sel_mode)
    hist_s = np.loadtxt(histogram_path, delimiter=',')[:,1:]
    
    histogram_path = pathlib.Path(path, dataset_name + "_histogram_l" + sel_mode)
    hist_l = np.loadtxt(histogram_path, delimiter=',')[:,1:]

    # 🛠️ 破損行対策：不正な列数の行を自動スキップして読み込む
    macd_path = pathlib.Path(path, dataset_name + "_macd_s" + sel_mode)
    md_s = np.genfromtxt(macd_path, delimiter=',', invalid_raise=False)
    
    # もし行数が元データと合わない場合、破損行を特定して除外する
    if len(md_s) != len(x):
        print(f"⚠️ Warning: Dimension mismatch in {macd_path.name}. Cleaning broken rows...")
        # どの行が正常に読み込めたかをインデックスで追跡
        valid_indices = []
        with open(macd_path, 'r') as f:
            for idx, line in enumerate(f):
                if len(line.split(',')) == (md_s.shape[1] if md_s.ndim > 1 else 0): 
                    valid_indices.append(idx)
        
        # 念のため安全な読み込み直し
        md_s = np.genfromtxt(macd_path, delimiter=',', invalid_raise=False)[:, 1:]
        
        # 正常な行だけを抽出して同期させる
        x = x[valid_indices]
        y = y[valid_indices]
        hist_s = hist_s[valid_indices]
        hist_l = hist_l[valid_indices]
    else:
        md_s = md_s[:, 1:]

    # 不正文字列によるNaNを0埋め
    if np.isnan(md_s).any():
        md_s = np.nan_to_num(md_s, nan=0.0)

    # 長期MACDも同様に処理
    macd_path = pathlib.Path(path, dataset_name + "_macd_l" + sel_mode)
    md_l = np.genfromtxt(macd_path, delimiter=',', invalid_raise=False)
    if len(md_l) != len(x):
        # 短期側で既にxやyの行数削減が行われている可能性があるため、現在のxの長さに合わせて再調整
        md_l = np.genfromtxt(macd_path, delimiter=',', invalid_raise=False)[:, 1:]
        min_len = min(len(x), len(md_l))
        x, y, hist_s, hist_l, md_s, md_l = x[:min_len], y[:min_len], hist_s[:min_len], hist_l[:min_len], md_s[:min_len], md_l[:min_len]
    else:
        md_l = md_l[:, 1:]

    if np.isnan(md_l).any():
        md_l = np.nan_to_num(md_l, nan=0.0)

    if(n_classes == 0):
        n_classes = len(np.unique(y))
    
    y = (y - y.min())/(y.max()-y.min())*(n_classes-1)
    y = to_categorical(y, n_classes)
    
    return x, y, hist_s, hist_l, n_classes, md_s, md_l

def generate_mcmhlite(MAX_SEQUENCE_LENGTH, NB_CLASS):
    lite = LITENetwork()
    input_layer, gap = lite.build_network(input_shape=(MAX_SEQUENCE_LENGTH, 5))
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

if __name__ == "__main__":
    flist = list(pathlib.Path("UCR_TS_Archive_2015").iterdir()) 
    flist.sort()

    basepath = pathlib.Path("mc2loutput")
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
        basepath = pathlib.Path("")
        x_train, y_train, hist_s_train, hist_l_train, n_classes, md_s_train, md_l_train = load_data(
            pathlib.Path(basepath, "UCR_TS_Archive_2015", fname.name), fname.name, "_TRAIN"
        )
        x_test, y_test, hist_s_test, hist_l_test, _, md_s_test, md_l_test = load_data(
            pathlib.Path(basepath, "UCR_TS_Archive_2015", fname.name), fname.name, "_TEST", n_classes
        )
        
        # 【最小限の修正】データ破損による行数のズレを、一番短い配列に合わせて切り揃える
        min_train = min(len(x_train), len(md_s_train), len(md_l_train), len(hist_s_train), len(hist_l_train))
        x_train, y_train = x_train[:min_train], y_train[:min_train]
        md_s_train, md_l_train = md_s_train[:min_train], md_l_train[:min_train]
        hist_s_train, hist_l_train = hist_s_train[:min_train], hist_l_train[:min_train]

        min_test = min(len(x_test), len(md_s_test), len(md_l_test), len(hist_s_test), len(hist_l_test))
        x_test, y_test = x_test[:min_test], y_test[:min_test]
        md_s_test, md_l_test = md_s_test[:min_test], md_l_test[:min_test]
        hist_s_test, hist_l_test = hist_s_test[:min_test], hist_l_test[:min_test]

        # 5つの特徴量を dstack で結合
        input_train = np.dstack([x_train, md_s_train, md_l_train, hist_s_train, hist_l_train])
        input_test = np.dstack([x_test, md_s_test, md_l_test, hist_s_test, hist_l_test])
        
        model = generate_mcmhlite(input_train.shape[1], n_classes)
        
        basepath = pathlib.Path("mc2loutput")
        secondpath = pathlib.Path(fname.name)
        p = pathlib.Path(basepath, secondpath)
        if(not p.exists()):
            p.mkdir(parents=True)
        
        # 🛠️ ロガーの引数を軽量版に変更
        logger = myCSVlogger( str(pathlib.Path(basepath, secondpath, 'training.log')), input_test, y_test, fname.name )
        checker = ModelCheckpoint(str(pathlib.Path(basepath, secondpath, 'model.h5')), 
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