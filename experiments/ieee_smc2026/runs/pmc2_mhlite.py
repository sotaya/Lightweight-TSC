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
from aeon.networks._lite import LITENetwork

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

    macd_path_s = pathlib.Path(path, dataset_name + "_macd_s" + sel_mode)
    md_s = np.genfromtxt(macd_path_s, delimiter=',', invalid_raise=False)
    md_s = md_s[:, 1:] if md_s.ndim > 1 else md_s

    macd_path_l = pathlib.Path(path, dataset_name + "_macd_l" + sel_mode)
    md_l = np.genfromtxt(macd_path_l, delimiter=',', invalid_raise=False)
    md_l = md_l[:, 1:] if md_l.ndim > 1 else md_l

    md_s = np.nan_to_num(md_s, nan=0.0)
    md_l = np.nan_to_num(md_l, nan=0.0)

    if(n_classes == 0):
        n_classes = len(np.unique(y))
    
    y = (y - y.min())/(y.max()-y.min())*(n_classes-1)
    y = to_categorical(y, n_classes)
    
    return x, y, hist_s, hist_l, n_classes, md_s, md_l

def generate_pmc2mhlite(time_series_length, n_classes):
    lite_a_net = LITENetwork()
    input_tensor_a, gap_output_a = lite_a_net.build_network(input_shape=(time_series_length, 3))
    model_a = Model(inputs=input_tensor_a, outputs=gap_output_a, name="LITE_Branch_A")
    
    lite_b_net = LITENetwork()
    input_tensor_b, gap_output_b = lite_b_net.build_network(input_shape=(time_series_length, 3))
    model_b = Model(inputs=input_tensor_b, outputs=gap_output_b, name="LITE_Branch_B")
    
    global_input_a = Input(shape=(time_series_length, 3), name="Input_Long_Pair")
    global_input_b = Input(shape=(time_series_length, 3), name="Input_Short_Pair")
    
    feat_a = model_a(global_input_a)
    feat_b = model_b(global_input_b)
    
    combined = Concatenate(name="concat_features")([feat_a, feat_b])
    
    output = Dense(n_classes, activation='softmax', name="classification_output")(combined)
    
    model = Model(inputs=[global_input_a, global_input_b], outputs=output)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
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

    basepath = pathlib.Path("pmc2loutput")
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
        
        min_train = min(len(x_train), len(md_s_train), len(md_l_train), len(hist_s_train), len(hist_l_train))
        x_train, y_train = x_train[:min_train], y_train[:min_train]
        md_s_train, md_l_train = md_s_train[:min_train], md_l_train[:min_train]
        hist_s_train, hist_l_train = hist_s_train[:min_train], hist_l_train[:min_train]

        min_test = min(len(x_test), len(md_s_test), len(md_l_test), len(hist_s_test), len(hist_l_test))
        x_test, y_test = x_test[:min_test], y_test[:min_test]
        md_s_test, md_l_test = md_s_test[:min_test], md_l_test[:min_test]
        hist_s_test, hist_l_test = hist_s_test[:min_test], hist_l_test[:min_test]

        input_1_train = np.dstack([x_train, md_l_train, hist_l_train])
        input_1_test  = np.dstack([x_test,  md_l_test,  hist_l_test])
        
        input_2_train = np.dstack([x_train, md_s_train, hist_s_train])
        input_2_test  = np.dstack([x_test,  md_s_test,  hist_s_test])
            
        input_train = [input_1_train, input_2_train]
        input_test  = [input_1_test,  input_2_test]

        # built model
        model = generate_pmc2mhlite(input_1_train.shape[1], n_classes)
        
        # make log file
        basepath = pathlib.Path("pmc2loutput")
        secondpath = pathlib.Path(fname.name)
        p = pathlib.Path(basepath, secondpath)
        if(not p.exists()):
            p.mkdir(parents=True)
        
        # fitting
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