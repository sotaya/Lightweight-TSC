# coding:utf-8

import time, os
import random
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import SeparableConv1D, BatchNormalization, GlobalAveragePooling1D, Concatenate, Input, Dense, Activation, DepthwiseConv1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']="True"

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

def make_hybrid_layer(input_tensor, input_channels, kernel_sizes=[2, 4, 8, 16, 32, 64]):

    conv_list = []
    for kernel_size in kernel_sizes:
        filter_ = np.ones(shape=(kernel_size, input_channels, 1))
        indices_ = np.arange(kernel_size)
        filter_[indices_ % 2 == 0] *= -1
        
        conv = DepthwiseConv1D(
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            depthwise_initializer=tf.keras.initializers.Constant(filter_.tolist()),
            trainable=False,
            name=f"hybrid_inc_{kernel_size}_{input_tensor.name.split(':')[0]}"
        )(input_tensor)
        conv_list.append(conv)

    for kernel_size in kernel_sizes:
        filter_ = np.ones(shape=(kernel_size, input_channels, 1))
        indices_ = np.arange(kernel_size)
        filter_[indices_ % 2 > 0] *= -1
        
        conv = DepthwiseConv1D(
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            depthwise_initializer=tf.keras.initializers.Constant(filter_.tolist()),
            trainable=False,
            name=f"hybrid_dec_{kernel_size}_{input_tensor.name.split(':')[0]}"
        )(input_tensor)
        conv_list.append(conv)

    for kernel_size in kernel_sizes[1:]:
        filter_ = np.zeros(shape=(kernel_size + kernel_size // 2, input_channels, 1))
        xmesh = np.linspace(start=0, stop=1, num=kernel_size // 4 + 1)[1:].reshape((-1, 1, 1))
        
        filter_left = xmesh**2
        filter_right = filter_left[::-1]

        filter_[0 : kernel_size // 4] = -filter_left
        filter_[kernel_size // 4 : kernel_size // 2] = -filter_right
        filter_[kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left
        filter_[3 * kernel_size // 4 : kernel_size] = 2 * filter_right
        filter_[kernel_size : 5 * kernel_size // 4] = -filter_left
        filter_[5 * kernel_size // 4 :] = -filter_right

        conv = DepthwiseConv1D(
            kernel_size=kernel_size + kernel_size // 2,
            padding="same",
            use_bias=False,
            depthwise_initializer=tf.keras.initializers.Constant(filter_.tolist()),
            trainable=False,
            name=f"hybrid_peak_{kernel_size}_{input_tensor.name.split(':')[0]}"
        )(input_tensor)
        conv_list.append(conv)

    x = Concatenate(axis=2)(conv_list)
    x = Activation("relu")(x)
    return x

def lite_inception_module(input_tensor, n_filters=32, kernel_size=40, use_custom_filters=True):
    
    kernel_sizes = [kernel_size // (2**i) for i in range(3)]
    
    conv_list = []
    for k in kernel_sizes:
        conv_list.append(
            SeparableConv1D(
                filters=n_filters,
                kernel_size=k,
                strides=1,
                padding="same",
                use_bias=False,
                activation="linear"
            )(input_tensor)
        )
    
    if use_custom_filters:
        input_channels = int(input_tensor.shape[-1])
        hybrid = make_hybrid_layer(input_tensor, input_channels)
        conv_list.append(hybrid)
        
    if len(conv_list) > 1:
        x = Concatenate(axis=2)(conv_list)
    else:
        x = conv_list[0]
        
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def lite_fcn_module(input_tensor, n_filters=32, kernel_size=20, dilation_rate=1):
    
    x = SeparableConv1D(
        filters=n_filters,
        kernel_size=kernel_size,
        padding="same",
        strides=1,
        dilation_rate=dilation_rate,
        use_bias=False
    )(input_tensor)
    
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def build_lite_backbone(input_tensor, n_filters=32, kernel_size=40):
    
    x = lite_inception_module(
        input_tensor, 
        n_filters=n_filters, 
        kernel_size=kernel_size, 
        use_custom_filters=True
    )
    
    _kernel_size = kernel_size // 2
    for i in range(2):
        dilation_rate = 2 ** (i + 1)
        
        x = lite_fcn_module(
            input_tensor=x,
            kernel_size=_kernel_size // (2**i),
            n_filters=n_filters,
            dilation_rate=dilation_rate
        )
        
    return x


def generate_pmc_mhlite_official(len_x, len_hist, nb_classes):
    
    input_x = Input(shape=(len_x, 1), name="Input_TimeSeries") 
    input_hist = Input(shape=(len_hist, 2), name="Input_Histogram")

    # --- Stream A: Time Series ---
    output_lite_a = build_lite_backbone(input_x, n_filters=32, kernel_size=40)
    
    # --- Stream B: Histogram ---
    output_lite_b = build_lite_backbone(input_hist, n_filters=32, kernel_size=40)

    # --- Integration (Late Fusion) ---
    combined = Concatenate(axis=1, name="Concat_TimeAxis")([output_lite_a, output_lite_b])
    
    # Global Average Pooling
    gap_layer = GlobalAveragePooling1D(name="Global_Average_Pooling")(combined)
    
    # Classification
    output_layer = Dense(nb_classes, activation='softmax', name="Output_Softmax")(gap_layer)

    model = Model(inputs=[input_x, input_hist], outputs=output_layer)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    return model


def normalization(data):
    ans = []
    for i in range(len(data)):
        m = np.average(data[i])
        s = np.std(data[i])
        if s == 0: s = 1e-10
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
                f.write(f"{i+1},{self.logs['loss'][i]},{self.logs['acc'][i]},{self.logs['val_loss'][i]},{self.logs['val_acc'][i]},\n")

if __name__ == "__main__":
    flist = list(pathlib.Path("UCR_TS_Archive_2015").iterdir()) 
    flist.sort()

    basepath = pathlib.Path("pmcloutput")
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

        data_basepath = pathlib.Path("") 
        x_train, y_train, hist_s_train, hist_l_train, n_classes = load_data( pathlib.Path(data_basepath, "UCR_TS_Archive_2015", fname.name), fname.name, "_TRAIN")
        x_test, y_test, hist_s_test, hist_l_test, _, = load_data( pathlib.Path(data_basepath, "UCR_TS_Archive_2015", fname.name), fname.name, "_TEST", n_classes)
        
        input_x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        input_x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
        input_hist_train = np.dstack([hist_s_train, hist_l_train])
        input_hist_test = np.dstack([hist_s_test, hist_l_test])

        input_train = [input_x_train, input_hist_train]
        input_test = [input_x_test, input_hist_test]
        
        model = generate_pmc_mhlite_official(input_x_train.shape[1], input_hist_train.shape[1], n_classes)
        
        secondpath = pathlib.Path(fname.name)
        p = pathlib.Path(basepath, secondpath)
        if(not p.exists()):
            p.mkdir(parents=True)
        
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