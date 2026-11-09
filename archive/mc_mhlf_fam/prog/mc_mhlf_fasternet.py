# coding:utf-8

import pathlib, os
import numpy as np
import tensorflow
from tensorflow import keras  
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Dropout, Layer
from tensorflow.keras.layers import Input, Dense, LSTM,concatenate, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

os.environ['KMP_DUPLICATE_LIB_OK']="True"

class PartialConv1D(Layer):
    def __init__(self, kernel_size, n_div=4, **kwargs):
        super(PartialConv1D, self).__init__(**kwargs)
        self.n_div = n_div
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channels = input_shape[-1]
        self.dim_conv = channels // self.n_div
        self.dim_untouched = channels - self.dim_conv
        
        self.partial_conv = Conv1D(self.dim_conv, self.kernel_size, padding='same', bias_initializer='zeros')

        super(PartialConv1D, self).build(input_shape)

    def call(self, x):
        x1, x2 = tensorflow.split(x, [self.dim_conv, self.dim_untouched], axis=-1)
        x1 = self.partial_conv(x1)
        out = concatenate([x1, x2], axis=-1)
        return out
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_div": self.n_div,
            "kernel_size": self.kernel_size,
        })
        return config


def FasterNetBlock(input_tensor, mlp_ratio=2.0, drop_path=0.0):
    ch_in = input_tensor.shape[-1]
    mlp_hidden_dim = int(ch_in * mlp_ratio)

    x = PartialConv1D(kernel_size=3)(input_tensor)

    shortcut = input_tensor
    
    x = Conv1D(filters=mlp_hidden_dim, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=ch_in, kernel_size=1, padding='same')(x)
    
    if drop_path > 0.:
        x = Dropout(drop_path)(x)
    
    x = Add()([shortcut, x])
    
    return x

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
    
    macd_path = pathlib.Path(path, dataset_name + "_macd_s" + sel_mode)
    md_s = np.loadtxt(macd_path, delimiter=',')[:,1:]

    macd_path = pathlib.Path(path, dataset_name + "_macd_l" + sel_mode)
    md_l = np.loadtxt(macd_path, delimiter=',')[:,1:]

    if(n_classes == 0):
        n_classes = len(np.unique(y))
        y = (y - y.min())/(y.max()-y.min())*(n_classes-1)
        y = to_categorical(y, n_classes)
    else:
        y = (y - y.min())/(y.max()-y.min())*(n_classes-1)
        y = to_categorical(y, n_classes)
    
    return x, y, hist_s, hist_l, n_classes, md_s, md_l

def generate_mcmhlf(MAX_SEQUENCE_LENGTH, NB_CLASS):

    ip = Input(shape=(MAX_SEQUENCE_LENGTH, 3))

    x = LSTM(32)(ip)
    x = Dropout(0.8)(x)

    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = FasterNetBlock(y, mlp_ratio=2.0)
    y = FasterNetBlock(y, mlp_ratio=2.0)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    x = Dense(NB_CLASS, activation=None)(x)
    out = Activation('softmax')(x)

    model = Model(ip, out)

    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

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

    for fname in flist:
        basepath = pathlib.Path("")
        x_train, y_train, hist_s_train, hist_l_train, n_classes, md_s_train, md_l_train = load_data( pathlib.Path(basepath, "UCR_TS_Archive_2015", fname.name), fname.name, "_TRAIN")
        x_test, y_test, hist_s_test, hist_l_test, _, md_s_test, md_l_test= load_data( pathlib.Path(basepath, "UCR_TS_Archive_2015", fname.name), fname.name, "_TEST", n_classes)

        input_train = np.dstack([x_train, hist_s_train, hist_l_train])
        input_test = np.dstack([x_test, hist_s_test, hist_l_test])


        # built model
        model = generate_mcmhlf(input_train.shape[1], n_classes)
        
        #make log file
        basepath = pathlib.Path("mcoutput")
        secondpath = pathlib.Path(fname.name)
        p = pathlib.Path(basepath, secondpath)
        if(not p.exists()):
            p.mkdir(parents=True)
        
        #fitting
        print(fname)
        logger = myCSVlogger( str(pathlib.Path(basepath, secondpath, 'training.log')), input_test, y_test, fname.name )
        # Using .h5 format for better compatibility
        checker = ModelCheckpoint(str(pathlib.Path(basepath, secondpath, 'model.h5')), 
                                  monitor='loss', verbose=0, save_best_only=True, 
                                  save_weights_only=False, mode='min')
        history = model.fit(input_train, y_train, validation_data = (input_test, y_test), 
                                epochs=2000, batch_size=batch_size_dic[fname.name],
                                callbacks=[logger, checker])

