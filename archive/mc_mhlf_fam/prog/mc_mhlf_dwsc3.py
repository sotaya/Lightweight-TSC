# coding:utf-8

import pathlib, sys

import numpy as np
import pandas as pd

import tensorflow

from tensorflow import keras 

from tensorflow.keras import backend as K  
from tensorflow.keras.layers import Conv1D, SeparableConv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
from tensorflow.keras.layers import Input, Dense, LSTM,concatenate, Activation, GRU, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from tensorflow.keras.utils import to_categorical

import time, os

os.environ['KMP_DUPLICATE_LIB_OK']="True"###

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

def generate_mcmhlfdwsc(MAX_SEQUENCE_LENGTH, NB_CLASS):

    ip = Input(shape=(MAX_SEQUENCE_LENGTH, 3))

    x = LSTM(32)(ip)
    x = Dropout(0.8)(x)

    y = SeparableConv1D(128, 8, padding='same', depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = SeparableConv1D(256, 5, padding='same', depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = SeparableConv1D(128, 3, padding='same', depthwise_initializer='he_uniform', pointwise_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    x = Dense(NB_CLASS, activation=None)(x)
    out = Activation('softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

batch_size_dic = {
    "StarLightCurves":128,
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
#         self.logs["acc"].append(logs.get('acc'))
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

    flist.sort()   #データセット辞書順並べ替え

    for fname in flist:
        if fname.name not in batch_size_dic:
            continue  # 辞書にないデータセットはスキップする
        #load data
#         basepath = pathlib.Path("./")
        basepath = pathlib.Path("")
        x_train, y_train, hist_s_train, hist_l_train, n_classes, md_s_train, md_l_train = load_data( pathlib.Path(basepath, "UCR_TS_Archive_2015", fname.name), fname.name, "_TRAIN")
        x_test, y_test, hist_s_test, hist_l_test, _, md_s_test, md_l_test= load_data( pathlib.Path(basepath, "UCR_TS_Archive_2015", fname.name), fname.name, "_TEST", n_classes)
            
        #input_train = np.dstack([x_train, hist_s_train, hist_l_train, md_s_train, md_l_train])
        #input_test = np.dstack([x_test, hist_s_test, hist_l_test, md_s_test, md_l_test])

        input_train = np.dstack([x_train, hist_s_train, hist_l_train])
        input_test = np.dstack([x_test, hist_s_test, hist_l_test])


        # built model
        model = generate_mcmhlfdwsc(input_train.shape[1], n_classes)
        
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

