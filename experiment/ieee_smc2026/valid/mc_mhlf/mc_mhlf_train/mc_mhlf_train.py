# coding:utf-8
import time, os
import pathlib
import numpy as np
import random
import tensorflow
from tensorflow import keras 
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Dropout
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# 環境変数の設定
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

# 固定のシード設定
SEED = 42
def set_seed(seed):
    tensorflow.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    
    hist_s = np.loadtxt(pathlib.Path(path, dataset_name + "_histogram_s" + sel_mode), delimiter=',')[:,1:]
    hist_l = np.loadtxt(pathlib.Path(path, dataset_name + "_histogram_l" + sel_mode), delimiter=',')[:,1:]
    
    if(n_classes == 0):
        n_classes = len(np.unique(y))
    
    y = (y - y.min())/(y.max()-y.min())*(n_classes-1)
    y = to_categorical(y, n_classes)
    return x, y, hist_s, hist_l, n_classes

def generate_mcmhlf(MAX_SEQUENCE_LENGTH, NB_CLASS):
    ip = Input(shape=(MAX_SEQUENCE_LENGTH, 3))
    x = LSTM(32)(ip)
    x = Dropout(0.8)(x)

    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)

    merged = concatenate([x, y])
    out = Dense(NB_CLASS, activation='softmax')(merged)

    model = Model(ip, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

batch_size_dic = {
    "Adiac":8, "CBF":16, "Beef":8, "Computers":128, "ECG5000":32, "Gun_Point":8, "wafer":8,
}

class myCSVlogger(keras.callbacks.Callback):
    def __init__(self, output_path, x_test, y_test):
        self.output_path = output_path
        self.logs = {"acc":[], "loss":[], "val_acc":[], "val_loss":[]}
        self.x_test = x_test
        self.y_test = y_test

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

    base_out = pathlib.Path("mcoutput5")
    base_out.mkdir(parents=True, exist_ok=True)

    # 実行時間ログの初期化
    output_time_path = base_out / "execution_times.csv"
    with open(output_time_path, "w") as f:
        f.write("dataset,time_seconds\n")

    for fname in flist:
        if fname.name not in batch_size_dic:
            continue
        
        print(f"--- Processing: {fname.name} ---")
        
        # 毎回セッションをクリアして計測を安定させる
        keras.backend.clear_session()
        set_seed(SEED)

        # データ準備
        data_dir = pathlib.Path("UCR_TS_Archive_2015", fname.name)
        x_train, y_train, hs_train, hl_train, n_classes = load_data(data_dir, fname.name, "_TRAIN")
        x_test, y_test, hs_test, hl_test, _ = load_data(data_dir, fname.name, "_TEST", n_classes)
        
        input_train = np.dstack([x_train, hs_train, hl_train])
        input_test = np.dstack([x_test, hs_test, hl_test])

        # モデル構築
        model = generate_mcmhlf(input_train.shape[1], n_classes)
        
        p = base_out / fname.name
        p.mkdir(parents=True, exist_ok=True)
        
        # --- 純粋な学習時間の計測開始 ---
        start_t = time.time()

        logger = myCSVlogger(str(p / 'training.log'), input_test, y_test)
        checker = ModelCheckpoint(str(p / 'model.h5'), monitor='loss', save_best_only=True)
        
        model.fit(input_train, y_train, validation_data=(input_test, y_test), 
                  epochs=2000, batch_size=batch_size_dic[fname.name],
                  callbacks=[logger, checker], verbose=1) # 進捗を表示
        
        elapsed = time.time() - start_t
        # --- 計測終了 ---

        print(f"Done. Time: {elapsed:.2f}s")

        # 結果を追記
        with open(output_time_path, "a") as f:
            f.write(f"{fname.name},{elapsed:.4f}\n")