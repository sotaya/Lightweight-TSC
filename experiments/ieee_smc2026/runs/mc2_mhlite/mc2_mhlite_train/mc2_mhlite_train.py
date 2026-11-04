# coding:utf-8
import time, os
import pathlib
import numpy as np
import random
import tensorflow
from tensorflow import keras 
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from aeon.networks._lite import LITENetwork

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

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

    macd_path = pathlib.Path(path, dataset_name + "_macd_s" + sel_mode)
    md_s_raw = np.genfromtxt(macd_path, delimiter=',', invalid_raise=False)
    
    if len(md_s_raw) != len(x):
        valid_indices = []
        with open(macd_path, 'r') as f:
            for idx, line in enumerate(f):
                if len(line.split(',')) == (md_s_raw.shape[1] if md_s_raw.ndim > 1 else 0): 
                    valid_indices.append(idx)
        x = x[valid_indices]
        y = y[valid_indices]
        hist_s = hist_s[valid_indices]
        hist_l = hist_l[valid_indices]
        md_s = np.genfromtxt(macd_path, delimiter=',', invalid_raise=False)[:, 1:]
    else:
        md_s = md_s_raw[:, 1:]

    md_s = np.nan_to_num(md_s, nan=0.0)

    macd_l_path = pathlib.Path(path, dataset_name + "_macd_l" + sel_mode)
    md_l = np.genfromtxt(macd_l_path, delimiter=',', invalid_raise=False)[:, 1:]
    md_l = np.nan_to_num(md_l, nan=0.0)

    if n_classes == 0:
        n_classes = len(np.unique(y))
    
    y = (y - y.min()) / (y.max() - y.min()) * (n_classes - 1)
    y = to_categorical(y, n_classes)
    
    return x, y, hist_s, hist_l, n_classes, md_s, md_l

def generate_mcmhlite(MAX_SEQUENCE_LENGTH, NB_CLASS):
    lite = LITENetwork()
    input_layer, gap = lite.build_network(input_shape=(MAX_SEQUENCE_LENGTH, 5))
    output_layer = tensorflow.keras.layers.Dense(NB_CLASS, activation='softmax')(gap)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="categorical_crossentropy", optimizer=tensorflow.keras.optimizers.Adam(), metrics=["accuracy"])
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
        self.logs["val_loss"].append(logs.get('val_loss', 0.0))
        self.logs["val_acc"].append(logs.get('val_accuracy', logs.get('val_acc', 0.0)))
        
    def on_train_end(self, logs):
        with open(self.output_path, "w") as f:
            f.write("epoch,loss,acc,val_loss,val_acc,\n")
            for i in range(len(self.logs["loss"])):
                f.write(f"{i+1},{self.logs['loss'][i]},{self.logs['acc'][i]},{self.logs['val_loss'][i]},{self.logs['val_acc'][i]},\n")

if __name__ == "__main__":
    flist = list(pathlib.Path("UCR_TS_Archive_2015").iterdir()) 
    flist.sort()
    
    # ベースディレクトリの定義
    base_out = pathlib.Path("mc2loutput1")
    base_out.mkdir(parents=True, exist_ok=True)

    # 🛠️ 出力CSVファイル名を mc_mhlite と同じ「times.csv」に統一
    output_time_path = base_out / "times.csv"
    if not output_time_path.exists():
        with open(output_time_path, "w") as f:
            # 🛠️ ヘッダーも mc_mhlite の形式（dataset,time）に統一
            f.write("dataset,time\n")

    for fname in flist:
        if fname.name not in batch_size_dic:
            continue
        
        print(f"--- Processing: {fname.name} ---")
        keras.backend.clear_session()
        set_seed(SEED)

        data_dir = pathlib.Path("UCR_TS_Archive_2015", fname.name)
        x_tr, y_tr, hs_tr, hl_tr, n_classes, ms_tr, ml_tr = load_data(data_dir, fname.name, "_TRAIN")
        x_te, y_te, hs_te, hl_te, _, ms_te, ml_te = load_data(data_dir, fname.name, "_TEST", n_classes=n_classes)
        
        min_tr = min(len(x_tr), len(ms_tr), len(ml_tr), len(hs_tr), len(hl_tr))
        min_te = min(len(x_te), len(ms_te), len(ml_te), len(hs_te), len(hl_te))
        
        input_train = np.dstack([x_tr[:min_tr], ms_tr[:min_tr], ml_tr[:min_tr], hs_tr[:min_tr], hl_tr[:min_tr]])
        input_test = np.dstack([x_te[:min_te], ms_te[:min_te], ml_te[:min_te], hs_te[:min_te], hl_te[:min_te]])
        y_train, y_test = y_tr[:min_tr], y_te[:min_te]

        model = generate_mcmhlite(input_train.shape[1], n_classes)
        p = base_out / fname.name
        p.mkdir(parents=True, exist_ok=True)
        
        # --- 純粋な学習時間の計測 ---
        start_t = time.time()

        logger = myCSVlogger(str(p / 'training.log'), input_test, y_test)
        checker = ModelCheckpoint(str(p / 'model.h5'), monitor='loss', save_best_only=True, verbose=0)
        
        model.fit(input_train, y_train, validation_data=(input_test, y_test), 
                  epochs=2000, batch_size=batch_size_dic[fname.name],
                  callbacks=[logger, checker], verbose=1)
        
        elapsed = time.time() - start_t
        # --- 計測終了 ---

        print(f"Done. Time: {elapsed:.2f}s")
        with open(output_time_path, "a") as f:
            # 🛠️ mc_mhlite と同じ書式で追記
            f.write(f"{fname.name},{elapsed}\n")