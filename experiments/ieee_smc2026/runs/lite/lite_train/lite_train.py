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

# 環境変数の設定（GPU動作の安定化と再現性のため）
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

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

    if(n_classes == 0):
        n_classes = len(np.unique(y))
    
    # ラベルを0からの範囲にスケーリングしてOne-hotエンコーディング
    y = (y - y.min())/(y.max()-y.min())*(n_classes-1)
    y = to_categorical(y, n_classes)
    
    return x, y, n_classes

def generate_lite(MAX_SEQUENCE_LENGTH, NB_CLASS):
    # aeonのLITENetworkを使用
    lite = LITENetwork()
    # LITEは標準で1チャネル（時系列データのみ）を使用
    input_layer, gap = lite.build_network(input_shape=(MAX_SEQUENCE_LENGTH, 1))
    output_layer = tensorflow.keras.layers.Dense(NB_CLASS, activation='softmax')(gap)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=tensorflow.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    return model

# バッチサイズ辞書（元の設定を維持）
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
        # テストセットが大きい場合のメモリ不足回避のためbatch_size固定
        scores = self.model.evaluate(self.x_test, self.y_test, batch_size=128, verbose=0)
        self.logs["val_loss"].append(scores[0])
        self.logs["val_acc"].append(scores[1])
        
    def on_train_end(self, logs):
        with open(self.output_path, "w") as f:
            f.write("epoch,loss,acc,val_loss,val_acc,\n")
            for i in range(len(self.logs["loss"])):
                f.write(f"{i+1},{self.logs['loss'][i]},{self.logs['acc'][i]},{self.logs['val_loss'][i]},{self.logs['val_acc'][i]},\n")

if __name__ == "__main__":
    archive_path = pathlib.Path("UCR_TS_Archive_2015")
    flist = list(archive_path.iterdir()) 
    flist.sort()

    # 出力ディレクトリの設定
    base_out = pathlib.Path("loutput5")
    base_out.mkdir(parents=True, exist_ok=True)

    # 実行時間ログの初期化
    output_time_path = base_out / "execution_times.csv"
    with open(output_time_path, "w") as f:
        f.write("dataset,time_seconds\n")

    for fname in flist:
        if fname.name not in batch_size_dic:
            continue
        
        print(f"--- Processing: {fname.name} ---")
        
        # 学習ごとにリセット
        keras.backend.clear_session()
        set_seed(SEED)

        # データ読み込み
        data_dir = archive_path / fname.name
        x_train, y_train, n_classes = load_data(data_dir, fname.name, "_TRAIN")
        x_test, y_test, _ = load_data(data_dir, fname.name, "_TEST", n_classes)
        
        # LITE用の形状変換 (samples, length, 1)
        input_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        input_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # モデル構築
        model = generate_lite(input_train.shape[1], n_classes)
        
        p = base_out / fname.name
        p.mkdir(parents=True, exist_ok=True)
        
        # --- 学習時間の計測開始 ---
        start_t = time.time()

        logger = myCSVlogger(str(p / 'training.log'), input_test, y_test)
        # 拡張性を考慮してsave_best_onlyを有効化
        checker = ModelCheckpoint(str(p / 'model.h5'), monitor='loss', save_best_only=True, verbose=0)
        
        model.fit(input_train, y_train, validation_data=(input_test, y_test), 
                  epochs=2000, batch_size=batch_size_dic[fname.name],
                  callbacks=[logger, checker], verbose=1)
        
        elapsed = time.time() - start_t
        # --- 計測終了 ---

        print(f"Done. Time: {elapsed:.2f}s")

        # 結果をCSVに保存
        with open(output_time_path, "a") as f:
            f.write(f"{fname.name},{elapsed:.4f}\n")

        # メモリリークを防ぐためにセッションをクリア（推奨）
        keras.backend.clear_session()