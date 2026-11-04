# coding:utf-8
import time, os
import pathlib
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Concatenate, Input, BatchNormalization, Activation, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical

# --- 安定化設定 ---
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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


# --- 改善版モデル生成関数（LITE風の構造を安全に独立構築） ---
def generate_pmc_mhlite_improved(time_series_length, n_classes):
    """
    各ブランチで独立した畳み込みネットワークを持ちつつ、
    レイヤー名の重複（名前の衝突）が起きないように安全に構築するモデル。
    """
    # 入力定義[cite: 3]
    global_input_a = Input(shape=(time_series_length, 2), name="Input_Long_Pair")
    global_input_b = Input(shape=(time_series_length, 2), name="Input_Short_Pair")
    
    # 独立したブランチを作成するためのヘルパー関数（プレフィックスで名前重複を防止）
    def create_branch(input_tensor, prefix):
        x = Conv1D(filters=32, kernel_size=3, padding='same', name=f"{prefix}_conv1")(input_tensor)
        x = BatchNormalization(name=f"{prefix}_bn1")(x)
        x = Activation('relu', name=f"{prefix}_relu1")(x)
        
        x = Conv1D(filters=64, kernel_size=3, padding='same', name=f"{prefix}_conv2")(x)
        x = BatchNormalization(name=f"{prefix}_bn2")(x)
        x = Activation('relu', name=f"{prefix}_relu2")(x)
        
        x = GlobalAveragePooling1D(name=f"{prefix}_gap")(x)
        return x

    # それぞれ異なるプレフィックスを与えて特徴量を抽出
    feat_a = create_branch(global_input_a, "branch_a")
    feat_b = create_branch(global_input_b, "branch_b")
   
    # 特徴量の結合[cite: 3]
    combined = Concatenate(name="concat_features")([feat_a, feat_b])
    
    # 分類層[cite: 3]
    output = Dense(n_classes, activation='softmax', name="classification_output")(combined)
    
    model = Model(inputs=[global_input_a, global_input_b], outputs=output)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

batch_size_dic = {
    "Adiac": 8, "Beef": 8, "CBF": 16, "Computers": 128, 
    "ECG200": 32, "Gun_Point": 8, "wafer": 8
}

if __name__ == "__main__":
    flist = list(pathlib.Path("UCR_TS_Archive_2015").iterdir()) 
    flist.sort()

    output_root_path = pathlib.Path("pmc2loutput_improved")
    if not output_root_path.exists():
        output_root_path.mkdir(parents=True)

    output_time_path = pathlib.Path(output_root_path, "execution_times.csv")
    with open(output_time_path, "w") as f:
        f.write("dataset,time_seconds\n")
    print(f"Time log file initialized at {output_time_path}")

    for fname in flist:
        if fname.name not in batch_size_dic:
            continue
        
        print(f"--- Processing: {fname.name} ---")
        start_time = time.time()

        # データロード[cite: 3]
        data_archive_path = pathlib.Path("UCR_TS_Archive_2015", fname.name)
        x_train, y_train, hist_s_train, hist_l_train, n_classes = load_data(data_archive_path, fname.name, "_TRAIN")
        x_test, y_test, hist_s_test, hist_l_test, _ = load_data(data_archive_path, fname.name, "_TEST", n_classes)
        
        # データの結合と整形[cite: 3]
        input_1_train = np.dstack([x_train, hist_l_train])
        input_1_test  = np.dstack([x_test,  hist_l_test])
        input_2_train = np.dstack([x_train, hist_s_train])
        input_2_test  = np.dstack([x_test,  hist_s_test])
            
        input_train = [input_1_train, input_2_train]
        input_test  = [input_1_test,  input_2_test]

        # モデル生成[cite: 3]
        model = generate_pmc_mhlite_improved(input_1_train.shape[1], n_classes)
        
        # ログ保存先[cite: 3]
        dataset_output_path = pathlib.Path(output_root_path, fname.name)
        if(not dataset_output_path.exists()):
            dataset_output_path.mkdir(parents=True)
        
        # 標準のCSVLoggerを使用[cite: 3]
        log_path = str(pathlib.Path(dataset_output_path, 'training.log'))
        csv_logger = CSVLogger(log_path, separator=',', append=False)
        
        checker = ModelCheckpoint(str(pathlib.Path(dataset_output_path, 'model.h5')), 
                                  monitor='val_loss', verbose=0, save_best_only=True, 
                                  save_weights_only=False, mode='min')
        
        # 訓練実行[cite: 3]
        history = model.fit(
            input_train, y_train, 
            validation_data=(input_test, y_test), 
            epochs=2000, 
            batch_size=batch_size_dic[fname.name],
            callbacks=[csv_logger, checker],
            verbose=1
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done. Time taken: {elapsed_time:.2f} seconds")

        with open(output_time_path, "a") as f:
            f.write(f"{fname.name},{elapsed_time}\n")

        keras.backend.clear_session()