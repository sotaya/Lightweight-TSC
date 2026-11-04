# coding:utf-8
import time, os
import pathlib
import numpy as np
import tensorflow
from tensorflow import keras 
from tensorflow.keras.models import load_model

# --- 環境設定 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def normalization(data):
    ans = []
    for i in range(len(data)):
        m = np.average(data[i])
        s = np.std(data[i])
        ans.append((data[i] - m) / s)
    return np.array(ans)

def load_data_for_inference(path, dataset_name):
    """LITE推論用にテストデータ（生データのみ）をロード"""
    data_path = pathlib.Path(path, dataset_name + "_TEST")
    data = np.loadtxt(data_path, delimiter=',')
    # 生データの正規化
    x = normalization(data[:, 1:])
    y = data[:, 0].astype(float)
    
    n_classes = len(np.unique(y))
    return x, n_classes

if __name__ == "__main__":
    archive_path = pathlib.Path("UCR_TS_Archive_2015")
    flist = list(archive_path.iterdir()) 
    flist.sort()

    # 学習済みLITEモデルが保存されているベースディレクトリ
    model_base_dir = pathlib.Path("loutput5")
    
    # 推論結果の保存先
    inference_out = pathlib.Path("test_loutput5")
    inference_out.mkdir(parents=True, exist_ok=True)

    # 推論時間ログの初期化
    output_time_path = inference_out / "inference_times.csv"
    with open(output_time_path, "w") as f:
        f.write("dataset,sample_count,total_inference_time_seconds,time_per_sample_ms\n")

    for fname in flist:
        dataset_name = fname.name
        model_path = model_base_dir / dataset_name / "model.h5"
        
        # モデルが存在しないデータセットはスキップ
        if not model_path.exists():
            continue
        
        print(f"--- Inference (LITE): {dataset_name} ---")
        
        # 1. テストデータの準備
        data_dir = archive_path / dataset_name
        x_test, _ = load_data_for_inference(data_dir, dataset_name)
        
        # LITE用の形状変換 (samples, length, 1チャネル)
        input_data = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        sample_count = input_data.shape[0]

        # 2. モデルのロード
        model = load_model(str(model_path))

        # --- 純粋な推論時間の計測開始 ---
        # ウォームアップ: GPUの初期化や最適化による遅延を計測から除外
        _ = model.predict(input_data, batch_size=128, verbose=0) 
        
        start_t = time.time()
        
        # テストセット全体に対して推論を実行
        _ = model.predict(input_data, batch_size=128, verbose=0)
        
        elapsed = time.time() - start_t
        # --- 計測終了 ---

        # 1サンプルあたりの平均推論時間 (ミリ秒)
        time_per_sample = (elapsed / sample_count) * 1000 
        
        print(f" Done. Samples: {sample_count}, Total: {elapsed:.4f}s, Per-sample: {time_per_sample:.4f}ms")

        # 結果をCSVに保存
        with open(output_time_path, "a") as f:
            f.write(f"{dataset_name},{sample_count},{elapsed:.6f},{time_per_sample:.6f}\n")
            
        # メモリリーク防止のためのセッションクリア
        keras.backend.clear_session()