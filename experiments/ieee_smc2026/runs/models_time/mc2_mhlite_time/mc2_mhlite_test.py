# coding:utf-8
import time, os
import pathlib
import numpy as np
import tensorflow
from tensorflow import keras 
from tensorflow.keras.models import load_model

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def normalization(data):
    ans = []
    for i in range(len(data)):
        m = np.average(data[i])
        s = np.std(data[i])
        ans.append((data[i] - m) / s)
    return np.array(ans)

def load_data_for_inference(path, dataset_name):
    data_path = pathlib.Path(path, dataset_name + "_TEST")
    data = np.loadtxt(data_path, delimiter=',')
    x = normalization(data[:, 1:])
    
    hist_s = np.loadtxt(pathlib.Path(path, dataset_name + "_histogram_s_TEST"), delimiter=',')[:,1:]
    hist_l = np.loadtxt(pathlib.Path(path, dataset_name + "_histogram_l_TEST"), delimiter=',')[:,1:]
    
    macd_path = pathlib.Path(path, dataset_name + "_macd_s_TEST")
    md_s_raw = np.genfromtxt(macd_path, delimiter=',', invalid_raise=False)
    
    if len(md_s_raw) != len(x):
        valid_indices = []
        with open(macd_path, 'r') as f:
            for idx, line in enumerate(f):
                if len(line.split(',')) == (md_s_raw.shape[1] if md_s_raw.ndim > 1 else 0): 
                    valid_indices.append(idx)
        x = x[valid_indices]
        hist_s = hist_s[valid_indices]
        hist_l = hist_l[valid_indices]
        md_s = np.genfromtxt(macd_path, delimiter=',', invalid_raise=False)[:, 1:]
    else:
        md_s = md_s_raw[:, 1:]

    md_s = np.nan_to_num(md_s, nan=0.0)

    macd_l_path = pathlib.Path(path, dataset_name + "_macd_l_TEST")
    md_l = np.genfromtxt(macd_l_path, delimiter=',', invalid_raise=False)[:, 1:]
    md_l = np.nan_to_num(md_l, nan=0.0)
    
    min_len = min(len(x), len(hist_s), len(hist_l), len(md_s), len(md_l))
    return x[:min_len], hist_s[:min_len], hist_l[:min_len], md_s[:min_len], md_l[:min_len]

if __name__ == "__main__":
    flist = list(pathlib.Path("UCR_TS_Archive_2015").iterdir()) 
    flist.sort()

    model_base_dir = pathlib.Path("mc2loutput5")
    inference_out = pathlib.Path("test_mc2loutput5")
    inference_out.mkdir(parents=True, exist_ok=True)

    output_time_path = inference_out / "inference_times.csv"
    with open(output_time_path, "w") as f:
        f.write("dataset,sample_count,total_inference_time_seconds,time_per_sample_ms\n")

    for fname in flist:
        dataset_name = fname.name
        model_path = model_base_dir / dataset_name / "model.h5"
        
        if not model_path.exists():
            continue
        
        print(f"--- Inference: {dataset_name} ---")
        
        data_dir = pathlib.Path("UCR_TS_Archive_2015", dataset_name)
        x_te, hs_te, hl_te, ms_te, ml_te = load_data_for_inference(data_dir, dataset_name)
        
        input_data = np.dstack([x_te, ms_te, ml_te, hs_te, hl_te])
        sample_count = input_data.shape[0]

        model = load_model(str(model_path))

        _ = model.predict(input_data, batch_size=128, verbose=0) 
        
        start_t = time.time()
        _ = model.predict(input_data, batch_size=128, verbose=0)
        elapsed = time.time() - start_t
        
        time_per_sample = (elapsed / sample_count) * 1000 
        print(f" Done. Samples: {sample_count}, Total: {elapsed:.4f}s, Per-sample: {time_per_sample:.4f}ms")

        with open(output_time_path, "a") as f:
            f.write(f"{dataset_name},{sample_count},{elapsed:.6f},{time_per_sample:.6f}\n")
            
        keras.backend.clear_session()