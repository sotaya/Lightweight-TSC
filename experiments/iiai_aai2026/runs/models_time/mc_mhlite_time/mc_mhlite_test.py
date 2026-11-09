# coding:utf-8

import time, os
import pathlib
import numpy as np
import tensorflow
from tensorflow import keras 
from tensorflow.keras.models import load_model

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
    data_path = pathlib.Path(path, dataset_name + "_TEST")
    data = np.loadtxt(data_path, delimiter=',')
    x = normalization(data[:, 1:])
    y = data[:, 0].astype(float)
    
    # Histogram features
    hist_s = np.loadtxt(pathlib.Path(path, dataset_name + "_histogram_s_TEST"), delimiter=',')[:,1:]
    hist_l = np.loadtxt(pathlib.Path(path, dataset_name + "_histogram_l_TEST"), delimiter=',')[:,1:]
    
    n_classes = len(np.unique(y))
    return x, hist_s, hist_l, n_classes

if __name__ == "__main__":
    flist = list(pathlib.Path("UCR_TS_Archive_2015").iterdir()) 
    flist.sort()

    model_base_dir = pathlib.Path("mcloutput3")
    
    inference_out = pathlib.Path("test_mcloutput3")
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
        x_test, hs_test, hl_test, _ = load_data_for_inference(data_dir, dataset_name)
        
        input_data = np.dstack([x_test, hs_test, hl_test])
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