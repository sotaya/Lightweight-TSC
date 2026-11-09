# coding:utf-8

import time, os
import pathlib
import numpy as np
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
    
    n_classes = len(np.unique(y))
    return x, n_classes

if __name__ == "__main__":
    archive_path = pathlib.Path("UCR_TS_Archive_2015")
    flist = list(archive_path.iterdir()) 
    flist.sort()

    model_base_dir = pathlib.Path("loutput5")
    
    inference_out = pathlib.Path("test_loutput5")
    inference_out.mkdir(parents=True, exist_ok=True)

    output_time_path = inference_out / "inference_times.csv"
    with open(output_time_path, "w") as f:
        f.write("dataset,sample_count,total_inference_time_seconds,time_per_sample_ms\n")

    for fname in flist:
        dataset_name = fname.name
        model_path = model_base_dir / dataset_name / "model.h5"
        
        if not model_path.exists():
            continue
        
        print(f"--- Inference (LITE): {dataset_name} ---")
        
        data_dir = archive_path / dataset_name
        x_test, _ = load_data_for_inference(data_dir, dataset_name)
        
        input_data = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
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