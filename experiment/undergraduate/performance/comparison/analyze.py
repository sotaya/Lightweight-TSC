import pandas as pd
import pathlib
import re
import sys
import os
import math

# TensorFlow の不要なログ（INFOやWARNING）を抑制する
# '1' = INFOを抑制, '2' = WARNINGも抑制, '3' = ERRORも抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("エラー: TensorFlow がインストールされていません。")
    print("このスクリプトを実行するには 'pip install tensorflow' が必要です。")
    sys.exit(1)


def count_hdf5_model_parameters(model_path):
    """Count saved layer-weight values without deserializing the model."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5読み込み用の h5py がありません。") from exc

    with h5py.File(model_path, "r") as model_file:
        model_weights = model_file.get("model_weights")
        if model_weights is None:
            raise ValueError("HDF5ファイルに model_weights グループがありません。")

        parameter_count = 0
        found_weights = False

        def add_weight_values(_, item):
            nonlocal parameter_count, found_weights
            if isinstance(item, h5py.Dataset):
                parameter_count += math.prod(item.shape)
                found_weights = True

        model_weights.visititems(add_weight_values)
        if not found_weights:
            raise ValueError("model_weights グループに重みデータがありません。")

    return parameter_count


def analyze_results():
    """
    10個のデータセットの実行結果（精度とパラメータ数）を集計し、
    analysis_summary.csv に保存します。
    
    変更点：ログファイルの代わりに 'model.h5' を直接読み込んでパラメータ数をカウントします。
    """
    
    # 1. 分析対象のデータセット
    dataset_dirs = [
    "50words",
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "ChlorineConcentration",
    "CinC_ECG_torso",
    "Coffee",
    "Computers",
    "Cricket_X",
    "Cricket_Y",
    "Cricket_Z",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FISH",
    "FordA",
    "FordB",
    "Gun_Point",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lighting2",
    "Lighting7",
    "MALLAT",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MoteStrain",
    "NonInvasiveFatalECG_Thorax1",
    "NonInvasiveFatalECG_Thorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "Plane",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "ScreenType",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SonyAIBORobotSurface",
    "SonyAIBORobotSurfaceII",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "synthetic_control",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "Two_Patterns",
    "TwoLeadECG",
    "uWaveGestureLibrary_X",
    "uWaveGestureLibrary_Y",
    "uWaveGestureLibrary_Z",
    "UWaveGestureLibraryAll",
    "wafer",
    "Wine",
    "WordsSynonyms",
    "Worms",
    "WormsTwoClass",
    "yoga"
    ]
    
    # 2. 結果が保存されているベースパス
    base_output_path = pathlib.Path("progress/valid/s_pmc_mhlite/pmc2loutput_v2")
    
    # 最終的な結果を格納するリスト
    results = []
    
    print(f"--- 分析を開始します ---")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"結果ディレクトリ: {base_output_path}")

    # --- 各データセットの training.log と model.h5 を分析 ---
    
    for dataset_name in dataset_dirs:
        print(f"処理中: {dataset_name}")
        
        best_acc = None
        param_count = None
        
        # --- ステップ 1: 最高精度の取得 (training.log) ---
        log_path = base_output_path / dataset_name / "training.log"
        try:
            if log_path.exists():
                log_df = pd.read_csv(log_path)
                
                # 'val_acc' または 'val_accuracy' 列を探す
                if 'val_acc' in log_df.columns:
                    best_acc = log_df['val_acc'].max()
                elif 'val_accuracy' in log_df.columns:
                    best_acc = log_df['val_accuracy'].max()
                else:
                    print(f"  警告: {log_path.name} に 'val_acc' または 'val_accuracy' 列がありません。")
            else:
                print(f"  警告: {log_path} が存在しません。")
                
        except pd.errors.EmptyDataError:
            print(f"  警告: {log_path.name} は空のファイルです。")
        except Exception as e:
            print(f"  エラー: {log_path.name} 読み込み中: {e}")

        # --- ステップ 2: パラメータ数の取得 (model.h5) ---
        model_path = base_output_path / dataset_name / "model.h5"
        try:
            if model_path.exists():
                # モデルを読み込む
                # compile=False にすると、オプティマイザの状態などを読み込まないため高速
                model = keras.models.load_model(str(model_path), compile=False) 
                
                # 総パラメータ数をカウント
                param_count = model.count_params()
                print(f"  {model_path.name} を読み込み成功。パラメータ数: {param_count}")
                
                # メモリを解放（ループ内で多数のモデルを読み込むため重要）
                keras.backend.clear_session()
                del model
            else:
                print(f"  警告: {model_path} が存在しません。")
                
        except Exception as e:
            print(f"  エラー: {model_path.name} の読み込み中に失敗しました: {e}")
            try:
                param_count = count_hdf5_model_parameters(model_path)
                print(f"  HDF5内の保存済み重みからパラメータ数を取得: {param_count}")
            except Exception as h5_error:
                print(f"  HDF5重みからの集計にも失敗しました: {h5_error}")
            finally:
                keras.backend.clear_session() # エラー時もセッションをクリア

        # 3. 結果の保存
        results.append({
            "dataset": dataset_name, 
            "best_accuracy": best_acc,
            "parameter_count": param_count
        })

    # --- ステップ 3: 最終的なCSVファイルに書き出し ---
    output_csv = "analysis_summary.csv"
    try:
        results_df = pd.DataFrame(results)
        
        columns_order = ["dataset", "best_accuracy", "parameter_count"]
        results_df = results_df[columns_order]
        
        # CSVファイルに保存
        results_df.to_csv(output_csv, index=False)
        
        print(f"\n--- 完了 ---")
        print(f"分析結果が '{output_csv}' に保存されました。")
        
        # 結果のプレビューを表示
        print("\n[分析結果の概要]")
        print(results_df.to_string())
        
    except Exception as e:
        print(f"\nエラー: '{output_csv}' の書き込み中にエラーが発生しました: {e}")
        
    return results_df

if __name__ == "__main__":
    analysis_summary = analyze_results()