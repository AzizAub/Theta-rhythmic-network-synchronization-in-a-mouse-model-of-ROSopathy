import os
import glob
import numpy as np
import pywt
import scipy.io as sio
from joblib import load
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Classes must be in the same order that was used during training:
CLASS_NAMES = ["SO", "SP", "SR", "SLM"]

# CWT parameters
scales = np.arange(9, 200)
fs = 1250.0
window_sec = 20.0

def extract_cwt_features(signal, scales, wavelet='morl', fs=1250.0, window_sec=20.0):
    window_size = int(window_sec * fs)
    trim_len = (len(signal) // window_size) * window_size
    sig_cut = signal[:trim_len]
    num_windows = trim_len // window_size

    feats = []
    for w in range(num_windows):
        start = w * window_size
        end   = start + window_size
        segment = sig_cut[start:end]
        coefs, freqs = pywt.cwt(segment, scales, wavelet)
        energies = np.sum(np.square(coefs), axis=1)
        log_energies = np.log1p(energies)
        feats.append(log_energies)

    return np.array(feats)

def main():
    # 1) Load the saved pipeline
    pca    = load("pca_modelHLayers_without_TG_data.joblib")
    scaler = load("scaler_modelHLayers_without_TG_data.joblib")
    svm_model = load("svm_modelHLayers_without_TG_data.joblib")
    print("Loaded PCA, Scaler, and SVM model from joblib files.")

    # 2) Specify the folder that contains .mat files to check
    folder_path = r"c:\Aziz\MasterThesis\MasterTh\Super\SO_SP_SR_SLMwithShufle"

    # 3) Gather all .mat files in that folder (or you can do e.g. *.mat recursively)
    mat_files = glob.glob(os.path.join(folder_path, "*.mat"))
    print(f"Found {len(mat_files)} .mat files in {folder_path}")

    # 4) Process each file
    for fpath in mat_files:
        print("\n--------------------------------------")
        print(f"Processing file: {fpath}")

        data = sio.loadmat(fpath)
        if 'thetaeeg' not in data:
            print(f"No 'thetaeeg' in {fpath}, skipping.")
            continue

        signal = data['thetaeeg'].squeeze()
        if len(signal) == 0:
            print("Empty signal, skipping.")
            continue

        # 5) Extract CWT features
        X_new = extract_cwt_features(signal, scales, 'morl', fs, window_sec)
        print(f"New data shape: {X_new.shape}")

        # 6) Apply PCA transform
        X_new_pca = pca.transform(X_new)
        print("After PCA:", X_new_pca.shape)

        # 7) Scale
        X_new_norm = scaler.transform(X_new_pca)

        # 8) Predict (each window gets a label)
        y_pred = svm_model.predict(X_new_norm)

        # 9) Majority vote
        pred_counts = np.bincount(y_pred)
        majority_label = np.argmax(pred_counts)
        majority_class = CLASS_NAMES[majority_label]

        # 10) Print results
        print(f"Prediction per window (first 30 shown): {y_pred[:30]} ...")
        print(f"Majority label: {majority_label} => Class = {majority_class}")

if __name__ == "__main__":
    main()