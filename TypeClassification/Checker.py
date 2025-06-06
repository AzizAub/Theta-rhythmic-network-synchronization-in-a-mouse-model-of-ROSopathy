import os
import glob
import numpy as np
import pywt
import scipy.io as sio
from joblib import load
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Classes must be in the same order
CLASS_NAMES = ["WT", "TG"]

# CWT Parameters
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
    # LOAD pca, scaler, svm
    pca    = load("pca_model.joblib")
    scaler = load("scaler_model.joblib")
    svm_model = load("svm_model.joblib")
    print("Loaded PCA, Scaler, and SVM model from joblib files.")

    # New .mat file without a label
    new_fpath = r"c:\Aziz\MasterThesis\MasterTh\Super\WTTGwithShufle\B966_day_3_1-20241118T2G_Tdet6_channel4.mat"  # Example
    
    data = sio.loadmat(new_fpath)
    if 'thetaeeg' not in data:
        print(f"No 'thetaeeg' in {new_fpath}")
        return
    signal = data['thetaeeg'].squeeze()
    if len(signal) == 0:
        print("Empty signal!")
        return
    
    # Extract features (CWT)
    X_new = extract_cwt_features(signal, scales, 'morl', fs, window_sec)
    print("New data shape:", X_new.shape)
    
    # PCA transform
    X_new_pca = pca.transform(X_new)

    # Scale
    X_new_norm = scaler.transform(X_new_pca)
    
    # Predict
    y_pred = svm_model.predict(X_new_norm)
    
    # y_pred contains the label for each window
    pred_counts = np.bincount(y_pred)
    majority_label = np.argmax(pred_counts)
    majority_class = CLASS_NAMES[majority_label]
    
    print(f"Prediction per window: {y_pred}")
    print("Majority label:", majority_label, " => Class =", majority_class)

if __name__ == "__main__":
    main()