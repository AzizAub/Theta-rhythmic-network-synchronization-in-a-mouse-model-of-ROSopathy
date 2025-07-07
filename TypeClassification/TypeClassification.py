import os
import glob
import numpy as np
import pywt
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from joblib import dump

# Parameters
DATA_DIR = r"c:\Aziz\MasterThesis\MasterTh\Super\WTTGwithShufle"  # Root folder containing train, val, test folders
scales = np.arange(9, 200)  # Example of optimized scales (depends on the task)
fs = 1250.0             # Signal sampling frequency
window_sec = 20.0       # Window size in seconds

# Function for feature extraction using CWT with parallelization over windows
def extract_cwt_features(signal, scales, wavelet='morl', fs=1250.0, window_sec=20.0):
    window_size = int(window_sec * fs)
    trim_len = (len(signal) // window_size) * window_size
    sig_cut = signal[:trim_len]
    
    num_windows = trim_len // window_size

    # Function to process a single window
    def process_window(w):
        start = w * window_size
        end = start + window_size
        segment = sig_cut[start:end]
        coefs, freqs = pywt.cwt(segment, scales, wavelet)
        energies = np.sum(np.square(coefs), axis=1)
        log_energies = np.log1p(energies)
        return log_energies

    # Parallel processing of windows
    feature_list = Parallel(n_jobs=-1)(delayed(process_window)(w) for w in range(num_windows))
    return np.array(feature_list)

# Function to process a single file (parallelized over files)
def process_file(fpath, scales_param, fs, window_sec):
    data = sio.loadmat(fpath)
    if 'thetaeeg' not in data:
        print(f"Skipping {fpath}, 'thetaeeg' not found")
        return None
    signal = data['thetaeeg'].squeeze()
    if len(signal) == 0:
        print(f"Empty signal in {fpath}")
        return None
    feat = extract_cwt_features(signal, scales_param, wavelet='morl', fs=fs, window_sec=window_sec)
    return feat

# Function to load data from a specified folder (e.g., train, val, test)
def load_data_from_folder(dataset_dir, scales_param):
    X_list = []
    y_list = []
    # Classes: 0 = WT, 1 = TG
    for label, class_folder in enumerate(["WT", "TG"]):
        folder_path = os.path.join(dataset_dir, class_folder)
        file_list = glob.glob(os.path.join(folder_path, "*.mat"))
        print(f"Folder: {folder_path} - found {len(file_list)} files")
        
        # Process files in parallel
        feats = Parallel(n_jobs=-1)(delayed(process_file)(fpath, scales_param, fs, window_sec) for fpath in file_list)
        # Filter out files where processing failed (None)
        feats = [f for f in feats if f is not None]
        if feats:
            X_file = np.vstack(feats)
            X_list.append(X_file)
            # For each file, the number of windows is f.shape[0]
            y_list.append(np.hstack([[label] * f.shape[0] for f in feats]))
    if not X_list:
        print(f"Error: No files loaded in folder {dataset_dir}!")
        return np.array([]), np.array([])
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y

# Function to load data for all sets (train, val, test)
def load_all_data(root_dir, scales_param):
    datasets = {}
    for ds in ["train", "val", "test"]:
        ds_dir = os.path.join(root_dir, ds)
        X_ds, y_ds = load_data_from_folder(ds_dir, scales_param)
        datasets[ds] = (X_ds, y_ds)
    return datasets

# Load data for all sets: train, val, test
data = load_all_data(DATA_DIR, scales)
X_train, y_train = data["train"]
X_val, y_val = data["val"]
X_test, y_test = data["test"]

print("Train X shape:", X_train.shape)
print("Val X shape:", X_val.shape)
print("Test X shape:", X_test.shape)

# PCA
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)
print("After PCA, Train X shape:", X_train_pca.shape)

# Normalization
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_pca)
X_val_norm = scaler.transform(X_val_pca)
X_test_norm = scaler.transform(X_test_pca)

# scaler = StandardScaler()
# X_train_norm = scaler.fit_transform(X_train)
# X_val_norm = scaler.transform(X_val)
# X_test_norm = scaler.transform(X_test)

# Train SVM
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_norm, y_train)

# Evaluate on Validation set
y_val_pred = svm_model.predict(X_val_norm)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on Test set
y_test_pred = svm_model.predict(X_test_norm)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

dump(pca,    "pca_model.joblib")
dump(scaler, "scaler_model.joblib")
dump(svm_model, "svm_model.joblib")
print("Saved pca_model.joblib, scaler_model.joblib, svm_model.joblib")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Test Confusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix
import seaborn as sns
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted WT", "Predicted TG"],
            yticklabels=["Actual WT", "Actual TG"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve on Test Set
y_prob = svm_model.predict_proba(X_test_norm)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (SVM)')
plt.legend()
plt.grid()
plt.show()