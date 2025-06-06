import os
import glob
import shutil
import numpy as np

# Root path where "AllData" is located
# Suppose we have:
#  c:\Aziz\MasterThesis\MasterTh\Super\AllData
#      ├── SO\  (all .mat files for SO)
#      ├── SP\  (all .mat files for SP)
#      ├── SR\  (all .mat files for SR)
#      └── SLM\ (all .mat files for SLM)
#
# We want to create folders:
#   train\SO, train\SP, train\SR, train\SLM
#   val\SO,   val\SP,   val\SR,   val\SLM
#   test\SO,  test\SP,  test\SR,  test\SLM
#
ALLDATA_PATH = r'c:\Aziz\MasterThesis\MasterTh\Shufle\AllDataSO_SP_SR_SLM'
OUTPUT_PATH  = r'c:\Aziz\MasterThesis\MasterTh\Shufle'  # Where we'll make train, val, test

CLASS_NAMES = ["SO", "SP", "SR", "SLM"]
# CLASS_NAMES = ["WT", "TG"]

# Ratios for train/val/test:
train_ratio = 0.6
val_ratio   = 0.2
# test_ratio = 0.15  (whatever is left)

np.random.seed(42)  # fix the seed for reproducibility (optional)

for class_name in CLASS_NAMES:
    # 1) Find all .mat files for this class
    class_folder = os.path.join(ALLDATA_PATH, class_name)
    file_list = glob.glob(os.path.join(class_folder, "*.mat"))
    
    print(f"Class: {class_name}, total files found: {len(file_list)}")
    if not file_list:
        continue

    # 2) Shuffle
    np.random.shuffle(file_list)

    # 3) Split
    N = len(file_list)
    train_size = int(train_ratio * N)
    val_size   = int(val_ratio * N)
    # remainder goes to test
    test_size  = N - train_size - val_size

    train_files = file_list[:train_size]
    val_files   = file_list[train_size : train_size + val_size]
    test_files  = file_list[train_size + val_size : ]

    print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # 4) Copy (or move) them into the new structure:
    for subset_name, subset_files in [
        ("train", train_files),
        ("val",   val_files),
        ("test",  test_files)
    ]:
        # e.g.  c:\Aziz\MasterThesis\MasterTh\Super\train\SO
        dest_folder = os.path.join(OUTPUT_PATH, subset_name, class_name)
        os.makedirs(dest_folder, exist_ok=True)

        # copy or move
        for fpath in subset_files:
            fname = os.path.basename(fpath)
            dst_path = os.path.join(dest_folder, fname)
            # either copy:
            shutil.copy2(fpath, dst_path)
            # or move:
            # shutil.move(fpath, dst_path)

print("Done! Now you have train/val/test subfolders with each class (SO, SP, SR, SLM).")

