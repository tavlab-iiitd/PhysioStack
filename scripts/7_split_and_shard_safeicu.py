#!/usr/bin/env python3
"""
Split SafeICU 60-min time-series files into Train / Test / Val
and shard each CSV into fixed-length segments.

Pipeline:
1. Split all CSV files from ORIGINAL_DIR into:
   - TRAIN_DIR (70%)
   - TEST_DIR  (20%)
   - VAL_DIR   (10%)

2. For each of Train / Test / Val:
   - For each CSV file:
       * If length < 510 rows  → save as a single shard
       * Else                  → split into non-overlapping shards of 509 rows
   - Shards for file `X.csv` are written into:
       SHARD_<SET>_DIR / X / X_1.csv, X_2.csv, ...

Edit the CONFIG block below to point to your SafeICU paths.
"""

import os
import shutil
import random
import pandas as pd

# ---------------------------------------------------
# CONFIG: update these paths for your environment
# ---------------------------------------------------
ORIGINAL_DIR = "/scratch/home/akshayai/Akshaya/Akshaya_Embedding/AKSHAYA_Data/SafeICU/60min/60min_Cluster"

TRAIN_DIR = "/scratch/home/akshayai/Akshaya/Akshaya_Embedding/AKSHAYA_Data/SafeICU/60min/Train"
TEST_DIR  = "/scratch/home/akshayai/Akshaya/Akshaya_Embedding/AKSHAYA_Data/SafeICU/60min/Test"
VAL_DIR   = "/scratch/home/akshayai/Akshaya/Akshaya_Embedding/AKSHAYA_Data/SafeICU/60min/Val"

SHARD_TRAIN_DIR = "/scratch/home/akshayai/Akshaya/Akshaya_Embedding/AKSHAYA_Data/SafeICU/60min/Train_Shard"
SHARD_TEST_DIR  = "/scratch/home/akshayai/Akshaya/Akshaya_Embedding/AKSHAYA_Data/SafeICU/60min/Test_Shard"
SHARD_VAL_DIR   = "/scratch/home/akshayai/Akshaya/Akshaya_Embedding/AKSHAYA_Data/SafeICU/60min/Val_Shard"

TRAIN_RATIO = 0.7
TEST_RATIO  = 0.2
VAL_RATIO   = 0.1   # implicitly: 1 - TRAIN_RATIO - TEST_RATIO


# ---------------------------------------------------
# Split CSV files into Train / Test / Val
# ---------------------------------------------------
def split_dataset(original_dir, train_dir, test_dir, val_dir,
                  train_ratio=0.7, test_ratio=0.2, seed=42):
    """
    Randomly split all CSV files in original_dir into Train / Test / Val.

    The split is done at file level (each CSV → exactly one split).
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # List all CSVs in ORIGINAL_DIR
    all_files = [
        f for f in os.listdir(original_dir)
        if f.lower().endswith(".csv")
    ]

    if not all_files:
        print(f"No CSV files found in {original_dir}. Nothing to split.")
        return

    random.seed(seed)
    random.shuffle(all_files)

    total = len(all_files)
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    val_count = total - train_count - test_count

    train_files = all_files[:train_count]
    test_files  = all_files[train_count:train_count + test_count]
    val_files   = all_files[train_count + test_count:]

    def copy_files(file_list, src_dir, dst_dir, label):
        for fname in file_list:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copy2(src, dst)
        print(f"{label}: {len(file_list)} files")

    print(f"Total files: {total}")
    copy_files(train_files, original_dir, train_dir, "Train")
    copy_files(test_files,  original_dir, test_dir,  "Test")
    copy_files(val_files,   original_dir, val_dir,   "Val")

    print("✅ Dataset split complete.")


# ---------------------------------------------------
# Sharding logic
# ---------------------------------------------------
def shard_csv(input_path: str, shard_root: str,
              shard_size: int = 509, min_full_len: int = 510):
    """
    Shard a single CSV into fixed-length segments.

    - Reads `input_path` into a DataFrame.
    - Creates a folder:
        shard_root / <basename_without_ext> /
    - If len(df) < min_full_len:
        Save entire df as a single CSV with original name.
    - Else:
        Save non-overlapping shards of length `shard_size` as:
            <name>_1.csv, <name>_2.csv, ...

    Parameters
    ----------
    input_path : str
        Full path to the input CSV.
    shard_root : str
        Root directory under which shards will be saved.
    shard_size : int
        Number of rows per shard (default: 509).
    min_full_len : int
        Minimum length to split; shorter files are saved as-is.
    """
    df = pd.read_csv(input_path)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    folder_path = os.path.join(shard_root, base_name)
    os.makedirs(folder_path, exist_ok=True)

    # If file is short, keep as one shard
    if len(df) < min_full_len:
        out_path = os.path.join(folder_path, os.path.basename(input_path))
        df.to_csv(out_path, index=False)
        return

    shard_number = 1
    for start in range(0, len(df), shard_size):
        shard_df = df.iloc[start:start + shard_size]
        if shard_df.empty:
            break

        out_name = f"{base_name}_{shard_number}.csv"
        out_path = os.path.join(folder_path, out_name)
        shard_df.to_csv(out_path, index=False)
        shard_number += 1


def shard_directory(input_dir: str, shard_root: str,
                    shard_size: int = 509, min_full_len: int = 510):
    """
    Apply sharding to all CSVs in `input_dir` and store
    results under `shard_root`.
    """
    os.makedirs(shard_root, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".csv")
    ]
    if not files:
        print(f"No CSV files found in {input_dir} to shard.")
        return

    for fname in files:
        in_path = os.path.join(input_dir, fname)
        try:
            shard_csv(in_path, shard_root,
                      shard_size=shard_size,
                      min_full_len=min_full_len)
        except Exception as e:
            print(f"⚠️  Error sharding {in_path}: {e}")

    print(f"✅ Sharding complete for: {input_dir}")


# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    # 1) Split original CSVs into Train / Test / Val
    split_dataset(
        ORIGINAL_DIR,
        TRAIN_DIR,
        TEST_DIR,
        VAL_DIR,
        train_ratio=TRAIN_RATIO,
        test_ratio=TEST_RATIO,
    )

    # 2) Shard each split into fixed-length sequences
    shard_directory(TRAIN_DIR, SHARD_TRAIN_DIR)
    shard_directory(TEST_DIR,  SHARD_TEST_DIR)
    shard_directory(VAL_DIR,   SHARD_VAL_DIR)

    print("🎯 All done: split + sharding pipeline finished.")
