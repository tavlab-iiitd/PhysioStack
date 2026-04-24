#!/usr/bin/env python3
"""
Discretize ICU vitals using K-Means clustering.

For each CSV in INPUT_DIR, this script:
- Reads continuous vitals: heartrate, sao2, respiration, Systolic_Abp
- Clusters each signal using KMeans (HR/Resp: 5 clusters, SaO2/ABP: 6 clusters)
- Orders clusters by mean value (low → high) and assigns letter tokens:
    e.g., 'a', 'b', 'c', ...
- Adds tokenized columns to the dataframe:
    C_heartrate, C_sao2, C_respiration, C_Systolic_Abp
- Writes the enriched CSVs to OUTPUT_DIR.

Edit INPUT_DIR and OUTPUT_DIR below to point to your data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# -----------------------------
# User configuration
# -----------------------------
INPUT_DIR = "path/to/input_csvs"   # folder with raw 60min CSVs
OUTPUT_DIR = "path/to/output_csvs" # folder to save clustered CSVs

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Helper: build 2D feature df
# -----------------------------
def _make_feature_df(series: pd.Series, value_col_name: str) -> pd.DataFrame:
    """
    Convert a 1D vitals series into a 2D dataframe with:
      - <value_col_name>: original numeric values
      - temp: constant feature (20) so that KMeans sees 2D input,
              but effectively clusters by the signal value only.
    """
    df_final = pd.DataFrame({value_col_name: series.values})
    df_final["temp"] = 20
    return df_final


def _cluster_and_tokenize(
    df: pd.DataFrame,
    value_col: str,
    n_clusters: int,
    token_col_name: str,
) -> pd.Series:
    """
    Run KMeans on a single vitals column and return a tokenized series.

    Steps:
    - Build 2D feature dataframe [value, temp]
    - Run KMeans with n_clusters
    - For each cluster, compute mean value
    - Sort clusters by mean value (low → high)
    - Assign tokens 'a', 'b', 'c', ... according to sorted order
    - Map each row’s cluster id to its corresponding token

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing `value_col`.
    value_col : str
        Name of the numeric column to cluster (e.g. "heartrate").
    n_clusters : int
        Desired number of clusters (e.g. 5 or 6).
    token_col_name : str
        Name of the resulting token column (e.g. "C_heartrate").

    Returns
    -------
    pd.Series
        Token labels for each row, aligned with df.index.
    """
    if value_col not in df.columns:
        # Column missing → return empty series so caller can skip
        return pd.Series(index=df.index, dtype="object")

    # Drop rows where the value is NaN for clustering
    series = df[value_col].astype(float)
    valid_mask = series.notna()

    if valid_mask.sum() == 0:
        # All NaN → nothing to cluster
        return pd.Series(index=df.index, dtype="object")

    feature_df = _make_feature_df(series[valid_mask], value_col_name=value_col.capitalize())

    # Fit KMeans on valid rows
    kmeans = KMeans(n_clusters=n_clusters, random_state=100, n_init=10)
    kmeans.fit(feature_df)
    cluster_labels = kmeans.labels_

    # Compute mean value per cluster
    feature_df["cluster"] = cluster_labels
    result = feature_df.groupby("cluster")[value_col.capitalize()].mean()
    result = pd.DataFrame(result)

    # How many distinct mean values actually exist?
    unique_values_count = result[value_col.capitalize()].nunique()

    # Generate tokens: at most n_clusters, but may be fewer if some means are identical
    token_letters = [chr(ord("a") + i) for i in range(n_clusters)]
    token_letters = token_letters[:unique_values_count]

    # Sort clusters by mean and assign tokens in order
    result = result.sort_values(by=value_col.capitalize())
    result["cluster_token"] = token_letters

    # Map original cluster ids → tokens
    cluster_to_token = dict(zip(result.index, result["cluster_token"]))
    token_series_valid = pd.Series(
        [cluster_to_token[label] for label in cluster_labels],
        index=series[valid_mask].index,
        name=token_col_name,
    )

    # Reindex to full dataframe (NaN where original values were NaN)
    token_series_full = pd.Series(index=df.index, dtype="object", name=token_col_name)
    token_series_full.loc[token_series_valid.index] = token_series_valid

    return token_series_full


def cluster_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clustering + tokenization to all four vitals:
    - heartrate        → C_heartrate (5 clusters: a–e)
    - sao2             → C_sao2      (6 clusters: a–f)
    - respiration      → C_respiration (5 clusters: a–e)
    - Systolic_Abp     → C_Systolic_Abp (6 clusters: a–f)
    """
    df = df.copy()

    # Heart rate: 5 clusters
    df["C_heartrate"] = _cluster_and_tokenize(
        df,
        value_col="heartrate",
        n_clusters=5,
        token_col_name="C_heartrate",
    )

    # SaO2: 6 clusters
    df["C_sao2"] = _cluster_and_tokenize(
        df,
        value_col="sao2",
        n_clusters=6,
        token_col_name="C_sao2",
    )

    # Respiration: 5 clusters
    df["C_respiration"] = _cluster_and_tokenize(
        df,
        value_col="respiration",
        n_clusters=5,
        token_col_name="C_respiration",
    )

    # Systolic ABP: 6 clusters
    df["C_Systolic_Abp"] = _cluster_and_tokenize(
        df,
        value_col="Systolic_Abp",
        n_clusters=6,
        token_col_name="C_Systolic_Abp",
    )

    return df


def process_directory(input_dir: str, output_dir: str) -> None:
    """
    Loop over all CSV files in input_dir, apply clustering, and save results to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(".csv"):
            # Skip non-CSV files quietly
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            df = pd.read_csv(input_path)
            df_clustered = cluster_vitals(df)
            df_clustered.to_csv(output_path, index=False)
            print(f"[OK]  {filename} -> {output_path}")
        except Exception as e:
            print(f"[ERR] {filename}: {e}")


if __name__ == "__main__":
    process_directory(INPUT_DIR, OUTPUT_DIR)
