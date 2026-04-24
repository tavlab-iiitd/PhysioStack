"""
Discretization of Heart Rate Time Series using K-Means + Silhouette

Features:
- Convert continuous heart rate into discrete tokens ('a', 'b', 'c', ...)
- Choose number of clusters using Silhouette score
  - CPU version (sklearn)
  - Torch version (GPU/CPU automatic)
- Process a batch of CSV files in a folder and report best k per file

Author(s): Anusha Tiwari, Kushagra, Akshaya, Pradeep Singh
"""

import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sk_silhouette_score
import torch

sns.set()

# -----------------------------
# Global configuration
# -----------------------------
MYSEED = 100
DEFAULT_CLUSTER_RANGE_SHORT = list(range(3, 11))         # 3..10 for quick tests (sklearn)
DEFAULT_CLUSTER_RANGE_LONG = list(range(3, 21))          # 3..20 for full search (torch)
HEART_RATE_COLUMN = "X.HR."                              # column name in CSV
CONSTANT_FEATURE_VALUE = 20                              # second feature dim
DEFAULT_CSV_GLOB = "../../Required1_CSV/*.csv"           # update as per your folder structure


# -----------------------------
# Data preparation & utilities
# -----------------------------
def data_extractor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract heart-rate column and create a 2D feature dataframe:
    - Heart_rate: original HR values from HEART_RATE_COLUMN
    - temp: constant feature (all rows = CONSTANT_FEATURE_VALUE)

    This makes the data compatible with 2D clustering, while
    effectively clustering only by heart rate.
    """
    if HEART_RATE_COLUMN not in df.columns:
        raise KeyError(f"Expected column '{HEART_RATE_COLUMN}' not found in dataframe.")
    df_temp = df[HEART_RATE_COLUMN]
    df_final = pd.DataFrame({"Heart_rate": df_temp.values})
    df_final["temp"] = CONSTANT_FEATURE_VALUE
    return df_final


def maptokens(df_final: pd.DataFrame, n_clusters: int):
    """
    Map numeric clusters (0..n-1) to ordered alphabet tokens ('a','b',...):

    Steps:
    - For each cluster, find the minimum heart rate belonging to that cluster.
    - Sort clusters by this minimum HR (low -> high).
    - Assign 'a' to lowest-HR cluster, 'b' to next, etc.

    Returns
    -------
    token_list : list[str]
        List of length n_clusters, where token_list[cluster_id]
        gives the corresponding character token.
    """
    # Initialize list with large values
    min_hr_per_cluster = [np.inf] * n_clusters

    # Find minimum HR in each cluster
    for hr, cl in zip(df_final["Heart_rate"], df_final["clusters"]):
        if cl < 0 or cl >= n_clusters:
            raise ValueError(f"Cluster id {cl} out of range [0,{n_clusters-1}]")
        if hr < min_hr_per_cluster[cl]:
            min_hr_per_cluster[cl] = hr

    # Map min-HR values to cluster indices
    value_to_cluster = {float(v): idx for idx, v in enumerate(min_hr_per_cluster)}

    # Sort min-HR values
    sorted_values = sorted(min_hr_per_cluster)

    # Assign tokens according to sorted order
    tokens = [None] * n_clusters
    for token_index, hr_value in enumerate(sorted_values):
        cluster_index = value_to_cluster[float(hr_value)]
        tokens[cluster_index] = chr(token_index + 97)  # 'a' = 97

    return tokens


def plot_both(df_final: pd.DataFrame, start: int, end: int, title: str = ""):
    """
    Plot (for a given index range):
    - The raw heart rate time series.
    - The cluster indices over time.
    - Token annotations at each point.
    """
    subset = df_final.iloc[start:end]

    plt.figure(figsize=(16, 8))
    plt.plot(subset.index, subset["Heart_rate"], label="Heart Rate")
    plt.plot(subset.index, subset["clusters"], linewidth=2, label="Cluster Index", color="green")
    plt.ylim([-4, max(120, subset["Heart_rate"].max() + 10)])

    for i in subset.index:
        plt.annotate(
            subset.loc[i, "token"],
            (i, subset.loc[i, "clusters"]),
            weight="bold",
            color="green",
            fontsize=10,
            ha="center",
            va="bottom",
        )

    plt.xlabel("Time index")
    plt.ylabel("Value / Cluster")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def clusterer(df: pd.DataFrame, n_clusters: int, show_plot: bool = True) -> pd.DataFrame:
    """
    Run K-Means clustering with n_clusters on the heart rate data.

    Returns a dataframe with:
        - Heart_rate
        - temp
        - clusters (cluster id per row)

    Optionally shows a scatter plot of the clustered points in (temp, HR) space.
    """
    df_final = data_extractor(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=MYSEED)
    labels = kmeans.fit_predict(df_final)

    df_final["clusters"] = labels

    if show_plot:
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(
            df_final["temp"],
            df_final["Heart_rate"],
            c=df_final["clusters"],
            cmap="rainbow",
        )
        plt.xlabel("Temp (constant feature)")
        plt.ylabel("Heart Rate")
        plt.title(f"KMeans Clustering (k={n_clusters})")
        plt.colorbar(scatter, label="Cluster ID")
        plt.tight_layout()
        plt.show()

    return df_final


def plotter(df: pd.DataFrame, n_clusters: int, start: int = 0, end: int = 1000, title: str = ""):
    """
    Given a dataframe with columns:
        - Heart_rate
        - temp
        - clusters

    Map clusters to tokens and plot both HR and clusters in the specified index range.
    """
    token_list = maptokens(df, n_clusters)
    df["token"] = df["clusters"].apply(lambda cl: token_list[cl])
    # Re-map clusters so that cluster index matches token order (0 for 'a', 1 for 'b', etc.)
    df["clusters"] = df["token"].apply(lambda t: ord(t) - 97)
    plot_both(df, start, end, title=title)


# -----------------------------
# Sklearn silhouette (CPU)
# -----------------------------
def calculate_silhouette_sklearn(df: pd.DataFrame, cluster_range=None) -> int:
    """
    Use sklearn's silhouette_score to choose the best number of clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with column HEART_RATE_COLUMN.
    cluster_range : list[int], optional
        List of k values to try. Defaults to 3..10.

    Returns
    -------
    best_k : int
        Cluster count in 'cluster_range' with the highest silhouette score.
    """
    if cluster_range is None:
        cluster_range = DEFAULT_CLUSTER_RANGE_SHORT

    df_final = data_extractor(df)
    silhouette_scores = []

    for k in cluster_range:
        # Not enough points to form k clusters
        if df_final.shape[0] <= k:
            silhouette_scores.append(-1)
            continue

        kmeans = KMeans(n_clusters=k, random_state=MYSEED)
        labels = kmeans.fit_predict(df_final)
        score = sk_silhouette_score(df_final, labels)
        silhouette_scores.append(score)

    # Plot silhouette scores
    plt.figure(figsize=(8, 4))
    plt.plot(cluster_range, silhouette_scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score (sklearn)")
    plt.title("Silhouette Scores vs k (sklearn)")
    plt.tight_layout()
    plt.show()

    # Choose best k
    best_index = int(np.argmax(silhouette_scores))
    best_k = cluster_range[best_index]
    return best_k


# -----------------------------
# Torch-based silhouette (GPU/CPU)
# -----------------------------
def _intra_cluster_distances_block_(subX: torch.Tensor) -> torch.Tensor:
    """
    Compute mean intra-cluster distance for each point in a single cluster.

    subX: (n_points_in_cluster, n_features)
    returns: (n_points_in_cluster,) mean distance to all other points in the cluster.
    """
    if subX.shape[0] <= 1:
        # Cluster of size 1: by convention, distance = 0
        return torch.zeros(subX.shape[0], device=subX.device, dtype=torch.float32)

    distances = torch.cdist(subX, subX)  # (n, n)
    return distances.sum(dim=1) / (distances.shape[0] - 1)


def _intra_cluster_distances_block(
    X: torch.Tensor,
    labels: torch.Tensor,
    unique_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute intra-cluster distance a(i) for each sample i in X.
    """
    intra_dist = torch.zeros(labels.shape[0], dtype=torch.float32, device=X.device)
    for lbl in unique_labels:
        idx = torch.where(labels == lbl)[0]
        if idx.numel() == 0:
            continue
        subX = X[idx]
        values = _intra_cluster_distances_block_(subX)
        intra_dist[idx] = values
    return intra_dist


def _nearest_cluster_distance_block_(subX_a: torch.Tensor, subX_b: torch.Tensor):
    """
    For two clusters A and B:
    - dist_a: mean distance from each point in A to all points in B
    - dist_b: mean distance from each point in B to all points in A
    """
    if subX_a.shape[0] == 0 or subX_b.shape[0] == 0:
        return (
            torch.full((subX_a.shape[0],), float("inf"), device=subX_a.device),
            torch.full((subX_b.shape[0],), float("inf"), device=subX_b.device),
        )

    dist = torch.cdist(subX_a, subX_b)  # (n_a, n_b)
    dist_a = dist.mean(dim=1)
    dist_b = dist.mean(dim=0)
    return dist_a, dist_b


def _nearest_cluster_distance_block(
    X: torch.Tensor,
    labels: torch.Tensor,
    unique_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute nearest-cluster distance b(i) for each sample i in X.
    """
    inter_dist = torch.full(labels.shape[0], float("inf"), device=X.device, dtype=torch.float32)

    # Get all pairwise combinations of cluster labels
    if unique_labels.numel() < 2:
        # Only one cluster -> by definition, b(i) is inf; silhouette will be 0.
        return inter_dist

    label_combinations = torch.combinations(unique_labels, r=2)

    for (label_a, label_b) in label_combinations:
        idx_a = torch.where(labels == label_a)[0]
        idx_b = torch.where(labels == label_b)[0]
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            continue

        subX_a = X[idx_a]
        subX_b = X[idx_b]
        dist_a, dist_b = _nearest_cluster_distance_block_(subX_a, subX_b)

        inter_dist[idx_a] = torch.minimum(inter_dist[idx_a], dist_a)
        inter_dist[idx_b] = torch.minimum(inter_dist[idx_b], dist_b)

    return inter_dist


def silhouette_score_torch(
    X,
    labels,
    loss: bool = False,
    device: str | None = None,
):
    """
    Compute the mean Silhouette Coefficient of all samples using PyTorch,
    with optional GPU acceleration.

    Parameters
    ----------
    X : array-like [n_samples, n_features]
    labels : array-like [n_samples]
        Cluster labels for each sample.
    loss : bool (default: False)
        If True, return negative silhouette score as a torch.Tensor (for autograd).
        If False, return positive silhouette score as a Python float.
    device : str or None
        "cuda" or "cpu". If None, will auto-select "cuda" if available else "cpu".

    Returns
    -------
    score : float or torch.Tensor
        Mean silhouette score (positive if loss=False, negative if loss=True).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)

    unique_labels = torch.unique(labels_tensor)

    # Compute a(i) and b(i)
    A = _intra_cluster_distances_block(X_tensor, labels_tensor, unique_labels)
    B = _nearest_cluster_distance_block(X_tensor, labels_tensor, unique_labels)

    # Silhouette for each sample
    max_AB = torch.maximum(A, B)
    # Avoid division by zero
    sil_samples = torch.where(max_AB > 0, (B - A) / max_AB, torch.zeros_like(max_AB))

    mean_sil_score = sil_samples.nan_to_num().mean()

    if loss:
        return -mean_sil_score
    else:
        return float(mean_sil_score.detach().cpu().numpy())


def calculate_silhouette_torch(
    df: pd.DataFrame,
    cluster_range=None,
    device: str | None = None,
) -> int:
    """
    Use custom torch-based silhouette_score_torch to choose the best number of clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with column HEART_RATE_COLUMN.
    cluster_range : list[int], optional
        List of k values to try. Defaults to 3..20.
    device : str or None
        "cuda" or "cpu". If None, auto-select.

    Returns
    -------
    best_k : int
        Cluster count in 'cluster_range' with the highest silhouette score.
    """
    if cluster_range is None:
        cluster_range = DEFAULT_CLUSTER_RANGE_LONG

    df_final = data_extractor(df)
    X = df_final[["Heart_rate", "temp"]].values
    silhouette_scores = []

    for k in cluster_range:
        # Not enough points to form k clusters
        if df_final.shape[0] <= k:
            silhouette_scores.append(-1)
            continue

        kmeans = KMeans(n_clusters=k, random_state=MYSEED)
        labels = kmeans.fit_predict(df_final)
        score = silhouette_score_torch(X, labels, loss=False, device=device)
        silhouette_scores.append(score)

    # Plot silhouette scores
    plt.figure(figsize=(8, 4))
    plt.plot(cluster_range, silhouette_scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score (torch)")
    plt.title("Silhouette Scores vs k (torch-based)")
    plt.tight_layout()
    plt.show()

    best_index = int(np.argmax(silhouette_scores))
    best_k = cluster_range[best_index]
    return best_k


# -----------------------------
# Batch processing of CSV files
# -----------------------------
def process_all_files(
    csv_glob_pattern: str = DEFAULT_CSV_GLOB,
    use_torch: bool = True,
    device: str | None = None,
    cluster_range=None,
):
    """
    Loop over all CSV files matching 'csv_glob_pattern', compute the best
    number of clusters for each file using silhouette score, and return:

    Returns
    -------
    silhouettes : list[tuple[int, str]]
        List of (best_k, filename) for each successfully processed file.
    na_files : list[str]
        List of filenames that contained NaNs in HEART_RATE_COLUMN and were skipped.
    """
    silhouettes = []
    na_files = []

    if cluster_range is None:
        cluster_range = DEFAULT_CLUSTER_RANGE_LONG if use_torch else DEFAULT_CLUSTER_RANGE_SHORT

    csv_paths = sorted(glob.glob(csv_glob_pattern))

    for fname in csv_paths:
        df = pd.read_csv(fname)

        # Skip files with NaN in heart rate
        if df[HEART_RATE_COLUMN].isna().sum() > 0:
            na_files.append(fname)
            continue

        print(f"Starting silhouette computation for: {fname}")

        if use_torch:
            best_k = calculate_silhouette_torch(df, cluster_range=cluster_range, device=device)
        else:
            best_k = calculate_silhouette_sklearn(df, cluster_range=cluster_range)

        print(f"Best k = {best_k} for file: {fname}")
        silhouettes.append((best_k, fname))

    return silhouettes, na_files


# -----------------------------
# Main: only batch over folder
# -----------------------------
if __name__ == "__main__":
    silhouettes_info, files_with_nans = process_all_files(
        csv_glob_pattern=DEFAULT_CSV_GLOB,
        use_torch=True,      # set False to use sklearn CPU version
        device=None,         # "cuda", "cpu", or None for auto
    )

    print("\nSilhouette results (best_k, filename):")
    for k, fname in silhouettes_info:
        print(k, fname)

    if files_with_nans:
        print("\nFiles skipped due to NaNs in heart rate column:")
        for fname in files_with_nans:
            print(fname)
