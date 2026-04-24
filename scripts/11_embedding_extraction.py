#!/usr/bin/env python
# coding: utf-8
"""
Compute BERT embeddings for HR sequences and estimate PCA dimensionality.

Workflow
--------
1. Try to load precomputed embeddings from DISK.
2. If not present:
   - Load a pretrained BERT model + tokenizer.
   - Read tokenized HR sequences from a text file (one sequence per line).
   - Encode with Hugging Face `datasets` + tokenizer.
   - Compute last_hidden_state for all sequences (batched).
   - Reduce each embedding with PCA to a fixed number of components.
   - Concatenate and save the resulting tensor.
3. Flatten the final embeddings and run PCA again with
   `n_components = desired_variance_ratio` to estimate how many components
   are needed to retain the desired fraction of variance (e.g. 97%).

This script is intended to be committed to a repo and then configured via the
CONFIG section below.
"""

import os
import json
from typing import List

import torch
import numpy as np
from sklearn.decomposition import PCA
from datasets import load_dataset
from transformers import BertModel, BertTokenizer


# ----------------------------------------------------------------------
# CONFIG – ADAPT THESE PATHS FOR YOUR ENVIRONMENT / REPO
# ----------------------------------------------------------------------

# Directory containing the pretrained BERT model
MODEL_DIR = (
    "models/SafeICU/HR/15min/Model"
)

# Text file: one tokenized HR sequence per line (space-separated tokens)
TRAIN_TEXT_FILE = (
    "data/SafeICU/Combined_All_Resolution/15min/HR_Train.txt"
)

# Where to store/load the (possibly PCA-reduced) embeddings tensor
EMBEDDING_OUTPUT_FILE = (
    "embeddings/SafeICU/HR/15min/train_15min_embeddings.pth"
)

# PCA components used *inside* the embedding extraction (per time step)
PCA_COMPONENTS_FOR_EMBEDDINGS = 9

# Target variance retained when searching the global PCA dimension
DESIRED_VARIANCE_RATIO = 0.97

# BERT max sequence length
MAX_SEQ_LEN = 512

# Batch size for embedding extraction
BATCH_SIZE = 32


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def get_device() -> torch.device:
    """Return GPU device if available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def reduce_dimension(tensor: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    Apply PCA to reduce the last dimension of a 3D tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape: (batch_size, seq_len, hidden_dim)
    n_components : int
        Number of components for PCA.

    Returns
    -------
    reduced_tensor : torch.Tensor
        Shape: (batch_size, seq_len, n_components)
    """
    # Flatten over batch and sequence dimensions
    batch_size, seq_len, hidden_dim = tensor.shape
    reshaped = tensor.view(-1, hidden_dim)  # (batch_size * seq_len, hidden_dim)

    pca = PCA(n_components=n_components)
    reduced_2d = pca.fit_transform(reshaped.cpu().numpy())

    reduced_tensor = torch.tensor(
        reduced_2d,
        device=tensor.device,
        dtype=tensor.dtype
    ).view(batch_size, seq_len, n_components)

    return reduced_tensor


def get_embedding_list(
    model_name: str,
    file_path: str,
    device: torch.device,
    pca_components: int
) -> List[torch.Tensor]:
    """
    Compute PCA-reduced BERT embeddings for all sequences in `file_path`.

    Parameters
    ----------
    model_name : str
        Path or name of the pretrained BERT model.
    file_path : str
        Text file with one sequence per line.
    device : torch.device
        GPU/CPU device to use.
    pca_components : int
        Number of PCA components for per-token embeddings.

    Returns
    -------
    embeddings_list : list[torch.Tensor]
        List of tensors, each of shape (batch_size, seq_len, pca_components).
    """
    print(f"Loading model from: {model_name}")
    model = BertModel.from_pretrained(model_name).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    print(f"Loading dataset from: {file_path}")
    dataset = load_dataset("text", data_files=[file_path])

    def encode_with_truncation(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_special_tokens_mask=True,
        )

    encoded_dataset = dataset.map(encode_with_truncation, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    input_ids = encoded_dataset["train"]["input_ids"].to(device)
    attention_masks = encoded_dataset["train"]["attention_mask"].to(device)

    model.eval()
    embeddings_list: List[torch.Tensor] = []

    print(f"Total sequences: {len(input_ids)}")
    for i in range(0, len(input_ids), BATCH_SIZE):
        batch_ids = input_ids[i : i + BATCH_SIZE]
        batch_masks = attention_masks[i : i + BATCH_SIZE]

        print(f"Processing batch {i} / {len(input_ids)}")

        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_masks)

        last_hidden_state = outputs.last_hidden_state  # (B, L, H)
        # Move to CPU for PCA, then back to original device in reduce_dimension
        last_hidden_state = last_hidden_state.cpu()
        reduced = reduce_dimension(last_hidden_state, pca_components)
        embeddings_list.append(reduced)

    return embeddings_list


def estimate_pca_components(
    tensor: torch.Tensor,
    desired_variance_ratio: float
) -> int:
    """
    Run PCA on a 3D tensor flattened across batch and time, and return
    the number of components needed to retain `desired_variance_ratio`.

    Uses sklearn's float `n_components` behavior.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape: (N, T, D)
    desired_variance_ratio : float
        e.g., 0.97 for 97% variance retained.

    Returns
    -------
    n_components : int
        Number of components chosen by PCA.
    """
    n, t, d = tensor.shape
    reshaped = tensor.view(-1, d)  # (N*T, D)
    print("Tensor shape for PCA:", reshaped.shape)

    pca = PCA(n_components=desired_variance_ratio, svd_solver="full")
    pca.fit(reshaped.cpu().numpy())
    n_components = pca.n_components_

    print(
        f"Best n_components to retain {desired_variance_ratio * 100:.1f}% "
        f"variance: {n_components}"
    )
    return n_components


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    device = get_device()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(EMBEDDING_OUTPUT_FILE), exist_ok=True)

    # 1. Try to load embeddings
    if os.path.isfile(EMBEDDING_OUTPUT_FILE):
        print(f"Loading precomputed embeddings from {EMBEDDING_OUTPUT_FILE}")
        train_embeddings = torch.load(EMBEDDING_OUTPUT_FILE, map_location=device)
    else:
        # 2. Generate embeddings if not present
        print("Precomputed embeddings not found. Generating new embeddings...")
        embeddings_list = get_embedding_list(
            model_name=MODEL_DIR,
            file_path=TRAIN_TEXT_FILE,
            device=device,
            pca_components=PCA_COMPONENTS_FOR_EMBEDDINGS,
        )
        train_embeddings = torch.cat(embeddings_list, dim=0)
        torch.save(train_embeddings, EMBEDDING_OUTPUT_FILE)
        print(f"Saved new embeddings to {EMBEDDING_OUTPUT_FILE}")

    print("Final embedding tensor shape:", train_embeddings.shape)

    # 3. Global PCA for dimensionality estimation
    n_components = estimate_pca_components(
        train_embeddings,
        desired_variance_ratio=DESIRED_VARIANCE_RATIO,
    )

    # Optionally save this metadata as JSON
    meta = {
        "embedding_file": EMBEDDING_OUTPUT_FILE,
        "pca_internal_components": PCA_COMPONENTS_FOR_EMBEDDINGS,
        "desired_variance_ratio": DESIRED_VARIANCE_RATIO,
        "estimated_components_for_global_pca": int(n_components),
        "embedding_shape": list(train_embeddings.shape),
    }

    meta_file = EMBEDDING_OUTPUT_FILE.replace(".pth", "_pca_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"PCA metadata saved to {meta_file}")


if __name__ == "__main__":
    main()
