#!/usr/bin/env python
# coding: utf-8
"""
Masked-token accuracy evaluation for HR 60min model.

What this script does
---------------------
1. Loads a pretrained BERT-MLM model (HR 60min) and its tokenizer.
2. Reads a text file where each line is a space-separated token sequence (HR_Val.txt).
3. For each sequence:
   - Randomly masks ~15% of tokens (excluding the first and last token when possible).
   - Uses a Hugging Face `fill-mask` pipeline to predict the masked tokens.
   - Computes accuracy over the masked positions.
4. Aggregates:
   - Line-wise accuracies (one scalar per sequence).
   - Token-wise accuracy for each token (how often it is correctly predicted).
5. Saves:
   - Line-wise accuracies (one per line).
   - Final token-wise accuracy dictionary (JSON) as the last line of the file.
"""

import os
import json
import random
from typing import List, Tuple, Dict

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from transformers import BertForMaskedLM, BertTokenizerFast, pipeline


# ----------------------------------------------------------------------
# CONFIG – UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# ----------------------------------------------------------------------

MODEL_DIR = (
    "/scratch/home/akshayai/Akshaya/Akshaya_Embedding/"
    "AKSHAYA_Optuna_TrainBERT/SafeICU/HR/60min/Model"
)

VAL_TXT_FILE = (
    "/scratch/home/akshayai/Akshaya/Akshaya_Embedding/"
    "AKSHAYA_Data/SafeICU/60min/HR_Val.txt"
)

ACCURACY_OUTPUT_FILE = (
    "/scratch/home/akshayai/Akshaya/Akshaya_Embedding/"
    "AKSHAYA_Codes/Resolution_Accuracy/Accuracy_HR_60min.txt"
)

MASK_FRACTION = 0.15   # fraction of tokens to mask per sequence
RANDOM_SEED = 42       # for reproducibility
MAX_SEQ_LEN = 512      # truncate sequences to this many tokens


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def get_pipeline_device() -> int:
    """
    Device index for Hugging Face pipeline:
    - 0 for first GPU if CUDA is available.
    - -1 for CPU.
    """
    return 0 if torch.cuda.is_available() else -1


def load_model_and_tokenizer(model_dir: str):
    """
    Load a pretrained MLM model and tokenizer from `model_dir`.
    """
    print(f"Loading model and tokenizer from: {model_dir}")
    model = BertForMaskedLM.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir, padding=True)
    print("Tokenizer is_fast:", tokenizer.is_fast)
    return model, tokenizer


def load_token_sequences(path: str, max_len: int) -> List[List[str]]:
    """
    Load sequences from a text file:
    - One space-separated token sequence per line.
    - Truncate each sequence to `max_len` tokens.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    lines: List[str] = []
    with open(path, "r") as f:
        for line in f:
            lines.append(line.strip())

    print(f"Total sequences loaded from {path}: {len(lines)}")

    random.shuffle(lines)
    token_seqs = [line.split()[:max_len] for line in lines]

    print(f"Total sequences after truncation to {max_len} tokens: {len(token_seqs)}")
    return token_seqs


def compute_accuracy_for_sequence(
    tokens: List[str],
    fill_mask_pipe,
    mask_fraction: float,
) -> Tuple[float, List[str], List[str]]:
    """
    For a single token sequence:
    - Randomly select ~mask_fraction of positions (excluding first/last if possible).
    - Replace each selected token with [MASK], and predict using `fill_mask_pipe`.
    - Return accuracy, list of actual tokens, list of predicted tokens.

    Returns
    -------
    accuracy : float
        Accuracy over masked positions.
    actual : list[str]
        Ground truth tokens at masked positions.
    preds : list[str]
        Predicted tokens for masked positions.
    """
    if len(tokens) < 3:
        # Too short to safely mask internal positions
        return float("nan"), [], []

    num_to_mask = max(1, int(mask_fraction * len(tokens)))
    candidate_indices = list(range(1, len(tokens) - 1))  # avoid first & last
    num_to_mask = min(num_to_mask, len(candidate_indices))

    if num_to_mask == 0:
        return float("nan"), [], []

    mask_indices = random.sample(candidate_indices, num_to_mask)
    mask_indices.sort()

    actual: List[str] = []
    test_cases: List[str] = []

    for idx in mask_indices:
        seq_copy = tokens.copy()
        actual.append(tokens[idx])
        seq_copy[idx] = "[MASK]"
        test_cases.append(" ".join(seq_copy))

    preds: List[str] = []
    for case in test_cases:
        res = fill_mask_pipe(case)
        # `res` is usually a list of candidates; use the top-1
        if isinstance(res, list) and len(res) > 0 and "token_str" in res[0]:
            preds.append(res[0]["token_str"])
        else:
            preds.append("")  # fallback

    acc = accuracy_score(actual, preds)
    return float(acc), actual, preds


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    # Reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        print(f"PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch is using CPU")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR)

    # Initialize fill-mask pipeline
    device_idx = get_pipeline_device()
    print(f"Initializing fill-mask pipeline on device: {device_idx}")
    fill_mask_pipe = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=device_idx,
    )

    # Load validation token sequences
    sequences = load_token_sequences(VAL_TXT_FILE, MAX_SEQ_LEN)
    num_sequences = len(sequences)
    print(f"Number of sequences to evaluate: {num_sequences}")

    # Evaluation
    sum_acc = 0.0
    num_valid_seqs = 0
    all_actuals: List[List[str]] = []
    all_preds: List[List[str]] = []

    # Write per-sequence accuracy
    with open(ACCURACY_OUTPUT_FILE, "w") as f_out:
        for idx, seq_tokens in enumerate(sequences, start=1):
            acc, actual, preds = compute_accuracy_for_sequence(
                seq_tokens,
                fill_mask_pipe,
                mask_fraction=MASK_FRACTION,
            )

            if np.isnan(acc):
                continue

            sum_acc += acc
            num_valid_seqs += 1
            all_actuals.append(actual)
            all_preds.append(preds)

            f_out.write(f"{acc:.4f}\n")

            if idx % 50 == 0:
                print(f"Processed {idx}/{num_sequences} sequences...")

        # Token-wise accuracy
        token_total: Dict[str, int] = {}
        token_correct: Dict[str, int] = {}

        for act_seq, pred_seq in zip(all_actuals, all_preds):
            for a, p in zip(act_seq, pred_seq):
                token_total[a] = token_total.get(a, 0) + 1
                if a == p:
                    token_correct[a] = token_correct.get(a, 0) + 1

        token_wise_acc: Dict[str, float] = {}
        for tok, total in token_total.items():
            correct = token_correct.get(tok, 0)
            token_wise_acc[tok] = correct / total if total > 0 else 0.0

        print("Token-wise accuracy (first 10 tokens):")
        for tok in list(token_wise_acc.keys())[:10]:
            print(f"  {tok}: {token_wise_acc[tok]:.4f}")

        # Append token-wise accuracy as JSON to the output file
        f_out.write(json.dumps(token_wise_acc) + "\n")

    if num_valid_seqs > 0:
        mean_acc = sum_acc / num_valid_seqs
        print(f"\nMean line-wise accuracy over {num_valid_seqs} sequences: {mean_acc:.4f}")
    else:
        print("\nNo valid sequences evaluated (all were too short).")

    # Optional: if you want the list of accuracies in Python as well:
    with open(ACCURACY_OUTPUT_FILE, "r") as f:
        lines = f.readlines()
    # Last line is JSON dict, previous lines are per-sequence floats
    pretrain_accuracy_values = [float(x.strip()) for x in lines[:-1]]
    print(f"Loaded {len(pretrain_accuracy_values)} per-sequence accuracy values from file.")


if __name__ == "__main__":
    main()
