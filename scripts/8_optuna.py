#!/usr/bin/env python3
"""
Pretraining BERT for Masked Language Modeling (MLM) on RR 60-min sequences
with Optuna-based hyperparameter tuning.

Pipeline:
1. Train a WordPiece tokenizer on RR_Val.txt.
2. Save tokenizer + config to MODEL_DIR.
3. Load text dataset via Hugging Face `datasets`.
4. Split into train / test (90% / 10%), create a "seen" split and save to file.
5. Tokenize text with truncation / padding to max_length.
6. Build a fresh `BertForMaskedLM` with given vocab size and max_position_embeddings.
7. Run Optuna hyperparameter search:
   - learning_rate (log-uniform)
   - batch_size (categorical)
   - weight_decay (log-uniform)
   - optimizer (adam / sgd / rmsprop)
   - scheduler (linear / cosine)
   - early stopping with patience
8. For each trial:
   - Log hyperparameters + per-epoch train/validation losses.
   - Save best-epoch model for that trial.
9. After all trials, save best-trial info + hyperparameters.

Edit the CONFIG section below to point to your data and output paths.
"""

import os
import json
import time
import math
import random
from itertools import chain

import numpy as np
import torch
import optuna
import pandas as pd  # only used if you later extend; kept for completeness
from tqdm.auto import tqdm
from datasets import load_dataset
from tokenizers import BertWordPieceTokenizer

from torch.utils.data import DataLoader
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

# -------------------------------------------------------------------
# CONFIG: update these paths for your environment
# -------------------------------------------------------------------
TEXT_FILE = "/path/to/RR_Val.txt"  # e.g. "/workspace/.../SafeICU/60min/RR_Val.txt"
MODEL_DIR = "/path/to/output/model_dir"  # e.g. "/workspace/.../SafeICU/RR/60min/Model"

LOSS_LOG_FILE = os.path.join(MODEL_DIR, "Train_Eval_Loss.txt")
TRIAL_PARAMS_FILE = os.path.join(MODEL_DIR, "All_trial_parameters.txt")
BEST_TRIAL_FILE = os.path.join(MODEL_DIR, "best_trial.txt")
SEEN_SAMPLES_FILE = os.path.join(MODEL_DIR, "Seen_PreTrain_RR_60min.txt")
OUTER_LOG_FILE = os.path.join(MODEL_DIR, "loss_text_60min.log")

# Training / Tokenizer hyperparams
VOCAB_SIZE = 30522
MAX_LENGTH = 512
TRUNCATE_LONGER_SAMPLES = True

# Optuna config
N_TRIALS = 100
MAX_EPOCHS = 100
PATIENCE = 5
WARMUP_STEPS = 1000
TRAINING_STEPS = 1000  # you can adjust after you see dataset size

# Seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# -------------------------------------------------------------------
# 1. Train WordPiece tokenizer on RR_Val.txt
# -------------------------------------------------------------------
def train_tokenizer(text_file: str, model_dir: str, vocab_size: int, max_length: int):
    """
    Train a WordPiece tokenizer on `text_file` and save it to `model_dir`.
    Also save a config.json with basic tokenizer settings.
    """
    os.makedirs(model_dir, exist_ok=True)

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]

    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files=text_file, vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.enable_truncation(max_length=max_length)

    tokenizer.save_model(model_dir)

    tokenizer_cfg = {
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": max_length,
        "max_len": max_length,
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(tokenizer_cfg, f)


# -------------------------------------------------------------------
# 2. Dataset loading & splitting
# -------------------------------------------------------------------
def prepare_datasets(text_file: str, tokenizer, max_length: int, truncate_longer_samples: bool):
    """
    Load text dataset, split into train/test (90/10), create 'seen' split,
    tokenize, and return train_dataset, test_dataset.
    """
    dataset = load_dataset("text", data_files=[text_file], split="train")

    # Train/Test split
    d = dataset.train_test_split(test_size=0.1, seed=RANDOM_SEED)
    d_train = d["train"]
    d_test = d["test"]

    # Save seen samples (test) to file
    with open(SEEN_SAMPLES_FILE, "w") as f:
        for t in d_test["text"]:
            print(t, file=f)

    # Load seen split
    d_seen = load_dataset("text", data_files=[SEEN_SAMPLES_FILE], split="train")
    d["seen"] = d_seen

    print(d)
    print("Train (unique):", len(set(d_train["text"])))
    print("Test  (unique):", len(set(d_test["text"])))
    print("Seen  (unique):", len(set(d_seen["text"])))
    print("Raw lengths:", len(d_train["text"]), len(d_test["text"]), len(d_seen["text"]))

    # Tokenization functions
    def encode_with_truncation(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    def encode_without_truncation(examples):
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=True,
        )

    encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

    train_dataset = d_train.map(encode, batched=True)
    test_dataset = d_seen.map(encode, batched=True)  # they used "seen" as eval

    if truncate_longer_samples:
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])
    else:
        # If not truncating here, group into chunks of max_length later
        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples["input_ids"])
            if total_length >= max_length:
                total_length = (total_length // max_length) * max_length
            return {
                k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
                for k, t in concatenated_examples.items()
            }

        train_dataset = train_dataset.map(group_texts, batched=True, desc=f"Grouping texts in chunks of {max_length}")
        test_dataset = test_dataset.map(group_texts, batched=True, desc=f"Grouping texts in chunks of {max_length}")
        train_dataset.set_format("torch")
        test_dataset.set_format("torch")

    print("Tokenized sizes:", len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset


# -------------------------------------------------------------------
# 3. Global state for Optuna
# -------------------------------------------------------------------
best_trial_number = None
best_trial_score = float("inf")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


# -------------------------------------------------------------------
# 4. Objective function for Optuna
# -------------------------------------------------------------------
def objective(trial):
    global best_trial_number, best_trial_score

    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform("learning_rate", 3e-5, 3e-3)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    scheduler_type = trial.suggest_categorical("scheduler_type", ["linear", "cosine"])

    # Fresh model for each trial
    model_config = BertConfig(vocab_size=VOCAB_SIZE, max_position_embeddings=MAX_LENGTH)
    model = BertForMaskedLM(config=model_config).to(device)

    # Optimizer
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Scheduler
    if scheduler_type == "linear":
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=TRAINING_STEPS,
        )
    elif scheduler_type == "cosine":
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=TRAINING_STEPS,
        )
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    # Data loaders
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        test_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )

    # Logging files
    os.makedirs(MODEL_DIR, exist_ok=True)
    outer_file_path = OUTER_LOG_FILE
    train_eval_loss_file = LOSS_LOG_FILE
    trial_model_path = os.path.join(MODEL_DIR, f"model_best_trial_{trial.number}.pt")

    best_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    total_start_time = time.time()

    with open(outer_file_path, "a") as f, \
         open(train_eval_loss_file, "a") as loss_file, \
         open(TRIAL_PARAMS_FILE, "a") as param_file:

        # Write hyperparameters
        f.write(f"\n===== Trial {trial.number} =====\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f"Scheduler Type: {scheduler_type}\n")

        param_file.write(f"Trial {trial.number} Parameters:\n")
        param_file.write(f"Learning Rate: {learning_rate}\n")
        param_file.write(f"Batch Size: {batch_size}\n")
        param_file.write(f"Weight Decay: {weight_decay}\n")
        param_file.write(f"Optimizer: {optimizer_name}\n")
        param_file.write(f"Scheduler Type: {scheduler_type}\n\n")

        for epoch in range(1, MAX_EPOCHS + 1):
            # -------- Training loop --------
            model.train()
            train_losses = []
            epoch_start_time = time.time()

            for batch in tqdm(train_dataloader, desc=f"Trial {trial.number} - Epoch {epoch} [Train]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss

                # (Optional) extra weight decay term per param named 'weight'
                for name, param in model.named_parameters():
                    if "weight" in name and param.requires_grad:
                        loss = loss + weight_decay * torch.norm(param)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

            # -------- Evaluation loop --------
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc=f"Trial {trial.number} - Epoch {epoch} [Eval]"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    eval_losses.append(loss.item())

            eval_loss = float(np.mean(eval_losses)) if eval_losses else float("inf")

            epoch_time = time.time() - epoch_start_time

            # Logging
            log_line = (
                f"Trial {trial.number}, Epoch {epoch}, "
                f"Train Loss: {train_loss:.6f}, Validation Loss: {eval_loss:.6f}, "
                f"Epoch Time: {epoch_time:.2f} sec\n"
            )
            f.write(log_line)
            loss_file.write(log_line)

            # Early stopping on validation loss
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), trial_model_path)
                best_trial_number = trial.number
                best_trial_score = eval_loss
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                f.write("Early stopping triggered.\n")
                break

        total_time = time.time() - total_start_time
        f.write(f"Best Epoch: {best_epoch}, Best Validation Loss: {best_loss:.6f}\n")
        f.write(f"Total training time: {total_time:.2f} seconds\n")

    return best_loss


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) Train tokenizer
    print("Training tokenizer...")
    train_tokenizer(TEXT_FILE, MODEL_DIR, VOCAB_SIZE, MAX_LENGTH)

    # 2) Load tokenizer & prepare datasets
    print("Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)

    print("Preparing datasets...")
    train_dataset, test_dataset = prepare_datasets(
        TEXT_FILE, tokenizer, MAX_LENGTH, TRUNCATE_LONGER_SAMPLES
    )

    # 3) Run Optuna study
    print("Starting Optuna optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    # 4) Save best-trial info
    with open(BEST_TRIAL_FILE, "w") as best_f:
        best_f.write(f"Best Trial: {best_trial_number}\n")
        best_f.write(f"Best Trial Score (val loss): {best_trial_score}\n")
        best_f.write(f"Best Hyperparameters: {study.best_trial.params}\n")

    print("Best hyperparameters:", study.best_trial.params)
    print("Best trial number:", best_trial_number)
    print("Best validation loss:", best_trial_score)
