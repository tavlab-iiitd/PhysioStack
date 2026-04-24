#!/usr/bin/env python3
"""
Pretraining BERT (Masked Language Modeling) on ABP 60-min symbolic sequences.

Pipeline:
1. Train a WordPiece tokenizer on ABP_Train.txt with special tokens.
2. Save tokenizer + config to MODEL_DIR.
3. Load text data with Hugging Face `datasets`.
4. Split into train / test (90% / 10%), define "seen" as test split and save.
5. Tokenize with truncation + padding to max_length.
6. Build a fresh `BertForMaskedLM` model from scratch.
7. Train with:
   - accelerate.Accelerator (GPU / multi-GPU support)
   - Linear learning-rate scheduler
   - Early stopping based on validation loss
   - Optional Weights & Biases logging

Edit the CONFIG section to point to your data/model paths.
"""

import os
import json
import math
import time
import random
from itertools import chain

import numpy as np
import torch
from torch.utils.data import DataLoader
from tokenizers import BertWordPieceTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm

from accelerate import Accelerator
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

# Optional: comment this out if you don't want to use Weights & Biases
import wandb


# -------------------------------------------------------------------
# CONFIG: update these paths for your environment
# -------------------------------------------------------------------
TRAIN_TEXT_FILE = "/path/to/ABP_Train.txt"  # e.g. "/workspace/.../ABP_Train.txt"

MODEL_DIR = "/path/to/output/model_dir"  # e.g. "/workspace/.../SafeICU/ABP/60min/Model"

SEEN_SAMPLES_FILE = os.path.join(MODEL_DIR, "PreTrain_seen_ABP_60min.txt")
LOSS_LOG_FILE = os.path.join(MODEL_DIR, "Train_Val_Loss_ABP_60min.txt")

# Tokenizer / Model hyperparameters
VOCAB_SIZE = 30522
MAX_LENGTH = 512
TRUNCATE_LONGER_SAMPLES = True  # your sequences are <= 512 anyway, but kept for clarity

# Training hyperparameters
LEARNING_RATE = 5.513462007374876e-4  # from your script
WEIGHT_DECAY = 2.6586444051788025e-05
BATCH_SIZE = 4
NUM_EPOCHS = 100
PATIENCE = 25  # early stopping patience
WARMUP_STEPS = 1000
TRAINING_STEPS = 1000  # used for scheduler; can be tuned

# wandb configuration
USE_WANDB = True
WANDB_PROJECT = "huggingface"
WANDB_ENTITY = "akshayaiiitd"  # change or remove if needed

# Random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# -------------------------------------------------------------------
# 1. Train WordPiece tokenizer on ABP_Train.txt
# -------------------------------------------------------------------
def train_tokenizer(text_file: str, model_dir: str, vocab_size: int, max_length: int):
    """
    Train a WordPiece tokenizer on `text_file` and save to `model_dir`.
    Also writes a tokenizer config JSON with special tokens and max length.
    """
    os.makedirs(model_dir, exist_ok=True)

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]

    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files=[text_file], vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.enable_truncation(max_length=max_length)

    # Save vocab + tokenizer files
    tokenizer.save_model(model_dir)

    # Save config for BertTokenizerFast
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
def prepare_datasets(text_file: str, tokenizer, max_length: int, truncate_longer: bool):
    """
    Load text dataset, split into train/test (90/10), create 'seen' split (= test),
    tokenize, and return train_dataset, eval_dataset.
    """
    dataset = load_dataset("text", data_files=[text_file], split="train")
    d = dataset.train_test_split(test_size=0.1, seed=RANDOM_SEED)

    d_train = d["train"]
    d_test = d["test"]

    # Save test ("seen") samples to file
    with open(SEEN_SAMPLES_FILE, "w") as f:
        for t in d_test["text"]:
            print(t, file=f)

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

    encode = encode_with_truncation if truncate_longer else encode_without_truncation

    train_dataset = d_train.map(encode, batched=True)
    eval_dataset = d_seen.map(encode, batched=True)

    if truncate_longer:
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])
    else:
        # Optional: group into fixed-length chunks if you don't truncate
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
        eval_dataset = eval_dataset.map(group_texts, batched=True, desc=f"Grouping texts in chunks of {max_length}")
        train_dataset.set_format("torch")
        eval_dataset.set_format("torch")

    print("Tokenized sizes:", len(train_dataset), len(eval_dataset))
    return train_dataset, eval_dataset


# -------------------------------------------------------------------
# 3. Build model, dataloaders, and training loop
# -------------------------------------------------------------------
def build_model(vocab_size: int, max_length: int) -> BertForMaskedLM:
    """
    Create a fresh BERT model for MLM with given vocab size and max position embeddings.
    """
    config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
    model = BertForMaskedLM(config=config)
    return model


def build_dataloaders(train_dataset, eval_dataset, tokenizer, batch_size: int):
    """
    Create PyTorch DataLoaders for train and eval using MLM data collator.
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    eval_loader = DataLoader(
        eval_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return train_loader, eval_loader


def train_mlm(
    model,
    tokenizer,
    train_loader,
    eval_loader,
    model_dir: str,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    patience: int,
    warmup_steps: int,
    training_steps: int,
    use_wandb: bool = True,
):
    """
    Train BERT-MLM with accelerate, early stopping, and optional W&B logging.
    Saves best model and tokenizer to `model_dir`.
    """
    accelerator = Accelerator()
    model, train_loader, eval_loader = accelerator.prepare(model, train_loader, eval_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
    )

    if use_wandb:
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
        config = wandb.config
        config.learning_rate = learning_rate
        config.batch_size = BATCH_SIZE
        config.weight_decay = weight_decay

    os.makedirs(model_dir, exist_ok=True)

    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    with open(LOSS_LOG_FILE, "w") as f:
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            model.train()
            train_losses = []

            print(f"\nEpoch {epoch}/{num_epochs}")
            for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss

                # Add weight decay penalty
                for name, param in model.named_parameters():
                    if "weight" in name and param.requires_grad:
                        loss = loss + weight_decay * torch.norm(param)

                accelerator.backward(loss)
                train_losses.append(accelerator.gather(loss.detach()))

                optimizer.step()
                lr_scheduler.step()

            # Compute mean train loss
            train_losses = torch.cat(train_losses)
            train_mean_loss = train_losses.mean().item()
            try:
                train_perplexity = math.exp(train_mean_loss)
            except OverflowError:
                train_perplexity = float("inf")

            print(f"  Train loss: {train_mean_loss:.6f}, perplexity: {train_perplexity:.4f}")
            f.write(f"Epoch {epoch}\n")
            f.write(f"train_loss: {train_mean_loss}\n")

            if use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_mean_loss})

            # ----------------- Validation -----------------
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"Epoch {epoch} [Eval]"):
                    outputs = model(**batch)
                    loss = outputs.loss
                    eval_losses.append(accelerator.gather(loss.detach()))

            eval_losses = torch.cat(eval_losses)
            eval_mean_loss = eval_losses.mean().item()
            try:
                eval_perplexity = math.exp(eval_mean_loss)
            except OverflowError:
                eval_perplexity = float("inf")

            print(f"  Val loss:   {eval_mean_loss:.6f}, perplexity: {eval_perplexity:.4f}")
            f.write(f"eval_loss: {eval_mean_loss}\n")
            f.write(f"learning_rate: {learning_rate}\n\n")

            if use_wandb:
                wandb.log({"val_loss": eval_mean_loss})

            # ----------------- Early stopping -----------------
            if eval_mean_loss < best_loss:
                best_loss = eval_mean_loss
                best_epoch = epoch
                patience_counter = 0

                # Save best model
                output_dir = model_dir
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)

                print("  → New best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"No improvement in {patience} epochs. Stopping early.")
                    break

            epoch_time = time.time() - epoch_start
            print(f"  Epoch time: {epoch_time:.2f} seconds")

        print(f"\nBest validation loss: {best_loss:.6f} at epoch {best_epoch}")

    if use_wandb:
        wandb.finish()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Training tokenizer...")
    train_tokenizer(TRAIN_TEXT_FILE, MODEL_DIR, VOCAB_SIZE, MAX_LENGTH)

    print("Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)

    print("Preparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(
        TRAIN_TEXT_FILE,
        tokenizer,
        MAX_LENGTH,
        TRUNCATE_LONGER_SAMPLES,
    )

    print("Building model...")
    model = build_model(VOCAB_SIZE, MAX_LENGTH)

    print("Building dataloaders...")
    train_loader, eval_loader = build_dataloaders(
        train_dataset,
        eval_dataset,
        tokenizer,
        batch_size=BATCH_SIZE,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Starting training...")
    train_mlm(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        model_dir=MODEL_DIR,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        warmup_steps=WARMUP_STEPS,
        training_steps=TRAINING_STEPS,
        use_wandb=USE_WANDB,
    )
