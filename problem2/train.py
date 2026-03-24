"""train.py — Train all three character-level name generation models

Trains VanillaRNN, BLSTMModel, and AttentionRNNModel on the
Indian names dataset and saves checkpoints.

Usage:
    python train.py

Outputs:
    results/p2_{model_name}_best.pt  - best model checkpoints
    plots/p2_training_loss.png       - loss curves comparison
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# add parent dir to path so we can import from sibling modules
sys.path.insert(0, os.path.dirname(__file__))
from dataset import NameDataset, collate_fn, CharVocab
from models import VanillaRNN, BLSTMModel, AttentionRNNModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "..", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# HYPERPARAMETERS
# ============================================================

# these values were chosen after some trial and error
HIDDEN_SIZE = 256
EMBED_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 0.001
EPOCHS = 50
BATCH_SIZE = 64
CLIP_GRAD = 5.0   # gradient clipping threshold - critical for RNNs
PATIENCE = 5      # early stopping patience

VAL_SPLIT = 0.1   # 10% of data for validation

# ============================================================
# TRAINING LOOP
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, clip=CLIP_GRAD, model_type="rnn"):
    """Run one training epoch, return average loss"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        inputs, targets, lengths = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # forward pass depends on model type
        if model_type == "blstm":
            logits, _ = model.forward(inputs)
        else:
            logits, _ = model(inputs)

        # reshape for loss: [B*T, vocab_size] and [B*T]
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), targets.reshape(B * T))

        loss.backward()

        # gradient clipping - really important for vanilla RNN especially
        # I learned the hard way that without this the loss goes to NaN
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def eval_epoch(model, loader, criterion, device, model_type="rnn"):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets, lengths = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            if model_type == "blstm":
                logits, _ = model.forward(inputs)
            else:
                logits, _ = model(inputs)

            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), targets.reshape(B * T))
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model(model, model_name, train_loader, val_loader, vocab, device, epochs=EPOCHS):
    """
    Full training loop with early stopping.
    Saves best checkpoint based on validation loss.
    """
    pad_idx = vocab.PAD_IDX
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # learning rate scheduler - reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    model.to(device)

    # figure out model type for forward pass
    model_type = "blstm" if "BLSTM" in type(model).__name__ else "rnn"

    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []

    print(f"\nTraining {model_name} ({model.count_parameters():,} params)")
    print(f"  Epochs: {epochs}, LR: {LR}, Batch: {BATCH_SIZE}")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                  model_type=model_type)
        val_loss = eval_epoch(model, val_loader, criterion, device, model_type=model_type)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch+1:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # save best checkpoint
            ckpt_path = os.path.join(RESULTS_DIR, f"p2_{model_name}_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "model_name": model_name,
            }, ckpt_path)
            print(" * saved", end="")
        else:
            patience_counter += 1

        print()

        # early stopping
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    print(f"  Best val loss: {best_val_loss:.4f}")
    return train_losses, val_losses


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load dataset
    names_path = os.path.join(DATA_DIR, "TrainingNames.txt")
    if not os.path.exists(names_path):
        print(f"ERROR: {names_path} not found")
        return

    dataset = NameDataset(names_path)
    vocab = dataset.vocab

    # train/val split
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val

    g = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn)

    print(f"Train: {n_train} names, Val: {n_val} names")
    print(f"Vocab size: {vocab.vocab_size}")

    # define all three models
    model_defs = [
        ("VanillaRNN", VanillaRNN(vocab.vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)),
        ("BLSTM",      BLSTMModel(vocab.vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)),
        ("AttnRNN",    AttentionRNNModel(vocab.vocab_size, EMBED_DIM, HIDDEN_SIZE, num_layers=1, dropout=DROPOUT)),
        # Note: AttentionRNN works better with 1 layer in my experiments
    ]

    all_train_losses = {}
    all_val_losses = {}

    for model_name, model in model_defs:
        train_losses, val_losses = train_model(
            model, model_name, train_loader, val_loader, vocab, device, epochs=EPOCHS
        )
        all_train_losses[model_name] = train_losses
        all_val_losses[model_name] = val_losses

    # plot loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"VanillaRNN": "#e74c3c", "BLSTM": "#3498db", "AttnRNN": "#2ecc71"}

    for name in all_train_losses:
        ax1.plot(all_train_losses[name], label=name, color=colors.get(name, "gray"))
        ax2.plot(all_val_losses[name], label=name, color=colors.get(name, "gray"))

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.legend()

    plt.suptitle("Character-Level Name Generation — Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "p2_training_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nLoss curve saved to plots/p2_training_loss.png")

if __name__ == "__main__":
    main()
