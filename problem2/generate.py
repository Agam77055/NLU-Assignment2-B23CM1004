"""generate.py — Generate names using trained models and compute metrics

Loads checkpoints and generates 500 names per model.
Computes novelty rate and diversity.

Usage:
    python generate.py

Outputs:
    results/p2_generated_{rnn,blstm,attn}.txt
    results/p2_metrics.csv
    plots/p2_metrics_comparison.png
"""

import os
import sys
import torch
import numpy as np
import csv
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dataset import NameDataset, CharVocab
from models import VanillaRNN, BLSTMModel, AttentionRNNModel

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
PLOTS_DIR = os.path.join(BASE_DIR, "..", "plots")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

HIDDEN_SIZE = 256
EMBED_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.0   # no dropout during generation
N_GENERATE = 500
MAX_LEN = 20
TEMPERATURE = 0.8  # controls randomness - lower = more predictable names

# ============================================================
# GENERATION FUNCTIONS
# ============================================================

def generate_name(model, vocab, device, max_len=MAX_LEN, temperature=TEMPERATURE, model_type="rnn"):
    """
    Autoregressively generate a single name.

    Starts with SOS token, samples next character from the output distribution
    (scaled by temperature), stops at EOS or max_len.
    """
    model.eval()

    with torch.no_grad():
        # start with SOS token
        current_idx = vocab.SOS_IDX
        hidden = None
        generated_indices = []

        for step in range(max_len):
            # input is just the current character: [1, 1]
            x = torch.tensor([[current_idx]], dtype=torch.long, device=device)

            if model_type == "blstm":
                logits, hidden = model.forward_only(x, hidden)
            else:
                logits, hidden = model(x, hidden)

            # logits: [1, 1, vocab_size] -> [vocab_size]
            logits_1d = logits[0, 0, :]

            # apply temperature scaling
            # lower temp = more confident/repetitive, higher = more random/diverse
            scaled = logits_1d / max(temperature, 1e-6)
            probs = torch.softmax(scaled, dim=0)

            # sample from distribution
            next_idx = torch.multinomial(probs, num_samples=1).item()

            if next_idx == vocab.EOS_IDX:
                break

            # skip PAD and SOS tokens if they somehow appear mid-generation
            if next_idx in (vocab.PAD_IDX, vocab.SOS_IDX):
                continue

            generated_indices.append(next_idx)
            current_idx = next_idx

        return vocab.decode([vocab.SOS_IDX] + generated_indices + [vocab.EOS_IDX])


def generate_names_batch(model, vocab, device, n=N_GENERATE, model_type="rnn"):
    """Generate n names using the given model"""
    names = []
    for i in range(n):
        name = generate_name(model, vocab, device, model_type=model_type)
        names.append(name)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n}...")
    return names


# ============================================================
# EVALUATION METRICS
# ============================================================

def novelty_rate(generated_names, training_names):
    """
    Novelty = fraction of generated names NOT in the training set.
    We want this to be high - the model should generalize, not memorize.
    """
    training_set = set(name.strip().lower() for name in training_names)
    novel = sum(1 for name in generated_names if name.lower() not in training_set)
    return novel / max(len(generated_names), 1)


def diversity(generated_names):
    """
    Diversity = number of unique names / total names generated.
    Higher is better - we don't want the model to generate the same name over and over.
    """
    cleaned = [n for n in generated_names if n]  # filter empty strings
    if not cleaned:
        return 0.0
    return len(set(cleaned)) / len(cleaned)


def load_model_from_checkpoint(ckpt_path, device):
    """Load a saved model from checkpoint file"""
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    model_name = ckpt.get("model_name", "unknown")

    # recreate the model based on name
    vocab_size = vocab.vocab_size

    if "VanillaRNN" in model_name:
        model = VanillaRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    elif "BLSTM" in model_name:
        model = BLSTMModel(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    elif "AttnRNN" in model_name:
        model = AttentionRNNModel(vocab_size, EMBED_DIM, HIDDEN_SIZE, num_layers=1, dropout=DROPOUT)
    else:
        print(f"  Unknown model type: {model_name}")
        return None, None

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"  Loaded {model_name} (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', 0):.4f})")
    return model, vocab


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load training names for novelty calculation
    names_path = os.path.join(DATA_DIR, "TrainingNames.txt")
    with open(names_path, "r") as f:
        training_names = [l.strip() for l in f if l.strip()]
    print(f"Training set: {len(training_names)} names")

    # model checkpoints and output files
    model_configs = [
        ("VanillaRNN", os.path.join(RESULTS_DIR, "p2_VanillaRNN_best.pt"), "rnn",  "p2_generated_rnn.txt"),
        ("BLSTM",      os.path.join(RESULTS_DIR, "p2_BLSTM_best.pt"),      "blstm","p2_generated_blstm.txt"),
        ("AttnRNN",    os.path.join(RESULTS_DIR, "p2_AttnRNN_best.pt"),    "rnn",  "p2_generated_attn.txt"),
    ]

    all_metrics = []

    for model_name, ckpt_path, model_type, out_fname in model_configs:
        print(f"\n=== {model_name} ===")

        model, vocab = load_model_from_checkpoint(ckpt_path, device)
        if model is None:
            print(f"  Skipping {model_name} - not trained yet")
            continue

        print(f"  Generating {N_GENERATE} names (temp={TEMPERATURE})...")
        generated = generate_names_batch(model, vocab, device, n=N_GENERATE, model_type=model_type)

        # filter out very short/empty names
        generated = [n for n in generated if len(n) >= 2]

        # save generated names
        out_path = os.path.join(RESULTS_DIR, out_fname)
        with open(out_path, "w") as f:
            for name in generated:
                f.write(name + "\n")
        print(f"  Saved {len(generated)} names to {out_path}")

        # compute metrics
        nov = novelty_rate(generated, training_names)
        div = diversity(generated)

        print(f"  Novelty  : {nov:.3f} ({nov*100:.1f}% of names not in training set)")
        print(f"  Diversity: {div:.3f} ({div*100:.1f}% unique names)")

        all_metrics.append({
            "model": model_name,
            "n_generated": len(generated),
            "novelty_rate": round(nov, 4),
            "diversity": round(div, 4),
        })

    if not all_metrics:
        print("\nNo models evaluated. Run train.py first!")
        return

    # save metrics to CSV
    csv_path = os.path.join(RESULTS_DIR, "p2_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "n_generated", "novelty_rate", "diversity"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nMetrics saved to {csv_path}")

    # bar chart comparison
    model_names = [m["model"] for m in all_metrics]
    novelties  = [m["novelty_rate"] for m in all_metrics]
    diversities = [m["diversity"] for m in all_metrics]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, novelties, width, label="Novelty Rate", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width/2, diversities, width, label="Diversity", color="#2ecc71", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Name Generation Metrics by Model")
    ax.legend()

    # add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "p2_metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metrics chart saved to plots/p2_metrics_comparison.png")

if __name__ == "__main__":
    main()
