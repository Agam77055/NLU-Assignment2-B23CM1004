"""word2vec_scratch.py — Train Word2Vec (CBOW + Skip-gram) from scratch in PyTorch

Implements both CBOW and Skip-gram with Negative Sampling using pure PyTorch.
Runs a hyperparameter sweep over embedding dims, window sizes, and neg samples.

Usage:
    python word2vec_scratch.py

Outputs:
    results/p1_hyperparameter_sweep.csv - all sweep results
    results/p1_{model}_{dim}_{win}_{neg}.pt - saved model checkpoints
"""

import os
import re
import time
import math
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it doesn't try to open a window
import matplotlib.pyplot as plt

# ============================================================
# SEEDS + PATHS
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# HYPERPARAMETER GRID
# ============================================================

# hyperparameter grid for the sweep
EMBEDDING_DIMS = [50, 100, 200]
WINDOW_SIZES = [2, 5, 10]
NEG_SAMPLES = [5, 10, 15]

MIN_FREQ = 2         # ignore words appearing fewer than this many times
SUBSAMPLE_T = 1e-4   # subsampling threshold (standard value from original paper)
NEG_TABLE_SIZE = 1_000_000  # precomputed noise distribution table size
EPOCHS = 5           # keeping low for sweep speed - can increase for final model
BATCH_SIZE = 512
LR_INITIAL = 0.025   # not sure if this is the optimal value but the paper uses 0.025


# ============================================================
# VOCABULARY
# ============================================================

class Vocabulary:
    """Builds vocab from token list with frequency info + negative sampling table"""

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = {}  # raw counts
        self.word_prob = {}  # normalized probs for subsampling
        self.neg_table = None

    def build(self, tokens):
        """Build vocabulary from a flat list of tokens"""
        from collections import Counter
        raw_counts = Counter(tokens)

        # filter by min frequency - words that appear once are basically noise anyway
        filtered = {w: c for w, c in raw_counts.items() if c >= self.min_freq}

        # sort by frequency descending - makes it easy to inspect top words later
        sorted_words = sorted(filtered.items(), key=lambda x: -x[1])

        self.word2idx = {"<UNK>": 0}
        self.idx2word = {0: "<UNK>"}
        self.word_freq = {}

        for i, (word, count) in enumerate(sorted_words, start=1):
            self.word2idx[word] = i
            self.idx2word[i] = word
            self.word_freq[word] = count

        # compute subsampling probabilities
        # P_discard(w) = 1 - sqrt(t / f(w)) where f(w) is normalized freq
        # the idea is that really common words like "the" don't add much signal
        total = sum(filtered.values())
        self.word_prob = {}
        for word, count in filtered.items():
            freq_normalized = count / total
            # words with freq > t get subsampled - the closer to 1.0 the more likely to drop
            p_discard = max(0.0, 1.0 - math.sqrt(SUBSAMPLE_T / freq_normalized))
            self.word_prob[word] = p_discard

        print(f"Vocabulary built: {len(self.word2idx)} words (including <UNK>)")
        return self

    def __len__(self):
        return len(self.word2idx)

    def get_negative_table(self, size=NEG_TABLE_SIZE):
        """
        Build the unigram^(3/4) noise table for negative sampling.
        Pre-computing this makes negative sampling O(1) at training time.

        The 3/4 power smooths out the distribution so rare words get
        sampled more often than they would under a pure unigram distribution.
        This is straight from Mikolov et al. 2013.
        """
        if self.neg_table is not None:
            return self.neg_table

        # unigram distribution raised to 3/4 power - this is what the paper uses
        # it smooths out the distribution so rare words get sampled more often
        words = list(self.word_freq.keys())
        freqs = np.array([self.word_freq[w] ** 0.75 for w in words])
        freqs = freqs / freqs.sum()  # normalize so it sums to 1

        # fill table with word indices according to this distribution
        word_indices = [self.word2idx[w] for w in words]
        table = []
        for idx, prob in zip(word_indices, freqs):
            count = int(prob * size)
            table.extend([idx] * count)

        # pad or trim to exact size - usually off by a small amount due to int rounding
        while len(table) < size:
            table.append(table[-1])
        table = table[:size]

        self.neg_table = np.array(table, dtype=np.int64)
        print(f"Negative sampling table built (size={size:,})")
        return self.neg_table

    def subsample_tokens(self, tokens):
        """Apply subsampling - skip frequent words with some probability"""
        result = []
        for token in tokens:
            if token not in self.word2idx:
                continue  # skip OOV words entirely
            # skip this token with probability p_discard
            p_disc = self.word_prob.get(token, 0.0)
            if random.random() < p_disc:
                continue
            result.append(self.word2idx[token])
        return result


# ============================================================
# DATASETS
# ============================================================

class CBOWDataset(Dataset):
    """Dataset for CBOW: (context_indices, center_idx) pairs"""

    def __init__(self, token_ids, window_size):
        self.samples = []
        self.window = window_size

        # build all (context, target) pairs - this takes a bit for large corpora
        n = len(token_ids)
        for i in range(window_size, n - window_size):
            # grab the window on both sides and concatenate them
            ctx = token_ids[i - window_size : i] + token_ids[i+1 : i + window_size + 1]
            center = token_ids[i]
            self.samples.append((ctx, center))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, center = self.samples[idx]
        return torch.tensor(ctx, dtype=torch.long), torch.tensor(center, dtype=torch.long)


class SkipgramDataset(Dataset):
    """Dataset for Skip-gram: (center_idx, context_idx) one pair at a time"""

    def __init__(self, token_ids, window_size):
        self.pairs = []
        n = len(token_ids)

        # NOTE: skip-gram generates way more pairs than CBOW because each center
        # word gets paired individually with each context word
        for i in range(len(token_ids)):
            center = token_ids[i]
            # randomly vary the actual window size - mentioned in the paper
            # this downweights context words that are farther away
            actual_win = random.randint(1, window_size)
            start = max(0, i - actual_win)
            end = min(n, i + actual_win + 1)
            for j in range(start, end):
                if j != i:
                    self.pairs.append((center, token_ids[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, ctx = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(ctx, dtype=torch.long)


# ============================================================
# MODELS
# ============================================================

class CBOWModel(nn.Module):
    """
    CBOW: predict center word from averaged context embeddings.
    Uses two embedding matrices (in and out) + negative sampling loss.

    In CBOW the context words are averaged together to form a single
    vector, then we try to predict the center word from that vector.
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # input embeddings (for context words)
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        # output embeddings (for center/negative words)
        # having separate in/out embeddings is important - learned it from the paper
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

        # init with small values - zeros for out_embed is from the original code
        nn.init.uniform_(self.in_embed.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, context_words, center_word, neg_words):
        """
        context_words: [B, 2*window]
        center_word: [B]
        neg_words: [B, num_neg]
        """
        # average context embeddings - this is the "continuous bag of words" part
        ctx_emb = self.in_embed(context_words)   # [B, 2*win, D]
        ctx_mean = ctx_emb.mean(dim=1)            # [B, D]

        # positive score - dot product with actual center word
        pos_emb = self.out_embed(center_word)     # [B, D]
        pos_score = (ctx_mean * pos_emb).sum(1)   # [B]

        # negative scores - dot product with sampled noise words
        neg_emb = self.out_embed(neg_words)       # [B, num_neg, D]
        # bmm: [B, num_neg, D] x [B, D, 1] -> [B, num_neg, 1]
        neg_score = torch.bmm(neg_emb, ctx_mean.unsqueeze(2)).squeeze(2)  # [B, num_neg]

        # NCE loss - maximize log sigma(pos) and log sigma(-neg)
        # I'm using negative sampling here because full softmax is too slow
        loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(1)).mean()
        return loss

    def get_embeddings(self):
        """Return the input embedding matrix as numpy array"""
        return self.in_embed.weight.detach().cpu().numpy()


class SkipgramModel(nn.Module):
    """
    Skip-gram: predict context words from center word embedding.
    Same two-matrix setup as CBOW but roles are reversed.
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

        # same init as CBOW
        nn.init.uniform_(self.in_embed.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center_words, context_words, neg_words):
        """
        center_words: [B]
        context_words: [B]
        neg_words: [B, num_neg]
        """
        ctr_emb = self.in_embed(center_words)            # [B, D]
        ctr_col = ctr_emb.unsqueeze(2)                   # [B, D, 1] - need this shape for bmm

        pos_emb = self.out_embed(context_words)          # [B, D]
        # dot product: [B, 1, D] x [B, D, 1] -> [B, 1, 1]
        pos_score = torch.bmm(pos_emb.unsqueeze(1), ctr_col).squeeze()   # [B]

        neg_emb = self.out_embed(neg_words)              # [B, num_neg, D]
        neg_score = torch.bmm(neg_emb, ctr_col).squeeze(2)               # [B, num_neg]

        # same NCE loss as CBOW
        loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(1)).mean()
        return loss

    def get_embeddings(self):
        return self.in_embed.weight.detach().cpu().numpy()


# ============================================================
# NEGATIVE SAMPLING HELPER
# ============================================================

def sample_negatives(batch_size, num_neg, neg_table, positive_indices):
    """
    Sample negative word indices from the precomputed noise table.
    Tries to avoid sampling actual positives - not perfect but good enough.

    The 2x buffer trick is to make sure we have enough samples even after
    filtering out positives. In practice collisions are rare enough that
    this almost never matters.
    """
    pos_set = set(positive_indices.tolist()) if hasattr(positive_indices, 'tolist') else set()

    # sample 2x more than needed so we have room to filter collisions
    negs = np.random.choice(neg_table, size=(batch_size * num_neg * 2,), replace=True)

    result = []
    i = 0
    for b in range(batch_size):
        row = []
        while len(row) < num_neg and i < len(negs):
            candidate = negs[i]
            i += 1
            # skip if it's a positive (rare collision but worth checking)
            if candidate not in pos_set:
                row.append(candidate)
        # if we ran out (shouldn't happen with 2x buffer), just fill with random samples
        while len(row) < num_neg:
            row.append(np.random.choice(neg_table))
        result.append(row)

    return torch.tensor(result, dtype=torch.long)


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(model, dataset, num_neg, neg_table, device, epochs=EPOCHS, lr=LR_INITIAL):
    """Train a Word2Vec model (works for both CBOW and Skipgram)"""

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # SGD with linear decay - using SGD because that's what the original paper uses
    # Adam would probably work too but SGD feels more faithful to the implementation
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    model.to(device)

    total_steps = len(loader) * epochs
    step = 0

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False)

        for batch in pbar:
            # figure out if this is CBOW or Skipgram by checking model type
            # (both return 2-element batches so we need isinstance)
            if len(batch) == 2:
                pass  # expected, just being explicit here

            # linear LR decay - lr goes from initial down to near 0 over training
            progress = step / total_steps
            current_lr = lr * (1.0 - progress)
            current_lr = max(current_lr, lr * 0.0001)  # don't go below 0.01% of initial
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            optimizer.zero_grad()

            if isinstance(model, CBOWModel):
                ctx, center = batch
                ctx, center = ctx.to(device), center.to(device)
                neg_words = sample_negatives(ctx.size(0), num_neg, neg_table, center).to(device)
                loss = model(ctx, center, neg_words)
            else:  # Skipgram
                center, context = batch
                center, context = center.to(device), context.to(device)
                neg_words = sample_negatives(center.size(0), num_neg, neg_table, context).to(device)
                loss = model(center, context, neg_words)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.5f}"})

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

    return losses


# ============================================================
# HYPERPARAMETER SWEEP
# ============================================================

def run_sweep(tokens, vocab, device):
    """
    Run the full hyperparameter sweep.
    For each combo of (embed_dim, window_size, num_neg), train both CBOW and Skipgram.
    This will take a while...

    TODO: could parallelize across configs if multiple GPUs are available
    """
    neg_table = vocab.get_negative_table()
    token_ids = vocab.subsample_tokens(tokens)

    print(f"After subsampling: {len(token_ids)} tokens (was {len(tokens)})")

    results = []

    for embed_dim in EMBEDDING_DIMS:
        for window_size in WINDOW_SIZES:
            for num_neg in NEG_SAMPLES:

                config_name = f"dim{embed_dim}_win{window_size}_neg{num_neg}"
                print(f"\n{'='*60}")
                print(f"Config: {config_name}")
                print(f"{'='*60}")

                # --- CBOW ---
                print("Training CBOW...")
                cbow_dataset = CBOWDataset(token_ids, window_size)
                cbow_model = CBOWModel(len(vocab), embed_dim)

                t0 = time.time()
                cbow_losses = train_model(cbow_model, cbow_dataset, num_neg, neg_table, device)
                cbow_time = time.time() - t0

                # save checkpoint - storing the vocab mapping so we can load + use embeddings later
                cbow_path = os.path.join(RESULTS_DIR, f"p1_cbow_{embed_dim}_{window_size}_{num_neg}.pt")
                torch.save({
                    "embeddings": cbow_model.get_embeddings(),
                    "vocab": vocab.word2idx,
                    "embed_dim": embed_dim,
                    "window_size": window_size,
                    "num_neg": num_neg,
                    "final_loss": cbow_losses[-1],
                }, cbow_path)

                results.append({
                    "model": "cbow",
                    "embed_dim": embed_dim,
                    "window_size": window_size,
                    "num_neg": num_neg,
                    "final_loss": cbow_losses[-1],
                    "train_time": cbow_time,
                })

                # --- Skipgram ---
                print("Training Skipgram...")
                sg_dataset = SkipgramDataset(token_ids, window_size)
                sg_model = SkipgramModel(len(vocab), embed_dim)

                t0 = time.time()
                sg_losses = train_model(sg_model, sg_dataset, num_neg, neg_table, device)
                sg_time = time.time() - t0

                sg_path = os.path.join(RESULTS_DIR, f"p1_skipgram_{embed_dim}_{window_size}_{num_neg}.pt")
                torch.save({
                    "embeddings": sg_model.get_embeddings(),
                    "vocab": vocab.word2idx,
                    "embed_dim": embed_dim,
                    "window_size": window_size,
                    "num_neg": num_neg,
                    "final_loss": sg_losses[-1],
                }, sg_path)

                results.append({
                    "model": "skipgram",
                    "embed_dim": embed_dim,
                    "window_size": window_size,
                    "num_neg": num_neg,
                    "final_loss": sg_losses[-1],
                    "train_time": sg_time,
                })

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load corpus - expecting a plain text file, one doc per line or just raw text
    corpus_path = os.path.join(DATA_DIR, "corpus.txt")
    if not os.path.exists(corpus_path):
        print("ERROR: corpus.txt not found. Run preprocess.py first!")
        return

    print("Loading corpus...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    # tokenize - just pulling out alphabetic tokens and lowercasing
    # could do something fancier but this is probably fine for the assignment
    tokens = re.findall(r'[a-z]+', text.lower())
    print(f"Total tokens: {len(tokens):,}")

    # build vocab
    vocab = Vocabulary(min_freq=MIN_FREQ)
    vocab.build(tokens)

    # run the full sweep
    print("\nStarting hyperparameter sweep...")
    print(f"Configs: {len(EMBEDDING_DIMS)} dims x {len(WINDOW_SIZES)} windows x {len(NEG_SAMPLES)} neg = "
          f"{len(EMBEDDING_DIMS)*len(WINDOW_SIZES)*len(NEG_SAMPLES)} combos x 2 models = "
          f"{len(EMBEDDING_DIMS)*len(WINDOW_SIZES)*len(NEG_SAMPLES)*2} total runs")

    results = run_sweep(tokens, vocab, device)

    # save results to CSV
    csv_path = os.path.join(RESULTS_DIR, "p1_hyperparameter_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "embed_dim", "window_size", "num_neg", "final_loss", "train_time"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSweep results saved to {csv_path}")

    # pull out best configs for each model type
    cbow_results = [r for r in results if r["model"] == "cbow"]
    sg_results = [r for r in results if r["model"] == "skipgram"]

    best_cbow = min(cbow_results, key=lambda x: x["final_loss"])
    best_sg = min(sg_results, key=lambda x: x["final_loss"])

    print(f"\nBest CBOW config: dim={best_cbow['embed_dim']}, win={best_cbow['window_size']}, neg={best_cbow['num_neg']} (loss={best_cbow['final_loss']:.4f})")
    print(f"Best Skipgram config: dim={best_sg['embed_dim']}, win={best_sg['window_size']}, neg={best_sg['num_neg']} (loss={best_sg['final_loss']:.4f})")


if __name__ == "__main__":
    main()
