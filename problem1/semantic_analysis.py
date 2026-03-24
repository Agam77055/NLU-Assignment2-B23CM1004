"""semantic_analysis.py — Cosine similarity neighbors and word analogies

Loads trained embeddings and performs semantic analysis:
1. Top-5 nearest neighbors for query words
2. Analogy experiments using 3CosAdd

Usage:
    python semantic_analysis.py
"""

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

# ============================================================
# PATHS AND SEED
# ============================================================

SEED = 42
BASE_DIR    = os.path.dirname(__file__)
PLOTS_DIR   = os.path.join(BASE_DIR, "..", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

# ============================================================
# QUERY WORDS AND ANALOGIES
# ============================================================

QUERY_WORDS = ["research", "student", "phd", "exam"]

ANALOGIES = [
    # (a, b, c) -> find d such that a:b :: c:d
    # i.e., vec(b) - vec(a) + vec(c) should be close to vec(d)
    ("ug",        "btech",    "pg"),          # UG:BTech :: PG:?
    ("professor", "research", "student"),     # professor:research :: student:?
    ("hostel",    "student",  "lab"),         # hostel:student :: lab:?
    ("lecture",   "professor","tutorial"),    # extra one just to be safe
]


# ============================================================
# LOADING EMBEDDINGS
# ============================================================

def load_embeddings(checkpoint_path):
    """Load embeddings and vocab from a saved .pt file"""
    if not os.path.exists(checkpoint_path):
        print(f"File not found: {checkpoint_path}")
        return None, None, None

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    embeddings = ckpt["embeddings"]  # numpy array [vocab_size, dim]
    word2idx   = ckpt["vocab"]
    idx2word   = {v: k for k, v in word2idx.items()}

    # normalize embeddings once for cosine similarity
    # doing it here saves time later since we reuse many times
    norms             = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings_normed = embeddings / norms

    return embeddings_normed, word2idx, idx2word


# ============================================================
# NEAREST NEIGHBORS
# ============================================================

def get_neighbors(word, embeddings, word2idx, idx2word, top_k=5):
    """Find top_k nearest neighbors by cosine similarity"""
    if word not in word2idx:
        return []

    query_idx = word2idx[word]
    query_vec = embeddings[query_idx]  # already normalized

    # dot product with normalized vectors = cosine similarity
    sims = embeddings @ query_vec

    # exclude the query word itself
    sims[query_idx] = -2.0

    top_k_indices = np.argsort(sims)[::-1][:top_k]
    neighbors = [(idx2word[i], float(sims[i])) for i in top_k_indices if i in idx2word]
    return neighbors


# ============================================================
# ANALOGY (3CosAdd)
# ============================================================

def analogy_3cosadd(a, b, c, embeddings, word2idx, idx2word, top_k=5):
    """
    3CosAdd analogy: a is to b as c is to ?
    Answer = argmax cos(d, b-a+c)
    This is the standard approach from the original word2vec paper.
    """
    # check all words exist
    for w in [a, b, c]:
        if w not in word2idx:
            return f"'{w}' not in vocabulary"

    # compute the analogy vector
    vec_a = embeddings[word2idx[a]]
    vec_b = embeddings[word2idx[b]]
    vec_c = embeddings[word2idx[c]]

    query_vec = vec_b - vec_a + vec_c
    # normalize
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    sims = embeddings @ query_vec

    # exclude a, b, c from results
    exclude = {word2idx[a], word2idx[b], word2idx[c]}
    for idx in exclude:
        sims[idx] = -2.0

    top_k_idx = np.argsort(sims)[::-1][:top_k]
    results   = [(idx2word[i], float(sims[i])) for i in top_k_idx if i in idx2word]
    return results


# ============================================================
# PRINTING / FORMATTING
# ============================================================

def print_neighbors_table(model_name, embeddings, word2idx, idx2word):
    """Print a formatted table of nearest neighbors for all query words"""
    print(f"\n{'='*60}")
    print(f"  Nearest Neighbors — {model_name}")
    print(f"{'='*60}")

    lines = [f"Nearest Neighbors — {model_name}", "="*60]

    for word in QUERY_WORDS:
        neighbors = get_neighbors(word, embeddings, word2idx, idx2word, top_k=5)

        if not neighbors:
            print(f"  '{word}': NOT IN VOCABULARY")
            lines.append(f"'{word}': NOT IN VOCABULARY")
            continue

        print(f"\n  '{word}' -> top 5 neighbors:")
        lines.append(f"\n'{word}' -> top 5 neighbors:")

        for rank, (neighbor, score) in enumerate(neighbors, 1):
            line = f"    {rank}. {neighbor:<20} (cos={score:.4f})"
            print(line)
            lines.append(line)

    return lines


def print_analogies_table(model_name, embeddings, word2idx, idx2word):
    """Run and print all analogy experiments"""
    print(f"\n{'='*60}")
    print(f"  Analogies — {model_name}")
    print(f"{'='*60}")

    lines = [f"\nAnalogies — {model_name}", "="*60]

    for (a, b, c) in ANALOGIES:
        results = analogy_3cosadd(a, b, c, embeddings, word2idx, idx2word)

        print(f"\n  {a}:{b} :: {c}:?")
        lines.append(f"\n{a}:{b} :: {c}:?")

        if isinstance(results, str):
            print(f"    -> {results}")
            lines.append(f"  -> {results}")
        else:
            for i, (word, score) in enumerate(results[:5], 1):
                line = f"    {i}. {word:<20} (score={score:.4f})"
                print(line)
                lines.append(line)

    return lines


# ============================================================
# MAIN
# ============================================================

def main():
    # we'll analyze both the best CBOW and Skipgram models
    # using the 100_5_5 config as the "best" (can change based on sweep results)

    model_configs = [
        ("CBOW (dim=100, win=5, neg=5)",     os.path.join(RESULTS_DIR, "p1_cbow_100_5_5.pt")),
        ("Skipgram (dim=100, win=5, neg=5)", os.path.join(RESULTS_DIR, "p1_skipgram_100_5_5.pt")),
    ]

    all_output = []

    for model_name, model_path in model_configs:
        print(f"\nAnalyzing: {model_name}")

        embeddings, word2idx, idx2word = load_embeddings(model_path)
        if embeddings is None:
            print(f"  Skipping {model_name} - model not found")
            print(f"  (run word2vec_scratch.py first)")
            continue

        print(f"  Vocab size: {len(word2idx)}, Embed dim: {embeddings.shape[1]}")

        # neighbors
        neighbor_lines = print_neighbors_table(model_name, embeddings, word2idx, idx2word)
        all_output.extend(neighbor_lines)

        # analogies
        analogy_lines = print_analogies_table(model_name, embeddings, word2idx, idx2word)
        all_output.extend(analogy_lines)
        all_output.append("\n" + "="*60 + "\n")

    # save to results file
    neighbors_path = os.path.join(RESULTS_DIR, "p1_neighbors.txt")
    with open(neighbors_path, "w") as f:
        f.write("\n".join(all_output))
    print(f"\nResults saved to {neighbors_path}")


if __name__ == "__main__":
    main()
