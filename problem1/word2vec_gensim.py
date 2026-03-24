"""word2vec_gensim.py — Train Word2Vec using gensim as a comparison baseline

Trains CBOW and Skip-gram using gensim's optimized implementation
for comparison with our scratch implementation.

Usage:
    python word2vec_gensim.py
"""

import os
import re
import numpy as np
import torch
from gensim.models import Word2Vec

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# PATHS AND SEED
# ============================================================

SEED = 42
BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR   = os.path.join(BASE_DIR, "..", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

# using the same best config as the scratch model for a fair comparison
BEST_DIM    = 100
BEST_WINDOW = 5
BEST_NEG    = 5


# ============================================================
# DATA LOADING
# ============================================================

def load_corpus_sentences(corpus_path):
    """Load corpus and split into sentences for gensim"""
    sentences = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # tokenize - same as in preprocess
                words = re.findall(r'[a-z]+', line.lower())
                if len(words) >= 3:
                    sentences.append(words)
    return sentences


# ============================================================
# MODEL TRAINING
# ============================================================

def train_gensim_model(sentences, sg=0, embed_dim=100, window=5, negative=5, model_name="cbow"):
    """
    Train a gensim Word2Vec model.
    sg=0 for CBOW, sg=1 for Skip-gram
    """
    print(f"\nTraining gensim {model_name} (dim={embed_dim}, win={window}, neg={negative})...")

    model = Word2Vec(
        sentences=sentences,
        vector_size=embed_dim,
        window=window,
        min_count=2,
        negative=negative,
        sg=sg,
        seed=SEED,
        workers=4,
        epochs=10,  # gensim is fast so we can do more epochs
    )

    return model


# ============================================================
# NEAREST NEIGHBOR HELPERS
# ============================================================

def get_neighbors_gensim(model, word, top_k=5):
    """Get top-k most similar words from a gensim model"""
    if word not in model.wv:
        return []
    return model.wv.most_similar(word, topn=top_k)


def get_neighbors_scratch(word, embeddings, word2idx, idx2word, top_k=5):
    """Get neighbors from our scratch embeddings using cosine similarity"""
    if word not in word2idx:
        return []

    idx = word2idx[word]
    query_vec = embeddings[idx]

    # compute cosine similarity with all words
    # normalize first
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms
    query_normed = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    sims = normed @ query_normed
    sims[idx] = -1  # exclude the word itself

    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(idx2word[i], float(sims[i])) for i in top_indices if i in idx2word]


# ============================================================
# COMPARISON
# ============================================================

def compare_models(query_words, gensim_cbow, gensim_sg, scratch_emb, scratch_word2idx, scratch_idx2word):
    """Print a side-by-side comparison table of nearest neighbors"""

    print("\n" + "="*80)
    print("NEAREST NEIGHBORS COMPARISON: Gensim CBOW vs Gensim Skipgram vs Scratch")
    print("="*80)

    comparison_lines = []

    for word in query_words:
        print(f"\n  Query: '{word}'")

        g_cbow_neighbors = get_neighbors_gensim(gensim_cbow, word)
        g_sg_neighbors   = get_neighbors_gensim(gensim_sg,   word)
        s_neighbors      = get_neighbors_scratch(word, scratch_emb, scratch_word2idx, scratch_idx2word)

        # format nicely
        g_cbow_str = ", ".join([f"{w}({s:.3f})" for w, s in g_cbow_neighbors[:3]])
        g_sg_str   = ", ".join([f"{w}({s:.3f})" for w, s in g_sg_neighbors[:3]])
        s_str      = ", ".join([f"{w}({s:.3f})" for w, s in s_neighbors[:3]])

        print(f"    Gensim CBOW   : {g_cbow_str}")
        print(f"    Gensim SG     : {g_sg_str}")
        print(f"    Scratch       : {s_str}")

        comparison_lines.append(f"Query: {word}")
        comparison_lines.append(f"  Gensim CBOW: {g_cbow_str}")
        comparison_lines.append(f"  Gensim SG  : {g_sg_str}")
        comparison_lines.append(f"  Scratch    : {s_str}")
        comparison_lines.append("")

    return comparison_lines


# ============================================================
# MAIN
# ============================================================

def main():
    corpus_path = os.path.join(DATA_DIR, "corpus.txt")
    if not os.path.exists(corpus_path):
        print("Need corpus.txt - run preprocess.py first")
        return

    print("Loading corpus sentences...")
    sentences = load_corpus_sentences(corpus_path)
    print(f"  Got {len(sentences)} sentences")

    # train both gensim models
    gensim_cbow = train_gensim_model(sentences, sg=0, embed_dim=BEST_DIM, window=BEST_WINDOW,
                                     negative=BEST_NEG, model_name="CBOW")
    gensim_sg   = train_gensim_model(sentences, sg=1, embed_dim=BEST_DIM, window=BEST_WINDOW,
                                     negative=BEST_NEG, model_name="Skipgram")

    # save gensim models
    gensim_cbow.save(os.path.join(RESULTS_DIR, "gensim_cbow.model"))
    gensim_sg.save(os.path.join(RESULTS_DIR,   "gensim_sg.model"))

    # load best scratch model for comparison (use the 100_5_5 config)
    scratch_path = os.path.join(RESULTS_DIR, "p1_cbow_100_5_5.pt")
    scratch_emb      = None
    scratch_word2idx = {}
    scratch_idx2word = {}

    if os.path.exists(scratch_path):
        ckpt = torch.load(scratch_path, map_location="cpu", weights_only=False)
        scratch_emb      = ckpt["embeddings"]  # numpy array
        scratch_word2idx = ckpt["vocab"]
        scratch_idx2word = {v: k for k, v in scratch_word2idx.items()}
        print(f"\nLoaded scratch model: {scratch_emb.shape}")
    else:
        print("Scratch model not found - run word2vec_scratch.py first for full comparison")

    # compare on these words from the assignment
    query_words = ["research", "student", "phd", "exam"]

    if scratch_emb is not None:
        lines    = compare_models(query_words, gensim_cbow, gensim_sg,
                                  scratch_emb, scratch_word2idx, scratch_idx2word)
        out_path = os.path.join(RESULTS_DIR, "p1_gensim_comparison.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\nComparison saved to {out_path}")
    else:
        # just show gensim results
        print("\n=== Gensim Results Only ===")
        for word in query_words:
            print(f"\n'{word}':")
            n = get_neighbors_gensim(gensim_cbow, word)
            print(f"  CBOW: {n[:5]}")
            n = get_neighbors_gensim(gensim_sg, word)
            print(f"  SG  : {n[:5]}")


if __name__ == "__main__":
    main()
