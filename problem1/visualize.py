"""visualize.py — PCA and t-SNE visualization of word embeddings

Projects high-dimensional word embeddings to 2D using PCA and t-SNE.
Colors word clusters by semantic category.

Usage:
    python visualize.py
"""

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
# WORD CLUSTERS FOR VISUALIZATION
# ============================================================

# these clusters are for semantic coloring in the plots
# I picked words that I expect to cluster together in an IITJ corpus
SEMANTIC_CLUSTERS = {
    "academic":  ["lecture", "course", "syllabus", "curriculum", "semester", "exam",
                  "grade", "class", "subject", "credit", "marks"],
    "student":   ["student", "undergraduate", "graduate", "phd", "btech", "mtech",
                  "hostel", "campus", "ug", "pg", "scholarship"],
    "research":  ["research", "thesis", "publication", "paper", "laboratory", "lab",
                  "journal", "conference", "project", "dissertation"],
    "admin":     ["department", "institute", "faculty", "professor", "director",
                  "dean", "committee", "board", "council", "office"],
    "misc":      []  # will be filled with high-freq words not in other clusters
}


# ============================================================
# LOADING
# ============================================================

def load_embeddings(path):
    """Load embeddings from .pt checkpoint"""
    if not os.path.exists(path):
        return None, None, None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    emb = ckpt["embeddings"]   # numpy [V, D]
    w2i = ckpt["vocab"]
    i2w = {v: k for k, v in w2i.items()}
    return emb, w2i, i2w


# ============================================================
# WORD SELECTION
# ============================================================

def select_words_for_viz(embeddings, word2idx, idx2word, n_top=100):
    """
    Pick which words to visualize.
    Take top-n by frequency (idx order since vocab is sorted by freq)
    plus all the cluster words we want to highlight.
    """
    cluster_words = set()
    for words in SEMANTIC_CLUSTERS.values():
        cluster_words.update(words)

    # top words by index (vocab is sorted by freq in our implementation)
    top_words = [idx2word[i] for i in range(1, min(n_top + 1, len(idx2word)))]  # skip idx 0 (<UNK>)

    # merge
    all_words = list(dict.fromkeys(top_words + list(cluster_words)))  # deduplicate but keep order

    # filter to only words in vocab
    all_words = [w for w in all_words if w in word2idx]

    return all_words


def get_word_color(word):
    """Return a cluster label for coloring"""
    for cluster_name, cluster_words in SEMANTIC_CLUSTERS.items():
        if cluster_name == "misc":
            continue
        if word in cluster_words:
            return cluster_name
    return "misc"


# ============================================================
# PLOTTING
# ============================================================

def plot_embedding_2d(coords, words, title, save_path, figsize=(14, 10)):
    """
    Scatter plot of 2D embeddings with word labels.
    Colors by semantic cluster.
    """
    cluster_colors = {
        "academic": "#e74c3c",   # red
        "student":  "#3498db",   # blue
        "research": "#2ecc71",   # green
        "admin":    "#9b59b6",   # purple
        "misc":     "#95a5a6",   # grey
    }

    fig, ax = plt.subplots(figsize=figsize)

    # group by cluster for legend
    cluster_data = {c: ([], []) for c in cluster_colors}
    word_labels  = []

    for i, word in enumerate(words):
        cluster = get_word_color(word)
        x, y    = coords[i]
        cluster_data[cluster][0].append(x)
        cluster_data[cluster][1].append(y)
        word_labels.append((x, y, word, cluster))

    # plot each cluster
    for cluster, (xs, ys) in cluster_data.items():
        if xs:
            ax.scatter(xs, ys, c=cluster_colors[cluster], label=cluster,
                       s=30, alpha=0.7, edgecolors="white", linewidths=0.5)

    # add word labels - only for cluster words and a few top words to avoid clutter
    shown = 0
    for x, y, word, cluster in word_labels:
        if cluster != "misc" or shown < 30:
            ax.annotate(word, (x, y), fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
            if cluster == "misc":
                shown += 1

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================

def run_pca(embeddings):
    """Reduce to 2D using PCA"""
    pca = PCA(n_components=2, random_state=SEED)
    return pca.fit_transform(embeddings)


def run_tsne(embeddings):
    """Reduce to 2D using t-SNE"""
    # not sure about the best perplexity value but 30 seems okay
    # also capping it so it doesn't blow up when there are fewer words
    perp = min(30, max(5, len(embeddings) // 10))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED,
                max_iter=1000, learning_rate="auto", init="pca")
    return tsne.fit_transform(embeddings)


# ============================================================
# PER-MODEL VISUALIZATION
# ============================================================

def visualize_model(model_name, model_path, method="both"):
    """Load a model and generate all visualizations for it"""

    emb, w2i, i2w = load_embeddings(model_path)
    if emb is None:
        print(f"  {model_name}: not found, skipping")
        return None, None, None

    words = select_words_for_viz(emb, w2i, i2w, n_top=100)
    print(f"  {model_name}: {len(words)} words selected for visualization")

    # grab embedding vectors for selected words
    word_embs = np.array([emb[w2i[w]] for w in words])

    pca_coords  = None
    tsne_coords = None

    if method in ("pca", "both"):
        print(f"  Running PCA...")
        pca_coords = run_pca(word_embs)
        safe_name  = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        plot_embedding_2d(
            pca_coords, words,
            title=f"PCA — {model_name}",
            save_path=os.path.join(PLOTS_DIR, f"p1_pca_{safe_name[:10]}.png")
        )

    if method in ("tsne", "both"):
        print(f"  Running t-SNE (this takes a bit)...")
        tsne_coords = run_tsne(word_embs)
        safe_name   = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        plot_embedding_2d(
            tsne_coords, words,
            title=f"t-SNE — {model_name}",
            save_path=os.path.join(PLOTS_DIR, f"p1_tsne_{safe_name[:10]}.png")
        )

    return words, pca_coords, tsne_coords


# ============================================================
# COMPARISON PLOT (CBOW vs SKIPGRAM side by side)
# ============================================================

def plot_comparison(words_cbow, coords_cbow, words_sg, coords_sg, method, title_prefix):
    """Side-by-side comparison of CBOW vs Skipgram"""

    # need common words for a fair comparison
    common = list(set(words_cbow) & set(words_sg))
    if len(common) < 10:
        print(f"  Not enough common words for comparison plot")
        return

    # get coords for common words only
    cbow_idx = [words_cbow.index(w) for w in common]
    sg_idx   = [words_sg.index(w)   for w in common]

    cbow_sub = coords_cbow[cbow_idx]
    sg_sub   = coords_sg[sg_idx]

    # re-run dim reduction on just the common set for consistency
    if method == "pca":
        reducer  = PCA(n_components=2, random_state=SEED)
        cbow_sub = reducer.fit_transform(cbow_sub)
        reducer2 = PCA(n_components=2, random_state=SEED)
        sg_sub   = reducer2.fit_transform(sg_sub)

    cluster_colors = {
        "academic": "#e74c3c", "student": "#3498db",
        "research": "#2ecc71", "admin":   "#9b59b6", "misc": "#95a5a6"
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, coords, label in [(ax1, cbow_sub, "CBOW"), (ax2, sg_sub, "Skipgram")]:
        shown = 0
        for i, word in enumerate(common):
            cluster = get_word_color(word)
            color   = cluster_colors[cluster]
            ax.scatter(coords[i, 0], coords[i, 1], c=color, s=25, alpha=0.7)
            if cluster != "misc" or shown < 20:
                ax.annotate(word, (coords[i, 0], coords[i, 1]),
                            fontsize=7, alpha=0.8, xytext=(2, 2), textcoords="offset points")
                if cluster == "misc":
                    shown += 1
        ax.set_title(f"{title_prefix} — {label}", fontsize=12)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    # add shared legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=name) for name, c in cluster_colors.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"p1_{method}_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Generating embedding visualizations...")

    cbow_path = os.path.join(RESULTS_DIR, "p1_cbow_100_5_5.pt")
    sg_path   = os.path.join(RESULTS_DIR, "p1_skipgram_100_5_5.pt")

    words_cbow, pca_cbow, tsne_cbow = visualize_model("CBOW",     cbow_path, method="both")
    words_sg,   pca_sg,   tsne_sg   = visualize_model("Skipgram", sg_path,   method="both")

    # comparison plots
    if pca_cbow is not None and pca_sg is not None:
        print("\nGenerating comparison plots...")
        plot_comparison(words_cbow, pca_cbow, words_sg, pca_sg, "pca", "PCA")

    if tsne_cbow is not None and tsne_sg is not None:
        plot_comparison(words_cbow, tsne_cbow, words_sg, tsne_sg, "tsne", "t-SNE")

    print("\nDone!")


if __name__ == "__main__":
    main()
