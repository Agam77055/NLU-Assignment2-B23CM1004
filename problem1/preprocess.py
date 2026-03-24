"""
preprocess.py — Clean raw crawled pages and build the corpus

Reads text from data/raw/, applies preprocessing steps,
and writes cleaned corpus to data/corpus.txt.
Also computes dataset stats and generates a word cloud.

Usage:
    python preprocess.py

Author: Agam Harpreet Singh (B23CM1004)
"""

import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # must set backend before importing pyplot — learned this the hard way
import matplotlib.pyplot as plt
from wordcloud import WordCloud

SEED = 42

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "..", "plots")
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# STOPWORDS
# ============================================================

# common stopwords - grabbed these from NLTK source and hardcoded to avoid dependency
# not 100% sure this is complete but should cover the most frequent ones
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "not", "no", "nor", "so", "as", "if", "then", "than", "too", "very",
    "just", "about", "also", "am", "any", "all", "more", "one", "two",
    "there", "here", "when", "where", "what", "which", "who", "how",
    "up", "out", "into", "over", "under", "after", "before", "between",
    "each", "both", "few", "some", "such", "own", "same", "other",
    "www", "http", "https", "iitj", "ac", "php", "html", "com",
    "s", "t", "re", "ve", "ll", "d"
])


# ============================================================
# FUNCTIONS
# ============================================================

def load_raw_pages(raw_dir):
    """Load all .txt files from the raw directory"""
    pages = []
    if not os.path.exists(raw_dir):
        print(f"raw dir doesn't exist: {raw_dir}")
        return pages

    files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".txt")])
    for fname in files:
        fpath = os.path.join(raw_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # skip the SOURCE: line we added during crawling
        lines = content.split("\n")
        if lines and lines[0].startswith("SOURCE:"):
            lines = lines[2:]  # skip url + blank line
        pages.append("\n".join(lines))

    print(f"Loaded {len(pages)} raw pages")
    return pages


def clean_document(text):
    """Clean a single document - remove URLs, emails, non-ASCII stuff"""
    # URLs
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    # emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # phone-like patterns
    text = re.sub(r'\+?[\d\-\(\)\s]{8,}', ' ', text)
    # multiple spaces / newlines
    text = re.sub(r'\s+', ' ', text)
    # lowercase
    text = text.lower().strip()
    return text


def tokenize(text):
    # wait, need to handle the case where text is empty
    if not text:
        return []
    # just grab alphabetic words - no numbers, no punctuation
    return re.findall(r'[a-z]+', text)


def compute_stats(docs, all_tokens, vocab):
    """Compute and return basic corpus statistics"""
    stats = {
        "num_docs": len(docs),
        "num_tokens": len(all_tokens),
        "vocab_size": len(vocab),
        "avg_doc_len": len(all_tokens) // max(len(docs), 1),
    }
    return stats


def save_corpus(docs, path):
    """Write cleaned documents to corpus file, one paragraph per line"""
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            # split by double newlines to get paragraphs
            paragraphs = [p.strip() for p in doc.split("\n") if len(p.strip()) > 20]
            for para in paragraphs:
                f.write(para + "\n")
    print(f"Corpus saved to {path}")


def plot_wordcloud(tokens, path):
    """Make a word cloud from token list"""
    # filter stopwords for a nicer cloud
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    text_blob = " ".join(filtered)

    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        max_words=150,
        colormap="viridis",
        random_state=SEED
    ).generate(text_blob)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequent Words in IITJ Corpus", fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Word cloud saved to {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    raw_dir = os.path.join(DATA_DIR, "raw")
    corpus_path = os.path.join(DATA_DIR, "corpus.txt")
    wc_path = os.path.join(PLOTS_DIR, "p1_wordcloud.png")

    # step 1: load
    raw_pages = load_raw_pages(raw_dir)
    if not raw_pages:
        print("No raw pages found! Run scraper.py first.")
        return

    # step 2: clean
    print("Cleaning documents...")
    cleaned_docs = [clean_document(pg) for pg in raw_pages]

    # step 3: tokenize all
    all_tokens = []
    for doc in cleaned_docs:
        all_tokens.extend(tokenize(doc))

    vocab = set(all_tokens)

    # step 4: stats
    stats = compute_stats(cleaned_docs, all_tokens, vocab)
    print("\n=== Corpus Statistics ===")
    print(f"  Documents  : {stats['num_docs']}")
    print(f"  Tokens     : {stats['num_tokens']:,}")
    print(f"  Vocab size : {stats['vocab_size']:,}")
    print(f"  Avg length : {stats['avg_doc_len']:,} tokens/doc")

    # step 5: save corpus
    save_corpus(cleaned_docs, corpus_path)

    # step 6: word cloud
    plot_wordcloud(all_tokens, wc_path)

    # also save freq distribution (top 30 words) as a bar chart
    word_freq = Counter(t for t in all_tokens if t not in STOPWORDS and len(t) > 2)
    top30 = word_freq.most_common(30)

    words_top, counts_top = zip(*top30)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(words_top, counts_top, color="steelblue", edgecolor="white")
    ax.set_xlabel("Word")
    ax.set_ylabel("Frequency")
    ax.set_title("Top 30 Words in IITJ Corpus")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    freq_path = os.path.join(PLOTS_DIR, "p1_word_freq.png")
    plt.savefig(freq_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Freq chart saved to {freq_path}")


if __name__ == "__main__":
    main()
