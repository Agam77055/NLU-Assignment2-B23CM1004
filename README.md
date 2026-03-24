# NLU Assignment 2 — Word Embeddings & Character-Level Name Generation

**CSL7640: Natural Language Understanding | IIT Jodhpur**
**Agam Harpreet Singh | B23CM1004**

---

## Problem 1: Word Embeddings from IITJ Data

Train Word2Vec (CBOW and Skip-gram with Negative Sampling) from scratch on text crawled from the IIT Jodhpur website. Perform semantic analysis and visualization.

### Structure

```
problem1/
  scraper.py          — BFS web crawler for iitj.ac.in
  preprocess.py       — Corpus cleaning, tokenization, word cloud
  word2vec_scratch.py — CBOW and Skip-gram implemented in PyTorch (no gensim)
  word2vec_gensim.py  — Gensim baseline for comparison
  semantic_analysis.py— Cosine similarity neighbors + 3CosAdd analogy experiments
  visualize.py        — PCA and t-SNE projections
data/
  corpus.txt          — Cleaned corpus (0.16 MB, ~23k tokens)
  raw/                — 100 raw scraped pages
```

### Running

```bash
python problem1/scraper.py          # crawl IITJ pages -> data/raw/
python problem1/preprocess.py       # clean + stats + word cloud
python problem1/word2vec_scratch.py # train all configs (hyperparameter sweep)
python problem1/word2vec_gensim.py  # gensim baseline
python problem1/semantic_analysis.py# neighbors + analogies -> results/
python problem1/visualize.py        # PCA/t-SNE plots -> plots/
```

### Key Results

- Corpus: 100 documents, 23,798 tokens, vocabulary size ~5,600
- Hyperparameter sweep: embedding dim ∈ {50, 100, 200}, window ∈ {2, 5, 10}, negatives ∈ {5, 10, 15}
- Best config: dim=100, win=5, neg=5 (lowest loss, most stable)
- Interesting analogy: `department : engineering :: school : science` (Skip-gram, score=0.887)

---

## Problem 2: Character-Level Name Generation using RNN Variants

Implement and compare three sequence models for generating Indian names, all built from scratch (no `nn.RNN`, `nn.LSTM`, `nn.GRU`).

### Structure

```
problem2/
  dataset.py   — CharVocab + PyTorch Dataset wrapping TrainingNames.txt
  models.py    — VanillaRNN, BLSTMModel, AttentionRNNModel (all from scratch)
  train.py     — Training loop with early stopping + LR scheduling
  generate.py  — Autoregressive generation + novelty/diversity metrics
  analysis.py  — Qualitative analysis and failure mode identification
data/
  TrainingNames.txt — 1,602 Indian names (generated via LLM)
```

### Running

```bash
python problem2/train.py    # train all three models -> results/p2_*_best.pt
python problem2/generate.py # generate names + compute metrics -> results/
python problem2/analysis.py # qualitative analysis
```

### Model Comparison

| Model      | Parameters | Novelty Rate | Diversity | Avg Realism |
|------------|-----------|-------------|-----------|-------------|
| VanillaRNN | 222,492   | 82.4%       | 96.2%     | 0.71        |
| BLSTM      | 1,731,384 | 100.0%      | 99.8%     | 0.48        |
| AttnRNN    | ~300k     | 99.6%       | 97.6%     | 0.61        |

**Best for generation quality: VanillaRNN.** The BLSTM achieves near-perfect novelty/diversity but generates mostly nonsense — it trains bidirectionally (seeing future tokens) but must generate unidirectionally, causing a train/inference distribution mismatch.

---

## Deliverables

| File | Description |
|------|-------------|
| `report/report.pdf` | Full report covering both problems |
| `data/corpus.txt` | Cleaned IITJ corpus |
| `data/TrainingNames.txt` | Indian names training set |
| `plots/` | Word cloud, PCA, t-SNE, training curves, metrics |
| `results/p1_neighbors.txt` | Semantic similarity results |
| `results/p1_hyperparameter_sweep.csv` | Full sweep results |
| `results/p2_metrics.csv` | Novelty and diversity per model |
| `results/p2_generated_*.txt` | Generated name samples |

> Model checkpoints (`*.pt`) are excluded from the repo due to size. Re-run the training scripts to reproduce them.

## Requirements

```bash
pip install -r requirements.txt
```
