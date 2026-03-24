"""analysis.py — Qualitative analysis of generated names

Loads generated name files and performs quality analysis:
- Realism scoring based on heuristics
- Failure mode identification
- Per-model representative samples

Usage:
    python analysis.py
"""

import os
import sys
import re

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

# ============================================================
# HEURISTICS FOR NAME QUALITY
# ============================================================

# typical Indian name ending patterns - I collected these by looking at the training set
INDIAN_NAME_ENDINGS = ["a", "i", "an", "ar", "av", "ay", "al", "am", "as",
                        "ra", "na", "ka", "ya", "va", "ma", "ha", "la",
                        "esh", "ish", "dev", "raj", "kumar", "priya", "deep"]

# consonant clusters that don't really appear in Indian names
IMPLAUSIBLE_CLUSTERS = ["zzz", "xxx", "bbb", "ccc", "xk", "xz", "zx", "qj"]


def realism_score(name):
    """
    Compute a rough 'realism' score for a single name (0 to 1).

    Checks:
    - Length between 3 and 14 characters
    - Has at least some vowels interspersed
    - Ends with a common Indian name ending
    - No implausible character clusters

    This is obviously a heuristic - no perfect way to judge realism without
    a human evaluator.
    """
    if not name or len(name) < 2:
        return 0.0

    score = 0.0

    # length check - most Indian names are 4-12 chars
    name_len = len(name)
    if 4 <= name_len <= 12:
        score += 0.3
    elif 3 <= name_len <= 14:
        score += 0.15

    # vowel check - should have some vowels spread through the name
    vowels = set("aeiou")
    vowel_count = sum(1 for c in name if c in vowels)
    vowel_ratio = vowel_count / name_len
    if 0.2 <= vowel_ratio <= 0.7:
        score += 0.3
    elif vowel_ratio > 0:
        score += 0.1

    # check for Indian-pattern ending
    has_good_ending = any(name.endswith(ending) for ending in INDIAN_NAME_ENDINGS)
    if has_good_ending:
        score += 0.2

    # penalize implausible clusters
    has_bad_cluster = any(cluster in name for cluster in IMPLAUSIBLE_CLUSTERS)
    if not has_bad_cluster:
        score += 0.2

    return min(score, 1.0)


def identify_failure_modes(names):
    """
    Categorize common generation failure modes.
    Returns a dict with counts.
    """
    failures = {
        "too_short": 0,          # length < 3
        "too_long": 0,           # length > 14
        "repetitive_chars": 0,   # same char 3+ times in a row like "aaa"
        "no_vowels": 0,          # no vowel at all
        "implausible_cluster": 0, # weird char combos
        "empty": 0,
    }

    for name in names:
        if not name:
            failures["empty"] += 1
            continue

        if len(name) < 3:
            failures["too_short"] += 1
        if len(name) > 14:
            failures["too_long"] += 1

        # check for repeating chars: 3 or more of same in a row
        if re.search(r'(.)\1{2,}', name):
            failures["repetitive_chars"] += 1

        # no vowels at all
        if not any(c in "aeiou" for c in name):
            failures["no_vowels"] += 1

        if any(cluster in name for cluster in IMPLAUSIBLE_CLUSTERS):
            failures["implausible_cluster"] += 1

    return failures


def load_generated_names(path):
    """Load names from a generated names file"""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def analyze_model(model_name, names_path, n_samples=20):
    """Full qualitative analysis for one model"""

    print(f"\n{'='*60}")
    print(f"  Analysis: {model_name}")
    print(f"{'='*60}")

    names = load_generated_names(names_path)
    if not names:
        print(f"  No names found at {names_path}")
        print(f"  (run generate.py first)")
        return {}

    print(f"  Total generated: {len(names)}")

    # compute realism scores for all names
    scores = [realism_score(n) for n in names]
    avg_score = sum(scores) / len(scores)

    print(f"  Avg realism score: {avg_score:.3f}")

    # failure modes
    failures = identify_failure_modes(names)
    total = len(names)
    print(f"\n  Failure Mode Analysis:")
    for mode, count in failures.items():
        pct = 100 * count / max(total, 1)
        print(f"    {mode:<25}: {count:4d} ({pct:.1f}%)")

    # sample some good and bad names
    sorted_by_score = sorted(zip(names, scores), key=lambda x: -x[1])

    good_samples = [n for n, s in sorted_by_score[:10] if s > 0.5]
    bad_samples  = [n for n, s in sorted_by_score[-10:] if s < 0.3]

    print(f"\n  High-quality samples (score > 0.5):")
    for name in good_samples[:n_samples//2]:
        print(f"    {name}")

    print(f"\n  Low-quality / failure samples:")
    for name in bad_samples[:n_samples//4]:
        print(f"    {name}")

    # random sample - more representative
    random_sample = names[:n_samples]
    print(f"\n  First {n_samples} generated names:")
    for i in range(0, len(random_sample), 5):
        row = random_sample[i:i+5]
        print("    " + ", ".join(row))

    return {
        "model": model_name,
        "n_names": len(names),
        "avg_realism": round(avg_score, 3),
        "failures": failures,
        "good_samples": good_samples[:10],
        "bad_samples": bad_samples[:5],
    }


def main():
    print("="*60)
    print("QUALITATIVE ANALYSIS OF GENERATED NAMES")
    print("="*60)

    model_files = [
        ("VanillaRNN",  os.path.join(RESULTS_DIR, "p2_generated_rnn.txt")),
        ("BLSTM",       os.path.join(RESULTS_DIR, "p2_generated_blstm.txt")),
        ("AttnRNN",     os.path.join(RESULTS_DIR, "p2_generated_attn.txt")),
    ]

    all_results = []

    for model_name, names_path in model_files:
        result = analyze_model(model_name, names_path, n_samples=20)
        if result:
            all_results.append(result)

    if not all_results:
        print("\nNo results to compare - make sure you've run generate.py first")
        return

    # cross-model comparison summary
    print(f"\n{'='*60}")
    print("CROSS-MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'N Names':<12} {'Avg Realism':<15}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['model']:<15} {r['n_names']:<12} {r['avg_realism']:<15}")

    print("\nNote: Realism scores are heuristic-based approximations.")
    print("See the report for detailed qualitative discussion.")

if __name__ == "__main__":
    main()
