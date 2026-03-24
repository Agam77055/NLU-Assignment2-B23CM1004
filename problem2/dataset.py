"""dataset.py — Character-level vocabulary and dataset for name generation

Builds a character vocabulary with special tokens and wraps
the TrainingNames.txt file as a PyTorch Dataset.

Usage:
    Called by train.py and generate.py
"""

# Agam Harpreet Singh | B23CM1004 | IIT Jodhpur
# NLU Assignment 2 — Problem 2

import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SEED = 42
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# ============================================================
# CHARACTER VOCABULARY
# ============================================================

class CharVocab:
    """
    Character-level vocabulary for name generation.

    Special tokens:
        PAD (0): padding for batch alignment
        SOS (1): start-of-sequence marker
        EOS (2): end-of-sequence marker
    """

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self):
        # start with just the three special tokens, real chars added in build()
        self.char2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.idx2char = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.vocab_size = 3  # grows once we call build() on actual data

    def build(self, names):
        """Build vocab from a list of name strings.

        Goes through every character in every name, collects unique chars,
        then assigns indices starting from 3 (after the special tokens).
        """
        # collect all unique characters across all names
        all_chars = set()
        for name in names:
            for ch in name.strip().lower():
                all_chars.add(ch)

        # sort to keep it deterministic - otherwise vocab order changes between runs
        # which would break any saved checkpoints
        sorted_chars = sorted(all_chars)

        # assign indices starting from 3 since 0,1,2 are already taken
        for i, ch in enumerate(sorted_chars, start=3):
            self.char2idx[ch] = i
            self.idx2char[i] = ch

        self.vocab_size = len(self.char2idx)
        print(f"Vocab built: {self.vocab_size} tokens ({len(sorted_chars)} chars + 3 special)")
        return self  # return self so we can chain: CharVocab().build(names)

    def encode(self, name):
        """Convert a name string to a list of token indices.

        Format: SOS + [char indices] + EOS
        This is the standard format for seq2seq / language model training.
        """
        indices = [self.SOS_IDX]
        for ch in name.strip().lower():
            if ch in self.char2idx:
                indices.append(self.char2idx[ch])
            # if char not in vocab just skip - shouldn't happen if build() was
            # called on the same data, but good to handle gracefully
        indices.append(self.EOS_IDX)
        return indices

    def decode(self, indices):
        """Convert a list of indices back to a name string.

        Skips PAD and SOS, stops at EOS.
        """
        chars = []
        for idx in indices:
            if idx in (self.PAD_IDX, self.SOS_IDX):
                continue
            if idx == self.EOS_IDX:
                break  # stop as soon as we hit EOS, don't include it
            if idx in self.idx2char:
                chars.append(self.idx2char[idx])
        return "".join(chars)


# ============================================================
# NAME DATASET
# ============================================================

class NameDataset(Dataset):
    """
    PyTorch Dataset wrapping TrainingNames.txt.

    Returns (input_seq, target_seq) pairs formatted for teacher forcing:
        input:  SOS + name chars          (i.e., full sequence except last char)
        target: name chars + EOS          (i.e., full sequence except first char)

    So for the name "alice":
        full encoded = [SOS, a, l, i, c, e, EOS]
        input  = [SOS, a, l, i, c, e]
        target = [a, l, i, c, e, EOS]
    The model predicts target[t] given input[0..t].
    """

    def __init__(self, filepath, vocab=None):
        self.filepath = filepath
        self.names = self._load_names(filepath)

        # allow passing in a pre-built vocab - useful when train/val share the same vocab
        if vocab is None:
            self.vocab = CharVocab().build(self.names)
        else:
            self.vocab = vocab

        # pre-encode everything upfront rather than encoding on-the-fly in __getitem__
        # makes training faster since encoding is done once
        self.encoded = [self.vocab.encode(name) for name in self.names]

        print(f"Dataset: {len(self.names)} names loaded, vocab size = {self.vocab.vocab_size}")

    def _load_names(self, path):
        """Read names line by line, skip blanks and single-char lines."""
        names = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                # skip empty lines and single chars - not useful training examples
                if name and len(name) >= 2:
                    names.append(name)
        return names

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        seq = self.encoded[idx]

        # Note to self: remember this is teacher forcing
        # input  = seq[:-1]  (all tokens except the last one)
        # target = seq[1:]   (all tokens except the first one)
        # so at each position t, input[t] is the current char, target[t] is what we want to predict next
        input_seq  = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:],  dtype=torch.long)
        return input_seq, target_seq


# ============================================================
# COLLATE FUNCTION (for DataLoader batching)
# ============================================================

def collate_fn(batch):
    """Pads a batch of variable-length sequences to the same length.

    DataLoader needs all samples in a batch to be the same size, so we
    pad shorter sequences with PAD_IDX (0) up to the longest one.

    Returns:
        inputs_padded  [B, T_max]
        targets_padded [B, T_max]
        lengths        [B] — original (unpadded) lengths, useful for pack_padded_sequence
                             though we don't use it with our custom cells
    """
    inputs, targets = zip(*batch)

    # record original lengths before padding - might be useful for masking the loss later
    lengths = torch.tensor([len(x) for x in inputs], dtype=torch.long)

    # pad_sequence handles the variable lengths for us
    # batch_first=True -> output is [B, T] not [T, B]
    inputs_padded  = pad_sequence(inputs,  batch_first=True, padding_value=CharVocab.PAD_IDX)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=CharVocab.PAD_IDX)

    return inputs_padded, targets_padded, lengths
