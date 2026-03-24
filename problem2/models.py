"""models.py — RNN model implementations from scratch for name generation

Implements three architectures:
  1. VanillaRNN     - basic tanh RNN cell, manually written
  2. BLSTMModel     - bidirectional LSTM with custom LSTM cells
  3. AttentionRNN   - RNN + Bahdanau-style additive attention

All cells are implemented from scratch - no nn.RNN/nn.LSTM/nn.GRU.
"""

# Agam Harpreet Singh | B23CM1004 | IIT Jodhpur
# NLU Assignment 2 — Problem 2

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42

# ============================================================
# VANILLA RNN (from scratch)
# ============================================================

class VanillaRNNCell(nn.Module):
    """
    A single vanilla RNN cell.
    h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)

    I'm implementing this from scratch (not using nn.RNN) as required.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # separate linear layers for input->hidden and hidden->hidden
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        # no bias on W_hh since W_ih already contributes a bias term - no point doubling up
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        # I went with tanh activation like in the original RNN paper
        # xavier init makes sense for the input weights since tanh is roughly linear near 0
        nn.init.xavier_uniform_(self.W_ih.weight)
        # orthogonal init for recurrent weights - standard advice to slow down vanishing/exploding
        nn.init.orthogonal_(self.W_hh.weight)
        nn.init.zeros_(self.W_ih.bias)

    def forward(self, x, h_prev):
        """
        x:      [B, input_size]
        h_prev: [B, hidden_size]
        returns h_t: [B, hidden_size]
        """
        # simple additive combination then nonlinearity - that's the whole RNN update
        return torch.tanh(self.W_ih(x) + self.W_hh(h_prev))


class VanillaRNN(nn.Module):
    """
    Multi-layer Vanilla RNN for character-level generation.
    Uses VanillaRNNCell internally, stacks multiple layers.
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # stack layers - first one takes embed_dim, subsequent ones take hidden_size
        # hmm, not sure if xavier is better here but it's a common choice for the cell weights
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_sz = embed_dim if i == 0 else hidden_size
            self.cells.append(VanillaRNNCell(in_sz, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)  # project hidden state to vocab distribution

    def forward(self, x, hidden=None):
        """
        x:      [B, T] — input token indices
        hidden: list of [B, H] tensors, one per layer (or None to start fresh)
        returns: logits [B, T, vocab_size], updated hidden states list
        """
        B, T = x.shape  # B = batch size, T = sequence length

        embeds = self.embed(x)  # [B, T, embed_dim]

        # zero init if no hidden state passed in (e.g., start of generation)
        if hidden is None:
            hidden = self.init_hidden(B, x.device)

        h = list(hidden)  # make a mutable copy so we can update layer by layer
        outputs = []

        for t in range(T):
            inp = embeds[:, t, :]  # current timestep embedding: [B, embed_dim]

            # feed through each stacked layer
            for layer_idx, cell in enumerate(self.cells):
                inp = cell(inp, h[layer_idx])
                h[layer_idx] = inp
                # apply dropout between layers but not on the last one
                if layer_idx < self.num_layers - 1:
                    inp = self.dropout(inp)

            outputs.append(h[-1])  # collect top layer's hidden state at each step

        # stack time steps back into a tensor
        out    = torch.stack(outputs, dim=1)   # [B, T, H]
        out    = self.dropout(out)
        logits = self.fc(out)                  # [B, T, vocab_size]

        return logits, h

    def init_hidden(self, batch_size, device):
        """Initialize all layer hidden states to zeros."""
        return [torch.zeros(batch_size, self.hidden_size, device=device)
                for _ in range(self.num_layers)]

    def count_parameters(self):
        # quick helper to print model size - useful for the report
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# LSTM CELL (from scratch)
# ============================================================

class LSTMCell(nn.Module):
    """
    Custom LSTM cell implementing all four gates manually.

    Gates: i (input), f (forget), g (cell candidate), o (output)
    c_t = f * c_{t-1} + i * g
    h_t = o * tanh(c_t)

    One trick: initialize forget gate bias to 1.0 - this helps the model
    remember longer sequences at the start of training. Read about it in
    Jozefowicz et al. 2015, seems to work well in practice.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # compute all 4 gates in one matrix multiply for efficiency
        # output has 4*H features: [i | f | g | o]
        self.W_ih = nn.Linear(input_size,   4 * hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size,  4 * hidden_size, bias=False)

        # TODO: look into whether layer norm here would help with training stability

        # initialize weights
        for name, param in self.W_ih.named_parameters():
            if 'weight' in name:
                # xavier for input weights - seems reasonable
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # set forget gate bias to 1.0 — standard LSTM trick
                # forget gate is second gate, so indices [H : 2H] in the bias vector
                param.data[hidden_size : 2 * hidden_size].fill_(1.0)

        # orthogonal init for recurrent weights, same reason as vanilla RNN
        nn.init.orthogonal_(self.W_hh.weight)

    def forward(self, x, h_prev, c_prev):
        """
        x:      [B, input_size]
        h_prev: [B, H]
        c_prev: [B, H]
        returns: (h_t, c_t) both [B, H]
        """
        # compute all gates at once then slice
        gates = self.W_ih(x) + self.W_hh(h_prev)  # [B, 4H]

        H = self.hidden_size
        # slice into individual gate activations
        i_gate = torch.sigmoid(gates[:, :H])          # input gate
        f_gate = torch.sigmoid(gates[:, H  : 2*H])    # forget gate
        g_gate = torch.tanh   (gates[:, 2*H : 3*H])   # cell candidate
        o_gate = torch.sigmoid(gates[:, 3*H :])        # output gate

        # cell update and hidden state
        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * torch.tanh(c_t)

        return h_t, c_t


# ============================================================
# BIDIRECTIONAL LSTM
# ============================================================

class BLSTMModel(nn.Module):
    """
    Bidirectional LSTM model built on top of custom LSTMCell.

    During training: runs both forward and backward LSTM passes,
    concatenates the two hidden state sequences for prediction.

    During generation: has to use only the forward direction since
    we don't have future tokens yet. This is a known limitation of
    BLSTM for autoregressive generation - I mention this in the report.
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # forward direction LSTM stack
        self.fwd_cells = nn.ModuleList()
        for i in range(num_layers):
            in_sz = embed_dim if i == 0 else hidden_size
            self.fwd_cells.append(LSTMCell(in_sz, hidden_size))

        # backward direction LSTM stack — separate parameters
        self.bwd_cells = nn.ModuleList()
        for i in range(num_layers):
            in_sz = embed_dim if i == 0 else hidden_size
            self.bwd_cells.append(LSTMCell(in_sz, hidden_size))

        # output projection: concat fwd+bwd so input is 2*H
        self.fc = nn.Linear(2 * hidden_size, vocab_size)

        # separate smaller projection for forward-only generation pass
        self.fc_gen = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Full bidirectional forward pass — used during training.
        x: [B, T]
        hidden: ignored here, we always reinit (training doesn't need continuity between batches)
        """
        B, T = x.shape
        emb = self.embed(x)  # [B, T, E]

        # init all LSTM states to zero for both directions
        fwd_h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        fwd_c = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        bwd_h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        bwd_c = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        fwd_outputs = []
        bwd_outputs = []

        # --- forward pass: left to right ---
        for t in range(T):
            inp = emb[:, t, :]
            for layer_idx, cell in enumerate(self.fwd_cells):
                fwd_h[layer_idx], fwd_c[layer_idx] = cell(inp, fwd_h[layer_idx], fwd_c[layer_idx])
                inp = fwd_h[layer_idx]
                if layer_idx < self.num_layers - 1:
                    inp = self.dropout(inp)
            fwd_outputs.append(fwd_h[-1])

        # --- backward pass: right to left ---
        for t in range(T - 1, -1, -1):
            inp = emb[:, t, :]
            for layer_idx, cell in enumerate(self.bwd_cells):
                bwd_h[layer_idx], bwd_c[layer_idx] = cell(inp, bwd_h[layer_idx], bwd_c[layer_idx])
                inp = bwd_h[layer_idx]
                if layer_idx < self.num_layers - 1:
                    inp = self.dropout(inp)
            # insert at front so bwd_outputs[0] corresponds to t=0 (maintains time order)
            bwd_outputs.insert(0, bwd_h[-1])

        # stack and concat along hidden dim
        fwd_tensor = torch.stack(fwd_outputs, dim=1)   # [B, T, H]
        bwd_tensor = torch.stack(bwd_outputs, dim=1)   # [B, T, H]
        combined   = torch.cat([fwd_tensor, bwd_tensor], dim=2)  # [B, T, 2H]

        combined = self.dropout(combined)
        logits   = self.fc(combined)  # [B, T, vocab_size]

        # return fwd states as the "hidden" - bwd states aren't useful for generation anyway
        return logits, (fwd_h, fwd_c)

    def forward_only(self, x, hidden=None):
        """
        Forward-direction-only pass for autoregressive generation.
        We can't run the backward pass when generating token by token
        since we don't have future tokens yet.
        """
        B, T = x.shape
        emb = self.embed(x)

        if hidden is None:
            fwd_h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            fwd_c = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            fwd_h, fwd_c = hidden

        outputs = []
        for t in range(T):
            inp = emb[:, t, :]
            for layer_idx, cell in enumerate(self.fwd_cells):
                fwd_h[layer_idx], fwd_c[layer_idx] = cell(inp, fwd_h[layer_idx], fwd_c[layer_idx])
                inp = fwd_h[layer_idx]
                if layer_idx < self.num_layers - 1:
                    inp = self.dropout(inp)
            outputs.append(fwd_h[-1])

        out    = torch.stack(outputs, dim=1)   # [B, T, H]
        logits = self.fc_gen(out)              # [B, T, vocab_size] — smaller projection

        return logits, (fwd_h, fwd_c)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# ATTENTION MECHANISM (Bahdanau-style)
# ============================================================

class BasicAttention(nn.Module):
    """
    Bahdanau-style additive attention.

    score(query, key) = v * tanh(W1*query + W2*key)

    The context vector is a weighted sum of the keys (prior hidden states).
    This should help the model focus on relevant past context, especially
    for names where certain prefixes might be strongly correlated.
    """

    def __init__(self, hidden_size):
        super().__init__()
        # no bias needed on these - the tanh nonlinearity handles expressiveness
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)  # projects query
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)  # projects keys
        self.v  = nn.Linear(hidden_size, 1,           bias=False)  # scalar scoring

    def forward(self, query, keys, mask=None):
        """
        query: [B, H]    — current hidden state (what we're querying from)
        keys:  [B, T, H] — previous hidden states to attend over
        mask:  [B, T]    — 1 for valid positions, 0 for padding (optional)

        returns: context [B, H], attention_weights [B, T]
        """
        # expand query to [B, 1, H] for broadcasting across the T key positions
        q_exp = query.unsqueeze(1)

        # additive attention: combine projected query and keys, then score
        # [B, T, H] -> [B, T, 1] -> [B, T]
        scores = self.v(torch.tanh(self.W1(q_exp) + self.W2(keys))).squeeze(2)

        if mask is not None:
            # fill padding positions with a large negative number before softmax
            # so they get ~0 attention weight
            scores = scores.masked_fill(mask == 0, -1e9)

        alpha = F.softmax(scores, dim=1)  # [B, T] — weights sum to 1 along time axis

        # weighted sum of keys gives us the context vector
        context = (alpha.unsqueeze(2) * keys).sum(dim=1)  # [B, H]

        return context, alpha


# ============================================================
# RNN WITH ATTENTION
# ============================================================

class AttentionRNNModel(nn.Module):
    """
    Vanilla RNN augmented with Bahdanau-style attention over past hidden states.

    At each timestep, the RNN cell input = [current_embed ; context_vector]
    where context is an attention-weighted sum of all *previous* hidden states.

    This is a self-attention-like mechanism applied to the RNN's own hidden history,
    different from encoder-decoder attention but similar idea.

    I think this could help with longer names where the character dependencies
    span many steps.
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        # Note: attention version usually works well with 1 layer - adding more layers
        # on top of the attention context might not buy much and makes it slower
        self.num_layers  = num_layers

        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # RNN input at each step = concat(embed, context), so first cell input is E+H
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_sz = (embed_dim + hidden_size) if i == 0 else hidden_size
            self.cells.append(VanillaRNNCell(in_sz, hidden_size))

        self.attention = BasicAttention(hidden_size)

        # final projection from hidden state to vocab logits
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x:      [B, T]
        hidden: list of [B, H] per layer (or None)
        returns: logits [B, T, vocab_size], hidden states
        """
        B, T = x.shape
        emb = self.embed(x)  # [B, T, E]

        if hidden is None:
            hidden = [torch.zeros(B, self.hidden_size, device=x.device)
                      for _ in range(self.num_layers)]

        h       = list(hidden)
        history = []    # stores top-layer hidden states from previous timesteps
        outputs = []

        for t in range(T):
            embed_t = emb[:, t, :]  # [B, E]

            # compute context from history of hidden states
            # at t=0 there is no history yet, so context is just zeros
            if len(history) > 0:
                hist_tensor = torch.stack(history, dim=1)   # [B, t, H]
                context, _  = self.attention(h[-1], hist_tensor)  # [B, H]
            else:
                context = torch.zeros(B, self.hidden_size, device=x.device)

            # cat embedding with context and feed into RNN cell
            rnn_input = torch.cat([embed_t, context], dim=1)  # [B, E+H]

            inp = rnn_input
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(inp, h[layer_idx])
                inp = h[layer_idx]
                if layer_idx < self.num_layers - 1:
                    inp = self.dropout(inp)

            # store current top-layer hidden state for future attention steps
            history.append(h[-1])
            outputs.append(h[-1])

        out    = torch.stack(outputs, dim=1)  # [B, T, H]
        out    = self.dropout(out)
        logits = self.fc(out)                 # [B, T, vocab_size]

        return logits, h

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
