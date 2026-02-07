"""
model.py — A minimal Decoder-Only Transformer (GPT) built from scratch.

This file contains every component of a GPT-style language model, from the
character-level tokenizer to the full architecture. Each class builds on the
last, forming a pipeline:

    Characters → Tokens → Embeddings → [Attention + FeedForward] × N → Logits → Next Character

Read top-to-bottom. Every tensor operation has a comment explaining the
"physical intuition" — what the data represents at that point.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# =============================================================================
# Hyperparameters — the "knobs" that control the model
# =============================================================================
# Keep these small so the model trains on a laptop in minutes.

batch_size  = 32        # How many independent sequences we process in parallel
block_size  = 64        # Maximum context length (how far back the model can look)
n_embd      = 384       # Dimensionality of the embedding vector (the "width" of the model)
n_head      = 6         # Number of attention heads (n_embd must be divisible by n_head)
n_layer     = 6         # How many Transformer blocks stacked on top of each other
dropout     = 0.2       # Fraction of neurons randomly turned off during training (regularization)
learning_rate = 3e-4    # Step size for the optimizer

# Training schedule
max_iters     = 5000    # Total training iterations
eval_interval = 100     # How often to print losses
eval_iters    = 200     # Number of batches to average when estimating loss

# =============================================================================
# Character-Level Tokenizer
# =============================================================================
# The simplest possible tokenizer: every unique character gets its own integer.
# For Shakespeare, the vocabulary is ~65 chars (a-z, A-Z, punctuation, newline).
# This means the final probability distribution the model outputs has only ~65
# entries — small enough to inspect by hand.


def build_tokenizer(text: str):
    """
    Build a character-level tokenizer from a body of text.

    Returns:
        stoi: dict mapping each character → unique integer
        itos: dict mapping each integer → its character
        vocab_size: total number of unique characters
        encode: function  string → list[int]
        decode: function  list[int] → string
    """
    # Collect every unique character and sort them so the mapping is deterministic
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # stoi: "string to integer" — e.g. {'a': 0, 'b': 1, ...}
    stoi = {ch: i for i, ch in enumerate(chars)}
    # itos: "integer to string" — the reverse lookup
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode: convert a string into a list of integers
    #   "hello" → [46, 43, 50, 50, 53]  (example indices)
    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    # Decode: convert a list of integers back into a string
    #   [46, 43, 50, 50, 53] → "hello"
    def decode(l: list[int]) -> str:
        return ''.join([itos[i] for i in l])

    return stoi, itos, vocab_size, encode, decode


# =============================================================================
# Model Components
# =============================================================================

class Head(nn.Module):
    """
    A single head of self-attention.

    Physical Intuition:
    ------------------
    Imagine a room full of people (tokens). Each person wants to gather
    information from the others. They do this by:
      1. Broadcasting a KEY   — "Here is what I contain"
      2. Broadcasting a QUERY — "Here is what I'm looking for"
      3. Holding a VALUE      — "Here is what I'll share if you ask"

    The attention score between two tokens is the dot product of one's QUERY
    with the other's KEY. High score = "these two tokens are relevant to each
    other." The scores become weights (via softmax) that determine how much
    VALUE each token receives from every other token.

    Input shape:  (batch, time, channels)  — B, T, C
    Output shape: (batch, time, head_size)
    """

    def __init__(self, head_size: int):
        super().__init__()
        # Linear projections: each one learns WHAT to extract from the input.
        # No bias — a common simplification that works fine in practice.
        self.key   = nn.Linear(n_embd, head_size, bias=False)   # What do I contain?
        self.query = nn.Linear(n_embd, head_size, bias=False)   # What am I looking for?
        self.value = nn.Linear(n_embd, head_size, bias=False)   # What do I share?

        # =====================================================================
        # *** AHA! THE MASK ***
        # This lower-triangular matrix is the heart of "causal" (autoregressive)
        # attention. It ensures that token at position t can ONLY attend to
        # tokens at positions <= t. Without this, the model could "cheat" by
        # looking at the answer (future tokens) during training.
        #
        #   tril = [[1, 0, 0, 0],
        #           [1, 1, 0, 0],
        #           [1, 1, 1, 0],
        #           [1, 1, 1, 1]]
        #
        # Zeros will be replaced with -infinity before softmax, which makes
        # softmax assign those positions a probability of exactly 0.
        # =====================================================================
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # B=batch, T=time (sequence length), C=channels (n_embd)

        # Step 1: Project input into key, query, value spaces
        # Each has shape (B, T, head_size)
        k = self.key(x)     # What information does each token advertise?
        q = self.query(x)   # What information is each token looking for?

        # Step 2: Compute raw attention scores
        # q @ k^T → (B, T, head_size) @ (B, head_size, T) → (B, T, T)
        # The result is a T×T matrix: score[i][j] = how much token i attends to token j
        #
        # We scale by 1/sqrt(head_size) to keep the variance of the scores stable.
        # Without this, large dot products would push softmax into regions where
        # gradients are extremely small (the "vanishing gradient" zone of softmax).
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # (B, T, T)

        # Step 3: Apply the causal mask
        # *** THIS IS THE LINE THAT PREVENTS CHEATING ***
        # For every position where tril == 0 (i.e., future tokens), we stuff in
        # -infinity. After softmax, e^(-inf) = 0, so future tokens contribute
        # absolutely nothing to the weighted sum.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        # Step 4: Softmax converts raw scores into a probability distribution
        # Each row sums to 1.0 — it's a weighted average recipe.
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Step 5: Weighted aggregation of values
        # For each token, sum up the values of all tokens it attends to,
        # weighted by the attention probabilities.
        v = self.value(x)            # (B, T, head_size) — what each token shares
        out = wei @ v                # (B, T, T) @ (B, T, head_size) → (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention running in parallel.

    Physical Intuition:
    ------------------
    One attention head can only learn one "type" of relationship (e.g., "what
    noun does this adjective modify?"). By running several heads in parallel,
    the model can simultaneously attend to different types of relationships:
      - Head 1 might learn syntax (subject-verb agreement)
      - Head 2 might learn semantics (words with similar meaning)
      - Head 3 might learn positional patterns (nearby words)

    We concatenate all heads' outputs and project them back to the model width.
    """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        # Create `num_heads` independent attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Final linear projection: combine the insights from all heads
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel, then concatenate along the channel dimension
        # Each head outputs (B, T, head_size); after concat → (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project back to n_embd so it can be added to the residual stream
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    A simple two-layer MLP (Multi-Layer Perceptron).

    Physical Intuition:
    ------------------
    Attention is about COMMUNICATION — tokens looking at each other.
    FeedForward is about COMPUTATION — each token independently "thinking"
    about what it just learned from attention.

    The hidden layer is 4× wider than the input, giving the network room to
    build complex internal representations before compressing back down.

    Think of it as: Attention = "gather information", FeedForward = "process it."
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # Expand: give the network room to think
            nn.ReLU(),                         # Non-linearity: allows learning complex patterns
            nn.Linear(4 * n_embd, n_embd),    # Compress: back to model dimension
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    One Transformer block = Attention + FeedForward, with residuals and norms.

    Physical Intuition:
    ------------------
    This is the core repeating unit. Each block does two things:
      1. COMMUNICATE: tokens look at each other (Multi-Head Attention)
      2. COMPUTE: each token processes what it learned (FeedForward)

    Two critical design choices:

    * Pre-Norm: We apply LayerNorm BEFORE each sub-layer. This stabilizes
      training by keeping the inputs to each layer well-scaled.

    =====================================================================
    *** AHA! THE RESIDUAL HIGHWAY ***
    * Residual connections: `x = x + layer(x)`. Instead of replacing x,
      we ADD the layer's output to it. This creates a "highway" for
      gradients to flow backward through the network.

      Without residuals, the gradient signal must pass through every
      layer's weights to reach the early layers — and it shrinks
      exponentially (vanishing gradient problem). With residuals,
      the gradient has a direct shortcut:

        Gradient path WITHOUT residuals:  → layer6 → layer5 → ... → layer1
        Gradient path WITH residuals:     → + → + → + → + → + → +  (express lane!)

      This is why deep Transformers (50+ layers) can actually train.
    =====================================================================
    """

    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head  # Each head gets an equal slice of the embedding
        self.sa   = MultiHeadAttention(n_head, head_size)  # Self-Attention
        self.ff   = FeedForward()                           # Feed-Forward
        self.ln1  = nn.LayerNorm(n_embd)                   # Norm before attention
        self.ln2  = nn.LayerNorm(n_embd)                   # Norm before feedforward

    def forward(self, x):
        # Residual connection around the attention sub-layer
        # x flows straight through; attention output is ADDED to it
        x = x + self.sa(self.ln1(x))    # Communicate: gather info from other tokens
        # Residual connection around the feedforward sub-layer
        x = x + self.ff(self.ln2(x))    # Compute: process what was gathered
        return x


class GPT(nn.Module):
    """
    The full GPT Language Model.

    Data flow (what happens during a forward pass):
    -----------------------------------------------
    1. Input token indices         (B, T)        — integers like [14, 52, 3, ...]
    2. Token embeddings            (B, T, C)     — each int becomes a learned vector
    3. + Position embeddings       (B, T, C)     — add "where am I" information
    4. Through N Transformer blocks (B, T, C)    — attention + feedforward × N
    5. Final LayerNorm             (B, T, C)     — stabilize before decoding
    6. Linear decode to logits     (B, T, vocab) — raw scores for each character
    7. Softmax → probabilities     (B, T, vocab) — "30% chance next char is 'e'"

    =====================================================================
    *** AHA! TWO EMBEDDINGS ***
    The model learns TWO things for every token:
      - Token Embedding:    WHAT is this character?   ('e' has meaning)
      - Position Embedding: WHERE is it in the text?  (position 5 has meaning)

    Without positional embeddings, the model would see "cat" and "tac" as
    identical — it would have no sense of order. The position embedding
    gives the model a sense of "first," "second," "third," etc.
    =====================================================================
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        # *** Token Embedding Table ***
        # Maps each token (character) to a dense vector of size n_embd.
        # Before training, these vectors are random. After training, similar
        # characters (like 'a' and 'e' — both vowels) will have nearby vectors.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # *** Position Embedding Table ***
        # Maps each position (0, 1, 2, ..., block_size-1) to a dense vector.
        # The model uses these to understand word ORDER.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of Transformer blocks — this is the "deep" part of deep learning
        # Each block refines the representation: communicate → compute → repeat
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])

        # Final layer norm for stability before the decoding step
        self.ln_f = nn.LayerNorm(n_embd)

        # Decoder head: project from the model's internal dimension (n_embd)
        # back to vocabulary size, producing one score per possible next character
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        Args:
            idx:     (B, T) tensor of token indices (integers)
            targets: (B, T) tensor of target token indices, or None for generation

        Returns:
            logits: (B, T, vocab_size) raw prediction scores
            loss:   scalar cross-entropy loss (or None if no targets)
        """
        B, T = idx.shape

        # Step 1: Look up token embeddings
        # (B, T) → (B, T, C) — each integer index becomes a learned vector
        # Physical intuition: we're converting "character #14" into a rich
        # representation that captures the character's meaning.
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)

        # Step 2: Look up position embeddings
        # torch.arange(T) = [0, 1, 2, ..., T-1] — the positions
        # (T,) → (T, C) — each position gets its own learned vector
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)

        # Step 3: COMBINE what each token IS with WHERE it IS
        # The addition broadcasts: (B, T, C) + (T, C) → (B, T, C)
        # Now every token's representation encodes both identity and position.
        x = tok_emb + pos_emb  # (B, T, C)

        # Step 4: Pass through all Transformer blocks
        # Each block lets tokens communicate (attention) and compute (feedforward).
        # After N blocks, each token's vector is a rich summary of the entire
        # context it's allowed to see (all tokens before it, thanks to the mask).
        x = self.blocks(x)     # (B, T, C)

        # Step 5: Final layer normalization
        x = self.ln_f(x)       # (B, T, C)

        # Step 6: Decode to vocabulary logits
        # (B, T, C) → (B, T, vocab_size) — one score per possible next character
        # "Logits" = raw unnormalized scores. Higher logit = model thinks that
        # character is more likely to come next.
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if we have targets
        if targets is None:
            loss = None
        else:
            # Reshape for cross_entropy: it expects (N, C) and (N,)
            B, T, C = logits.shape
            logits_flat  = logits.view(B * T, C)    # (B*T, vocab_size)
            targets_flat = targets.view(B * T)       # (B*T,)

            # Cross-entropy loss: measures how far the model's probability
            # distribution is from the true answer. Lower = better predictions.
            # At initialization (random weights), loss ≈ -ln(1/65) ≈ 4.17
            # because the model assigns equal probability to all 65 characters.
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        """
        Generate new text autoregressively.

        Physical Intuition:
        -------------------
        This is the core generation loop. At each step:
          1. Feed the current context into the model
          2. Look at the model's prediction for the LAST position only
          3. Convert logits → probabilities (softmax)
          4. SAMPLE one token from that distribution (not argmax — sampling
             gives variety and creativity to the output)
          5. Append the sampled token to the context
          6. Repeat

        The model only ever sees the past and predicts one token into the future.
        It has no "plan" for what comes next — each token is generated fresh.

        Args:
            idx: (B, T) tensor of token indices — the starting context
            max_new_tokens: how many characters to generate

        Returns:
            idx: (B, T + max_new_tokens) tensor — original context + generated tokens
        """
        for _ in range(max_new_tokens):
            # Crop the context to the last block_size tokens
            # (the model can't handle inputs longer than block_size because
            # our positional embedding table only has block_size entries)
            idx_cond = idx[:, -block_size:]  # (B, T') where T' <= block_size

            # Forward pass — get predictions for all positions
            logits, _ = self(idx_cond)  # (B, T', vocab_size)

            # We only care about the LAST time step's prediction
            # (what comes after the entire context we've seen so far)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # Sample from the distribution
            # This randomness is what gives language models their creativity.
            # argmax would always pick the most likely character — boring and repetitive.
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the new token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
