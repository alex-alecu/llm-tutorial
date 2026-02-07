from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """All core model sizes live here so architecture choices are explicit."""

    vocab_size: int
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4


class Head(nn.Module):
    """One self-attention head: one way for tokens to "look around" in context."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)

        # Causal mask: 1 means "can attend", 0 means "future token, must hide".
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.scale = head_size**-0.5

    def forward(self, x: Tensor) -> Tensor:
        # x is (B, T, C): B sequences, each with T tokens, each token with C features.
        _, t, _ = x.shape

        # Each token emits three vectors:
        # - query: what this position is asking for
        # - key: what this position offers
        # - value: the content to pass if selected
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Pairwise match score between every query and every key.
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, T, T)

        # Core anti-cheating line:
        # force all future positions to -inf so softmax gives them probability 0.
        att = att.masked_fill(self.tril[:t, :t] == 0, float("-inf"))

        # Turn scores into probabilities over "which previous token to read from".
        att = F.softmax(att, dim=-1)  # (B, T, T)

        # Weighted sum of values -> each token gathers context from allowed past tokens.
        out = att @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Many heads in parallel, then mix them back together."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        # Different heads can focus on different relationships in the same sequence.
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, C)
        out = self.proj(out)  # (B, T, C)
        return out


class FeedForward(nn.Module):
    """Per-token MLP: after communication, each token computes locally."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block (pre-norm):
    1) Attention = communication between tokens
    2) FeedForward = computation inside each token
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        # Residual highway: keep old representation and add refinement.
        # This helps gradients flow through deep stacks without vanishing.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """Minimal decoder-only GPT."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        # idx is (B, T): each number is a character id.
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.config.block_size}")

        # Token embedding answers "what is this token?"
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)

        # Position embedding answers "where is this token in the window?"
        pos = torch.arange(t, device=idx.device)
        pos_emb = self.position_embedding_table(pos)  # (T, C)

        # Sum means each token representation carries identity + location together.
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy expects (N, classes), so flatten batch and time together.
            logits_flat = logits.view(b * t, self.config.vocab_size)
            targets_flat = targets.view(b * t)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            # Only keep the latest block_size tokens (the model's attention window).
            idx_cond = idx[:, -self.config.block_size :]

            # Predict logits for every position; we only need the last position.
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample one token id

            # Autoregressive step: append sampled token and continue.
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
