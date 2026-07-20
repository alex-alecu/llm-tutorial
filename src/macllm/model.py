from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


class Attention(nn.Module):
    """Grouped-query causal attention with rotary position information."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_base)

    def __call__(self, x: mx.array, cache=None):
        batch, length, _ = x.shape
        # Each projection gives every token a different role in attention:
        # query = what I need, key = what I contain, value = what I can share.
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        # Attention expects heads before time: [batch, heads, tokens, head_dim].
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Cached keys and values are the transformer loop's memory during generation.
        offset = 0 if cache is None else cache[0].shape[2]
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k = mx.concatenate((cache[0], k), axis=2)
            v = mx.concatenate((cache[1], v), axis=2)

        mask = "causal" if length > 1 else None  # A token cannot inspect future answers.
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=mask
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.out_proj(attended), (k, v)


class FeedForward(nn.Module):
    """SwiGLU: a gated per-token computation stage."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # SiLU opens a learned gate; multiplication chooses which features pass through.
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

    def __call__(self, x: mx.array, cache=None):
        attended, cache = self.attention(self.attention_norm(x), cache)
        x = x + attended  # The residual keeps the old stream and adds new context.
        return x + self.feed_forward(self.ffn_norm(x)), cache


class TinyLM(nn.Module):
    """A compact Llama-style decoder-only language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

    def __call__(self, tokens: mx.array, cache=None):
        if tokens.shape[1] > self.config.max_seq_len and cache is None:
            raise ValueError(
                f"Sequence has {tokens.shape[1]} tokens; maximum is {self.config.max_seq_len}"
            )
        caches = [None] * len(self.layers) if cache is None else cache
        x = self.embed(tokens)  # Integer token IDs become learned vectors.
        next_caches = []
        for layer, layer_cache in zip(self.layers, caches, strict=True):
            # Every layer refines the same residual stream and grows its own KV cache.
            x, layer_cache = layer(x, layer_cache)
            next_caches.append(layer_cache)
        # Weight tying reuses the embedding table to score every possible next token.
        return self.embed.as_linear(self.norm(x)), next_caches


def parameter_count(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return sum(value.size for _, value in tree_flatten(model.parameters()))
