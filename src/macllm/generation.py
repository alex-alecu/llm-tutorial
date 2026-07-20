from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from .checkpoint import load_checkpoint


def generate(
    checkpoint: Path,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    seed: int = 42,
) -> str:
    model, config = load_checkpoint(checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    prompt_ids = tokenizer.encode(prompt).ids  # Text becomes vocabulary indexes.
    if bos_id is not None:
        prompt_ids.insert(0, bos_id)
    prompt_ids = prompt_ids[-config.model.max_seq_len :]
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens")

    mx.random.seed(seed)
    tokens = list(prompt_ids)
    # Prefill: process the full prompt once and remember every layer's keys and values.
    logits, cache = model(mx.array([prompt_ids]))
    for _ in range(max_new_tokens):
        # Only the final position predicts what should be appended next.
        next_logits = logits[:, -1, :].astype(mx.float32)
        if temperature <= 0:
            next_token = int(mx.argmax(next_logits, axis=-1).item())
        else:
            next_token = int(mx.random.categorical(next_logits / temperature).item())
        if eos_id is not None and next_token == eos_id:
            break
        tokens.append(next_token)  # The output immediately becomes the next input.
        # Decode one token at a time; the cache avoids recomputing the whole prefix.
        logits, cache = model(mx.array([[next_token]]), cache)
    return tokenizer.decode(tokens, skip_special_tokens=True)
