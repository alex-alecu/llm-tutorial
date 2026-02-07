# Minimal Char-Level GPT (PyTorch)

This project builds and trains a tiny decoder-only Transformer from scratch, with code written for readability and intuition.

## Files

- `input.txt`: training corpus (auto-downloaded if missing or empty)
- `model.py`: GPT architecture (attention, MLP, residual blocks, generation)
- `train.py`: tokenizer, batching, training loop, and text generation
- `requirements.txt`: Python dependency

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 train.py
```

What happens on run:

1. Picks device automatically (`cuda` > `mps` > `cpu`).
2. Downloads Tiny Shakespeare into `input.txt` only if the file is missing or empty.
3. Builds a character-level tokenizer (`stoi`, `itos`, `encode`, `decode`).
4. Trains GPT and prints loss every 100 steps.
5. Generates sample text from seed `"ROMEO: "`.

## Manual Dataset Fallback

If download fails (no internet, blocked URL, etc.), place your own text in `input.txt` and run again.

## Aha Checklist

- **Causal mask (no cheating):** `model.py` line with `masked_fill(..., -inf)`
- **Residual highways:** `model.py` lines with `x = x + ...`
- **Token + position meaning:** `model.py` lines adding token and position embeddings

