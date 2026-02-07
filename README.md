# Nano-GPT Merge Notes

## Goal

Build a minimal, readable, and trainable character-level GPT (decoder-only Transformer) from scratch in PyTorch, with comments that explain the physical intuition of what data is doing at each step.  
This repo keeps two implementations so we can compare design choices and combine their strengths into one final teaching-friendly baseline.

This repository now contains both implementations side by side:

- `codex/`: minimal, spec-driven implementation with clean APIs
- `opus/`: tutorial-style implementation with more training features and heavier defaults

## Folder Layout

- `codex/model.py`, `codex/train.py`, `codex/README.md`, `codex/requirements.txt`
- `opus/model.py`, `opus/train.py`, `opus/input.txt`

## Key Differences

| Area                              | `codex/`                                                                   | `opus/`                                                                   | Why it matters                                    |
| --------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------- |
| Configuration style               | Uses `GPTConfig` dataclass passed into model                               | Uses module-level global hyperparameters                                  | `codex/` is easier to reuse and test as a library |
| Dataset handling                  | Auto-downloads Tiny Shakespeare if `input.txt` is missing/empty            | Expects local `input.txt` already present                                 | `codex/` is more robust in fresh environments     |
| Default size/runtime              | Smaller defaults (`n_embd=128`, `n_head=4`, `n_layer=4`, `max_iters=2000`) | Larger defaults (`n_embd=384`, `n_head=6`, `n_layer=6`, `max_iters=5000`) | `codex/` trains faster; `opus/` has more capacity |
| Regularization                    | No dropout                                                                 | Attention/MLP dropout enabled (`dropout=0.2`)                             | `opus/` is closer to realistic training behavior  |
| Metrics during training           | Logs training loss every 100 steps                                         | Estimates both train/val loss across `eval_iters`                         | `opus/` gives a clearer quality signal            |
| Code organization in train script | Function-based with `main()`                                               | Mostly top-level script flow                                              | `codex/` is cleaner for import/reuse              |
| Educational style                 | Concise comments focused on core intuition                                 | Very extensive "physical intuition" narration                             | `opus/` is better for long-form teaching          |

## Recommended Merged Direction

If you are keeping one final implementation, the strongest combined shape is:

1. Keep `codex/` structure and APIs (`GPTConfig`, `main()`, `download_if_needed`).
2. Keep `opus/` training improvements (dropout + train/val loss estimation).
3. Keep the physical-intuition comments from both, but trim repetition.
4. Preserve the small default profile for quick runs, and add an optional larger preset.

## Quick Run

Minimal/spec-oriented run:

```bash
cd codex
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

Tutorial/heavier run:

```bash
cd opus
python train.py
```
