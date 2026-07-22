# Mac LLM Lab

Build a small language model on an Apple-silicon Mac. No machine-learning background is needed.

## The one idea

A language model guesses the next piece of text.

1. Split text into small pieces called **tokens**.
2. Let the model guess the next token.
3. Measure the error. That score is **loss**.
4. Adjust the model’s stored numbers, called **weights**.
5. Repeat.

This project turns that loop into readable Python and real commands.

## Open the visual course

The site uses plain HTML, CSS, and JavaScript. It has no build step or web framework.

```bash
make serve
open http://localhost:8000/
```

Or use the [published course](https://alex-alecu.github.io/llm-tutorial/).

Follow the chapters in order:

| Step | You do | You get |
|---:|---|---|
| 00 | Set up Python | A checked local environment |
| 01 | Prepare stories | Tokenizer and token files |
| 02 | Learn neural-network basics, then build the transformer | A model with random weights |
| 03 | Trace backpropagation, then pretrain | A model that learned story patterns |
| 04 | Fine-tune | A narrower behavior |
| 05 | Quantize | A smaller checkpoint |
| 06 | Generate | New text from a prompt |
| 07 | Trace the code | Every command mapped to Python |

## What each tool does

- **`macllm`** is this repository’s Python package and command. It connects every stage.
- [**MLX**](https://ml-explore.github.io/mlx/) runs arrays and neural-network math on Apple silicon.
- [**MLX-LM**](https://github.com/ml-explore/mlx-lm) runs, fine-tunes, and converts larger MLX language models.
- [**Tokenizers**](https://huggingface.co/docs/tokenizers/) builds a vocabulary and converts text to token IDs.
- [**Datasets**](https://huggingface.co/docs/datasets/) streams TinyStories without loading it all at once.
- [**Hugging Face Hub**](https://huggingface.co/docs/huggingface_hub/) downloads public model and dataset files.
- [**NumPy**](https://numpy.org/doc/stable/) stores long token-ID arrays efficiently.
- **pytest** tests behavior. **Ruff** checks Python code. Both are developer-only tools.

`macllm` is the part you will read and change. The other libraries handle fast, well-tested building blocks.

## Set up once

Python 3.11–3.13 is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
macllm doctor
```

The editable install makes `macllm` use the code under `src/macllm/`.

Chapter 02 animates a small neural network from numeric inputs to a prediction. Chapter 03 follows the transformer forward to loss, then backward through gradients and an optimizer update. Chapter 06 animates the complete inference loop, including token encoding, the KV cache, decoding, and feeding each generated token back as input.

The source readers place explanation beside exact repository code. Select an explanation to highlight only the lines it describes; comments inside the core model, training, and generation files provide a second layer of detail.

## Your first complete model

Start with the `quick` preset. It proves the whole path with about 8 million weights.

```bash
macllm prepare --preset quick
macllm inspect --data data/prepared/quick --text "Once upon a little time"
macllm train --preset quick
macllm generate --checkpoint runs/quick --prompt "Once upon a time"
```

Training creates `runs/quick/`. That checkpoint contains weights, model settings, and the tokenizer.

Build a local loss report:

```bash
macllm dashboard --run runs/quick
open runs/quick/dashboard.html
```

When `quick` works, use the recommended 57M model:

```bash
macllm prepare --preset standard
macllm train --preset standard
macllm generate --checkpoint runs/standard --prompt "Once upon a time"
```

Times are planning ranges, not benchmarks:

| Preset | Weights | Tokens seen | Rough M5 Pro time |
|---|---:|---:|---:|
| `quick` | 8M | 6M | 10–30 min |
| `standard` | 57M | 82M | 3–8 h |
| `overnight` | 113M | 205M | 12–30 h |

Your first training report shows measured tokens per second.

## Fine-tune

Fine-tuning keeps training a saved model on a smaller, focused dataset.

```bash
macllm finetune \
  --checkpoint runs/standard \
  --data data/our-model/train.jsonl \
  --output runs/standard-story-tuned \
  --steps 300
```

The course also shows QLoRA: small adapter weights teach the 4-bit Qwen3-4B model while its main weights stay frozen. See [Chapter 04](site/chapters/04-finetune.html).

## Quantize

Quantization stores weights with fewer bits. It reduces size but can change output slightly.

```bash
macllm quantize \
  --checkpoint runs/standard-story-tuned \
  --output runs/standard-story-tuned-4bit \
  --bits 4
```

Compare original and 4-bit models with the same prompt and `--temperature 0`.

## Repository map

```text
data/                     small fine-tuning examples
docs/                     written deep dives
site/                     interactive course
src/macllm/               local package and CLI
tests/                    fast behavior checks
```

## Honest limits

- The from-scratch model learns simple TinyStories patterns. It is not a general assistant.
- All model math stays on this Mac. Dataset and model commands download files from Hugging Face.
- Keep at least 15 GB free for `standard`. Qwen conversion needs more.
- Closing the lid pauses useful training. Keep the Mac powered and awake.
- Compare models with fixed prompts, seeds, and settings. One nice sample is not proof.

If a command fails, read [troubleshooting](docs/troubleshooting.md), then run:

```bash
macllm doctor
```
