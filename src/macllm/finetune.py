from __future__ import annotations

from pathlib import Path
import json
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tokenizers import Tokenizer

from .checkpoint import load_checkpoint, save_checkpoint


def load_examples(path: Path) -> list[dict[str, str]]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("prompt"), str) or not isinstance(
                item.get("completion"), str
            ):
                raise ValueError(f"{path}:{line_number} needs prompt and completion strings")
            examples.append(item)
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def make_batch(
    tokenizer: Tokenizer,
    examples: list[dict[str, str]],
    indices: np.ndarray,
    sequence_length: int,
) -> tuple[mx.array, mx.array, mx.array]:
    pad_id = tokenizer.token_to_id("<pad>") or 0
    eos_id = tokenizer.token_to_id("<eos>") or pad_id
    inputs = np.full((len(indices), sequence_length), pad_id, dtype=np.int32)
    targets = np.full((len(indices), sequence_length), pad_id, dtype=np.int32)
    masks = np.zeros((len(indices), sequence_length), dtype=np.float32)

    for row, index in enumerate(indices):
        example = examples[int(index)]
        prompt_ids = tokenizer.encode(example["prompt"]).ids
        completion_ids = tokenizer.encode(example["completion"]).ids + [eos_id]
        joined = (prompt_ids + completion_ids)[: sequence_length + 1]
        if len(joined) < 2:
            continue
        x = joined[:-1]
        y = joined[1:]
        length = min(len(x), sequence_length)
        inputs[row, :length] = x[:length]
        targets[row, :length] = y[:length]
        completion_start = max(0, min(length, len(prompt_ids) - 1))
        masks[row, completion_start:length] = 1.0
    return mx.array(inputs), mx.array(targets), mx.array(masks)


def masked_loss(model, inputs, targets, mask):
    logits, _ = model(inputs)
    losses = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    return mx.sum(losses * mask) / mx.maximum(mx.sum(mask), mx.array(1.0))


def finetune(
    checkpoint: Path,
    data_path: Path,
    output_dir: Path,
    *,
    steps: int = 300,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    seed: int = 7,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; choose a fresh run directory")
    model, config = load_checkpoint(checkpoint)
    if (checkpoint / "quantization.json").exists():
        raise ValueError("Fine-tune the unquantized checkpoint, then quantize the result")
    model.train()
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    examples = load_examples(data_path)
    rng = np.random.default_rng(seed)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, masked_loss)
    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.jsonl"
    started = time.perf_counter()

    for step in range(1, steps + 1):
        indices = rng.integers(0, len(examples), size=batch_size)
        batch = make_batch(tokenizer, examples, indices, config.model.max_seq_len)
        loss, grads = loss_and_grad(model, *batch)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 1 or step % 25 == 0 or step == steps:
            record = {
                "step": step,
                "train_loss": loss.item(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"fine-tune step {step:>4}/{steps} | loss {record['train_loss']:.3f}")

    save_checkpoint(
        model,
        config,
        checkpoint / "tokenizer.json",
        output_dir,
        metadata={"status": "complete", "base_checkpoint": str(checkpoint), "steps": steps},
    )
