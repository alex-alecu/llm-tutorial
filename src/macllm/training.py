from __future__ import annotations

from collections import deque
from functools import partial
from pathlib import Path
import json
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tokenizers import Tokenizer

from .checkpoint import save_checkpoint
from .config import TrainConfig
from .data import TokenDataset
from .model import TinyLM, parameter_count


def cosine_learning_rate(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def language_model_loss(model: TinyLM, inputs: mx.array, targets: mx.array) -> mx.array:
    logits, _ = model(inputs)
    return nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="mean")


def evaluate(
    model: TinyLM,
    dataset: TokenDataset,
    rng: np.random.Generator,
    config: TrainConfig,
) -> float:
    losses = []
    for _ in range(config.eval_batches):
        x, y = dataset.batch(rng, config.batch_size, config.model.max_seq_len)
        loss = language_model_loss(model, mx.array(x), mx.array(y))
        losses.append(loss.item())
    return float(np.mean(losses))


def train(
    config: TrainConfig,
    data_dir: Path,
    output_dir: Path,
    *,
    steps: int | None = None,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; choose a fresh run directory")
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = config.with_vocab_size(tokenizer.get_vocab_size())
    if steps is not None:
        config = TrainConfig(**{**config.to_dict(), "model": config.model, "steps": steps})

    mx.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = TinyLM(config.model)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        betas=[0.9, 0.95],
        weight_decay=config.weight_decay,
    )
    loss_and_grad = nn.value_and_grad(model, language_model_loss)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs, targets):
        loss, grads = loss_and_grad(model, inputs, targets)
        grads, grad_norm = optim.clip_grad_norm(grads, config.grad_clip)
        optimizer.update(model, grads)
        return loss, grad_norm

    train_data = TokenDataset(data_dir / "train.bin")
    valid_data = TokenDataset(data_dir / "valid.bin")
    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.jsonl"
    recent_losses: deque[float] = deque(maxlen=max(1, config.eval_every))
    started = time.perf_counter()
    interval_started = started
    interval_tokens = 0

    print(
        f"Training {parameter_count(model) / 1e6:.1f}M parameters | "
        f"{config.model.n_layers} layers | {config.model.max_seq_len} token context"
    )

    try:
        for index in range(config.steps):
            optimizer.learning_rate = cosine_learning_rate(index, config)
            x, y = train_data.batch(rng, config.batch_size, config.model.max_seq_len)
            loss, grad_norm = step(mx.array(x), mx.array(y))
            mx.eval(state, loss, grad_norm)
            recent_losses.append(loss.item())
            interval_tokens += x.size
            completed = index + 1

            should_report = completed == 1 or completed % config.eval_every == 0
            if not should_report and completed != config.steps:
                continue

            now = time.perf_counter()
            tokens_per_second = interval_tokens / max(now - interval_started, 1e-9)
            val_loss = evaluate(model, valid_data, rng, config)
            record = {
                "step": completed,
                "train_loss": float(np.mean(recent_losses)),
                "val_loss": val_loss,
                "perplexity": math.exp(min(val_loss, 20)),
                "learning_rate": float(optimizer.learning_rate),
                "tokens_per_second": tokens_per_second,
                "tokens_seen": completed * config.batch_size * config.model.max_seq_len,
                "peak_memory_gb": mx.get_peak_memory() / 1024**3,
                "elapsed_seconds": now - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"step {completed:>6}/{config.steps} | "
                f"train {record['train_loss']:.3f} | val {val_loss:.3f} | "
                f"{tokens_per_second:,.0f} tok/s | {record['peak_memory_gb']:.1f} GB"
            )
            save_checkpoint(
                model,
                config,
                tokenizer_path,
                output_dir,
                metadata={"status": "training", "last_step": completed},
            )
            interval_started = time.perf_counter()
            interval_tokens = 0
    except KeyboardInterrupt:
        save_checkpoint(
            model,
            config,
            tokenizer_path,
            output_dir,
            metadata={"status": "interrupted"},
        )
        print("\nSaved an interrupted checkpoint.")
        return

    save_checkpoint(
        model,
        config,
        tokenizer_path,
        output_dir,
        metadata={"status": "complete", "last_step": config.steps},
    )
