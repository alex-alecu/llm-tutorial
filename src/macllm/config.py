from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 320
    hidden_dim: int = 896
    n_layers: int = 6
    n_heads: int = 5
    n_kv_heads: int = 1
    max_seq_len: int = 256
    rope_base: float = 10_000.0
    norm_eps: float = 1e-5

    def validate(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")


@dataclass(frozen=True)
class TrainConfig:
    name: str
    model: ModelConfig
    batch_size: int
    steps: int
    learning_rate: float
    warmup_steps: int
    eval_every: int
    eval_batches: int
    max_stories: int
    validation_stories: int
    tokenizer_vocab: int
    seed: int = 42
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    def with_vocab_size(self, vocab_size: int) -> "TrainConfig":
        return replace(self, model=replace(self.model, vocab_size=vocab_size))

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, TrainConfig] = {
    "quick": TrainConfig(
        name="quick",
        model=ModelConfig(),
        batch_size=16,
        steps=1_500,
        learning_rate=4e-4,
        warmup_steps=100,
        eval_every=100,
        eval_batches=10,
        max_stories=20_000,
        validation_stories=2_000,
        tokenizer_vocab=4_096,
    ),
    "standard": TrainConfig(
        name="standard",
        model=ModelConfig(
            vocab_size=8_192,
            dim=640,
            hidden_dim=1_728,
            n_layers=12,
            n_heads=10,
            n_kv_heads=2,
            max_seq_len=512,
        ),
        batch_size=8,
        steps=20_000,
        learning_rate=3e-4,
        warmup_steps=500,
        eval_every=500,
        eval_batches=20,
        max_stories=500_000,
        validation_stories=10_000,
        tokenizer_vocab=8_192,
    ),
    "overnight": TrainConfig(
        name="overnight",
        model=ModelConfig(
            vocab_size=16_384,
            dim=768,
            hidden_dim=2_048,
            n_layers=16,
            n_heads=12,
            n_kv_heads=4,
            max_seq_len=512,
        ),
        batch_size=8,
        steps=50_000,
        learning_rate=2.5e-4,
        warmup_steps=1_000,
        eval_every=1_000,
        eval_batches=20,
        max_stories=1_500_000,
        validation_stories=20_000,
        tokenizer_vocab=16_384,
    ),
}


def get_preset(name: str) -> TrainConfig:
    try:
        config = PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset {name!r}; choose from {', '.join(PRESETS)}") from exc
    config.model.validate()
    return config


def save_config(config: TrainConfig, path: Path) -> None:
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_config(path: Path) -> TrainConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["model"] = ModelConfig(**raw["model"])
    config = TrainConfig(**raw)
    config.model.validate()
    return config
