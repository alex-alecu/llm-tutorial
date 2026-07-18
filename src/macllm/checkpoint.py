from __future__ import annotations

from pathlib import Path
import json
import shutil

import mlx.core as mx
import mlx.nn as nn

from .config import TrainConfig, load_config, save_config
from .model import TinyLM


def save_checkpoint(
    model: TinyLM,
    config: TrainConfig,
    tokenizer_path: Path,
    output_dir: Path,
    *,
    metadata: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "weights.tmp.safetensors"
    model.save_weights(str(temporary))
    temporary.replace(output_dir / "weights.safetensors")
    save_config(config, output_dir / "config.json")
    if tokenizer_path.resolve() != (output_dir / "tokenizer.json").resolve():
        shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    if metadata is not None:
        (output_dir / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


def load_checkpoint(path: Path) -> tuple[TinyLM, TrainConfig]:
    config = load_config(path / "config.json")
    model = TinyLM(config.model)
    quantization_path = path / "quantization.json"
    if quantization_path.exists():
        quantization = json.loads(quantization_path.read_text(encoding="utf-8"))
        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
        )
    model.load_weights(str(path / "weights.safetensors"))
    mx.eval(model.parameters())
    model.eval()
    return model, config
