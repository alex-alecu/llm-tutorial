from __future__ import annotations

from pathlib import Path
import json
import shutil

import mlx.core as mx
import mlx.nn as nn

from .checkpoint import load_checkpoint


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def quantize_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    bits: int = 4,
    group_size: int = 64,
) -> tuple[int, int]:
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; choose a fresh output directory")
    if bits not in {2, 3, 4, 5, 6, 8}:
        raise ValueError("bits must be one of 2, 3, 4, 5, 6, or 8")
    model, _ = load_checkpoint(checkpoint)
    before = directory_size(checkpoint)
    nn.quantize(model, group_size=group_size, bits=bits, mode="affine")
    mx.eval(model.parameters())
    output_dir.mkdir(parents=True)
    model.save_weights(str(output_dir / "weights.safetensors"))
    shutil.copy2(checkpoint / "config.json", output_dir / "config.json")
    shutil.copy2(checkpoint / "tokenizer.json", output_dir / "tokenizer.json")
    quantization = {"bits": bits, "group_size": group_size, "mode": "affine"}
    (output_dir / "quantization.json").write_text(
        json.dumps(quantization, indent=2) + "\n", encoding="utf-8"
    )
    after = directory_size(output_dir)
    return before, after
