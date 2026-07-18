from __future__ import annotations

import argparse
from pathlib import Path
import json

from tokenizers import Tokenizer

from .config import PRESETS, get_preset


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="macllm", description="Learn LLMs by building one on Apple silicon."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Check this Mac and Python environment")

    prepare = commands.add_parser("prepare", help="Download text and build a tokenizer")
    prepare.add_argument("--preset", choices=PRESETS, default="quick")
    prepare.add_argument("--output", type=_path)
    prepare.add_argument("--text", type=_path, help="Use a local UTF-8 text file")
    prepare.add_argument("--max-stories", type=int)

    inspect = commands.add_parser("inspect", help="See the next-token training shift")
    inspect.add_argument("--data", type=_path, default=Path("data/prepared/quick"))
    inspect.add_argument("--text", default="Once upon a time")

    train = commands.add_parser("train", help="Pretrain our model from random weights")
    train.add_argument("--preset", choices=PRESETS, default="quick")
    train.add_argument("--data", type=_path)
    train.add_argument("--output", type=_path)
    train.add_argument("--steps", type=int, help="Override the preset for a smoke run")

    generate = commands.add_parser("generate", help="Generate with our checkpoint")
    generate.add_argument("--checkpoint", type=_path, required=True)
    generate.add_argument("--prompt", default="Once upon a time")
    generate.add_argument("--max-new-tokens", type=int, default=200)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--seed", type=int, default=42)

    tune = commands.add_parser("finetune", help="Full fine-tune our small model")
    tune.add_argument("--checkpoint", type=_path, required=True)
    tune.add_argument("--data", type=_path, default=Path("data/our-model/train.jsonl"))
    tune.add_argument("--output", type=_path, default=Path("runs/our-model-finetuned"))
    tune.add_argument("--steps", type=int, default=300)
    tune.add_argument("--batch-size", type=int, default=4)
    tune.add_argument("--learning-rate", type=float, default=2e-5)

    quantize = commands.add_parser("quantize", help="Quantize our model's weights")
    quantize.add_argument("--checkpoint", type=_path, required=True)
    quantize.add_argument("--output", type=_path, required=True)
    quantize.add_argument("--bits", type=int, default=4)
    quantize.add_argument("--group-size", type=int, default=64)

    dashboard = commands.add_parser("dashboard", help="Build a visual training report")
    dashboard.add_argument("--run", type=_path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        from .doctor import run_doctor

        return 0 if run_doctor() else 1

    if args.command == "prepare":
        from .data import prepare_data

        config = get_preset(args.preset)
        output = args.output or Path(f"data/prepared/{args.preset}")
        summary = prepare_data(config, output, local_text=args.text, max_stories=args.max_stories)
        (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "inspect":
        tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
        ids = tokenizer.encode(args.text).ids
        if len(ids) < 2:
            raise ValueError("Choose text that produces at least two tokens")
        pieces = [tokenizer.decode([token]) for token in ids]
        print("INPUT  ", " │ ".join(repr(piece) for piece in pieces[:-1]))
        print("TARGET ", " │ ".join(repr(piece) for piece in pieces[1:]))
        print("Every column asks: given everything to the left, what comes next?")
        return 0

    if args.command == "train":
        from .training import train

        data = args.data or Path(f"data/prepared/{args.preset}")
        output = args.output or Path(f"runs/{args.preset}")
        train(get_preset(args.preset), data, output, steps=args.steps)
        return 0

    if args.command == "generate":
        from .generation import generate

        print(
            generate(
                args.checkpoint,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )
        )
        return 0

    if args.command == "finetune":
        from .finetune import finetune

        finetune(
            args.checkpoint,
            args.data,
            args.output,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        return 0

    if args.command == "quantize":
        from .quantization import quantize_checkpoint

        before, after = quantize_checkpoint(
            args.checkpoint,
            args.output,
            bits=args.bits,
            group_size=args.group_size,
        )
        print(
            f"{before / 1024**2:.1f} MB → {after / 1024**2:.1f} MB "
            f"({after / before:.0%} of original)"
        )
        return 0

    if args.command == "dashboard":
        from .dashboard import build_dashboard

        print(build_dashboard(args.run))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
