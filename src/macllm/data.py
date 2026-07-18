from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .config import TrainConfig


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def _documents_from_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    documents = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(documents) < 2:
        documents = [line.strip() for line in text.splitlines() if line.strip()]
    if len(documents) < 2:
        raise ValueError("Local text needs at least two non-empty documents or lines")
    return documents


def _stream_tiny_stories(split: str, limit: int) -> Iterator[str]:
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        text = row["text"].strip()
        if text:
            yield text


def _write_documents(path: Path, documents: Iterable[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(document.replace("\x00", ""))
            handle.write("\n\n")
            count += 1
    return count


def train_tokenizer(files: list[str], vocab_size: int, output: Path) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train(files, trainer)
    tokenizer.save(str(output))
    return tokenizer


def encode_file(text_path: Path, tokenizer: Tokenizer, output: Path) -> int:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer has no <eos> token")
    total = 0
    with text_path.open("r", encoding="utf-8") as source, output.open("wb") as target:
        buffer: list[str] = []

        def flush() -> None:
            nonlocal total
            if not buffer:
                return
            encoded = tokenizer.encode_batch(buffer)
            token_buffer = array("I")
            for item in encoded:
                token_buffer.extend(item.ids)
                token_buffer.append(eos_id)
            token_buffer.tofile(target)
            total += len(token_buffer)
            buffer.clear()

        document: list[str] = []
        for line in source:
            if line.strip():
                document.append(line)
            elif document:
                buffer.append("".join(document).strip())
                document.clear()
                if len(buffer) >= 1_000:
                    flush()
        if document:
            buffer.append("".join(document).strip())
        flush()
    return total


def prepare_data(
    config: TrainConfig,
    output_dir: Path,
    *,
    local_text: Path | None = None,
    max_stories: int | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; choose another directory or remove it intentionally"
        )
    output_dir.mkdir(parents=True)
    raw_train = output_dir / "train.txt"
    raw_valid = output_dir / "valid.txt"

    if local_text is None:
        train_limit = max_stories or config.max_stories
        train_count = _write_documents(raw_train, _stream_tiny_stories("train", train_limit))
        valid_count = _write_documents(
            raw_valid,
            _stream_tiny_stories("validation", config.validation_stories),
        )
        source = "roneneldan/TinyStories"
    else:
        documents = _documents_from_text(local_text)
        split = max(1, int(len(documents) * 0.95))
        if split == len(documents):
            split -= 1
        train_count = _write_documents(raw_train, documents[:split])
        valid_count = _write_documents(raw_valid, documents[split:])
        source = str(local_text)

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer = train_tokenizer([str(raw_train)], config.tokenizer_vocab, tokenizer_path)
    train_tokens = encode_file(raw_train, tokenizer, output_dir / "train.bin")
    valid_tokens = encode_file(raw_valid, tokenizer, output_dir / "valid.bin")

    raw_train.unlink()
    raw_valid.unlink()
    return {
        "source": source,
        "train_documents": train_count,
        "valid_documents": valid_count,
        "train_tokens": train_tokens,
        "valid_tokens": valid_tokens,
        "vocab_size": tokenizer.get_vocab_size(),
    }


class TokenDataset:
    def __init__(self, path: Path):
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")

    def __len__(self) -> int:
        return len(self.tokens)

    def batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        sequence_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(self.tokens) <= sequence_length + 1:
            raise ValueError(
                f"Dataset has {len(self.tokens)} tokens but needs more than {sequence_length + 1}"
            )
        starts = rng.integers(0, len(self.tokens) - sequence_length - 1, size=batch_size)
        windows = np.stack(
            [self.tokens[start : start + sequence_length + 1] for start in starts]
        ).astype(np.int32)
        return windows[:, :-1], windows[:, 1:]
