from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import torch

from model import GPT, GPTConfig


# Fast demo defaults: small enough to run quickly, but still a real Transformer.
BLOCK_SIZE = 64
BATCH_SIZE = 32
MAX_ITERS = 2000
EVAL_INTERVAL = 100
LEARNING_RATE = 3e-4
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
MAX_NEW_TOKENS = 300

DATA_PATH = Path("input.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_device() -> torch.device:
    """Pick the fastest available backend in priority order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download_if_needed(path: str) -> None:
    """
    Download Tiny Shakespeare if the file is missing or effectively empty.
    If download fails, raise a clear error telling the user to place input.txt manually.
    """
    data_path = Path(path)
    needs_download = not data_path.exists()

    if not needs_download:
        text = data_path.read_text(encoding="utf-8")
        needs_download = not text.strip()

    if not needs_download:
        print(f"Using existing dataset at {data_path}.")
        return

    print(f"Downloading Tiny Shakespeare to {data_path} ...")
    try:
        with urlopen(DATA_URL, timeout=30) as response:
            text = response.read().decode("utf-8")
    except URLError as exc:
        raise RuntimeError(
            "Failed to download dataset. Please place your own text in input.txt and rerun."
        ) from exc

    data_path.write_text(text, encoding="utf-8")
    print(f"Downloaded {len(text):,} characters.")


def build_tokenizer(text: str):
    """Character-level tokenizer: tiny vocab, easy to reason about."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    def decode(ids: list[int]) -> str:
        return "".join(itos[i] for i in ids)

    return stoi, itos, encode, decode


def get_batch(
    split_data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random training windows.
    x: current tokens
    y: same window shifted by 1 char (the next-token targets)
    """
    if len(split_data) <= block_size:
        raise ValueError("Dataset split is too small for the configured block_size.")

    starts = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i : i + block_size] for i in starts])
    y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def main() -> None:
    torch.manual_seed(1337)
    device = get_device()
    print(f"Device: {device}")

    download_if_needed(str(DATA_PATH))
    text = DATA_PATH.read_text(encoding="utf-8")

    stoi, itos, encode, decode = build_tokenizer(text)
    vocab_size = len(stoi)
    print(f"Loaded {len(text):,} characters with vocab size {vocab_size}.")

    data = torch.tensor(encode(text), dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]  # Kept for future experiments; training loop stays minimal.
    print(f"Train chars: {len(train_data):,} | Val chars: {len(val_data):,}")

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for step in range(MAX_ITERS + 1):
        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, device)

        # Forward pass: model predicts next-char logits and computes CE loss.
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Loss should never be None during training.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    model.eval()
    seed_text = "ROMEO: "
    seed_ids = encode(seed_text)
    context = torch.tensor([seed_ids], dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()
    generated_text = decode(generated_ids)

    print("\n--- Generated sample ---")
    print(generated_text)


if __name__ == "__main__":
    main()
