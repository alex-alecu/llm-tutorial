"""
train.py — Training loop for the minimal GPT.

This script:
  1. Loads the Shakespeare text and builds a character-level tokenizer
  2. Creates train/val splits
  3. Instantiates the GPT model and moves it to the best available device
  4. Trains the model with AdamW, logging loss every 100 iterations
  5. Generates a sample of text after training

Run with:  python train.py
"""

import torch
from model import (
    GPT, build_tokenizer,
    batch_size, block_size, n_embd, n_head, n_layer,
    learning_rate, max_iters, eval_interval, eval_iters,
)

# =============================================================================
# Device Detection
# =============================================================================
# Automatically pick the fastest available hardware:
#   - CUDA  → NVIDIA GPU (fastest)
#   - MPS   → Apple Silicon GPU (fast on Mac)
#   - CPU   → fallback (slowest but always works)
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Using device: {device}")

# =============================================================================
# Data Loading & Tokenization
# =============================================================================
# Read the entire Shakespeare text file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")

# Build the character-level tokenizer
# This creates a unique integer for each of the ~65 characters in Shakespeare
stoi, itos, vocab_size, encode, decode = build_tokenizer(text)
print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Characters: {''.join(sorted(stoi.keys()))}")

# Encode the ENTIRE text into a single tensor of integers
# "First Citizen..." → [18, 47, 56, 57, 58, 1, 15, 47, ...]
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Encoded data shape: {data.shape}")

# =============================================================================
# Train / Validation Split
# =============================================================================
# Use 90% of the data for training, 10% for validation.
# The validation set lets us check if the model is actually learning patterns
# (not just memorizing the training data).
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]
print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")


# =============================================================================
# Batching — How We Feed Data to the Model
# =============================================================================
def get_batch(split: str):
    """
    Grab a random batch of training examples.

    Physical Intuition:
    -------------------
    We carve random windows of text out of the dataset.

    For each window of length `block_size`, the TARGET for position i is
    the character at position i+1. So one window gives us block_size
    training examples simultaneously:

        Input:   "To be or n"      (positions 0-9)
        Target:  "o be or no"      (positions 1-10)

        At position 0: given "T", predict "o"
        At position 1: given "To", predict " "
        At position 2: given "To ", predict "b"
        ... and so on.

    This is incredibly efficient — one forward pass trains on block_size
    different context lengths at once!

    Returns:
        x: (batch_size, block_size) — input token indices
        y: (batch_size, block_size) — target token indices (shifted by 1)
    """
    data_split = train_data if split == 'train' else val_data

    # Pick `batch_size` random starting positions
    # We subtract block_size so we don't run off the end of the text
    ix = torch.randint(len(data_split) - block_size, (batch_size,))

    # Stack the windows into a batch
    x = torch.stack([data_split[i   : i + block_size]     for i in ix])  # inputs
    y = torch.stack([data_split[i+1 : i + block_size + 1] for i in ix])  # targets

    # Move to device (GPU/MPS/CPU)
    x, y = x.to(device), y.to(device)
    return x, y


# =============================================================================
# Loss Estimation — How We Measure Progress
# =============================================================================
@torch.no_grad()  # Disable gradient tracking — we're only measuring, not training
def estimate_loss():
    """
    Estimate the average loss over several batches for train and val sets.

    We average over `eval_iters` batches to get a smoother, more reliable
    estimate (a single batch is too noisy to judge progress).
    """
    out = {}
    model.eval()  # Switch to evaluation mode (disables dropout)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Switch back to training mode
    return out


# =============================================================================
# Model Initialization
# =============================================================================
print(f"\nInitializing GPT with {n_layer} layers, {n_head} heads, {n_embd} embedding dim...")
model = GPT(vocab_size)
model = model.to(device)

# Count parameters — a rough measure of the model's capacity
num_params = sum(p.numel() for p in model.parameters())
print(f"Model has {num_params:,} parameters")

# AdamW optimizer — the standard choice for training Transformers.
# It combines Adam (adaptive learning rates for each parameter) with
# weight decay (L2 regularization to prevent overfitting).
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# =============================================================================
# Training Loop
# =============================================================================
# This is the heart of the training process. For each iteration:
#   1. Grab a random batch of data
#   2. Forward pass: compute predictions and loss
#   3. Backward pass: compute gradients (how to adjust each weight)
#   4. Optimizer step: actually adjust the weights
#
# The loss should steadily decrease from ~4.17 (random guessing among 65 chars,
# since -ln(1/65) ≈ 4.17) down to ~1.5-2.0 (the model has learned English!).

print(f"\nTraining for {max_iters} iterations...\n")

for iter in range(max_iters):

    # Every eval_interval steps, check how we're doing
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

    # --- The four sacred steps of training ---

    # 1. Get a fresh batch of random text windows
    xb, yb = get_batch('train')

    # 2. Forward pass: feed data through the model, get predictions + loss
    logits, loss = model(xb, yb)

    # 3. Zero out old gradients (PyTorch accumulates gradients by default)
    optimizer.zero_grad(set_to_none=True)

    # 4. Backward pass: compute the gradient of the loss w.r.t. every parameter
    #    This is backpropagation — the chain rule applied recursively through
    #    every layer, every attention head, every embedding.
    loss.backward()

    # 5. Optimizer step: nudge every parameter in the direction that reduces loss
    optimizer.step()

# Final loss estimate
losses = estimate_loss()
print(f"Step {max_iters:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

# =============================================================================
# Generation — Let's See What the Model Learned!
# =============================================================================
print("\n" + "=" * 60)
print("GENERATED TEXT (500 characters)")
print("=" * 60 + "\n")

# Start with a single newline character as the "seed"
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate 500 new tokens autoregressively
generated_indices = model.generate(context, max_new_tokens=500)

# Decode back from integers to characters and print
print(decode(generated_indices[0].tolist()))
