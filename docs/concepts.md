# The model, drawn before it is coded

## One training example is many predictions

```text
input   Once | upon | a | time
target  upon | a    | time | ,
          ↑      ↑      ↑    ↑
        four supervised next-token predictions in one pass
```

The causal mask prevents a token from reading the answer to its right:

```text
keys →       Once  upon  a  time
query Once     ●     ·   ·    ·
      upon     ●     ●   ·    ·
      a        ●     ●   ●    ·
      time     ●     ●   ●    ●

● visible    · hidden
```

## One transformer block

```mermaid
flowchart LR
    X["token vectors"] --> N1["RMSNorm"]
    N1 --> A["causal grouped-query attention"]
    X --> R1((+))
    A --> R1
    R1 --> N2["RMSNorm"]
    N2 --> F["SwiGLU feed-forward"]
    R1 --> R2((+))
    F --> R2
    R2 --> Y["richer token vectors"]
```

- **Attention is communication.** Each position chooses useful earlier positions.
- **Feed-forward is computation.** Each position transforms what it collected.
- **Residual `+` paths are memory highways.** New information is added without erasing the old representation.
- **RMSNorm controls scale.** It keeps deep updates numerically manageable.
- **RoPE gives order.** Query and key vectors rotate according to their positions.
- **Grouped-query attention shares keys and values.** It keeps most of multi-head attention's flexibility with less memory.

The full model repeats this block, normalizes once more, then uses the token embedding table backward as the next-token classifier. Sharing that table saves parameters.

## Why these three presets?

Model weights fit easily in 48 GB. Training time is the real constraint because every token touches nearly every parameter during forward and backward passes.

```mermaid
quadrantChart
    title Local pretraining trade-off
    x-axis Short wait --> Long wait
    y-axis Pipeline lesson --> Better story quality
    quadrant-1 Bigger learning project
    quadrant-2 Ideal
    quadrant-3 Smoke tests
    quadrant-4 Too expensive locally
    quick: [0.18, 0.28]
    standard: [0.52, 0.72]
    overnight: [0.84, 0.84]
```

The `standard` preset is the practical center. Scaling much beyond it without far more high-quality data mostly buys waiting.
