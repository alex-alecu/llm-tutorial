from __future__ import annotations

from html import escape
from pathlib import Path
import json


def _points(values: list[float], width: int, height: int, pad: int) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    spread = max(high - low, 1e-9)
    usable_width = width - 2 * pad
    usable_height = height - 2 * pad
    return " ".join(
        f"{pad + index * usable_width / max(1, len(values) - 1):.1f},"
        f"{pad + (high - value) * usable_height / spread:.1f}"
        for index, value in enumerate(values)
    )


def build_dashboard(run_dir: Path) -> Path:
    metrics_path = run_dir / "metrics.jsonl"
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No metrics in {metrics_path}")
    train = [float(item["train_loss"]) for item in records]
    valid = [float(item["val_loss"]) for item in records if "val_loss" in item]
    steps = [int(item["step"]) for item in records]
    latest = records[-1]
    width, height, pad = 900, 320, 36
    valid_line = (
        f'<polyline class="valid" points="{_points(valid, width, height, pad)}" />' if valid else ""
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training dashboard</title>
<style>
:root {{ color-scheme: light dark; --bg:#f7f5ef; --fg:#17212b; --muted:#66707a; --grid:#c8c4b8; --train:#1565c0; --valid:#d1495b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111820; --fg:#edf2f7; --muted:#a5b0bb; --grid:#39434d; --train:#6fb1ff; --valid:#ff8493; }} }}
* {{ box-sizing:border-box }} body {{ margin:0; padding:clamp(16px,4vw,48px); background:var(--bg); color:var(--fg); font:16px/1.5 system-ui,sans-serif }}
main {{ max-width:1000px; margin:auto }} h1 {{ font-size:clamp(24px,5vw,42px); margin:.2em 0 }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:16px; margin:24px 0 }}
.stat {{ border-top:3px solid var(--grid); padding-top:10px }} .value {{ font-size:1.7rem; font-variant-numeric:tabular-nums }}
.label, .legend {{ color:var(--muted) }} svg {{ width:100%; height:auto; overflow:visible }}
.axis {{ stroke:var(--grid); stroke-width:1 }} polyline {{ fill:none; stroke-width:3; stroke-linejoin:round }}
.train {{ stroke:var(--train) }} .valid {{ stroke:var(--valid) }} .swatch {{ display:inline-block; width:28px; border-top:3px solid; margin-right:7px }}
</style>
</head>
<body><main>
<p class="label">{escape(run_dir.name)}</p><h1>Loss is the learning signal</h1>
<div class="stats">
  <div class="stat"><div class="value">{steps[-1]:,}</div><div class="label">steps</div></div>
  <div class="stat"><div class="value">{train[-1]:.3f}</div><div class="label">train loss</div></div>
  <div class="stat"><div class="value">{float(latest.get("val_loss", 0)):.3f}</div><div class="label">validation loss</div></div>
  <div class="stat"><div class="value">{float(latest.get("peak_memory_gb", 0)):.1f} GB</div><div class="label">peak unified memory</div></div>
</div>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Training and validation loss by checkpoint">
  <line class="axis" x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" />
  <line class="axis" x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" />
  <polyline class="train" points="{_points(train, width, height, pad)}" />
  {valid_line}
</svg>
<p class="legend"><span class="swatch" style="border-color:var(--train)"></span>train loss &nbsp; <span class="swatch" style="border-color:var(--valid)"></span>validation loss</p>
</main></body></html>
"""
    output = run_dir / "dashboard.html"
    output.write_text(html, encoding="utf-8")
    return output
