from pathlib import Path
import json

from macllm.dashboard import build_dashboard


def test_dashboard_contains_metrics(tmp_path: Path):
    records = [
        {"step": 1, "train_loss": 4.2, "val_loss": 4.3, "peak_memory_gb": 1.2},
        {"step": 10, "train_loss": 3.1, "val_loss": 3.2, "peak_memory_gb": 1.3},
    ]
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(json.dumps(item) for item in records), encoding="utf-8"
    )
    output = build_dashboard(tmp_path)
    html = output.read_text(encoding="utf-8")
    assert "3.100" in html
    assert "validation loss" in html
    assert "<polyline" in html
