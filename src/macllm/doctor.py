from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys


def run_doctor() -> bool:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("macOS", platform.system() == "Darwin", platform.platform()))
    checks.append(("Apple silicon", platform.machine() == "arm64", platform.machine()))
    checks.append(
        ("Python 3.11-3.13", (3, 11) <= sys.version_info[:2] < (3, 14), platform.python_version())
    )
    for package in ("mlx", "mlx-lm", "datasets", "tokenizers"):
        try:
            version = importlib.metadata.version(package)
            checks.append((package, True, version))
        except importlib.metadata.PackageNotFoundError:
            checks.append((package, False, "not installed"))
    try:
        memory = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
        memory_gb = int(memory.stdout.strip()) / 1024**3
        checks.append(("Unified memory", memory_gb >= 16, f"{memory_gb:.0f} GB"))
    except (OSError, subprocess.SubprocessError, ValueError):
        checks.append(("Unified memory", False, "could not read"))

    for name, passed, detail in checks:
        print(f"{'✓' if passed else '✗'} {name:<18} {detail}")
    return all(passed for _, passed, _ in checks)
