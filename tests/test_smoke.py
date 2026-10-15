"""End-to-end smoke pipeline test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_smoke_pipeline(tmp_path: Path) -> None:
    """Full smoke pipeline must complete and produce a valid smoke.json."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_smoke.py"), "--smoke"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Smoke pipeline failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    report_path = ROOT / "reports" / "smoke.json"
    assert report_path.exists(), "smoke.json was not created"

    with open(report_path) as f:
        report = json.load(f)

    required_keys = {"git_commit", "config", "seed", "dataset_scale", "timestamp", "results"}
    missing = required_keys - set(report.keys())
    assert not missing, f"smoke.json missing keys: {missing}"

    required_result_keys = {"popularity", "user_most_listened", "user_most_recent"}
    missing_results = required_result_keys - set(report["results"].keys())
    assert not missing_results, f"smoke.json results missing baselines: {missing_results}"

    scale = report["dataset_scale"]
    assert scale["users"] > 0
    assert scale["items"] > 0
    assert scale["events"] > 0
