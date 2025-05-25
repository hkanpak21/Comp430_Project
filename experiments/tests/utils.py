import json
import os
import pathlib
import shutil
import subprocess
import uuid
import yaml
import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "out"

def run_experiment(cfg_path: str):
    """Launch one SFL run, return metrics dict."""
    run_id = f"{pathlib.Path(cfg_path).stem}__{uuid.uuid4().hex[:6]}"
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. spawn training as subprocess
    cmd = [
        "python", "experiments/train_secure_sfl.py",
        "--config", cfg_path,
        "--run_id", run_id                  
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # 2. collect metrics saved by the runner
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file) as f:
        metrics = json.load(f)

    # 3. additionally log stdout / stderr
    (run_dir / "stdout.txt").write_text(result.stdout)
    (run_dir / "stderr.txt").write_text(result.stderr)

    return metrics
