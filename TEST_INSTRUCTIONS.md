# ✅  Secure Split-FL — Sanity & Correctness Test Plan
This document explains:

1. **Scope** – what we consider “correct” for a Split-Federated-Learning (SFL) run.  
2. **Folder conventions** – where code & results must live.  
3. **Writing tests** – PyTest skeletons you can copy-paste.  
4. **Success criteria** – hard thresholds that decide PASS / FAIL.  
5. **Artefact logging** – how every test must deposit a summary in `experiments/out/`.

> **Context** – The repo already follows a clean layout (`configs/`, `experiments/`, `src/` etc.) and exposes the main runner `experiments/train_secure_sfl.py`. :contentReference[oaicite:0]{index=0}

---

## 1  Environment checklist
```bash
python -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt                  # PyTorch, PyYAML, pytest, …
````

---

## 2  Folder conventions

```
Comp430_Project/
├── configs/               # *.yaml – experiment hyper-params
├── experiments/
│   ├── train_secure_sfl.py
│   ├── tests/             # ←  add all *test_*.py here
│   └── out/               # ←  auto-generated; one sub-dir per run
└── src/                   # framework code
```

*Create `experiments/tests/` and `experiments/out/` if they are missing.*

---

## 3  Writing tests

We rely on **pytest**.
Each test script must:

1. Import the high-level helper `run_experiment()` (defined below).
2. Call it with a YAML file living in `configs/`.
3. *Assert* on metrics returned by the helper.

### 3.1 Helper – `experiments/tests/utils.py`

```python
import json, os, pathlib, shutil, subprocess, uuid, yaml, datetime

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
        "--run_id", run_id                  # make runner accept this flag
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
```

> **💡 Hook required:**
> `train_secure_sfl.py` must accept `--run_id` and write a single JSON file
> (`metrics.json`) with at least:
>
> ```json
> {"final_test_acc": 0.91,
>  "epsilon": 3.8,
>  "delta": 1e-5,
>  "sigma": 1.25,
>  "rounds": 20}
> ```

### 3.2 Sanity pipeline – `test_pipeline.py`

```python
from .utils import run_experiment

def test_forward_backward():
    m = run_experiment("configs/default.yaml")
    # —— Success criteria ——
    assert 0.0 < m["final_test_acc"] < 1.0,  "accuracy not logged"
    assert m["rounds"] >= 1,                 "training never started"
```

### 3.3 DP correctness – `test_dp_budget.py`

```python
from .utils import run_experiment

# fixed-DP budget must not blow up
MAX_EPS = 8.0
def test_fixed_dp():
    m = run_experiment("configs/fixed_dp.yaml")
    assert m["epsilon"] <= MAX_EPS, "ε budget exceeded"

# adaptive-DP must strictly *decrease* σ over time
def test_adaptive_sigma():
    m = run_experiment("configs/adaptive_dp.yaml")
    assert m["sigma"] <  m["sigma_init"], "σ did not decay"
```

### 3.4 Aggregation sanity – `test_fedavg_equiv.py`

A tiny toy dataset (e.g. 200 samples of MNIST) is trained both **centrally** and with
**1 client, 1 round split-FL**.  Losses must match to < 1 × 10⁻³.

```python
import torch, math
from .utils import run_experiment

THRESH = 1e-3
def test_single_client_equivalence():
    m_split = run_experiment("configs/one_client.yaml")
    m_central = run_experiment("configs/central.yaml")
    diff = abs(m_split["final_test_acc"] - m_central["final_test_acc"])
    assert diff < THRESH, f"Split = {m_split}, Central = {m_central}"
```

---

## 4  Success criteria (per test)

| Category                   | Metric / Check                           | PASS Threshold            |        |      |
| -------------------------- | ---------------------------------------- | ------------------------- | ------ | ---- |
| **Pipeline**               | script completes; rounds ≥ 1             | —                         |        |      |
| **Privacy (fixed)**        | ε ≤ `MAX_EPS` (default 8.0)              | True                      |        |      |
| **Privacy (adaptive)**     | σ final < σ initial                      | True                      |        |      |
| **Utility**                | Final test‐accuracy ≥ config’s `min_acc` | True                      |        |      |
| **One-client equivalence** |                                          | Acc\_split − Acc\_central | < 1e-3 | True |

If any assert fails the PyTest run returns **non-zero** and CI marks it red.

---

## 5  Artefact logging

Every test run lands in:

```
experiments/out/
└── <config_stem>__<uuid>/
    ├── metrics.json
    ├── stdout.txt
    ├── stderr.txt
    └── model.pth            # (optional, if runner exports)
```

*Use `uuid` so parallel jobs never clash.  For long-running sweeps you can
garbage-collect sub-dirs older than N days.*

---

## 6  Running the whole suite

```bash
pytest -q experiments/tests
```

A green exit-code 0 ⇒ all sanity & correctness checks hold.

---

## 7  (Opt.) Continuous Integration

Add `.github/workflows/ci.yml`:

```yaml
name: Split-FL tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -r requirements.txt pytest
      - run: pytest -q experiments/tests
```

---

### ✨ Outcome

Following this plan gives you a reproducible pass/fail gate on:

* **Correct Split-FL plumbing** (client → main server → fed server loops).
* **Sound DP accounting** for both fixed and adaptive regimes.
* **Federated AVG correctness** via one-client equivalence.

All results are self-documented inside `experiments/out/`, easing later paper
figures or debugging sessions.
