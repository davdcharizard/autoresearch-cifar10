#!/usr/bin/env python3
"""EXP-010 parameter sweep: PEAK_LR_MUON for the Muon conv-weight optimizer.

Grid sweep (itertools.product) over a single high-leverage knob — the Muon-group
peak LR — which EXP-009 showed is the only variable that broke the metric (0.24
diverged-and-recovered to 94.11). Each trial is a FULL 300s training run (real
best_test_acc, no proxy), with PEAK_LR_MUON supplied via env var so no tracked
file is edited by a trial. Sequential (single GPU 1). Resumable via trials.tsv.

Direction: higher is better. A failed/crashed trial -> objective 0.0 (worst).
Outputs: trials.tsv (one row per trial) and best.json (winning params+objective).
"""
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# --- Sweep configuration ---------------------------------------------------
OPTIMIZER = "grid"
DIRECTION = "maximize"  # best_test_acc, higher is better
MAX_PARALLEL = 1        # single GPU (GPU 1); trials contend, so run sequentially
PEAK_LR_MUON_GRID = [0.03, 0.06, 0.10, 0.14]  # anchored below EXP-009's divergent 0.24

PROJECT_ROOT = Path("/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8")
EXP_DIR = PROJECT_ROOT / ".autoresearch/goals/maximize-cifar10-test-accuracy/experiments/010"
TRIALS_TSV = EXP_DIR / "trials.tsv"
BEST_JSON = EXP_DIR / "best.json"
LOG_DIR = EXP_DIR / "trial_logs"
WORST = 0.0
PER_TRIAL_TIMEOUT = 600  # external wall kill (the 300s training budget is internal)

# --- Point generation (library, not hand-picked) ---------------------------
POINTS = [{"PEAK_LR_MUON": v} for (v,) in itertools.product(PEAK_LR_MUON_GRID)]


def parse_metric(log_text):
    """Return (best_test_acc, num_epochs, total_seconds) or (None,...) on failure."""
    def grab(pat):
        m = re.search(pat, log_text, re.M)
        return float(m.group(1)) if m else None
    return (
        grab(r"^best_test_acc:\s+([0-9.]+)%"),
        grab(r"^num_epochs:\s+([0-9]+)"),
        grab(r"^total_seconds:\s+([0-9.]+)"),
    )


def load_done():
    """Resume: map trial_index -> row dict for already-completed trials."""
    done = {}
    if TRIALS_TSV.exists():
        for line in TRIALS_TSV.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            done[int(parts[0])] = parts
    return done


def write_trials(rows):
    header = "trial\tpeak_lr_muon\tbest_test_acc\tnum_epochs\ttotal_seconds\tstatus"
    TRIALS_TSV.write_text(header + "\n" + "\n".join("\t".join(map(str, r)) for r in rows) + "\n")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done()
    rows = [done[i] for i in sorted(done)] if done else []

    for idx, point in enumerate(POINTS):
        if idx in done:
            print(f"[trial {idx}] PEAK_LR_MUON={point['PEAK_LR_MUON']} already done -> skip", flush=True)
            continue
        lr = point["PEAK_LR_MUON"]
        log_path = LOG_DIR / f"trial_{idx}_lr{lr}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PEAK_LR_MUON=str(lr))
        print(f"[trial {idx}] launching PEAK_LR_MUON={lr} -> {log_path.name}", flush=True)
        status = "ok"
        try:
            with open(log_path, "w") as fh:
                subprocess.run(
                    ["uv", "run", "train.py"],
                    cwd=str(PROJECT_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
                    timeout=PER_TRIAL_TIMEOUT, check=False,
                )
        except subprocess.TimeoutExpired:
            status = "timeout"
        log_text = log_path.read_text() if log_path.exists() else ""
        best, epochs, secs = parse_metric(log_text)
        if best is None:
            status = "crash" if status == "ok" else status
            obj = WORST
            epochs, secs = epochs or -1, secs or -1
        else:
            obj = best
        rows.append([idx, lr, obj, int(epochs), secs, status])
        write_trials(rows)  # incremental -> resumable
        print(f"[trial {idx}] PEAK_LR_MUON={lr} best_test_acc={obj} epochs={epochs} status={status}", flush=True)

    # Winner = max objective (ties -> lower index, i.e. earlier grid value)
    best_row = max(rows, key=lambda r: (float(r[2]), -int(r[0])))
    best = {
        "trial": int(best_row[0]),
        "params": {"PEAK_LR_MUON": float(best_row[1])},
        "best_test_acc": float(best_row[2]),
        "num_epochs": int(best_row[3]),
        "total_seconds": float(best_row[4]),
        "status": best_row[5],
    }
    BEST_JSON.write_text(json.dumps(best, indent=2) + "\n")
    print("BEST:", json.dumps(best), flush=True)


if __name__ == "__main__":
    sys.exit(main())
