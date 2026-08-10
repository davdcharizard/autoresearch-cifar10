#!/usr/bin/env python3
"""Resumable grid sweep for EXP-003 using ephemeral train.py variants."""

import csv
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.py"
TRIALS_PATH = EXPERIMENT_DIR / "trials.tsv"
BEST_PATH = EXPERIMENT_DIR / "best.json"
TIMEOUT_SECONDS = 600
FAILED_OBJECTIVE = -1_000_000_000.0

PARAM_SPACE = {
    "cutmix_prob": {"values": [0.35, 0.50, 0.65]},
    "max_drop_path": {"values": [0.04, 0.08]},
}
PARENT_PARAMS = {"cutmix_prob": 0.50, "max_drop_path": 0.08}
PARENT_RESULT = {
    "objective": 95.23,
    "status": "parent",
    "final_test_acc": 95.19,
    "final_test_loss": 0.2044,
    "training_seconds": 300.0,
    "total_seconds": 467.1,
    "startup_seconds": "",
    "peak_vram_mb": 1178.9,
    "num_epochs": 144,
    "num_steps": 27950,
    "num_params": 2748890,
}
FIELDS = [
    "trial_idx",
    "cutmix_prob",
    "max_drop_path",
    "objective",
    "status",
    "final_test_acc",
    "final_test_loss",
    "training_seconds",
    "total_seconds",
    "startup_seconds",
    "peak_vram_mb",
    "num_epochs",
    "num_steps",
    "num_params",
    "return_code",
    "error",
]
SUMMARY_PATTERNS = {
    "objective": r"^best_test_acc:\s+([0-9.]+)%$",
    "final_test_acc": r"^final_test_acc:\s+([0-9.]+)%$",
    "final_test_loss": r"^final_test_loss:\s+([0-9.]+)$",
    "training_seconds": r"^training_seconds:\s+([0-9.]+)$",
    "total_seconds": r"^total_seconds:\s+([0-9.]+)$",
    "startup_seconds": r"^startup_seconds:\s+([0-9.]+)$",
    "peak_vram_mb": r"^peak_vram_mb:\s+([0-9.]+)$",
    "num_epochs": r"^num_epochs:\s+([0-9]+)$",
    "num_steps": r"^num_steps:\s+([0-9]+)$",
    "num_params": r"^num_params:\s+([0-9,]+)$",
}


def grid_points():
    names = list(PARAM_SPACE)
    axes = [PARAM_SPACE[name]["values"] for name in names]
    return [
        dict(zip(names, values, strict=False))
        for values in itertools.product(*axes)
    ]


def read_rows():
    if not TRIALS_PATH.exists():
        return []
    with TRIALS_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_row(row):
    new_file = not TRIALS_PATH.exists()
    with TRIALS_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def update_best():
    valid = [
        row
        for row in read_rows()
        if row["status"] in {"ok", "parent"}
        and float(row["objective"]) > FAILED_OBJECTIVE
    ]
    if not valid:
        return
    winner = max(valid, key=lambda row: float(row["objective"]))
    payload = {
        "trial_idx": int(winner["trial_idx"]),
        "params": {
            "cutmix_prob": float(winner["cutmix_prob"]),
            "max_drop_path": float(winner["max_drop_path"]),
        },
        "objective": float(winner["objective"]),
        "status": winner["status"],
    }
    temporary = BEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(BEST_PATH)


def parse_summary(output):
    parsed = {}
    for name, pattern in SUMMARY_PATTERNS.items():
        match = re.search(pattern, output, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing summary field: {name}")
        value = match.group(1).replace(",", "")
        parsed[name] = int(value) if name.startswith("num_") else float(value)
    return parsed


def trial_source(params):
    source = TRAIN_PATH.read_text()
    replacements = {
        "MAX_DROP_PATH = 0.08": f"MAX_DROP_PATH = {params['max_drop_path']}",
        "CUTMIX_PROB = 0.5": f"CUTMIX_PROB = {params['cutmix_prob']}",
    }
    for original, replacement in replacements.items():
        if source.count(original) != 1:
            raise RuntimeError(f"expected exactly one occurrence of {original!r}")
        source = source.replace(original, replacement)
    return source


def run_trial(index, params):
    script_path = EXPERIMENT_DIR / f".trial-{index:02d}.py"
    log_path = EXPERIMENT_DIR / f".trial-{index:02d}.log"
    script_path.write_text(trial_source(params))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = str(ROOT)

    return_code = -1
    error = ""
    try:
        with log_path.open("w") as log_handle:
            process = subprocess.Popen(
                ["uv", "run", "python", str(script_path)],
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                error = f"timeout after {TIMEOUT_SECONDS}s"

        output = log_path.read_text(errors="replace")
        if return_code != 0:
            error = error or f"process exited {return_code}"
            raise RuntimeError(error)
        result = parse_summary(output)
        result.update(status="ok", return_code=return_code, error="")
        return result
    except Exception as exc:
        excerpt = ""
        if log_path.exists():
            excerpt = " | ".join(log_path.read_text(errors="replace").splitlines()[-5:])
        return {
            "objective": FAILED_OBJECTIVE,
            "status": "failed",
            "return_code": return_code,
            "error": f"{exc}: {excerpt}"[:1000],
        }
    finally:
        script_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)


def main():
    points = grid_points()
    completed = {int(row["trial_idx"]) for row in read_rows()}

    for index, params in enumerate(points):
        if index in completed:
            print(f"trial {index}: already complete", flush=True)
            continue
        row = {"trial_idx": index, **params}
        if params == PARENT_PARAMS:
            result = {**PARENT_RESULT, "return_code": 0, "error": ""}
            print(f"trial {index}: reuse parent objective={result['objective']:.2f}", flush=True)
        else:
            print(
                f"trial {index}: start cutmix_prob={params['cutmix_prob']} "
                f"max_drop_path={params['max_drop_path']} GPU=0",
                flush=True,
            )
            result = run_trial(index, params)
            print(
                f"trial {index}: {result['status']} "
                f"objective={result['objective']:.2f}",
                flush=True,
            )
        append_row({**row, **result})
        update_best()

    update_best()
    print(f"complete: {len(read_rows())}/{len(points)} trials", flush=True)
    print(BEST_PATH.read_text().strip(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
