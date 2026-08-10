import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def train_step(model, optimizer, fast, slow, completed_step, cpu_inputs, cpu_targets, candidate):
    started = time.perf_counter()
    inputs = cpu_inputs.to("cuda", non_blocking=True)
    targets = cpu_targets.to("cuda", non_blocking=True)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    optimizer.step()
    if candidate and completed_step % train.LOOKAHEAD_K == 0:
        with torch.no_grad():
            torch._foreach_lerp_(slow, fast, train.LOOKAHEAD_ALPHA)
            torch._foreach_copy_(fast, slow)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not torch.isfinite(loss).item():
        raise RuntimeError("non-finite timing loss")
    return elapsed


def run_region(candidate, batches, measured_steps, warmup_steps, alternate_soft):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER).cuda()
    fast = list(model.parameters())
    slow = [parameter.detach().clone() for parameter in fast] if candidate else []
    optimizer = torch.optim.SGD(
        fast, lr=train.LR, momentum=train.MOMENTUM, weight_decay=train.WEIGHT_DECAY
    )
    hard_inputs, hard_targets = batches["hard"]
    soft_inputs, soft_targets = batches["soft"]
    hard_inputs = hard_inputs.pin_memory()
    hard_targets = hard_targets.pin_memory()
    soft_inputs = soft_inputs.pin_memory()
    soft_targets = soft_targets.pin_memory()
    completed_step = 0
    torch.cuda.reset_peak_memory_stats()
    for index in range(warmup_steps):
        completed_step += 1
        use_soft = alternate_soft and index % 2 == 0
        inputs, targets = (soft_inputs, soft_targets) if use_soft else (hard_inputs, hard_targets)
        train_step(model, optimizer, fast, slow, completed_step, inputs, targets, candidate)
    times = []
    for index in range(measured_steps):
        completed_step += 1
        use_soft = alternate_soft and index % 2 == 0
        inputs, targets = (soft_inputs, soft_targets) if use_soft else (hard_inputs, hard_targets)
        times.append(train_step(model, optimizer, fast, slow, completed_step, inputs, targets, candidate))
    result = {
        "mean_ms": 1000 * statistics.mean(times),
        "median_ms": 1000 * statistics.median(times),
        "p95_ms": 1000 * percentile(times, 0.95),
        "steps": measured_steps,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }
    del model, optimizer, fast, slow
    torch.cuda.empty_cache()
    return result


def child(arm, conditioning):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    batches = torch.load(EXPERIMENT_DIR / "timing-batches.pt", map_location="cpu", weights_only=False)
    candidate = arm == "candidate"
    if conditioning:
        strong_steps, weak_steps, warmup = 80, 20, 20
    else:
        strong_steps, weak_steps, warmup = 800, 200, 100
    result = {
        "arm": arm,
        "strong": run_region(candidate, batches, strong_steps, warmup, alternate_soft=True),
        "weak": run_region(candidate, batches, weak_steps, max(20, warmup // 2), alternate_soft=False),
    }
    print(json.dumps(result))


def coefficient_of_variation(values):
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if len(values) > 1 else 0.0


def run_child(arm, conditioning=False):
    command = [sys.executable, str(Path(__file__).resolve()), "--child", arm]
    if conditioning:
        command.append("--conditioning")
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def parent():
    if not (EXPERIMENT_DIR / "timing-batches.pt").exists():
        raise RuntimeError("run preflight_lookahead.py before timing")
    conditioning = run_child("candidate", conditioning=True)
    trials = []
    for trial_index in range(5):
        order = ["control", "candidate"] if trial_index % 2 == 0 else ["candidate", "control"]
        arms = {}
        for arm in order:
            arms[arm] = run_child(arm)
        control_weighted = 0.8 * arms["control"]["strong"]["mean_ms"] + 0.2 * arms["control"]["weak"]["mean_ms"]
        candidate_weighted = 0.8 * arms["candidate"]["strong"]["mean_ms"] + 0.2 * arms["candidate"]["weak"]["mean_ms"]
        trials.append(
            {
                "trial": trial_index + 1,
                "order": order,
                "control": arms["control"],
                "candidate": arms["candidate"],
                "control_weighted_mean_ms": control_weighted,
                "candidate_weighted_mean_ms": candidate_weighted,
                "ratio": candidate_weighted / control_weighted,
            }
        )

    control_means = [trial["control_weighted_mean_ms"] for trial in trials]
    candidate_means = [trial["candidate_weighted_mean_ms"] for trial in trials]
    overall_ratio = statistics.mean(candidate_means) / statistics.mean(control_means)
    projected_steps = math.floor(26_898 / overall_ratio)
    candidate_peak = max(
        max(trial["candidate"]["strong"]["peak_vram_mb"], trial["candidate"]["weak"]["peak_vram_mb"])
        for trial in trials
    )
    report = {
        "status": "pass",
        "conditioning": conditioning,
        "trials": trials,
        "overall_weighted_ratio": overall_ratio,
        "control_trial_mean_cv": coefficient_of_variation(control_means),
        "candidate_trial_mean_cv": coefficient_of_variation(candidate_means),
        "historical_projected_steps": projected_steps,
        "candidate_peak_vram_mb": candidate_peak,
        "projected_total_seconds": 330.7 * overall_ratio,
    }
    report_path = EXPERIMENT_DIR / "timing-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if overall_ratio > 1.01:
        raise RuntimeError(f"weighted timing ratio {overall_ratio:.6f} exceeded 1.01")
    if projected_steps < 26_629:
        raise RuntimeError(f"historical projected steps {projected_steps} below 26629")
    if report["control_trial_mean_cv"] > 0.02 or report["candidate_trial_mean_cv"] > 0.02:
        raise RuntimeError("trial-mean timing CV exceeded 2%")
    if candidate_peak >= 650:
        raise RuntimeError(f"candidate peak {candidate_peak:.2f} MiB exceeded limit")
    if report["projected_total_seconds"] >= 540:
        raise RuntimeError("projected total runtime exceeded limit")
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("control", "candidate"))
    parser.add_argument("--conditioning", action="store_true")
    args = parser.parse_args()
    if args.child:
        child(args.child, args.conditioning)
    else:
        parent()


if __name__ == "__main__":
    main()
