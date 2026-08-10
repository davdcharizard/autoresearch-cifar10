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


def build_model(arm):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if arm == "control":
        return train.ResNet(3, train.NUM_CLASSES, 2).cuda()
    return train.ResNet(2, train.NUM_CLASSES, 3).cuda()


def run_step(model, optimizer, cpu_inputs, cpu_targets):
    started = time.perf_counter()
    inputs = cpu_inputs.to("cuda", non_blocking=True)
    targets = cpu_targets.to("cuda", non_blocking=True)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not torch.isfinite(loss).item():
        raise RuntimeError("non-finite timing loss")
    return elapsed


def run_region(arm, batches, measured_steps, warmup_steps, alternate_soft):
    model = build_model(arm)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=train.LR, momentum=train.MOMENTUM, weight_decay=train.WEIGHT_DECAY
    )
    hard_inputs, hard_targets = batches["hard"]
    soft_inputs, soft_targets = batches["soft"]
    hard_inputs = hard_inputs.pin_memory()
    hard_targets = hard_targets.pin_memory()
    soft_inputs = soft_inputs.pin_memory()
    soft_targets = soft_targets.pin_memory()
    torch.cuda.reset_peak_memory_stats()
    for index in range(warmup_steps):
        use_soft = alternate_soft and index % 2 == 0
        inputs, targets = (soft_inputs, soft_targets) if use_soft else (hard_inputs, hard_targets)
        run_step(model, optimizer, inputs, targets)
    times = []
    for index in range(measured_steps):
        use_soft = alternate_soft and index % 2 == 0
        inputs, targets = (soft_inputs, soft_targets) if use_soft else (hard_inputs, hard_targets)
        times.append(run_step(model, optimizer, inputs, targets))
    result = {
        "mean_ms": 1000 * statistics.mean(times),
        "median_ms": 1000 * statistics.median(times),
        "p95_ms": 1000 * percentile(times, 0.95),
        "steps": measured_steps,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def child(arm, conditioning):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    batches = torch.load(EXPERIMENT_DIR / "timing-batches.pt", map_location="cpu", weights_only=False)
    if conditioning:
        strong_steps, weak_steps, warmup = 80, 20, 20
    else:
        strong_steps, weak_steps, warmup = 800, 200, 100
    result = {
        "arm": arm,
        "constructor": "ResNet(3,10,2)" if arm == "control" else "ResNet(2,10,3)",
        "strong": run_region(arm, batches, strong_steps, warmup, alternate_soft=True),
        "weak": run_region(arm, batches, weak_steps, max(20, warmup // 2), alternate_soft=False),
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
        timeout=100,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def parent():
    if not (EXPERIMENT_DIR / "timing-batches.pt").exists():
        raise RuntimeError("run preflight_rebalance.py before timing")
    conditioning = run_child("candidate", conditioning=True)
    trials = []
    for trial_index in range(5):
        order = ["control", "candidate"] if trial_index % 2 == 0 else ["candidate", "control"]
        arms = {arm: run_child(arm) for arm in order}
        for values in arms.values():
            values["weighted_mean_ms"] = 0.8 * values["strong"]["mean_ms"] + 0.2 * values["weak"]["mean_ms"]
            values["weighted_p95_ms"] = 0.8 * values["strong"]["p95_ms"] + 0.2 * values["weak"]["p95_ms"]
        trials.append(
            {
                "trial": trial_index + 1,
                "order": order,
                "control": arms["control"],
                "candidate": arms["candidate"],
                "ratio": arms["candidate"]["weighted_mean_ms"] / arms["control"]["weighted_mean_ms"],
                "candidate_p95_control_mean_ratio": arms["candidate"]["weighted_p95_ms"]
                / arms["control"]["weighted_mean_ms"],
            }
        )

    control_means = [trial["control"]["weighted_mean_ms"] for trial in trials]
    candidate_means = [trial["candidate"]["weighted_mean_ms"] for trial in trials]
    overall_ratio = statistics.mean(candidate_means) / statistics.mean(control_means)
    candidate_peak = max(
        max(trial["candidate"]["strong"]["peak_vram_mb"], trial["candidate"]["weak"]["peak_vram_mb"])
        for trial in trials
    )
    report = {
        "status": "pass",
        "conditioning": conditioning,
        "trials": trials,
        "overall_weighted_ratio": overall_ratio,
        "max_pair_ratio": max(trial["ratio"] for trial in trials),
        "max_candidate_p95_control_mean_ratio": max(
            trial["candidate_p95_control_mean_ratio"] for trial in trials
        ),
        "control_trial_mean_cv": coefficient_of_variation(control_means),
        "candidate_trial_mean_cv": coefficient_of_variation(candidate_means),
        "historical_projected_steps": math.floor(26_898 / overall_ratio),
        "candidate_peak_vram_mb": candidate_peak,
        "projected_total_seconds": 330.7 * overall_ratio,
    }
    report_path = EXPERIMENT_DIR / "timing-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    failures = []
    if overall_ratio > 1.345:
        failures.append(f"weighted ratio {overall_ratio:.6f} > 1.345")
    if report["max_pair_ratio"] >= 1.38:
        failures.append(f"max pair ratio {report['max_pair_ratio']:.6f} >= 1.38")
    if report["control_trial_mean_cv"] > 0.02 or report["candidate_trial_mean_cv"] > 0.02:
        failures.append("trial-mean CV > 2%")
    if report["max_candidate_p95_control_mean_ratio"] > 1.45:
        failures.append("candidate p95/control mean > 1.45")
    if candidate_peak >= 1280:
        failures.append(f"candidate peak {candidate_peak:.2f} MiB >= 1280")
    if report["historical_projected_steps"] < 20_000:
        failures.append(f"projected steps {report['historical_projected_steps']} < 20000")
    if report["projected_total_seconds"] >= 540:
        failures.append(f"projected total {report['projected_total_seconds']:.2f}s >= 540")
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError("; ".join(failures))
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
