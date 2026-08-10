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
REPORT_PATH = EXPERIMENT_DIR / "timing-report.json"
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def summarize(values):
    return {
        "mean_ms": 1000 * statistics.mean(values),
        "median_ms": 1000 * statistics.median(values),
        "p95_ms": 1000 * percentile(values, 0.95),
        "samples": len(values),
    }


def build_model(arm):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if arm == "control":
        return train.ResNet(3, train.NUM_CLASSES, 2).cuda()
    return train.ResNet(3, train.NUM_CLASSES, 2, 160).cuda()


def timed_step(model, optimizer, cpu_inputs, cpu_targets):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    started = time.perf_counter()
    events[0].record()
    inputs = cpu_inputs.to("cuda", non_blocking=True)
    targets = cpu_targets.to("cuda", non_blocking=True)
    events[1].record()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    events[2].record()
    loss.backward()
    events[3].record()
    optimizer.step()
    events[4].record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    if not torch.isfinite(loss).item():
        raise RuntimeError("non-finite timing loss")
    return {
        "wall": wall,
        "h2d": events[0].elapsed_time(events[1]) / 1000,
        "forward_loss": events[1].elapsed_time(events[2]) / 1000,
        "backward": events[2].elapsed_time(events[3]) / 1000,
        "optimizer": events[3].elapsed_time(events[4]) / 1000,
    }


def inference_step(model, cpu_inputs):
    started = time.perf_counter()
    inputs = cpu_inputs.to("cuda", non_blocking=True)
    with torch.inference_mode():
        outputs = model(inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if outputs.shape != (cpu_inputs.shape[0], 10) or not torch.isfinite(outputs).all():
        raise RuntimeError("invalid inference output")
    return elapsed


def run_workload(arm, batches, conditioning):
    model = build_model(arm)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    pinned = {
        name: tuple(tensor.pin_memory() for tensor in batch)
        for name, batch in batches.items()
    }
    if conditioning:
        warmups, strong_hard, strong_soft, weak_hard = 20, 40, 40, 20
    else:
        warmups, strong_hard, strong_soft, weak_hard = 100, 400, 400, 200

    warm_sequence = ["strong_hard", "strong_soft"]
    for index in range(warmups):
        inputs, targets = pinned[warm_sequence[index % 2]]
        timed_step(model, optimizer, inputs, targets)

    sequence = (
        [name for _ in range(strong_hard) for name in ("strong_hard",)]
        + [name for _ in range(strong_soft) for name in ("strong_soft",)]
        + [name for _ in range(weak_hard) for name in ("weak_hard",)]
    )
    # Interleave strong hard/soft exactly 50/50, then mirror production's weak tail.
    sequence[: strong_hard + strong_soft] = [
        name
        for pair in zip(
            ["strong_hard"] * strong_hard,
            ["strong_soft"] * strong_soft,
            strict=True,
        )
        for name in pair
    ]
    records = {name: [] for name in ("strong_hard", "strong_soft", "weak_hard")}
    torch.cuda.reset_peak_memory_stats()
    for name in sequence:
        inputs, targets = pinned[name]
        records[name].append(timed_step(model, optimizer, inputs, targets))

    all_records = [record for name in sequence for record in [records[name].pop(0)]]
    # Reconstruct per-bucket records after preserving the exact sequence above.
    bucket_records = {name: [] for name in records}
    for name, record in zip(sequence, all_records, strict=True):
        bucket_records[name].append(record)

    model.eval()
    inference_inputs = torch.cat(
        [pinned["weak_hard"][0], pinned["weak_hard"][0]], dim=0
    )
    inference_warmups = 20 if conditioning else 100
    inference_samples = 50 if conditioning else 500
    for _ in range(inference_warmups):
        inference_step(model, inference_inputs)
    inference_times = [
        inference_step(model, inference_inputs) for _ in range(inference_samples)
    ]

    metric_names = ("wall", "h2d", "forward_loss", "backward", "optimizer")
    result = {
        "arm": arm,
        "constructor": (
            "ResNet(3, 10, 2)" if arm == "control" else "ResNet(3, 10, 2, 160)"
        ),
        "warmups": warmups,
        "sequence_counts": {
            "strong_hard": strong_hard,
            "strong_soft": strong_soft,
            "weak_hard": weak_hard,
        },
        "overall": {
            metric: summarize(
                [
                    record[metric]
                    for bucket in bucket_records.values()
                    for record in bucket
                ]
            )
            for metric in metric_names
        },
        "buckets": {
            name: {
                metric: summarize([record[metric] for record in bucket])
                for metric in metric_names
            }
            for name, bucket in bucket_records.items()
        },
        "inference": summarize(inference_times),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
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
        timeout=120,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def parent():
    batch_path = EXPERIMENT_DIR / "timing-batches.pt"
    if not batch_path.exists():
        raise RuntimeError("run preflight_stage_width.py before timing")
    conditioning = run_child("candidate", conditioning=True)
    trials = []
    for trial_index in range(5):
        order = (
            ["control", "candidate"]
            if trial_index % 2 == 0
            else ["candidate", "control"]
        )
        arms = {arm: run_child(arm) for arm in order}
        control_mean = arms["control"]["overall"]["wall"]["mean_ms"]
        candidate_mean = arms["candidate"]["overall"]["wall"]["mean_ms"]
        candidate_p95 = arms["candidate"]["overall"]["wall"]["p95_ms"]
        trials.append(
            {
                "trial": trial_index + 1,
                "order": order,
                "control": arms["control"],
                "candidate": arms["candidate"],
                "ratio": candidate_mean / control_mean,
                "candidate_p95_control_mean_ratio": candidate_p95 / control_mean,
                "inference_ratio": arms["candidate"]["inference"]["mean_ms"]
                / arms["control"]["inference"]["mean_ms"],
            }
        )

    control_means = [trial["control"]["overall"]["wall"]["mean_ms"] for trial in trials]
    candidate_means = [
        trial["candidate"]["overall"]["wall"]["mean_ms"] for trial in trials
    ]
    control_inference = [trial["control"]["inference"]["mean_ms"] for trial in trials]
    candidate_inference = [
        trial["candidate"]["inference"]["mean_ms"] for trial in trials
    ]
    overall_ratio = statistics.mean(candidate_means) / statistics.mean(control_means)
    inference_ratio = statistics.mean(candidate_inference) / statistics.mean(
        control_inference
    )
    candidate_peak = max(trial["candidate"]["peak_vram_mb"] for trial in trials)
    non_training_baseline_seconds = 330.7 - 300.0
    projected_total = 300.0 + non_training_baseline_seconds * max(
        overall_ratio, inference_ratio
    )
    report = {
        "status": "pass",
        "conditioning": conditioning,
        "trials": trials,
        "overall_weighted_ratio": overall_ratio,
        "inference_ratio": inference_ratio,
        "max_pair_ratio": max(trial["ratio"] for trial in trials),
        "max_candidate_p95_control_mean_ratio": max(
            trial["candidate_p95_control_mean_ratio"] for trial in trials
        ),
        "control_trial_mean_cv": coefficient_of_variation(control_means),
        "candidate_trial_mean_cv": coefficient_of_variation(candidate_means),
        "historical_projected_steps": math.floor(26_898 / overall_ratio),
        "candidate_peak_vram_mb": candidate_peak,
        "projection_terms": {
            "training_seconds": 300.0,
            "accepted_non_training_seconds": non_training_baseline_seconds,
            "charged_overhead_ratio": max(overall_ratio, inference_ratio),
        },
        "projected_total_seconds": projected_total,
    }
    failures = []
    if overall_ratio > 1.12:
        failures.append(f"weighted ratio {overall_ratio:.6f} > 1.12")
    if report["max_pair_ratio"] >= 1.15:
        failures.append(f"max pair ratio {report['max_pair_ratio']:.6f} >= 1.15")
    if (
        report["control_trial_mean_cv"] >= 0.02
        or report["candidate_trial_mean_cv"] >= 0.02
    ):
        failures.append("trial-mean CV >= 2%")
    if report["max_candidate_p95_control_mean_ratio"] >= 1.20:
        failures.append("candidate p95/control mean >= 1.20")
    if report["historical_projected_steps"] < 24_000:
        failures.append(
            f"projected steps {report['historical_projected_steps']} < 24000"
        )
    if candidate_peak >= 1024:
        failures.append(f"candidate peak {candidate_peak:.2f} MiB >= 1024")
    if projected_total >= 540:
        failures.append(f"projected total {projected_total:.2f}s >= 540")
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("control", "candidate"))
    parser.add_argument("--conditioning", action="store_true")
    args = parser.parse_args()
    if args.child:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        batches = torch.load(
            EXPERIMENT_DIR / "timing-batches.pt",
            map_location="cpu",
            weights_only=False,
        )
        run_workload(args.child, batches, args.conditioning)
    else:
        parent()


if __name__ == "__main__":
    main()
