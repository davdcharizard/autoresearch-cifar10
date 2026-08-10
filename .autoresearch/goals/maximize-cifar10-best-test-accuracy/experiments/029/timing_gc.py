import argparse
import gc
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
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORT_PATH = EXPERIMENT_DIR / "timing-report.json"
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(values):
    return {
        "count": len(values),
        "mean_ms": 1000 * statistics.mean(values),
        "median_ms": 1000 * statistics.median(values),
        "p95_ms": 1000 * percentile(values, 0.95),
    }


def coefficient_of_variation(values):
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if len(values) > 1 else 0.0


def make_transforms():
    mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)
    weak = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    strong = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return strong, weak


def next_batch(loader, iterator):
    started = time.perf_counter()
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator, time.perf_counter() - started


def train_step(model, optimizer, cpu_inputs, cpu_targets, candidate):
    started = time.perf_counter()
    inputs = cpu_inputs.to("cuda", non_blocking=True)
    targets = cpu_targets.to("cuda", non_blocking=True)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    if candidate:
        train.centralize_conv_weight_gradients(model)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not torch.isfinite(loss).item():
        raise RuntimeError("non-finite timing loss")
    return elapsed


def run_steps(
    model,
    optimizer,
    loader,
    iterator,
    candidate,
    warmups,
    measured,
    first_batch=None,
):
    for index in range(warmups):
        if index == 0 and first_batch is not None:
            cpu_inputs, cpu_targets = first_batch
        else:
            (cpu_inputs, cpu_targets), iterator, _ = next_batch(loader, iterator)
        train_step(model, optimizer, cpu_inputs, cpu_targets, candidate)

    hard_times = []
    soft_times = []
    waits = []
    for _ in range(measured):
        (cpu_inputs, cpu_targets), iterator, wait = next_batch(loader, iterator)
        elapsed = train_step(model, optimizer, cpu_inputs, cpu_targets, candidate)
        waits.append(wait)
        (soft_times if cpu_targets.ndim == 2 else hard_times).append(elapsed)
    return {
        "hard": summarize(hard_times) if hard_times else None,
        "soft": summarize(soft_times) if soft_times else None,
        "all": summarize(hard_times + soft_times),
        "wait": summarize(waits),
    }, iterator


def child(arm, conditioning):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    candidate = arm == "candidate"
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    strong_transform, weak_transform = make_transforms()
    model = train.ResNet(
        train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER
    ).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    strong_loader = train.make_train_loader(
        strong_transform, collate_fn=train.cutmix_collate
    )
    strong_iterator = iter(strong_loader)
    if conditioning:
        strong_warmups, strong_measured = 20, 80
        weak_warmups, weak_measured = 5, 20
    else:
        strong_warmups, strong_measured = 100, 800
        weak_warmups, weak_measured = 20, 200

    process_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    strong, strong_iterator = run_steps(
        model,
        optimizer,
        strong_loader,
        strong_iterator,
        candidate,
        strong_warmups,
        strong_measured,
    )
    strong_iterator = None
    switch_started = time.perf_counter()
    strong_stopped = train.shutdown_train_loader(strong_loader)
    del strong_loader
    gc.collect()
    weak_loader = train.make_train_loader(weak_transform)
    weak_iterator = iter(weak_loader)
    first_weak, weak_iterator, _ = next_batch(weak_loader, weak_iterator)
    weak_rebuild_seconds = time.perf_counter() - switch_started
    weak, weak_iterator = run_steps(
        model,
        optimizer,
        weak_loader,
        weak_iterator,
        candidate,
        weak_warmups,
        weak_measured,
        first_batch=first_weak,
    )
    weak_iterator = None
    weak_stopped = train.shutdown_train_loader(weak_loader)
    process_seconds = time.perf_counter() - process_started
    if len(strong_stopped) != train.NUM_WORKERS or len(weak_stopped) != train.NUM_WORKERS:
        raise RuntimeError("timing worker shutdown count mismatch")
    if strong["hard"] is None or strong["soft"] is None or weak["soft"] is not None:
        raise RuntimeError("timing target-path coverage mismatch")
    result = {
        "arm": arm,
        "conditioning": conditioning,
        "strong": strong,
        "weak": weak,
        "weak_rebuild_seconds": weak_rebuild_seconds,
        "strong_workers_stopped": len(strong_stopped),
        "weak_workers_stopped": len(weak_stopped),
        "process_seconds": process_seconds,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }
    print(json.dumps(result))


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


def weighted_mean(arm):
    return (
        0.4 * arm["strong"]["hard"]["mean_ms"]
        + 0.4 * arm["strong"]["soft"]["mean_ms"]
        + 0.2 * arm["weak"]["hard"]["mean_ms"]
    )


def write_report(report):
    with REPORT_PATH.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parent():
    conditioning = run_child("candidate", conditioning=True)
    trials = []
    for trial_index in range(5):
        order = (
            ["control", "candidate"]
            if trial_index % 2 == 0
            else ["candidate", "control"]
        )
        arms = {arm: run_child(arm) for arm in order}
        control_mean = weighted_mean(arms["control"])
        candidate_mean = weighted_mean(arms["candidate"])
        trials.append(
            {
                "trial": trial_index + 1,
                "order": order,
                "control": arms["control"],
                "candidate": arms["candidate"],
                "control_weighted_mean_ms": control_mean,
                "candidate_weighted_mean_ms": candidate_mean,
                "ratio": candidate_mean / control_mean,
            }
        )

    control_means = [trial["control_weighted_mean_ms"] for trial in trials]
    candidate_means = [trial["candidate_weighted_mean_ms"] for trial in trials]
    overall_ratio = statistics.mean(candidate_means) / statistics.mean(control_means)
    candidate_peak = max(trial["candidate"]["peak_vram_mb"] for trial in trials)
    candidate_wait_medians = [
        max(
            trial["candidate"]["strong"]["wait"]["median_ms"],
            trial["candidate"]["weak"]["wait"]["median_ms"],
        )
        for trial in trials
    ]
    candidate_wait_p95s = [
        max(
            trial["candidate"]["strong"]["wait"]["p95_ms"],
            trial["candidate"]["weak"]["wait"]["p95_ms"],
        )
        for trial in trials
    ]
    max_rebuild = max(trial["candidate"]["weak_rebuild_seconds"] for trial in trials)
    candidate_mean_ms = statistics.mean(candidate_means)
    delivery_ratio = candidate_mean_ms / max(
        statistics.mean(candidate_wait_medians), 1e-30
    )
    report = {
        "status": "pass",
        "conditioning": conditioning,
        "trials": trials,
        "overall_weighted_ratio": overall_ratio,
        "max_pair_ratio": max(trial["ratio"] for trial in trials),
        "control_trial_mean_cv": coefficient_of_variation(control_means),
        "candidate_trial_mean_cv": coefficient_of_variation(candidate_means),
        "historical_projected_steps": math.floor(26_898 / overall_ratio),
        "candidate_peak_vram_mb": candidate_peak,
        "candidate_delivery_consumption_ratio": delivery_ratio,
        "candidate_max_wait_median_ms": max(candidate_wait_medians),
        "candidate_max_wait_p95_ms": max(candidate_wait_p95s),
        "candidate_mean_step_ms": candidate_mean_ms,
        "max_weak_rebuild_seconds": max_rebuild,
        "projected_training_lifecycle_wall_count_ratio": (300.0 + max_rebuild) / 300.0,
        "projected_total_seconds": 330.7 + max_rebuild,
    }
    write_report(report)

    failures = []
    if overall_ratio > 1.01:
        failures.append("aggregate timing ratio")
    if report["max_pair_ratio"] > 1.04:
        failures.append("paired timing ratio")
    if report["control_trial_mean_cv"] >= 0.03 or report["candidate_trial_mean_cv"] >= 0.03:
        failures.append("trial-mean CV")
    if candidate_peak >= 650:
        failures.append("peak memory")
    if delivery_ratio < 1.2:
        failures.append("loader delivery ratio")
    if report["candidate_max_wait_median_ms"] >= 0.10 * candidate_mean_ms:
        failures.append("median iterator wait")
    if report["candidate_max_wait_p95_ms"] >= 0.20 * candidate_mean_ms:
        failures.append("p95 iterator wait")
    if max_rebuild >= 5.0:
        failures.append("weak loader rebuild")
    if report["projected_training_lifecycle_wall_count_ratio"] > 1.07:
        failures.append("training lifecycle wall/count")
    if report["projected_total_seconds"] >= 540:
        failures.append("projected total runtime")
    if any(
        trial[arm]["strong_workers_stopped"] != train.NUM_WORKERS
        or trial[arm]["weak_workers_stopped"] != train.NUM_WORKERS
        for trial in trials
        for arm in ("control", "candidate")
    ):
        failures.append("worker lifecycle")
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
        write_report(report)
        raise RuntimeError(f"timing gates failed: {failures}")
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
