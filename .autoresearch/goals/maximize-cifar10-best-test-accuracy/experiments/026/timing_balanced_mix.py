import argparse
import json
import math
import multiprocessing
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORT_PATH = EXPERIMENT_DIR / "timing-report.json"
PROGRESS_PATH = EXPERIMENT_DIR / "timing-progress.json"
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


THRESHOLDS = {
    "mean_counted_ratio": 1.01,
    "pair_counted_ratio": 1.04,
    "trial_cv": 0.03,
    "projected_steps": 26_629,
    "loader_headroom": 1.20,
    "median_wait_gpu_fraction": 0.10,
    "p95_wait_gpu_fraction": 0.20,
    "wall_counted_ratio": 1.07,
    "wall_counted_control_delta": 0.02,
    "hard_fraction_min": 0.485,
    "hard_fraction_max": 0.515,
    "cutmix_fraction_min": 0.235,
    "cutmix_fraction_max": 0.265,
    "mixup_fraction_min": 0.235,
    "mixup_fraction_max": 0.265,
    "peak_vram_mb": 650.0,
    "allocation_growth_mb": 1.0,
    "weak_rebuild_seconds": 5.0,
    "projected_total_seconds": 540.0,
}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def summary(values):
    return {
        "mean_ms": 1_000 * statistics.mean(values),
        "median_ms": 1_000 * statistics.median(values),
        "p95_ms": 1_000 * percentile(values, 0.95),
        "count": len(values),
    }


def coefficient_of_variation(values):
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if len(values) > 1 else 0.0


def backend_flags():
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def configure_backend():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def control_policy(inputs, targets):
    draw = torch.rand(()).item()
    if draw < 0.5:
        inputs, targets = train.cutmix(inputs, targets)
        kind = train.CUTMIX
    else:
        kind = train.HARD
    return inputs, targets, kind


def control_collate(batch):
    inputs, targets = default_collate(batch)
    with torch.random.fork_rng(devices=[]):
        return control_policy(inputs, targets)


def strong_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
        ]
    )


def weak_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
        ]
    )


def validate_batch(inputs, targets, kind=None):
    if inputs.shape != (128, 3, 32, 32) or inputs.dtype != torch.float32:
        raise RuntimeError("invalid timing input contract")
    if kind is None or kind == train.HARD:
        if targets.shape != (128,) or targets.dtype != torch.int64:
            raise RuntimeError("invalid hard target contract")
    elif (
        targets.shape != (128, 10)
        or not torch.is_floating_point(targets)
        or not torch.isfinite(targets).all()
        or targets.min() < 0
        or not torch.allclose(targets.sum(1), torch.ones(128), atol=1e-6, rtol=0)
    ):
        raise RuntimeError("invalid mixed target contract")


def next_strong(loader, iterator):
    started = time.perf_counter()
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    waited = time.perf_counter() - started
    inputs, targets, kind = batch
    validate_batch(inputs, targets, kind)
    return iterator, inputs, targets, kind, waited


def next_weak(loader, iterator):
    started = time.perf_counter()
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    waited = time.perf_counter() - started
    inputs, targets = batch
    validate_batch(inputs, targets)
    return iterator, inputs, targets, waited


def train_step(model, optimizer, cpu_inputs, cpu_targets):
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


def stop_loader(loader):
    workers = train.shutdown_train_loader(loader)
    if len(workers) != train.NUM_WORKERS:
        raise RuntimeError(
            f"stopped {len(workers)} workers, expected {train.NUM_WORKERS}"
        )
    return workers


def run_workload(arm, conditioning):
    configure_backend()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    collator = control_collate if arm == "control" else train.mixed_collate
    strong_loader = train.make_train_loader(strong_transform(), collate_fn=collator)
    model = (
        train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER)
        .cuda()
        .train()
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    strong_iterator = iter(strong_loader)
    warmups, measured, weak_measured = (
        (20, 100, 20) if conditioning else (100, 1_000, 100)
    )
    counts = [0, 0, 0]
    for _ in range(warmups):
        strong_iterator, inputs, targets, kind, _wait = next_strong(
            strong_loader, strong_iterator
        )
        counts[kind] += 1
        train_step(model, optimizer, inputs, targets)

    allocated_after_warmup = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    strong_waits = []
    strong_gpu = []
    integrated_started = time.perf_counter()
    for _ in range(measured):
        strong_iterator, inputs, targets, kind, wait = next_strong(
            strong_loader, strong_iterator
        )
        counts[kind] += 1
        strong_waits.append(wait)
        strong_gpu.append(train_step(model, optimizer, inputs, targets))
    strong_iterator = None
    strong_workers = stop_loader(strong_loader)

    rebuild_started = time.perf_counter()
    weak_loader = train.make_train_loader(weak_transform())
    weak_iterator = iter(weak_loader)
    weak_iterator, first_inputs, first_targets, first_wait = next_weak(
        weak_loader, weak_iterator
    )
    weak_rebuild_seconds = time.perf_counter() - rebuild_started
    weak_waits = [first_wait]
    weak_gpu = [train_step(model, optimizer, first_inputs, first_targets)]
    for _ in range(weak_measured - 1):
        weak_iterator, inputs, targets, wait = next_weak(weak_loader, weak_iterator)
        weak_waits.append(wait)
        weak_gpu.append(train_step(model, optimizer, inputs, targets))
    integrated_wall = time.perf_counter() - integrated_started
    weak_iterator = None
    weak_workers = stop_loader(weak_loader)
    ending_allocation = torch.cuda.memory_allocated()
    live_children = [
        child.pid for child in multiprocessing.active_children() if child.is_alive()
    ]
    all_gpu = strong_gpu + weak_gpu
    all_waits = strong_waits + weak_waits
    result = {
        "arm": arm,
        "policy_identity": (
            "one CPU draw; u<0.5 CutMix(alpha=1); else hard"
            if arm == "control"
            else "production mixed_collate: u<0.25 CutMix(alpha=1); u<0.5 Mixup(alpha=0.4); else hard"
        ),
        "backend": backend_flags(),
        "warmups": warmups,
        "measured_strong_steps": measured,
        "measured_weak_steps": weak_measured,
        "counts_including_warmup": counts,
        "counted": summary(all_gpu),
        "strong_counted": summary(strong_gpu),
        "weak_counted": summary(weak_gpu),
        "iterator_wait": summary(all_waits),
        "strong_iterator_wait": summary(strong_waits),
        "integrated_wall_seconds": integrated_wall,
        "counted_seconds": sum(all_gpu),
        "wall_counted_ratio": integrated_wall / sum(all_gpu),
        "loader_headroom": statistics.mean(strong_gpu) / statistics.mean(strong_waits),
        "median_wait_gpu_fraction": statistics.median(strong_waits)
        / statistics.mean(strong_gpu),
        "p95_wait_gpu_fraction": percentile(strong_waits, 0.95)
        / statistics.mean(strong_gpu),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "allocation_growth_mb": max(0, ending_allocation - allocated_after_warmup)
        / 1024
        / 1024,
        "strong_workers_stopped": len(strong_workers),
        "weak_workers_stopped": len(weak_workers),
        "weak_rebuild_seconds": weak_rebuild_seconds,
        "live_children": live_children,
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
    return json.loads(
        [line for line in completed.stdout.splitlines() if line.strip()][-1]
    )


def parent(reanalyze=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if reanalyze:
        progress = json.loads(PROGRESS_PATH.read_text())
        conditioning = progress["conditioning"]
        trials = progress["trials"]
        if len(trials) != 5:
            raise RuntimeError(f"expected five persisted trials, got {len(trials)}")
    else:
        conditioning = run_child("candidate", conditioning=True)
        PROGRESS_PATH.write_text(
            json.dumps({"conditioning": conditioning, "trials": []}, indent=2) + "\n"
        )
        print(json.dumps({"stage": "conditioning", "status": "complete"}), flush=True)
        trials = []
        for index in range(5):
            order = (
                ["control", "candidate"] if index % 2 == 0 else ["candidate", "control"]
            )
            arms = {}
            for arm in order:
                arms[arm] = run_child(arm)
                print(
                    json.dumps({"stage": "trial", "trial": index + 1, "arm": arm}),
                    flush=True,
                )
            counted_ratio = (
                arms["candidate"]["counted"]["mean_ms"]
                / arms["control"]["counted"]["mean_ms"]
            )
            integrated_ratio = (
                arms["candidate"]["integrated_wall_seconds"]
                / arms["control"]["integrated_wall_seconds"]
            )
            trials.append(
                {
                    "trial": index + 1,
                    "order": order,
                    "control": arms["control"],
                    "candidate": arms["candidate"],
                    "counted_ratio": counted_ratio,
                    "integrated_wall_ratio": integrated_ratio,
                    "wall_counted_delta": arms["candidate"]["wall_counted_ratio"]
                    - arms["control"]["wall_counted_ratio"],
                }
            )
            PROGRESS_PATH.write_text(
                json.dumps({"conditioning": conditioning, "trials": trials}, indent=2)
                + "\n"
            )
            with PROGRESS_PATH.open("rb") as handle:
                os.fsync(handle.fileno())

    projected_strong_steps = round(26_898 * train.LR_HOLD_FRACTION)
    projected_weak_steps = 26_898 - projected_strong_steps
    for trial in trials:
        for arm_name in ("control", "candidate"):
            arm = trial[arm_name]
            strong_wait_sum_ms = (
                arm["strong_iterator_wait"]["mean_ms"]
                * arm["strong_iterator_wait"]["count"]
            )
            all_wait_sum_ms = (
                arm["iterator_wait"]["mean_ms"] * arm["iterator_wait"]["count"]
            )
            weak_wait_mean_ms = (all_wait_sum_ms - strong_wait_sum_ms) / arm[
                "weak_counted"
            ]["count"]
            measured_wait_seconds = all_wait_sum_ms / 1_000
            one_time_overhead = max(
                0.0,
                arm["integrated_wall_seconds"]
                - arm["counted_seconds"]
                - measured_wait_seconds,
            )
            projected_wait_seconds = (
                projected_strong_steps * arm["strong_iterator_wait"]["mean_ms"]
                + projected_weak_steps * weak_wait_mean_ms
            ) / 1_000
            arm["one_time_overhead_seconds"] = one_time_overhead
            arm["projected_wait_seconds"] = projected_wait_seconds
            arm["projected_wall_counted_ratio"] = (
                train.TIME_BUDGET_S + projected_wait_seconds + one_time_overhead
            ) / train.TIME_BUDGET_S
        trial["projected_wall_counted_delta"] = (
            trial["candidate"]["projected_wall_counted_ratio"]
            - trial["control"]["projected_wall_counted_ratio"]
        )

    controls = [trial["control"]["counted"]["mean_ms"] for trial in trials]
    candidates = [trial["candidate"]["counted"]["mean_ms"] for trial in trials]
    counted_ratio = statistics.mean(candidates) / statistics.mean(controls)
    candidate_counts = [
        sum(trial["candidate"]["counts_including_warmup"][kind] for trial in trials)
        for kind in range(3)
    ]
    total_candidate = sum(candidate_counts)
    fractions = [count / total_candidate for count in candidate_counts]
    max_projection_ratio = max(
        counted_ratio,
        statistics.mean(trial["integrated_wall_ratio"] for trial in trials),
    )
    report = {
        "status": "pass",
        "thresholds": THRESHOLDS,
        "conditioning": conditioning,
        "trials": trials,
        "mean_counted_ratio": counted_ratio,
        "max_pair_counted_ratio": max(trial["counted_ratio"] for trial in trials),
        "control_cv": coefficient_of_variation(controls),
        "candidate_cv": coefficient_of_variation(candidates),
        "projected_steps": math.floor(26_898 / counted_ratio),
        "candidate_counts": candidate_counts,
        "candidate_fractions": fractions,
        "mean_integrated_wall_ratio": statistics.mean(
            trial["integrated_wall_ratio"] for trial in trials
        ),
        "max_projection_ratio": max_projection_ratio,
        "projected_total_seconds": 330.7 * max_projection_ratio,
        "failures": [],
    }
    failures = report["failures"]
    if counted_ratio > THRESHOLDS["mean_counted_ratio"]:
        failures.append(f"mean counted ratio {counted_ratio:.6f} > 1.01")
    if report["max_pair_counted_ratio"] > THRESHOLDS["pair_counted_ratio"]:
        failures.append(
            f"pair counted ratio {report['max_pair_counted_ratio']:.6f} > 1.04"
        )
    if (
        report["control_cv"] >= THRESHOLDS["trial_cv"]
        or report["candidate_cv"] >= THRESHOLDS["trial_cv"]
    ):
        failures.append("per-arm trial CV >= 3%")
    if report["projected_steps"] < THRESHOLDS["projected_steps"]:
        failures.append(f"projected steps {report['projected_steps']} < 26629")
    hard, cutmix, mixup = fractions
    for value, name, lower, upper in (
        (
            hard,
            "hard",
            THRESHOLDS["hard_fraction_min"],
            THRESHOLDS["hard_fraction_max"],
        ),
        (
            cutmix,
            "CutMix",
            THRESHOLDS["cutmix_fraction_min"],
            THRESHOLDS["cutmix_fraction_max"],
        ),
        (
            mixup,
            "Mixup",
            THRESHOLDS["mixup_fraction_min"],
            THRESHOLDS["mixup_fraction_max"],
        ),
    ):
        if not lower <= value <= upper:
            failures.append(f"{name} fraction {value:.6f} outside [{lower}, {upper}]")
    for trial in trials:
        candidate = trial["candidate"]
        control = trial["control"]
        if candidate["backend"] != control["backend"]:
            failures.append(f"trial {trial['trial']} backend mismatch")
        if candidate["loader_headroom"] < THRESHOLDS["loader_headroom"]:
            failures.append(f"trial {trial['trial']} loader headroom < 1.20")
        if (
            candidate["median_wait_gpu_fraction"]
            >= THRESHOLDS["median_wait_gpu_fraction"]
        ):
            failures.append(f"trial {trial['trial']} median wait/GPU >= 10%")
        if candidate["p95_wait_gpu_fraction"] >= THRESHOLDS["p95_wait_gpu_fraction"]:
            failures.append(f"trial {trial['trial']} p95 wait/GPU >= 20%")
        if candidate["projected_wall_counted_ratio"] > THRESHOLDS["wall_counted_ratio"]:
            failures.append(f"trial {trial['trial']} projected wall/count > 1.07")
        if (
            trial["projected_wall_counted_delta"]
            > THRESHOLDS["wall_counted_control_delta"]
        ):
            failures.append(f"trial {trial['trial']} projected wall/count delta > 0.02")
        if candidate["peak_vram_mb"] >= THRESHOLDS["peak_vram_mb"]:
            failures.append(f"trial {trial['trial']} peak VRAM >= 650 MiB")
        if candidate["allocation_growth_mb"] > THRESHOLDS["allocation_growth_mb"]:
            failures.append(f"trial {trial['trial']} allocation growth > 1 MiB")
        if candidate["weak_rebuild_seconds"] >= THRESHOLDS["weak_rebuild_seconds"]:
            failures.append(f"trial {trial['trial']} weak rebuild >= 5s")
        if candidate["live_children"] or control["live_children"]:
            failures.append(f"trial {trial['trial']} live children remain")
    if report["projected_total_seconds"] >= THRESHOLDS["projected_total_seconds"]:
        failures.append(
            f"projected total {report['projected_total_seconds']:.2f}s >= 540"
        )
    if failures:
        report["status"] = "failed"
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    with REPORT_PATH.open("rb") as handle:
        os.fsync(handle.fileno())
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("control", "candidate"))
    parser.add_argument("--conditioning", action="store_true")
    parser.add_argument("--reanalyze", action="store_true")
    args = parser.parse_args()
    if args.child:
        run_workload(args.child, args.conditioning)
    else:
        parent(reanalyze=args.reanalyze)


if __name__ == "__main__":
    main()
