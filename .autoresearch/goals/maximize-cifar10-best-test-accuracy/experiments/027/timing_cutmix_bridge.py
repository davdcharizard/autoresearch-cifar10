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
from torch.utils.data import get_worker_info
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORT_PATH = EXPERIMENT_DIR / "timing-report.json"
PROGRESS_PATH = EXPERIMENT_DIR / "timing-progress.json"
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


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


def cv(values):
    return (
        statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0
    )


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


class TaggedCollator(train.PhaseCutMixCollator):
    def __call__(self, batch):
        inputs, targets, policy = super().__call__(batch)
        return inputs, targets, policy, get_worker_info().id


def configure_backend():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def backend_flags():
    return {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def validate_batch(inputs, targets, policy=None):
    if inputs.shape != (128, 3, 32, 32) or inputs.dtype != torch.float32:
        raise RuntimeError("invalid inputs")
    if policy == train.POLICY_OFF or policy is None:
        if targets.shape != (128,) or targets.dtype != torch.int64:
            raise RuntimeError("invalid hard targets")
    elif targets.ndim == 2:
        if (
            targets.shape != (128, 10)
            or not torch.isfinite(targets).all()
            or targets.min() < 0
            or not torch.allclose(targets.sum(1), torch.ones(128), atol=1e-6, rtol=0)
        ):
            raise RuntimeError("invalid soft targets")
    elif targets.shape != (128,) or targets.dtype != torch.int64:
        raise RuntimeError("invalid eligible hard targets")


def next_strong(loader, iterator):
    started = time.perf_counter()
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    wait = time.perf_counter() - started
    inputs, targets, policy, worker_id = batch
    validate_batch(inputs, targets, policy)
    return iterator, inputs, targets, policy, worker_id, wait


def next_weak(loader, iterator):
    started = time.perf_counter()
    try:
        inputs, targets = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        inputs, targets = next(iterator)
    wait = time.perf_counter() - started
    validate_batch(inputs, targets)
    return iterator, inputs, targets, wait


def train_step(model, optimizer, cpu_inputs, cpu_targets):
    started = time.perf_counter()
    inputs = cpu_inputs.cuda(non_blocking=True)
    targets = cpu_targets.cuda(non_blocking=True)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item():
        raise RuntimeError("non-finite timing loss")
    return time.perf_counter() - started


def stop_loader(loader):
    workers = train.shutdown_train_loader(loader)
    if len(workers) != train.NUM_WORKERS:
        raise RuntimeError("incomplete worker shutdown")
    return workers


def run_workload(arm, conditioning):
    configure_backend()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    context = multiprocessing.get_context("forkserver")
    flag = context.Value("b", True, lock=True)
    strong_loader = train.make_train_loader(
        strong_transform(), collate_fn=TaggedCollator(flag)
    )
    model = train.ResNet(3, 10, 2).cuda().train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    iterator = iter(strong_loader)
    warmups, strong_steps, weak_steps = (
        (20, 80, 20) if conditioning else (100, 800, 200)
    )
    flip_step = 70 if conditioning else 700
    for _ in range(warmups):
        iterator, inputs, targets, _policy, _worker_id, _wait = next_strong(
            strong_loader, iterator
        )
        train_step(model, optimizer, inputs, targets)
    allocated_after_warmup = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    counts = {"on_hard": 0, "on_cutmix": 0, "off_hard": 0}
    waits, gpu_times, post_request = [], [], []
    integrated_started = time.perf_counter()
    for index in range(strong_steps):
        if arm == "candidate" and index == flip_step:
            with flag.get_lock():
                flag.value = False
        iterator, inputs, targets, policy, worker_id, wait = next_strong(
            strong_loader, iterator
        )
        waits.append(wait)
        gpu_times.append(train_step(model, optimizer, inputs, targets))
        if policy == train.POLICY_ON:
            counts["on_cutmix" if targets.ndim == 2 else "on_hard"] += 1
        else:
            counts["off_hard"] += 1
        if arm == "candidate" and index >= flip_step:
            post_request.append(
                {
                    "offset": index - flip_step + 1,
                    "policy": policy,
                    "worker_id": worker_id,
                }
            )
    iterator = None
    strong_workers = stop_loader(strong_loader)
    rebuild_started = time.perf_counter()
    weak_loader = train.make_train_loader(weak_transform())
    weak_iterator = iter(weak_loader)
    weak_waits, weak_gpu = [], []
    for _ in range(weak_steps):
        weak_iterator, inputs, targets, wait = next_weak(weak_loader, weak_iterator)
        weak_waits.append(wait)
        weak_gpu.append(train_step(model, optimizer, inputs, targets))
    weak_rebuild = (
        time.perf_counter() - rebuild_started - sum(weak_gpu[1:]) - sum(weak_waits[1:])
    )
    integrated_wall = time.perf_counter() - integrated_started
    weak_iterator = None
    weak_workers = stop_loader(weak_loader)
    ending_allocation = torch.cuda.memory_allocated()
    live = [
        child.pid for child in multiprocessing.active_children() if child.is_alive()
    ]
    last_on = max(
        (item["offset"] for item in post_request if item["policy"]), default=0
    )
    off_workers = {item["worker_id"] for item in post_request if not item["policy"]}
    all_gpu, all_waits = gpu_times + weak_gpu, waits + weak_waits
    return {
        "arm": arm,
        "policy_identity": "always accepted CutMix-on"
        if arm == "control"
        else "CutMix-on 0-70%, same-loader off 70-80%",
        "backend": backend_flags(),
        "warmups": warmups,
        "strong_steps": strong_steps,
        "weak_steps": weak_steps,
        "counts": counts,
        "post_request": post_request,
        "last_policy_on_offset": last_on,
        "off_worker_ids": sorted(off_workers),
        "counted": summary(all_gpu),
        "strong_counted": summary(gpu_times),
        "weak_counted": summary(weak_gpu),
        "iterator_wait": summary(all_waits),
        "strong_iterator_wait": summary(waits),
        "weak_iterator_wait": summary(weak_waits),
        "integrated_wall_seconds": integrated_wall,
        "counted_seconds": sum(all_gpu),
        "wall_counted_ratio": integrated_wall / sum(all_gpu),
        "loader_headroom": statistics.mean(gpu_times) / statistics.mean(waits),
        "median_wait_gpu_fraction": statistics.median(waits)
        / statistics.mean(gpu_times),
        "p95_wait_gpu_fraction": percentile(waits, 0.95) / statistics.mean(gpu_times),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "allocation_growth_mb": max(0, ending_allocation - allocated_after_warmup)
        / 1024
        / 1024,
        "strong_workers_stopped": len(strong_workers),
        "weak_workers_stopped": len(weak_workers),
        "weak_rebuild_seconds": weak_rebuild,
        "live_children": live,
    }


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


def projected_wall_ratio(arm):
    strong_steps = round(26_898 * 0.8)
    weak_steps = 26_898 - strong_steps
    projected_wait = (
        strong_steps * arm["strong_iterator_wait"]["mean_ms"]
        + weak_steps * arm["weak_iterator_wait"]["mean_ms"]
    ) / 1_000
    measured_wait = (
        arm["iterator_wait"]["mean_ms"] * arm["iterator_wait"]["count"] / 1_000
    )
    one_time = max(
        0, arm["integrated_wall_seconds"] - arm["counted_seconds"] - measured_wait
    )
    return (300 + projected_wait + one_time) / 300


def parent():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    conditioning = run_child("candidate", True)
    trials = []
    for index in range(5):
        order = ["control", "candidate"] if index % 2 == 0 else ["candidate", "control"]
        arms = {}
        for arm in order:
            arms[arm] = run_child(arm)
            print(json.dumps({"trial": index + 1, "arm": arm}), flush=True)
        trial = {
            "trial": index + 1,
            "order": order,
            "control": arms["control"],
            "candidate": arms["candidate"],
            "counted_ratio": arms["candidate"]["counted"]["mean_ms"]
            / arms["control"]["counted"]["mean_ms"],
            "integrated_ratio": arms["candidate"]["integrated_wall_seconds"]
            / arms["control"]["integrated_wall_seconds"],
        }
        for arm in ("control", "candidate"):
            trial[arm]["projected_wall_counted_ratio"] = projected_wall_ratio(
                trial[arm]
            )
        trial["projected_wall_delta"] = (
            trial["candidate"]["projected_wall_counted_ratio"]
            - trial["control"]["projected_wall_counted_ratio"]
        )
        trials.append(trial)
        PROGRESS_PATH.write_text(
            json.dumps({"conditioning": conditioning, "trials": trials}, indent=2)
            + "\n"
        )
        with PROGRESS_PATH.open("rb") as handle:
            os.fsync(handle.fileno())
    controls = [trial["control"]["counted"]["mean_ms"] for trial in trials]
    candidates = [trial["candidate"]["counted"]["mean_ms"] for trial in trials]
    ratio = statistics.mean(candidates) / statistics.mean(controls)
    report = {
        "status": "pass",
        "conditioning": conditioning,
        "trials": trials,
        "mean_counted_ratio": ratio,
        "max_pair_ratio": max(trial["counted_ratio"] for trial in trials),
        "control_cv": cv(controls),
        "candidate_cv": cv(candidates),
        "projected_steps": math.floor(26_898 / ratio),
        "mean_integrated_ratio": statistics.mean(
            trial["integrated_ratio"] for trial in trials
        ),
        "projected_total_seconds": 330.7
        * max(ratio, statistics.mean(trial["integrated_ratio"] for trial in trials)),
        "failures": [],
    }
    failures = report["failures"]
    if ratio > 1.01:
        failures.append(f"mean ratio {ratio:.6f} > 1.01")
    if report["max_pair_ratio"] > 1.04:
        failures.append("pair ratio > 1.04")
    if report["control_cv"] >= 0.03 or report["candidate_cv"] >= 0.03:
        failures.append("CV >= 3%")
    if report["projected_steps"] < 26_629:
        failures.append("projected steps < 26629")
    total_on = sum(
        trial["candidate"]["counts"]["on_hard"]
        + trial["candidate"]["counts"]["on_cutmix"]
        for trial in trials
    )
    total_cutmix = sum(trial["candidate"]["counts"]["on_cutmix"] for trial in trials)
    report["candidate_policy_on_cutmix_fraction"] = total_cutmix / total_on
    if not 0.475 <= report["candidate_policy_on_cutmix_fraction"] <= 0.525:
        failures.append("candidate CutMix fraction outside interval")
    for trial in trials:
        candidate, control = trial["candidate"], trial["control"]
        number = trial["trial"]
        if candidate["backend"] != control["backend"]:
            failures.append(f"trial {number} backend mismatch")
        if candidate["last_policy_on_offset"] > 24 or candidate[
            "off_worker_ids"
        ] != list(range(8)):
            failures.append(f"trial {number} propagation failure")
        if candidate["counts"]["off_hard"] < 70:
            failures.append(f"trial {number} insufficient off-policy exposure")
        if (
            candidate["loader_headroom"] < 1.2
            or candidate["median_wait_gpu_fraction"] >= 0.1
            or candidate["p95_wait_gpu_fraction"] >= 0.2
        ):
            failures.append(f"trial {number} loader gate")
        if (
            candidate["projected_wall_counted_ratio"] > 1.07
            or trial["projected_wall_delta"] > 0.02
        ):
            failures.append(f"trial {number} wall/count gate")
        if candidate["peak_vram_mb"] >= 650 or candidate["allocation_growth_mb"] > 1:
            failures.append(f"trial {number} memory gate")
        if (
            candidate["weak_rebuild_seconds"] >= 5
            or candidate["live_children"]
            or control["live_children"]
        ):
            failures.append(f"trial {number} lifecycle gate")
    if report["projected_total_seconds"] >= 540:
        failures.append("projected total >= 540s")
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
    args = parser.parse_args()
    if args.child:
        print(json.dumps(run_workload(args.child, args.conditioning)))
    else:
        parent()


if __name__ == "__main__":
    main()
