import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
TRAINING_SECONDS = 300.0
ONLINE_STOP_SECONDS = 294.0
SWA_START_SECONDS = 258.0
STEPS_PER_EPOCH = 390


def weak_transform():
    mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def training_step(model, optimizer, inputs, targets, device):
    started = time.perf_counter()
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad()
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    return 1000 * (time.perf_counter() - started), float(loss.item())


def child(output_path):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = train.ResNet(
        train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER
    ).to(device)
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=0.001,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    inputs = torch.randn(train.BATCH_SIZE, 3, 32, 32, pin_memory=True)
    targets = torch.randint(0, train.NUM_CLASSES, (train.BATCH_SIZE,), pin_memory=True)

    for _ in range(100):
        training_step(model, optimizer, inputs, targets, device)
    measured = []
    for _ in range(500):
        elapsed_ms, _ = training_step(model, optimizer, inputs, targets, device)
        measured.append(elapsed_ms)

    accumulator = train.SWAAccumulator(parameters)
    snapshot_seconds = []
    for index in range(9):
        training_step(model, optimizer, inputs, targets, device)
        snapshot_seconds.append(
            train.timed_swa_snapshot(accumulator, parameters, 0.86 + index * 0.012)
        )

    install_seconds, momenta = train.timed_swa_install(accumulator, model, parameters)
    loader = train.make_train_loader(weak_transform())
    refresh_seconds, refresh_batches = train.refresh_batch_norm(
        model, loader, device, max_batches=390
    )
    finish_seconds, bn_batches = train.finish_batch_norm_refresh(momenta, parameters)
    stopped_workers = len(train.shutdown_train_loader(loader))
    assert refresh_batches == bn_batches == 390
    assert stopped_workers == train.NUM_WORKERS
    eval_started = time.perf_counter()
    train.evaluator.evaluate(model, device)
    eval_seconds = time.perf_counter() - eval_started

    result = {
        "mean_step_ms": statistics.mean(measured),
        "p95_step_ms": sorted(measured)[int(0.95 * len(measured)) - 1],
        "snapshot_seconds": snapshot_seconds,
        "snapshot_total_seconds": sum(snapshot_seconds),
        "install_seconds": install_seconds + finish_seconds,
        "refresh_390_seconds": refresh_seconds,
        "refresh_batches": refresh_batches,
        "eval_seconds": eval_seconds,
        "median_consecutive_rms": accumulator.median_consecutive_rms,
        "first_last_rms": accumulator.first_last_rms,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")


def projected_snapshot_count(epoch_seconds, snapshot_seconds):
    elapsed = 240.0
    count = 0
    while True:
        elapsed += epoch_seconds
        if elapsed >= ONLINE_STOP_SECONDS:
            return count
        if elapsed >= SWA_START_SECONDS:
            count += 1
            elapsed += snapshot_seconds


def projected_evaluation_count(epoch_seconds):
    elapsed = 240.0
    weak_epoch_evaluations = 0
    while elapsed + epoch_seconds < ONLINE_STOP_SECONDS:
        elapsed += epoch_seconds
        weak_epoch_evaluations += 1
    return 4 + 1 + weak_epoch_evaluations + 1


def controller():
    script = Path(__file__).resolve()
    conditioning_path = OUTPUT_DIR / "timing-conditioning.json"
    subprocess.run(
        [sys.executable, str(script), "--child", str(conditioning_path)],
        check=True,
        timeout=90,
    )

    trials = []
    for index in range(5):
        path = OUTPUT_DIR / f"timing-{index + 1}.json"
        subprocess.run(
            [sys.executable, str(script), "--child", str(path)],
            check=True,
            timeout=90,
        )
        trials.append(json.loads(path.read_text()))

    step_means = [trial["mean_step_ms"] for trial in trials]
    conservative_step_ms = max(step_means) * 1.01
    conservative_snapshot_total = max(
        trial["snapshot_total_seconds"] for trial in trials
    )
    conservative_snapshot_each = conservative_snapshot_total / 9
    projected_snapshots = projected_snapshot_count(
        STEPS_PER_EPOCH * conservative_step_ms / 1000,
        conservative_snapshot_each,
    )
    projected_steps = math.floor(
        1000
        * (ONLINE_STOP_SECONDS - conservative_snapshot_total)
        / conservative_step_ms
    )
    refresh_390_seconds = max(trial["refresh_390_seconds"] for trial in trials)
    peak_memory_mb = max(trial["peak_memory_mb"] for trial in trials)
    step_cv = statistics.pstdev(step_means) / statistics.mean(step_means)
    epoch_seconds = STEPS_PER_EPOCH * conservative_step_ms / 1000
    projected_evaluations = projected_evaluation_count(epoch_seconds)
    conservative_eval_seconds = max(trial["eval_seconds"] for trial in trials) * 1.05
    projected_wall_seconds = (
        1.0 + TRAINING_SECONDS + projected_evaluations * conservative_eval_seconds + 8.0
    )

    gates = {
        "snapshot_time": conservative_snapshot_total < 0.5,
        "refresh_time": refresh_390_seconds < 4.5,
        "snapshot_count": projected_snapshots >= 8,
        "steps": projected_steps >= 26200,
        "step_cv": step_cv < 0.03,
        "memory": peak_memory_mb < 700,
        "evaluations": projected_evaluations <= 19,
        "wall": projected_wall_seconds < 540,
        "spread": all(
            trial["median_consecutive_rms"] >= train.SWA_MIN_CONSECUTIVE_RMS
            and trial["first_last_rms"] >= train.SWA_MIN_FIRST_LAST_RMS
            for trial in trials
        ),
    }
    result = {
        "status": "pass" if all(gates.values()) else "fail",
        "step_mean_ms": statistics.mean(step_means),
        "conservative_step_ms": conservative_step_ms,
        "step_cv": step_cv,
        "snapshot_total_seconds": conservative_snapshot_total,
        "refresh_390_seconds": refresh_390_seconds,
        "projected_snapshots": projected_snapshots,
        "snapshot_margin": projected_snapshots - train.SWA_MIN_SNAPSHOTS,
        "projected_steps": projected_steps,
        "projected_evaluations": projected_evaluations,
        "conservative_eval_seconds": conservative_eval_seconds,
        "projected_wall_seconds": projected_wall_seconds,
        "peak_memory_mb": peak_memory_mb,
        "gates": gates,
    }
    (OUTPUT_DIR / "timing-swa.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(gates.values()):
        raise SystemExit("TIMING_GATE_FAIL")
    print("TIMING_GATE_PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child)
    else:
        controller()


if __name__ == "__main__":
    main()
