import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torch.optim as optim

import train


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "timing-bf16.json"


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def backend_state(model):
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "parameter_contiguous_nchw": all(p.is_contiguous() for p in model.parameters()),
    }


def make_workload():
    generator = torch.Generator().manual_seed(16042)
    inputs = torch.randn(
        train.BATCH_SIZE, 3, 32, 32, generator=generator, pin_memory=True
    )
    hard = torch.randint(
        train.NUM_CLASSES,
        (train.BATCH_SIZE,),
        generator=generator,
        pin_memory=True,
    )
    left = F.one_hot(hard, train.NUM_CLASSES).float()
    right_labels = torch.randint(
        train.NUM_CLASSES, (train.BATCH_SIZE,), generator=generator
    )
    right = F.one_hot(right_labels, train.NUM_CLASSES).float()
    soft = (0.63 * left + 0.37 * right).pin_memory()
    return inputs, hard, soft


def production_step(model, optimizer, cpu_inputs, cpu_targets, bf16, progress):
    start = time.perf_counter()
    inputs = cpu_inputs.cuda(non_blocking=True)
    targets = cpu_targets.cuda(non_blocking=True)
    if progress <= train.LR_HOLD_FRACTION:
        lr = train.LR
    else:
        cosine_progress = (progress - train.LR_HOLD_FRACTION) / (
            1.0 - train.LR_HOLD_FRACTION
        )
        lr = train.MIN_LR + 0.5 * (train.ANNEAL_START_LR - train.MIN_LR) * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad()
    if bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
    else:
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return time.perf_counter() - start, loss


def staged_step(model, optimizer, cpu_inputs, cpu_targets, bf16):
    names = ["start", "transfer", "forward", "loss", "backward", "optimizer"]
    events = {name: torch.cuda.Event(enable_timing=True) for name in names}
    events["start"].record()
    inputs = cpu_inputs.cuda(non_blocking=True)
    targets = cpu_targets.cuda(non_blocking=True)
    events["transfer"].record()
    optimizer.zero_grad()
    if bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            events["forward"].record()
            loss = F.cross_entropy(outputs, targets)
    else:
        outputs = model(inputs)
        events["forward"].record()
        loss = F.cross_entropy(outputs, targets)
    events["loss"].record()
    loss.backward()
    events["backward"].record()
    optimizer.step()
    events["optimizer"].record()
    torch.cuda.synchronize()
    return {
        "transfer_ms": events["start"].elapsed_time(events["transfer"]),
        "forward_ms": events["transfer"].elapsed_time(events["forward"]),
        "loss_ms": events["forward"].elapsed_time(events["loss"]),
        "backward_ms": events["loss"].elapsed_time(events["backward"]),
        "optimizer_ms": events["backward"].elapsed_time(events["optimizer"]),
    }


def arm_run(arm, output, warmup, measure, stage_steps):
    assert arm in {"A", "B", "C"}
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    width = 2 if arm == "A" else 3
    bf16 = arm == "C"
    model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, width).cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    cpu_inputs, hard, soft = make_workload()
    state = backend_state(model)
    torch.cuda.reset_peak_memory_stats()
    for index in range(warmup):
        target = hard if index % 2 == 0 else soft
        _, loss = production_step(model, optimizer, cpu_inputs, target, bf16, 0.5)
        assert torch.isfinite(loss)

    durations = []
    allocation_samples = []
    for index in range(measure):
        target = hard if index % 2 == 0 else soft
        duration, loss = production_step(
            model, optimizer, cpu_inputs, target, bf16, index / max(measure, 1)
        )
        assert torch.isfinite(loss)
        durations.append(duration)
        if (index + 1) % 100 == 0:
            allocation_samples.append(torch.cuda.memory_allocated() / 1024 / 1024)

    stage_rows = []
    for index in range(stage_steps):
        target = hard if index % 2 == 0 else soft
        stage_rows.append(staged_step(model, optimizer, cpu_inputs, target, bf16))
    stage_means = {
        key: statistics.mean(row[key] for row in stage_rows)
        for key in stage_rows[0]
    } if stage_rows else {}
    strictly_growing = len(allocation_samples) > 1 and all(
        right > left for left, right in zip(allocation_samples, allocation_samples[1:])
    )
    result = {
        "arm": arm,
        "width": width,
        "bf16": bf16,
        "warmup_steps": warmup,
        "measured_steps": measure,
        "stage_steps": stage_steps,
        "mean_ms": 1000 * statistics.mean(durations),
        "median_ms": 1000 * statistics.median(durations),
        "p95_ms": 1000 * percentile(durations, 0.95),
        "images_per_second": train.BATCH_SIZE / statistics.mean(durations),
        "stage_means": stage_means,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "allocation_samples_mb": allocation_samples,
        "allocation_strictly_growing": strictly_growing,
        "backend_state": state,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")


def cv(values):
    return statistics.pstdev(values) / statistics.mean(values)


def controller():
    conditioning = HERE / "timing-conditioning.json"
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--arm",
            "A",
            "--output",
            str(conditioning),
            "--warmup",
            "100",
            "--measure",
            "20",
            "--stage-steps",
            "0",
        ],
        check=True,
        timeout=120,
    )

    orders = ["ABC", "BCA", "CAB", "ACB", "CBA"]
    trials = []
    for trial_index, order in enumerate(orders):
        trial = {"trial": trial_index + 1, "order": order, "arms": {}}
        for arm in order:
            output = HERE / f"timing-trial-{trial_index + 1}-{arm}.json"
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--arm",
                    arm,
                    "--output",
                    str(output),
                    "--warmup",
                    "100",
                    "--measure",
                    "500",
                    "--stage-steps",
                    "100",
                ],
                check=True,
                timeout=120,
            )
            trial["arms"][arm] = json.loads(output.read_text())
        trials.append(trial)

    states = [trial["arms"][arm]["backend_state"] for trial in trials for arm in "ABC"]
    assert all(state == states[0] for state in states)
    assert states[0]["parameter_contiguous_nchw"]
    assert states[0]["cudnn_allow_tf32"] is True
    assert states[0]["matmul_allow_tf32"] is False
    assert states[0]["cudnn_benchmark"] is False

    means = {
        arm: [trial["arms"][arm]["mean_ms"] for trial in trials] for arm in "ABC"
    }
    medians = {
        arm: [trial["arms"][arm]["median_ms"] for trial in trials] for arm in "ABC"
    }
    p95s = {arm: [trial["arms"][arm]["p95_ms"] for trial in trials] for arm in "ABC"}
    mean_step = {arm: statistics.mean(values) for arm, values in means.items()}
    median_trial_mean = {arm: statistics.median(values) for arm, values in means.items()}
    trial_speedups = [
        trial["arms"]["B"]["mean_ms"] / trial["arms"]["C"]["mean_ms"]
        for trial in trials
    ]
    trial_exposure_ratios = [
        trial["arms"]["C"]["mean_ms"] / trial["arms"]["A"]["mean_ms"]
        for trial in trials
    ]
    funding_ratio = median_trial_mean["C"] / median_trial_mean["B"]

    stage = {}
    for arm in "ABC":
        stage[arm] = {
            key: statistics.median(
                trial["arms"][arm]["stage_means"][key] for trial in trials
            )
            for key in trials[0]["arms"][arm]["stage_means"]
        }
    b_fb = stage["B"]["forward_ms"] + stage["B"]["backward_ms"]
    c_fb = stage["C"]["forward_ms"] + stage["C"]["backward_ms"]
    b_total = sum(stage["B"].values())
    c_total = sum(stage["C"].values())
    stage_total_savings = b_total - c_total
    stage_savings_fraction = (
        (b_fb - c_fb) / stage_total_savings if stage_total_savings > 0 else float("-inf")
    )

    ratio_projection = math.floor(26_898 * mean_step["A"] / mean_step["C"])
    absolute_projection = math.floor(300 / (1.025 * mean_step["C"] / 1000))
    ratio_values = [
        trial["arms"]["C"]["mean_ms"] / trial["arms"]["B"]["mean_ms"]
        for trial in trials
    ]
    peak_memory = max(
        trial["arms"]["C"]["peak_allocated_mb"] for trial in trials
    )
    growing = any(
        trial["arms"]["C"]["allocation_strictly_growing"] for trial in trials
    )

    gates = {
        "funding_ratio": funding_ratio <= 0.86957,
        "every_triplet_speedup": min(trial_speedups) >= 1.12,
        "forward_backward_ratio": c_fb / b_fb <= 0.85,
        "backward_ratio": stage["C"]["backward_ms"] / stage["B"]["backward_ms"] <= 0.90,
        "stage_total_savings": stage_total_savings >= 0.05 * b_total,
        "stage_savings_fraction": stage_savings_fraction >= 0.90,
        "ratio_projection": ratio_projection >= 22_863,
        "absolute_projection": absolute_projection >= 22_863,
        "every_triplet_exposure": max(trial_exposure_ratios) <= 1.17647,
        "candidate_p95": statistics.median(p95s["C"])
        <= 1.25 * statistics.median(medians["A"]),
        "trial_cv": all(cv(means[arm]) < 0.03 for arm in "ABC"),
        "ratio_cv": cv(ratio_values) < 0.02,
        "memory": peak_memory < 2048,
        "allocation_not_growing": not growing,
    }
    result = {
        "status": "pass" if all(gates.values()) else "fail",
        "conditioning_processes": 1,
        "orders": orders,
        "backend_state": states[0],
        "mean_step_ms": mean_step,
        "trial_mean_cv": {arm: cv(means[arm]) for arm in "ABC"},
        "candidate_control_ratio_cv": cv(ratio_values),
        "funding_ratio_c_over_b": funding_ratio,
        "triplet_b_over_c_speedups": trial_speedups,
        "triplet_c_over_a_ratios": trial_exposure_ratios,
        "stage_median_ms": stage,
        "forward_backward_c_over_b": c_fb / b_fb,
        "backward_c_over_b": stage["C"]["backward_ms"] / stage["B"]["backward_ms"],
        "stage_total_savings_ms": stage_total_savings,
        "stage_savings_fraction": stage_savings_fraction,
        "ratio_projected_steps": ratio_projection,
        "absolute_projected_steps": absolute_projection,
        "candidate_peak_allocated_mb": peak_memory,
        "gates": gates,
        "trials": trials,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))
    if not all(gates.values()):
        raise SystemExit("TIMING_GATE_FAIL")
    print("TIMING_GATE_PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "B", "C"])
    parser.add_argument("--output")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measure", type=int, default=500)
    parser.add_argument("--stage-steps", type=int, default=100)
    args = parser.parse_args()
    if args.arm:
        if not args.output:
            raise SystemExit("--output is required with --arm")
        arm_run(args.arm, args.output, args.warmup, args.measure, args.stage_steps)
    else:
        controller()


if __name__ == "__main__":
    main()
