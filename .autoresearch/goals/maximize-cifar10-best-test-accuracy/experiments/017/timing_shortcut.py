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

import train
from preflight_shortcut import ControlResNet, optimizer


HERE = Path(__file__).resolve().parent


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def backend_state():
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def training_step(model, opt, cpu_inputs, cpu_targets, progress):
    start = time.perf_counter()
    inputs = cpu_inputs.cuda(non_blocking=True)
    targets = cpu_targets.cuda(non_blocking=True)
    if progress <= train.LR_HOLD_FRACTION:
        lr = train.LR
    else:
        cosine_progress = (progress - train.LR_HOLD_FRACTION) / (1 - train.LR_HOLD_FRACTION)
        lr = train.MIN_LR + 0.5 * (train.ANNEAL_START_LR - train.MIN_LR) * (1 + math.cos(math.pi * cosine_progress))
    for group in opt.param_groups: group["lr"] = lr
    opt.zero_grad(set_to_none=True)
    outputs = model(inputs); loss = F.cross_entropy(outputs, targets)
    loss.backward(); opt.step(); torch.cuda.synchronize()
    return time.perf_counter() - start, loss


def child(arm, output, warmup, measured, inference_steps):
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    started = time.perf_counter()
    model = (ControlResNet() if arm == "control" else train.ResNet(3, 10, 2)).cuda()
    opt = optimizer(model)
    cpu_inputs = torch.randn(128, 3, 32, 32, pin_memory=True, generator=torch.Generator().manual_seed(17042))
    hard = torch.randint(10, (128,), pin_memory=True, generator=torch.Generator().manual_seed(17043))
    soft = F.one_hot(hard, 10).float().pin_memory()
    startup = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    for index in range(warmup):
        _, loss = training_step(model, opt, cpu_inputs, hard if index % 2 == 0 else soft, 0.5)
        assert torch.isfinite(loss)
    durations = []
    for index in range(measured):
        duration, loss = training_step(model, opt, cpu_inputs, hard if index % 2 == 0 else soft, index / measured)
        assert torch.isfinite(loss); durations.append(duration)
    inference_input = torch.randn(256, 3, 32, 32, device="cuda", generator=torch.Generator(device="cuda").manual_seed(17044))
    model.eval()
    with torch.no_grad():
        for _ in range(100): model(inference_input)
        torch.cuda.synchronize()
        inference = []
        for _ in range(inference_steps):
            start = time.perf_counter(); model(inference_input); torch.cuda.synchronize(); inference.append(time.perf_counter() - start)
    result = {
        "arm": arm,
        "mean_ms": 1000 * statistics.mean(durations),
        "median_ms": 1000 * statistics.median(durations),
        "p95_ms": 1000 * percentile(durations, 0.95),
        "inference_mean_ms": 1000 * statistics.mean(inference),
        "inference_p95_ms": 1000 * percentile(inference, 0.95),
        "peak_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "startup_seconds": startup,
        "backend": backend_state(),
        "warmup": warmup,
        "measured": measured,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")


def cv(values):
    return statistics.pstdev(values) / statistics.mean(values)


def controller():
    conditioning = HERE / "timing-conditioning.json"
    subprocess.run([sys.executable, __file__, "--arm", "control", "--output", str(conditioning), "--warmup", "100", "--measured", "20", "--inference-steps", "20"], check=True, timeout=120)
    trials = []
    for index in range(5):
        order = ("control", "candidate") if index % 2 == 0 else ("candidate", "control")
        trial = {"trial": index + 1, "order": list(order), "arms": {}}
        for arm in order:
            output = HERE / f"timing-{index + 1}-{arm}.json"
            subprocess.run([sys.executable, __file__, "--arm", arm, "--output", str(output), "--warmup", "100", "--measured", "500", "--inference-steps", "500"], check=True, timeout=120)
            trial["arms"][arm] = json.loads(output.read_text())
        trials.append(trial)
    states = [trial["arms"][arm]["backend"] for trial in trials for arm in ("control", "candidate")]
    assert all(state == states[0] for state in states)
    control_means = [t["arms"]["control"]["mean_ms"] for t in trials]
    candidate_means = [t["arms"]["candidate"]["mean_ms"] for t in trials]
    ratios = [c / a for c, a in zip(candidate_means, control_means)]
    control_mean, candidate_mean = statistics.mean(control_means), statistics.mean(candidate_means)
    ratio = candidate_mean / control_mean
    projected_steps = math.floor(26_898 / ratio)
    control_p95 = statistics.mean(t["arms"]["control"]["p95_ms"] for t in trials)
    candidate_p95 = statistics.mean(t["arms"]["candidate"]["p95_ms"] for t in trials)
    control_inference = [t["arms"]["control"]["inference_mean_ms"] for t in trials]
    candidate_inference = [t["arms"]["candidate"]["inference_mean_ms"] for t in trials]
    inference_ratio = statistics.mean(candidate_inference) / statistics.mean(control_inference)
    peak_control = max(t["arms"]["control"]["peak_mb"] for t in trials)
    peak_candidate = max(t["arms"]["candidate"]["peak_mb"] for t in trials)
    tail_epochs = math.ceil(60 / (390 * candidate_mean / 1000))
    expected_evals = 4 + tail_epochs
    startup = max(t["arms"]["candidate"]["startup_seconds"] for t in trials)
    inference_full = 40 * max(t["arms"]["candidate"]["inference_p95_ms"] for t in trials) / 1000
    projected_total = 300 + startup + 5 + expected_evals * inference_full + 10
    gates = {
        "ratio": ratio <= 1.0548,
        "steps": projected_steps >= 25_500,
        "p95": candidate_p95 <= 1.10 * control_p95,
        "control_cv": cv(control_means) < 0.03,
        "candidate_cv": cv(candidate_means) < 0.03,
        "ratio_cv": cv(ratios) < 0.02,
        "memory": peak_candidate < 700 and peak_candidate - peak_control <= 96,
        "inference_ratio": inference_ratio <= 1.08,
        "inference_cv": cv(candidate_inference) < 0.03,
        "evaluations": expected_evals <= 19,
        "wall": projected_total < 540,
    }
    result = {"status": "pass" if all(gates.values()) else "fail", "backend": states[0], "training_ratio": ratio, "projected_steps": projected_steps, "control_mean_ms": control_mean, "candidate_mean_ms": candidate_mean, "control_cv": cv(control_means), "candidate_cv": cv(candidate_means), "ratio_cv": cv(ratios), "p95_ratio": candidate_p95 / control_p95, "inference_ratio": inference_ratio, "inference_cv": cv(candidate_inference), "peak_control_mb": peak_control, "peak_candidate_mb": peak_candidate, "tail_epochs": tail_epochs, "expected_evaluations": expected_evals, "projected_total_seconds": projected_total, "gates": gates, "trials": trials}
    (HERE / "timing-shortcut.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))
    if not all(gates.values()): raise SystemExit("TIMING_GATE_FAIL")
    print("TIMING_GATE_PASS")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=("control", "candidate")); parser.add_argument("--output"); parser.add_argument("--warmup", type=int, default=100); parser.add_argument("--measured", type=int, default=500); parser.add_argument("--inference-steps", type=int, default=500); args = parser.parse_args()
    if args.arm: child(args.arm, args.output, args.warmup, args.measured, args.inference_steps)
    else: controller()


if __name__ == "__main__": main()
