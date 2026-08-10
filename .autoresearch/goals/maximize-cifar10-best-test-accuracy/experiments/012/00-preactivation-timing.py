import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

import train as candidate


BASELINE_COMMIT = "7c1e7d8"
BATCH_SIZE = 128
WARMUP_STEPS = 100
TIMED_STEPS = 500
TRIALS = 5
EVAL_BATCHES = math.ceil(10_000 / BATCH_SIZE)
CONSERVATIVE_EVAL_PASSES = 20


def load_baseline():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:train.py"], text=True
    )
    namespace = {"__name__": "baseline_train", "__file__": "baseline_train.py"}
    exec(compile(source, "baseline_train.py", "exec"), namespace)
    return namespace


def model_class(name):
    if name == "candidate":
        return candidate.ResNet
    return load_baseline()["ResNet"]


def make_inputs():
    generator = torch.Generator().manual_seed(20260806)
    inputs = torch.randn(BATCH_SIZE, 3, 32, 32, generator=generator).pin_memory()
    hard_targets = torch.randint(0, 10, (BATCH_SIZE,), generator=generator).pin_memory()
    labels_a = torch.randint(0, 10, (BATCH_SIZE,), generator=generator)
    labels_b = torch.randint(0, 10, (BATCH_SIZE,), generator=generator)
    lam = torch.rand(BATCH_SIZE, 1, generator=generator)
    soft_targets = (
        lam * F.one_hot(labels_a, 10) + (1 - lam) * F.one_hot(labels_b, 10)
    ).pin_memory()
    return inputs, hard_targets, soft_targets


def make_model(name):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    return model_class(name)(3, 10, 2).cuda()


def train_worker(name):
    inputs, hard_targets, soft_targets = make_inputs()
    model = make_model(name)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    durations = []
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for step in range(WARMUP_STEPS + TIMED_STEPS):
        targets = hard_targets if step % 2 == 0 else soft_targets
        start = time.perf_counter()
        batch = inputs.to("cuda", non_blocking=True)
        target_batch = targets.to("cuda", non_blocking=True)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(batch), target_batch)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert torch.isfinite(loss)
        if step >= WARMUP_STEPS:
            durations.append(elapsed)
    result = {
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
        "p95": sorted(durations)[math.ceil(0.95 * len(durations)) - 1],
        "images_s": BATCH_SIZE / statistics.mean(durations),
        "peak_mb": torch.cuda.max_memory_allocated() / (1024**2),
    }
    print(json.dumps(result))


def inference_worker(name):
    inputs, _, _ = make_inputs()
    model = make_model(name)
    durations = []
    model.eval()
    with torch.inference_mode():
        for step in range(WARMUP_STEPS + TIMED_STEPS):
            start = time.perf_counter()
            model(inputs.to("cuda", non_blocking=True))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= WARMUP_STEPS:
                durations.append(elapsed)
    print(json.dumps({"mean": statistics.mean(durations)}))


def cv(values):
    return statistics.stdev(values) / statistics.mean(values)


def run_fresh(name, mode):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."
    output = subprocess.check_output(
        [sys.executable, __file__, "--worker", name, "--mode", mode],
        text=True,
        env=environment,
    )
    return json.loads(output.strip().splitlines()[-1])


def orchestrate():
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_name() == "NVIDIA H20"
    results = {"accepted": [], "candidate": []}
    for trial in range(TRIALS):
        order = ("accepted", "candidate") if trial % 2 == 0 else ("candidate", "accepted")
        for name in order:
            result = run_fresh(name, "train")
            results[name].append(result)
            print(f"train trial {trial + 1} {name}: {result}", flush=True)

    accepted_means = [result["mean"] for result in results["accepted"]]
    candidate_means = [result["mean"] for result in results["candidate"]]
    accepted_mean = statistics.median(accepted_means)
    candidate_mean = statistics.median(candidate_means)
    ratio = candidate_mean / accepted_mean
    projected_steps = math.floor(26_898 / ratio)
    training_summary = {
        "accepted_mean_s": accepted_mean,
        "candidate_mean_s": candidate_mean,
        "ratio": ratio,
        "projected_steps": projected_steps,
        "accepted_cv": cv(accepted_means),
        "candidate_cv": cv(candidate_means),
        "p95_ratio": statistics.median(
            result["p95"] for result in results["candidate"]
        )
        / statistics.median(result["p95"] for result in results["accepted"]),
        "accepted_peak_mb": statistics.median(
            result["peak_mb"] for result in results["accepted"]
        ),
        "candidate_peak_mb": statistics.median(
            result["peak_mb"] for result in results["candidate"]
        ),
    }
    print("training summary:", training_summary, flush=True)

    inference = {"accepted": [], "candidate": []}
    for trial in range(TRIALS):
        order = ("accepted", "candidate") if trial % 2 == 0 else ("candidate", "accepted")
        for name in order:
            result = run_fresh(name, "inference")
            inference[name].append(result["mean"])
            print(f"inference trial {trial + 1} {name}: {result}", flush=True)
    accepted_inference = statistics.median(inference["accepted"])
    candidate_inference = statistics.median(inference["candidate"])
    projected_total = 330.7 + max(0, candidate_inference - accepted_inference) * (
        EVAL_BATCHES * CONSERVATIVE_EVAL_PASSES
    )
    inference_summary = {
        "accepted_mean_s": accepted_inference,
        "candidate_mean_s": candidate_inference,
        "ratio": candidate_inference / accepted_inference,
        "accepted_cv": cv(inference["accepted"]),
        "candidate_cv": cv(inference["candidate"]),
        "projected_total_s": projected_total,
    }
    print("inference summary:", inference_summary, flush=True)

    assert training_summary["accepted_cv"] <= 0.02
    assert training_summary["candidate_cv"] <= 0.02
    assert ratio <= 1.03
    assert projected_steps >= 26_091
    assert inference_summary["accepted_cv"] <= 0.02
    assert inference_summary["candidate_cv"] <= 0.02
    assert projected_total < 540
    print("PREACTIVATION TIMING GATE: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("accepted", "candidate"))
    parser.add_argument("--mode", choices=("train", "inference"))
    args = parser.parse_args()
    if args.worker:
        if args.mode == "train":
            train_worker(args.worker)
        else:
            inference_worker(args.worker)
    else:
        orchestrate()


if __name__ == "__main__":
    main()
