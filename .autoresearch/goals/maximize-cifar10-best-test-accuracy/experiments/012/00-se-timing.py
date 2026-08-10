import gc
import math
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F

import train as candidate


BASELINE_COMMIT = "7c1e7d8"
BATCH_SIZE = 128
WARMUP_STEPS = 100
TIMED_STEPS = 500
TRIALS = 5


def load_baseline():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:train.py"], text=True
    )
    namespace = {"__name__": "baseline_train", "__file__": "baseline_train.py"}
    exec(compile(source, "baseline_train.py", "exec"), namespace)
    return namespace


def make_model(resnet_cls):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    return resnet_cls(3, 10, 2).cuda()


def verify_alignment(baseline):
    torch.manual_seed(42)
    reference = baseline["ResNet"](3, 10, 2)
    reference_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    trial = candidate.ResNet(3, 10, 2)
    trial_rng = torch.random.get_rng_state().clone()

    reference_state = reference.state_dict()
    trial_state = trial.state_dict()
    common = set(reference_state) & set(trial_state)
    assert common == set(reference_state)
    assert all(torch.equal(reference_state[key], trial_state[key]) for key in common)
    assert torch.equal(reference_rng, trial_rng)
    assert sum(parameter.numel() for parameter in trial.parameters()) == 1_082_740
    gates = [module for module in trial.modules() if isinstance(module, candidate.SqueezeExcitation)]
    assert len(gates) == 9
    assert [gate.reduce.out_features for gate in gates] == [2] * 3 + [4] * 3 + [8] * 3
    print("alignment: shared tensors and global CPU RNG are identical")
    print("candidate: 9 gates, hidden widths 2/4/8, params 1,082,740")


def train_trial(resnet_cls, inputs, hard_targets, soft_targets):
    model = make_model(resnet_cls)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    model.train()
    durations = []
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
        if step >= WARMUP_STEPS:
            durations.append(elapsed)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
        "p95": sorted(durations)[math.ceil(0.95 * len(durations)) - 1],
        "images_s": BATCH_SIZE / statistics.mean(durations),
        "peak_mb": peak_mb,
    }


def inference_trial(resnet_cls, inputs):
    model = make_model(resnet_cls)
    model.eval()
    durations = []
    with torch.inference_mode():
        for step in range(WARMUP_STEPS + TIMED_STEPS):
            start = time.perf_counter()
            model(inputs.to("cuda", non_blocking=True))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= WARMUP_STEPS:
                durations.append(elapsed)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return statistics.mean(durations)


def cv(values):
    return statistics.stdev(values) / statistics.mean(values)


def main():
    assert torch.cuda.get_device_name() == "NVIDIA H20"
    baseline = load_baseline()
    verify_alignment(baseline)

    generator = torch.Generator().manual_seed(20260806)
    inputs = torch.randn(BATCH_SIZE, 3, 32, 32, generator=generator).pin_memory()
    hard_targets = torch.randint(0, 10, (BATCH_SIZE,), generator=generator).pin_memory()
    labels_a = torch.randint(0, 10, (BATCH_SIZE,), generator=generator)
    labels_b = torch.randint(0, 10, (BATCH_SIZE,), generator=generator)
    lam = torch.rand(BATCH_SIZE, 1, generator=generator)
    soft_targets = (
        lam * F.one_hot(labels_a, 10) + (1 - lam) * F.one_hot(labels_b, 10)
    ).pin_memory()

    results = {"accepted": [], "candidate": []}
    classes = {"accepted": baseline["ResNet"], "candidate": candidate.ResNet}
    for trial in range(TRIALS):
        order = ("accepted", "candidate") if trial % 2 == 0 else ("candidate", "accepted")
        for name in order:
            result = train_trial(classes[name], inputs, hard_targets, soft_targets)
            results[name].append(result)
            print(f"train trial {trial + 1} {name}: {result}")

    accepted_means = [result["mean"] for result in results["accepted"]]
    candidate_means = [result["mean"] for result in results["candidate"]]
    accepted_mean = statistics.median(accepted_means)
    candidate_mean = statistics.median(candidate_means)
    ratio = candidate_mean / accepted_mean
    projected_steps = math.floor(26_898 / ratio)
    accepted_p95 = statistics.median(result["p95"] for result in results["accepted"])
    candidate_p95 = statistics.median(result["p95"] for result in results["candidate"])
    accepted_peak = statistics.median(result["peak_mb"] for result in results["accepted"])
    candidate_peak = statistics.median(result["peak_mb"] for result in results["candidate"])
    print(
        "training summary:",
        {
            "accepted_mean_s": accepted_mean,
            "candidate_mean_s": candidate_mean,
            "ratio": ratio,
            "projected_steps": projected_steps,
            "accepted_cv": cv(accepted_means),
            "candidate_cv": cv(candidate_means),
            "p95_ratio": candidate_p95 / accepted_p95,
            "accepted_peak_mb": accepted_peak,
            "candidate_peak_mb": candidate_peak,
            "peak_delta_mb": candidate_peak - accepted_peak,
        },
    )

    inference = {"accepted": [], "candidate": []}
    for trial in range(TRIALS):
        order = ("accepted", "candidate") if trial % 2 == 0 else ("candidate", "accepted")
        for name in order:
            mean = inference_trial(classes[name], inputs)
            inference[name].append(mean)
            print(f"inference trial {trial + 1} {name}: {mean}")
    inference_ratio = statistics.median(inference["candidate"]) / statistics.median(
        inference["accepted"]
    )
    print(
        "inference summary:",
        {
            "ratio": inference_ratio,
            "accepted_cv": cv(inference["accepted"]),
            "candidate_cv": cv(inference["candidate"]),
        },
    )

    assert cv(accepted_means) < 0.03
    assert cv(candidate_means) < 0.03
    assert ratio <= 1.0526
    assert projected_steps >= 25_553
    assert candidate_p95 / accepted_p95 <= 1.10
    assert candidate_peak < 700
    assert candidate_peak - accepted_peak <= 64
    assert inference_ratio <= 1.10
    assert cv(inference["accepted"]) < 0.03
    assert cv(inference["candidate"]) < 0.03
    print("TIMING GATE: PASS")


if __name__ == "__main__":
    main()
