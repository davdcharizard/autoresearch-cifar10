import argparse
import copy
import hashlib
import inspect
import json
import math
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets


ROOT = Path(__file__).resolve().parents[5]
BASE_COMMIT = "67c8e98"
DEVICE = torch.device("cuda")
ORIGINAL_CIFAR10 = datasets.CIFAR10


class GuardEval:
    constructions = 0

    def __init__(self):
        type(self).constructions += 1

    def evaluate(self, model, device):
        raise AssertionError("preflight may not evaluate")


class GuardedCIFAR10(ORIGINAL_CIFAR10):
    def __init__(self, *args, **kwargs):
        train_split = kwargs.get("train", args[1] if len(args) > 1 else True)
        if not train_split:
            raise AssertionError("preflight may not construct test data")
        super().__init__(*args, **kwargs)


sys.path.insert(0, str(ROOT))
import prepare


prepare.Eval = GuardEval
datasets.CIFAR10 = GuardedCIFAR10
import train


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def load_accepted():
    source = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:train.py"], cwd=ROOT, text=True
    )
    module = types.ModuleType("accepted_train")
    module.__file__ = f"git:{BASE_COMMIT}:train.py"
    module.__source__ = source
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def optimizer_for(module, model):
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return optim.SGD(
        [
            {"params": decay, "weight_decay": module.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=module.MIN_LR,
        momentum=module.MOMENTUM,
        nesterov=True,
    )


def optimizer_signature(optimizer, model):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    signature = []
    for group in optimizer.param_groups:
        values = {key: value for key, value in group.items() if key != "params"}
        values["params"] = [names[id(parameter)] for parameter in group["params"]]
        signature.append(values)
    return signature


def assert_tensor_dict_equal(left, right, label):
    assert left.keys() == right.keys(), label
    for key in left:
        assert torch.equal(left[key], right[key]), f"{label}.{key}"


def assert_tree_equal(left, right, label):
    assert type(left) is type(right), label
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), label
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), label
        for key in left:
            assert_tree_equal(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), label
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_tree_equal(left_item, right_item, f"{label}.{index}")
    else:
        assert left == right, label


def static_scope_checks(accepted):
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_COMMIT, "--"], cwd=ROOT, text=True
    ).splitlines()
    assert changed == ["train.py"], changed
    subprocess.run(
        ["git", "diff", "--exit-code", BASE_COMMIT, "--", "prepare.py"],
        cwd=ROOT,
        check=True,
    )
    candidate_source = (ROOT / "train.py").read_text()
    assert accepted.__source__.count("MIXUP_ALPHA = 0.2") == 1
    expected_source = accepted.__source__.replace(
        "MIXUP_ALPHA = 0.2", "MIXUP_ALPHA = 0.1"
    )
    assert candidate_source == expected_source
    diff = subprocess.check_output(
        ["git", "diff", "--unified=0", BASE_COMMIT, "--", "train.py"],
        cwd=ROOT,
        text=True,
    )
    removed = [
        line
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    added = [
        line
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert removed == ["-MIXUP_ALPHA = 0.2"], removed
    assert added == ["+MIXUP_ALPHA = 0.1"], added
    assert train.MIXUP_ALPHA == 0.1 and accepted.MIXUP_ALPHA == 0.2


def construction_checks(accepted):
    torch.empty(1, device=DEVICE)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    accepted_model = accepted.WideResNet(
        accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES
    )
    accepted_cpu_state = torch.random.get_rng_state().clone()
    accepted_cuda_state = torch.cuda.get_rng_state().clone()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate_model = train.WideResNet(
        train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES
    )
    assert torch.equal(torch.random.get_rng_state(), accepted_cpu_state)
    assert torch.equal(torch.cuda.get_rng_state(), accepted_cuda_state)
    assert_tensor_dict_equal(
        accepted_model.state_dict(), candidate_model.state_dict(), "initial_model"
    )
    assert sum(parameter.numel() for parameter in candidate_model.parameters()) == 987_098
    assert all(parameter.dtype == torch.float32 for parameter in candidate_model.parameters())

    accepted_optimizer = optimizer_for(accepted, accepted_model)
    candidate_optimizer = optimizer_for(train, candidate_model)
    assert optimizer_signature(accepted_optimizer, accepted_model) == optimizer_signature(
        candidate_optimizer, candidate_model
    )
    for progress in (0.0, 0.025, 0.05, 0.5, 0.65, 1.0):
        seconds = progress * prepare.TIME_BUDGET_S
        assert train.learning_rate(seconds) == accepted.learning_rate(seconds)

    unchanged = (
        "STAGE_BLOCKS",
        "WIDEN_FACTOR",
        "NUM_CLASSES",
        "BATCH_SIZE",
        "LR",
        "MIN_LR",
        "WARMUP_FRACTION",
        "MOMENTUM",
        "WEIGHT_DECAY",
        "MAX_STEPS",
        "EVAL_EVERY",
        "MIXUP_END_FRACTION",
        "RANDAUGMENT_END_FRACTION",
    )
    for name in unchanged:
        assert getattr(train, name) == getattr(accepted, name), name
    for name in (
        "EarlyRandAugment",
        "make_train_transform",
        "PreActBlock",
        "WideResNet",
        "learning_rate",
        "mixup_batch",
        "main",
    ):
        assert inspect.getsource(getattr(train, name)) in accepted.__source__, name

    accepted_transform = accepted.make_train_transform(None)
    candidate_transform = train.make_train_transform(None)
    assert [type(op).__name__ for op in accepted_transform.transforms] == [
        type(op).__name__ for op in candidate_transform.transforms
    ]
    accepted_active_transform = accepted.make_train_transform(
        __import__("multiprocessing").get_context().Value("b", 1, lock=False)
    )
    candidate_active_transform = train.make_train_transform(
        __import__("multiprocessing").get_context().Value("b", 1, lock=False)
    )
    assert [type(op).__name__ for op in accepted_active_transform.transforms] == [
        type(op).__name__ for op in candidate_active_transform.transforms
    ]


def beta_distribution(alpha):
    concentration = torch.tensor(alpha, device=DEVICE)
    assert concentration.ndim == 0
    assert concentration.dtype == torch.float32
    assert concentration.device.type == "cuda"
    return torch.distributions.Beta(concentration, concentration)


def distribution_statistics(alpha, seed):
    torch.cuda.manual_seed(seed)
    samples = beta_distribution(alpha).sample((100_000,))
    assert samples.shape == (100_000,)
    assert torch.isfinite(samples).all()
    assert ((samples >= 0.0) & (samples <= 1.0)).all()
    result = {
        "mean": samples.mean().item(),
        "variance": samples.var(unbiased=False).item(),
        "central_mass": ((samples >= 0.2) & (samples <= 0.8)).float().mean().item(),
        "endpoint_mass": ((samples <= 0.1) | (samples >= 0.9)).float().mean().item(),
    }
    del samples
    return result


def clone_grads(model):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def replay_early_step(model, optimizer, inputs, targets, base_model, base_optimizer, rng):
    model.load_state_dict(base_model)
    optimizer.load_state_dict(base_optimizer)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.set_rng_state(rng)
    distribution = beta_distribution(train.MIXUP_ALPHA)
    mixed, target_a, target_b, coefficient = train.mixup_batch(
        inputs, targets, distribution
    )
    outputs = model(mixed)
    loss = coefficient * F.cross_entropy(outputs, target_a) + (
        1.0 - coefficient
    ) * F.cross_entropy(outputs, target_b)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = clone_grads(model)
    optimizer.step()
    torch.cuda.synchronize()
    return {
        "coefficient": coefficient.detach().clone(),
        "mixed": mixed.detach().clone(),
        "target_a": target_a.detach().clone(),
        "target_b": target_b.detach().clone(),
        "loss": loss.detach().clone(),
        "grads": gradients,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cuda_rng": torch.cuda.get_rng_state().clone(),
    }


def run_hard_step(module, base_state, inputs, targets, rng):
    model = module.WideResNet(
        module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES
    ).to(DEVICE)
    model.load_state_dict(base_state)
    optimizer = optimizer_for(module, model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.set_rng_state(rng)
    cpu_rng = torch.random.get_rng_state().clone()
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    gradients = clone_grads(model)
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "loss": loss.detach().clone(),
        "grads": gradients,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cpu_rng_before": cpu_rng,
        "cpu_rng_after": torch.random.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state().clone(),
    }
    del model, optimizer
    return result


def training_semantic_checks(accepted):
    # Warm the exact kernels before requiring bitwise replay.
    torch.manual_seed(35_010)
    torch.cuda.manual_seed(35_010)
    warm_model = train.WideResNet(
        train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES
    ).to(DEVICE)
    warm_inputs = torch.randn(8, 3, 32, 32, device=DEVICE)
    F.cross_entropy(warm_model(warm_inputs), torch.arange(8, device=DEVICE)).backward()
    torch.cuda.synchronize()
    del warm_model, warm_inputs

    generator = torch.Generator().manual_seed(35_011)
    host_inputs = torch.randn(
        train.BATCH_SIZE, 3, 32, 32, generator=generator
    ).pin_memory()
    host_targets = (torch.arange(train.BATCH_SIZE) % train.NUM_CLASSES).pin_memory()
    inputs = host_inputs.to(DEVICE, non_blocking=True)
    targets = host_targets.to(DEVICE, non_blocking=True)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = train.WideResNet(
        train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES
    ).to(DEVICE)
    optimizer = optimizer_for(train, model)
    base_model = copy.deepcopy(model.state_dict())
    base_optimizer = copy.deepcopy(optimizer.state_dict())
    replay_rng = torch.cuda.get_rng_state().clone()
    first = replay_early_step(
        model, optimizer, inputs, targets, base_model, base_optimizer, replay_rng
    )
    second = replay_early_step(
        model, optimizer, inputs, targets, base_model, base_optimizer, replay_rng
    )
    assert_tree_equal(first, second, "candidate_replay")
    assert first["coefficient"].ndim == 0

    torch.cuda.manual_seed(35_012)
    oracle_inputs = torch.arange(
        8, dtype=torch.float32, device=DEVICE
    ).reshape(8, 1, 1, 1).expand(8, 3, 2, 2)
    oracle_targets = torch.arange(8, device=DEVICE)
    oracle_mixed, oracle_a, oracle_b, oracle_coefficient = train.mixup_batch(
        oracle_inputs, oracle_targets, beta_distribution(train.MIXUP_ALPHA)
    )
    permutation = oracle_b
    expected_mixed = oracle_coefficient * oracle_inputs + (
        1.0 - oracle_coefficient
    ) * oracle_inputs[permutation]
    assert oracle_coefficient.ndim == 0
    assert torch.equal(oracle_a, oracle_targets)
    assert torch.equal(oracle_mixed, expected_mixed)
    logits = torch.randn(8, train.NUM_CLASSES, device=DEVICE)
    production_loss = oracle_coefficient * F.cross_entropy(logits, oracle_a) + (
        1.0 - oracle_coefficient
    ) * F.cross_entropy(logits, oracle_b)
    reference_loss = (
        oracle_coefficient * F.cross_entropy(logits, oracle_targets)
        + (1.0 - oracle_coefficient) * F.cross_entropy(logits, permutation)
    )
    assert torch.equal(production_loss, reference_loss)

    hard_base = copy.deepcopy(base_model)
    hard_rng = torch.cuda.get_rng_state().clone()
    cpu_rng = torch.random.get_rng_state().clone()
    accepted_hard = run_hard_step(accepted, hard_base, inputs, targets, hard_rng)
    torch.random.set_rng_state(cpu_rng)
    candidate_hard = run_hard_step(train, hard_base, inputs, targets, hard_rng)
    assert_tree_equal(accepted_hard, candidate_hard, "hard_path")

    probes = [0.65 - 1e-12, 0.65, 0.65 + 1e-12]
    assert [progress < train.MIXUP_END_FRACTION for progress in probes] == [
        True,
        False,
        False,
    ]
    for progress in probes:
        seconds = progress * prepare.TIME_BUDGET_S
        assert train.learning_rate(seconds) == accepted.learning_rate(seconds)
    main_source = inspect.getsource(train.main)
    assert "use_mixup = progress < MIXUP_END_FRACTION" in main_source
    assert main_source.count("randaugment_active.value = 0") == 1
    assert "and not budget_exhausted" in main_source
    assert main_source.count("evaluator.evaluate(model, device)") == 1
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    assert peak < 90_000
    coefficient_hash = hashlib.sha256(
        first["coefficient"].cpu().numpy().tobytes()
    ).hexdigest()
    return {"peak_vram_mb": peak, "replay_coefficient_sha256": coefficient_hash}


def semantic_checks():
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    accepted = load_accepted()
    assert GuardEval.constructions == 2
    static_scope_checks(accepted)
    construction_checks(accepted)
    candidate_stats = distribution_statistics(0.1, 35_001)
    accepted_stats = distribution_statistics(0.2, 35_001)
    print(
        json.dumps(
            {"alpha_0.1": candidate_stats, "alpha_0.2": accepted_stats},
            sort_keys=True,
        )
    )
    assert 0.495 <= candidate_stats["mean"] <= 0.505
    assert 0.203 <= candidate_stats["variance"] <= 0.214
    assert 0.115 <= candidate_stats["central_mass"] <= 0.127
    assert 0.806 <= candidate_stats["endpoint_mass"] <= 0.820
    assert candidate_stats["variance"] > accepted_stats["variance"]
    assert candidate_stats["central_mass"] < accepted_stats["central_mass"]
    assert candidate_stats["endpoint_mass"] > accepted_stats["endpoint_mass"]
    training_checks = training_semantic_checks(accepted)
    print(
        json.dumps(
            {
                "params": 987_098,
                "candidate_alpha": train.MIXUP_ALPHA,
                "accepted_alpha": accepted.MIXUP_ALPHA,
                **training_checks,
            },
            sort_keys=True,
        )
    )
    print("SEMANTICS PASS")


def timed_step(module, model, optimizer, host_inputs, host_targets, distribution, mixup):
    inputs = host_inputs.to(DEVICE, non_blocking=True)
    targets = host_targets.to(DEVICE, non_blocking=True)
    for group in optimizer.param_groups:
        group["lr"] = module.MIN_LR
    optimizer.zero_grad(set_to_none=True)
    if mixup:
        mixed, target_a, target_b, coefficient = module.mixup_batch(
            inputs, targets, distribution
        )
        outputs = model(mixed)
        loss = coefficient * F.cross_entropy(outputs, target_a) + (
            1.0 - coefficient
        ) * F.cross_entropy(outputs, target_b)
    else:
        loss = F.cross_entropy(model(inputs), targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()


def make_timing_arm(module, fixture_seed):
    torch.manual_seed(fixture_seed)
    torch.cuda.manual_seed(fixture_seed)
    model = module.WideResNet(
        module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES
    ).to(DEVICE).train()
    optimizer = optimizer_for(module, model)
    generator = torch.Generator().manual_seed(fixture_seed + 1)
    host_inputs = torch.randn(
        module.BATCH_SIZE, 3, 32, 32, generator=generator
    ).pin_memory()
    host_targets = (
        torch.arange(module.BATCH_SIZE) % module.NUM_CLASSES
    ).pin_memory()
    distribution = beta_distribution(module.MIXUP_ALPHA)
    torch.cuda.manual_seed(fixture_seed + 2)
    return model, optimizer, host_inputs, host_targets, distribution


def warm_timing_arm(module, mixup, fixture_seed):
    arm = make_timing_arm(module, fixture_seed)
    for _ in range(20):
        timed_step(module, *arm, mixup)
    del arm
    torch.cuda.empty_cache()


def timing_window(module, mixup, fixture_seed):
    model, optimizer, host_inputs, host_targets, distribution = make_timing_arm(
        module, fixture_seed
    )
    started = time.perf_counter()
    for _ in range(50):
        timed_step(
            module,
            model,
            optimizer,
            host_inputs,
            host_targets,
            distribution,
            mixup,
        )
    elapsed_ms = 1000.0 * (time.perf_counter() - started) / 50
    del model, optimizer, host_inputs, host_targets, distribution
    torch.cuda.empty_cache()
    return elapsed_ms


def population_cv(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else float("inf")


def timing_checks():
    assert torch.cuda.is_available()
    accepted = load_accepted()
    static_scope_checks(accepted)
    torch.cuda.reset_peak_memory_stats()
    results = {}
    for mixup, regime_index, regime in ((True, 0, "mixup"), (False, 1, "hard")):
        warm_timing_arm(accepted, mixup, 35_100 + regime_index * 100)
        warm_timing_arm(train, mixup, 35_110 + regime_index * 100)
        windows = {"accepted": [], "candidate": []}
        for replicate in range(4):
            order = (
                ("accepted", "candidate")
                if replicate % 2 == 0
                else ("candidate", "accepted")
            )
            pair_seed = 35_200 + regime_index * 100 + replicate * 10
            for kind in order:
                module = accepted if kind == "accepted" else train
                windows[kind].append(timing_window(module, mixup, pair_seed))
        medians = {
            kind: statistics.median(values) for kind, values in windows.items()
        }
        cvs = {kind: population_cv(values) for kind, values in windows.items()}
        results[regime] = {
            "windows_ms_per_step": windows,
            "medians_ms_per_step": medians,
            "population_cvs": cvs,
        }

    accepted_mixup = results["mixup"]["medians_ms_per_step"]["accepted"]
    candidate_mixup = results["mixup"]["medians_ms_per_step"]["candidate"]
    accepted_hard = results["hard"]["medians_ms_per_step"]["accepted"]
    candidate_hard = results["hard"]["medians_ms_per_step"]["candidate"]
    retention = (0.65 / candidate_mixup + 0.35 / candidate_hard) / (
        0.65 / accepted_mixup + 0.35 / accepted_hard
    )
    projected_passes = 133.00736 * retention
    hard_ratio = candidate_hard / accepted_hard
    payload = {
        "results": results,
        "retention": retention,
        "projected_passes": projected_passes,
        "hard_candidate_over_accepted": hard_ratio,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }
    print(json.dumps(payload, sort_keys=True))
    assert all(
        math.isfinite(value)
        for regime in results.values()
        for values in regime["windows_ms_per_step"].values()
        for value in values
    )
    assert all(
        cv <= 0.05
        for regime in results.values()
        for cv in regime["population_cvs"].values()
    ), results
    assert 0.98 <= hard_ratio <= 1.02, hard_ratio
    assert retention >= 0.9774, retention
    assert projected_passes >= 130.0, projected_passes
    assert payload["peak_vram_mb"] < 90_000
    print("TIMING PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("semantics", "timing"))
    args = parser.parse_args()
    if args.mode == "semantics":
        semantic_checks()
    else:
        timing_checks()


if __name__ == "__main__":
    main()
