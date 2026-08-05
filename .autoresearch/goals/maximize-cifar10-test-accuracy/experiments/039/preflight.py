import argparse
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
BASE = "a7c42dc"
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
        if not kwargs.get("train", args[1] if len(args) > 1 else True):
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
        ["git", "show", f"{BASE}:train.py"], cwd=ROOT, text=True
    )
    module = types.ModuleType("accepted_train")
    module.__file__ = f"git:{BASE}:train.py"
    module.__source__ = source
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def construct(module):
    return module.WideResNet(
        module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES
    )


def optimizer_for(module, model):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return optim.SGD(
        [
            {"params": decay, "weight_decay": module.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=module.MIN_LR,
        momentum=module.MOMENTUM,
        nesterov=True,
    )


def assert_state_equal(left, right, label):
    assert left.keys() == right.keys(), label
    for name in left:
        assert torch.equal(left[name], right[name]), f"{label}.{name}"


def static_scope(accepted):
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "--"], cwd=ROOT, text=True
    ).splitlines()
    assert changed == ["train.py"], changed
    subprocess.run(
        ["git", "diff", "--exit-code", BASE, "--", "prepare.py"],
        cwd=ROOT,
        check=True,
    )
    old = '''def learning_rate(training_time):
    progress = min(max(training_time / TIME_BUDGET_S, 0.0), 1.0)
    if progress < WARMUP_FRACTION:
        warmup_progress = progress / WARMUP_FRACTION
        return MIN_LR + (LR - MIN_LR) * warmup_progress

    cosine_progress = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )
'''
    new = '''def learning_rate(training_time):
    progress = min(max(training_time / TIME_BUDGET_S, 0.0), 1.0)
    if progress < WARMUP_FRACTION:
        warmup_progress = progress / WARMUP_FRACTION
        return MIN_LR + (LR - MIN_LR) * warmup_progress

    cosine_progress = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
    scheduled_lr = MIN_LR + 0.5 * (LR - MIN_LR) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )
    if progress < MIXUP_END_FRACTION:
        return scheduled_lr

    transition_cosine_progress = (
        MIXUP_END_FRACTION - WARMUP_FRACTION
    ) / (1.0 - WARMUP_FRACTION)
    transition_lr = MIN_LR + 0.5 * (LR - MIN_LR) * (
        1.0 + math.cos(math.pi * transition_cosine_progress)
    )
    tail_progress = (progress - MIXUP_END_FRACTION) / (
        1.0 - MIXUP_END_FRACTION
    )
    return MIN_LR + 0.5 * (transition_lr - MIN_LR) * (
        1.0 + math.cos(math.pi * tail_progress)
    )
'''
    candidate = (ROOT / "train.py").read_text()
    assert accepted.__source__.count(old) == 1 and candidate.count(new) == 1
    assert candidate.replace(new, old) == accepted.__source__
    lr_pos = candidate.index("            lr = learning_rate(total_training_time)")
    progress_pos = candidate.index(
        "            progress = min(total_training_time / TIME_BUDGET_S, 1.0)"
    )
    mixup_pos = candidate.index("            use_mixup = progress < MIXUP_END_FRACTION")
    assert lr_pos < progress_pos < mixup_pos
    assert "iterator_exhausted=true" in candidate


def accepted_formula(module, progress):
    if progress < module.WARMUP_FRACTION:
        return module.MIN_LR + (module.LR - module.MIN_LR) * (
            progress / module.WARMUP_FRACTION
        )
    cosine_progress = (progress - module.WARMUP_FRACTION) / (
        1.0 - module.WARMUP_FRACTION
    )
    return module.MIN_LR + 0.5 * (module.LR - module.MIN_LR) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )


def candidate_formula(module, progress):
    accepted_lr = accepted_formula(module, progress)
    if progress < module.MIXUP_END_FRACTION:
        return accepted_lr
    transition_lr = accepted_formula(module, module.MIXUP_END_FRACTION)
    tail_progress = (progress - module.MIXUP_END_FRACTION) / (
        1.0 - module.MIXUP_END_FRACTION
    )
    return module.MIN_LR + 0.5 * (transition_lr - module.MIN_LR) * (
        1.0 + math.cos(math.pi * tail_progress)
    )


def schedule_checks(accepted):
    refs = [0.0, 0.05, 0.65 - 1e-12, 0.65, 0.65 + 1e-12, 0.70, 0.75, 0.825, 0.90, 0.95, 1.0]
    values = []
    for progress in refs:
        old = accepted.learning_rate(progress * accepted.TIME_BUDGET_S)
        new = train.learning_rate(progress * train.TIME_BUDGET_S)
        expected_old = accepted_formula(accepted, progress)
        expected_new = candidate_formula(train, progress)
        assert abs(old - expected_old) <= 1e-12
        assert abs(new - expected_new) <= 1e-12
        if progress < train.MIXUP_END_FRACTION:
            assert old == new
        values.append({"progress": progress, "accepted": old, "candidate": new})

    grid = [index / 10000 for index in range(10001)]
    candidate_values = [train.learning_rate(p * train.TIME_BUDGET_S) for p in grid]
    assert all(math.isfinite(v) and train.MIN_LR <= v <= train.LR for v in candidate_values)
    warmup_end = int(train.WARMUP_FRACTION * 10000)
    post_warmup = candidate_values[warmup_end:]
    maximum_post_warmup_rise = max(b - a for a, b in zip(post_warmup, post_warmup[1:]))
    assert maximum_post_warmup_rise <= 1e-15
    for progress in (0.70, 0.75, 0.825, 0.90, 0.95):
        assert train.learning_rate(progress * train.TIME_BUDGET_S) > accepted.learning_rate(
            progress * accepted.TIME_BUDGET_S
        )
    boundary = train.MIXUP_END_FRACTION
    anchor = accepted_formula(accepted, boundary)
    assert abs(train.learning_rate(boundary * train.TIME_BUDGET_S) - anchor) <= 1e-12
    assert abs(train.learning_rate(train.TIME_BUDGET_S) - train.MIN_LR) <= 1e-12
    left = train.learning_rate((boundary - 1e-12) * train.TIME_BUDGET_S)
    right = train.learning_rate((boundary + 1e-12) * train.TIME_BUDGET_S)
    assert abs(left - right) <= 1e-12

    w, b = train.WARMUP_FRACTION, train.MIXUP_END_FRACTION
    delta = train.LR - train.MIN_LR
    accepted_area = train.MIN_LR * (1 - b) + 0.5 * delta * (
        (1 - b)
        + (1 - w)
        / math.pi
        * (
            math.sin(math.pi) - math.sin(math.pi * (b - w) / (1 - w))
        )
    )
    candidate_area = (1 - b) * (train.MIN_LR + 0.5 * (anchor - train.MIN_LR))
    ratio = candidate_area / accepted_area
    assert abs(ratio - 1.3946300912086436) <= 1e-10
    return {
        "references": values,
        "maximum_post_warmup_grid_rise": maximum_post_warmup_rise,
        "boundary_left_right_gap": abs(left - right),
        "accepted_tail_area": accepted_area,
        "candidate_tail_area": candidate_area,
        "tail_area_ratio": ratio,
    }


def fixture(module, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = construct(module).to(DEVICE).train()
    optimizer = optimizer_for(module, model)
    generator = torch.Generator().manual_seed(seed + 1)
    inputs = torch.randn(64, 3, 32, 32, generator=generator).to(DEVICE)
    targets = torch.arange(64, device=DEVICE) % module.NUM_CLASSES
    return model, optimizer, inputs, targets


def backward_only(module, model, optimizer, inputs, targets, mixup, rng):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.set_rng_state(rng)
    if mixup:
        distribution = torch.distributions.Beta(
            torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
            torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
        )
        mixed, target_a, target_b, coefficient = module.mixup_batch(
            inputs, targets, distribution
        )
        output = model(mixed)
        loss = coefficient * F.cross_entropy(output, target_a) + (
            1.0 - coefficient
        ) * F.cross_entropy(output, target_b)
    else:
        output = model(inputs)
        loss = F.cross_entropy(output, targets)
    loss.backward()
    return loss.detach().clone(), {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def oracle_fixture(accepted, progress, mixup, preseeded):
    seed = 39_100 + int(100 * progress) + 10 * int(preseeded) + int(mixup)
    model_a, opt_a, inputs_a, targets_a = fixture(accepted, seed)
    model_c, opt_c, inputs_c, targets_c = fixture(train, seed)
    assert torch.equal(inputs_a, inputs_c) and torch.equal(targets_a, targets_c)
    assert_state_equal(model_a.state_dict(), model_c.state_dict(), "fixture")
    if preseeded:
        for optimizer, model in ((opt_a, model_a), (opt_c, model_c)):
            for index, parameter in enumerate(model.parameters()):
                optimizer.state[parameter]["momentum_buffer"] = torch.full_like(
                    parameter, 0.0001 * (index + 1)
                )
    lr_a = accepted.learning_rate(progress * accepted.TIME_BUDGET_S)
    lr_c = train.learning_rate(progress * train.TIME_BUDGET_S)
    for optimizer, lr in ((opt_a, lr_a), (opt_c, lr_c)):
        for group in optimizer.param_groups:
            group["lr"] = lr
    rng = torch.cuda.get_rng_state().clone()
    loss_a, grads_a = backward_only(
        accepted, model_a, opt_a, inputs_a, targets_a, mixup, rng
    )
    end_rng_a = torch.cuda.get_rng_state().clone()
    loss_c, grads_c = backward_only(
        train, model_c, opt_c, inputs_c, targets_c, mixup, rng
    )
    end_rng_c = torch.cuda.get_rng_state().clone()
    assert torch.equal(loss_a, loss_c)
    assert torch.equal(end_rng_a, end_rng_c)
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_c[name]), name
    assert_state_equal(model_a.state_dict(), model_c.state_dict(), "pre_step")
    before_a = {name: parameter.detach().clone() for name, parameter in model_a.named_parameters()}
    before_c = {name: parameter.detach().clone() for name, parameter in model_c.named_parameters()}
    previous_a = {
        name: opt_a.state[parameter].get("momentum_buffer", torch.zeros_like(parameter)).clone()
        for name, parameter in model_a.named_parameters()
    }
    previous_c = {
        name: opt_c.state[parameter].get("momentum_buffer", torch.zeros_like(parameter)).clone()
        for name, parameter in model_c.named_parameters()
    }
    opt_a.step()
    opt_c.step()
    names_a, names_c = dict(model_a.named_parameters()), dict(model_c.named_parameters())
    max_parameter_delta = 0.0
    for name in names_a:
        wd = accepted.WEIGHT_DECAY if before_a[name].ndim >= 2 else 0.0
        direction = grads_a[name] + wd * before_a[name]
        if preseeded:
            buffer_a = accepted.MOMENTUM * previous_a[name] + direction
            buffer_c = train.MOMENTUM * previous_c[name] + direction
        else:
            buffer_a = direction
            buffer_c = direction
        expected_a = before_a[name] - lr_a * (
            direction + accepted.MOMENTUM * buffer_a
        )
        expected_c = before_c[name] - lr_c * (
            direction + train.MOMENTUM * buffer_c
        )
        torch.testing.assert_close(names_a[name], expected_a, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(names_c[name], expected_c, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            opt_a.state[names_a[name]]["momentum_buffer"], buffer_a, rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            opt_c.state[names_c[name]]["momentum_buffer"], buffer_c, rtol=1e-6, atol=1e-7
        )
        assert torch.equal(
            opt_a.state[names_a[name]]["momentum_buffer"],
            opt_c.state[names_c[name]]["momentum_buffer"],
        )
        max_parameter_delta = max(
            max_parameter_delta,
            (names_a[name] - names_c[name]).abs().max().item(),
        )
    if progress < train.MIXUP_END_FRACTION or progress == train.MIXUP_END_FRACTION:
        assert lr_a == lr_c and max_parameter_delta == 0.0
    else:
        assert lr_c > lr_a and max_parameter_delta > 0.0
    return {
        "progress": progress,
        "accepted_lr": lr_a,
        "candidate_lr": lr_c,
        "max_parameter_delta": max_parameter_delta,
    }


def semantics():
    accepted = load_accepted()
    assert GuardEval.constructions == 2
    static_scope(accepted)
    torch.empty(1, device=DEVICE)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_a = construct(accepted)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_c = construct(train)
    assert_state_equal(model_a.state_dict(), model_c.state_dict(), "model")
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert sum(p.numel() for p in model_c.parameters()) == 1_003_482
    schedule_cpu = torch.random.get_rng_state().clone()
    schedule_cuda = torch.cuda.get_rng_state().clone()
    schedule = schedule_checks(accepted)
    assert torch.equal(torch.random.get_rng_state(), schedule_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), schedule_cuda)
    updates = [
        oracle_fixture(accepted, 0.50, True, False),
        oracle_fixture(accepted, 0.65, False, False),
        oracle_fixture(accepted, 0.75, False, False),
        oracle_fixture(accepted, 0.90, False, False),
        oracle_fixture(accepted, 0.75, False, True),
    ]
    print(json.dumps({"schedule": schedule, "updates": updates}, sort_keys=True))
    print("SEMANTICS PASS")


def timed_step(module, model, optimizer, host_x, host_y, distribution, mixup, progress):
    inputs = host_x.to(DEVICE, non_blocking=True)
    targets = host_y.to(DEVICE, non_blocking=True)
    lr = module.learning_rate(progress * module.TIME_BUDGET_S)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    if mixup:
        mixed, target_a, target_b, coefficient = module.mixup_batch(
            inputs, targets, distribution
        )
        output = model(mixed)
        loss = coefficient * F.cross_entropy(output, target_a) + (
            1.0 - coefficient
        ) * F.cross_entropy(output, target_b)
    else:
        loss = F.cross_entropy(model(inputs), targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()


def timing_arm(module, mixup, progress, seed, steps):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = construct(module).to(DEVICE).train()
    optimizer = optimizer_for(module, model)
    generator = torch.Generator().manual_seed(seed + 1)
    host_x = torch.randn(256, 3, 32, 32, generator=generator).pin_memory()
    host_y = (torch.arange(256) % 10).pin_memory()
    distribution = torch.distributions.Beta(
        torch.tensor(0.2, device=DEVICE), torch.tensor(0.2, device=DEVICE)
    )
    torch.cuda.manual_seed(seed + 2)
    started = time.perf_counter()
    for _ in range(steps):
        timed_step(
            module, model, optimizer, host_x, host_y, distribution, mixup, progress
        )
    value = 1000 * (time.perf_counter() - started) / steps
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    del model, optimizer, host_x, host_y, distribution
    torch.cuda.empty_cache()
    return value, peak


def timing():
    accepted = load_accepted()
    static_scope(accepted)
    results, peaks = {}, []
    for mixup, progress, regime, regime_index in (
        (True, 0.50, "mixup", 0),
        (False, 0.75, "hard", 1),
    ):
        timing_arm(accepted, mixup, progress, 39_200 + 100 * regime_index, 20)
        timing_arm(train, mixup, progress, 39_210 + 100 * regime_index, 20)
        windows = {"accepted": [], "candidate": []}
        for repetition in range(4):
            order = (
                ("accepted", accepted), ("candidate", train)
            ) if repetition % 2 == 0 else (
                ("candidate", train), ("accepted", accepted)
            )
            for kind, module in order:
                if kind == "candidate":
                    torch.cuda.reset_peak_memory_stats()
                value, peak = timing_arm(
                    module,
                    mixup,
                    progress,
                    39_300 + 100 * regime_index + 10 * repetition,
                    50,
                )
                windows[kind].append(value)
                if kind == "candidate":
                    peaks.append(peak)
        results[regime] = {
            "progress": progress,
            "windows_ms": windows,
            "medians_ms": {
                kind: statistics.median(values) for kind, values in windows.items()
            },
            "cvs": {
                kind: statistics.pstdev(values) / statistics.fmean(values)
                for kind, values in windows.items()
            },
        }
    accepted_mix = results["mixup"]["medians_ms"]["accepted"]
    candidate_mix = results["mixup"]["medians_ms"]["candidate"]
    accepted_hard = results["hard"]["medians_ms"]["accepted"]
    candidate_hard = results["hard"]["medians_ms"]["candidate"]
    retention = (0.65 / candidate_mix + 0.35 / candidate_hard) / (
        0.65 / accepted_mix + 0.35 / accepted_hard
    )
    payload = {
        "results": results,
        "retention": retention,
        "projected_passes": 130.304 * retention,
        "peak_mb": max(peaks),
    }
    print(json.dumps(payload, sort_keys=True))
    assert all(
        cv <= 0.05
        for regime in results.values()
        for cv in regime["cvs"].values()
    )
    assert retention >= 0.974644 and payload["projected_passes"] >= 127
    assert max(peaks) < 2048
    print("TIMING PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("semantics", "timing"))
    args = parser.parse_args()
    semantics() if args.mode == "semantics" else timing()
