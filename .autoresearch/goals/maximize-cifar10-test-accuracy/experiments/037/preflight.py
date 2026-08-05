import argparse
import copy
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


def optimizer_for(module, model, candidate):
    if candidate:
        decay = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.ndim >= 2 and name != "fc.weight"
        ]
        no_decay = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and (parameter.ndim < 2 or name == "fc.weight")
        ]
    else:
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
    old = (
        "    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]\n"
        "    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]\n"
    )
    new = (
        "    decay_params = [\n"
        "        p\n"
        "        for name, p in model.named_parameters()\n"
        "        if p.requires_grad and p.ndim >= 2 and name != \"fc.weight\"\n"
        "    ]\n"
        "    no_decay_params = [\n"
        "        p\n"
        "        for name, p in model.named_parameters()\n"
        "        if p.requires_grad and (p.ndim < 2 or name == \"fc.weight\")\n"
        "    ]\n"
    )
    candidate = (ROOT / "train.py").read_text()
    assert accepted.__source__.count(old) == 1 and candidate.count(new) == 1
    assert candidate.replace(new, old) == accepted.__source__


def group_checks(accepted):
    torch.empty(1, device=DEVICE)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_a = construct(accepted)
    cpu = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state().clone()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_c = construct(train)
    assert torch.equal(torch.random.get_rng_state(), cpu)
    assert torch.equal(torch.cuda.get_rng_state(), cuda)
    assert_state_equal(model_a.state_dict(), model_c.state_dict(), "model")
    assert sum(p.numel() for p in model_c.parameters()) == 1_003_482
    opt_a = optimizer_for(accepted, model_a, False)
    opt_c = optimizer_for(train, model_c, True)
    names = {id(p): name for name, p in model_c.named_parameters()}
    candidate_groups = [
        [names[id(p)] for p in group["params"]] for group in opt_c.param_groups
    ]
    flat = [name for group in candidate_groups for name in group]
    assert len(flat) == len(set(flat)) == len(list(model_c.parameters()))
    assert sum(p.numel() for p in opt_c.param_groups[0]["params"]) == 999_856
    assert sum(p.numel() for p in opt_c.param_groups[1]["params"]) == 3_626
    assert "fc.weight" not in candidate_groups[0] and "fc.weight" in candidate_groups[1]
    assert {"pooled_head.0.weight", "pooled_head.2.weight"} <= set(candidate_groups[0])
    assert [g["weight_decay"] for g in opt_c.param_groups] == [5e-4, 0.0]
    for key in ("lr", "momentum", "dampening", "nesterov", "maximize", "foreach"):
        assert [g[key] for g in opt_a.param_groups] == [g[key] for g in opt_c.param_groups]


def fixture(module, candidate, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = construct(module).to(DEVICE).train()
    optimizer = optimizer_for(module, model, candidate)
    generator = torch.Generator().manual_seed(seed + 1)
    inputs = torch.randn(64, 3, 32, 32, generator=generator).to(DEVICE)
    targets = torch.arange(64, device=DEVICE) % module.NUM_CLASSES
    return model, optimizer, inputs, targets


def backward_step(module, model, optimizer, inputs, targets, mixup, rng):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.set_rng_state(rng)
    if mixup:
        dist = torch.distributions.Beta(
            torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
            torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
        )
        mixed, a, b, coefficient = module.mixup_batch(inputs, targets, dist)
        output = model(mixed)
        loss = coefficient * F.cross_entropy(output, a) + (1 - coefficient) * F.cross_entropy(output, b)
    else:
        output = model(inputs)
        loss = F.cross_entropy(output, targets)
    loss.backward()
    return loss.detach().clone(), {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def oracle_fixture(accepted, mixup, preseeded):
    model_a, opt_a, inputs_a, targets_a = fixture(accepted, False, 37_100 + int(mixup) + 10 * int(preseeded))
    model_c, opt_c, inputs_c, targets_c = fixture(train, True, 37_100 + int(mixup) + 10 * int(preseeded))
    assert torch.equal(inputs_a, inputs_c) and torch.equal(targets_a, targets_c)
    if preseeded:
        for optimizer, model in ((opt_a, model_a), (opt_c, model_c)):
            for index, parameter in enumerate(model.parameters()):
                optimizer.state[parameter]["momentum_buffer"] = torch.full_like(parameter, 0.001 * (index + 1))
    lr = 0.037
    for optimizer in (opt_a, opt_c):
        for group in optimizer.param_groups:
            group["lr"] = lr
    rng = torch.cuda.get_rng_state().clone()
    loss_a, grads_a = backward_step(accepted, model_a, opt_a, inputs_a, targets_a, mixup, rng)
    loss_c, grads_c = backward_step(train, model_c, opt_c, inputs_c, targets_c, mixup, rng)
    assert torch.equal(loss_a, loss_c)
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_c[name]), name
    before_a = {n: p.detach().clone() for n, p in model_a.named_parameters()}
    before_c = {n: p.detach().clone() for n, p in model_c.named_parameters()}
    prev_a = {n: opt_a.state[p].get("momentum_buffer", torch.zeros_like(p)).clone() for n, p in model_a.named_parameters()}
    prev_c = {n: opt_c.state[p].get("momentum_buffer", torch.zeros_like(p)).clone() for n, p in model_c.named_parameters()}
    opt_a.step()
    opt_c.step()
    names_a = dict(model_a.named_parameters())
    names_c = dict(model_c.named_parameters())
    max_fc_delta = 0.0
    for name in names_a:
        pa0, pc0 = before_a[name], before_c[name]
        ga, gc = grads_a[name], grads_c[name]
        wd_a = 5e-4 if pa0.ndim >= 2 else 0.0
        wd_c = 5e-4 if pc0.ndim >= 2 and name != "fc.weight" else 0.0
        da, dc = ga + wd_a * pa0, gc + wd_c * pc0
        if preseeded:
            ba, bc = 0.9 * prev_a[name] + da, 0.9 * prev_c[name] + dc
        else:
            ba, bc = da, dc
        expected_a = pa0 - lr * (da + 0.9 * ba)
        expected_c = pc0 - lr * (dc + 0.9 * bc)
        torch.testing.assert_close(names_a[name], expected_a, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(names_c[name], expected_c, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(opt_a.state[names_a[name]]["momentum_buffer"], ba, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(opt_c.state[names_c[name]]["momentum_buffer"], bc, rtol=1e-6, atol=1e-7)
        if name != "fc.weight":
            assert torch.equal(names_a[name], names_c[name]), name
        else:
            max_fc_delta = (names_a[name] - names_c[name]).abs().max().item()
    assert max_fc_delta > 0 and math.isfinite(max_fc_delta)
    return max_fc_delta


def semantics():
    accepted = load_accepted()
    assert GuardEval.constructions == 2
    static_scope(accepted)
    group_checks(accepted)
    deltas = {
        "first_mixup_fc_delta": oracle_fixture(accepted, True, False),
        "preseeded_hard_fc_delta": oracle_fixture(accepted, False, True),
    }
    print(json.dumps(deltas, sort_keys=True))
    print("SEMANTICS PASS")


def timed_step(module, model, optimizer, host_x, host_y, dist, mixup):
    x = host_x.to(DEVICE, non_blocking=True)
    y = host_y.to(DEVICE, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    if mixup:
        mixed, a, b, coefficient = module.mixup_batch(x, y, dist)
        output = model(mixed)
        loss = coefficient * F.cross_entropy(output, a) + (1 - coefficient) * F.cross_entropy(output, b)
    else:
        loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()


def timing_arm(module, candidate, mixup, seed, steps):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = construct(module).to(DEVICE).train()
    optimizer = optimizer_for(module, model, candidate)
    generator = torch.Generator().manual_seed(seed + 1)
    host_x = torch.randn(256, 3, 32, 32, generator=generator).pin_memory()
    host_y = (torch.arange(256) % 10).pin_memory()
    dist = torch.distributions.Beta(torch.tensor(0.2, device=DEVICE), torch.tensor(0.2, device=DEVICE))
    torch.cuda.manual_seed(seed + 2)
    started = time.perf_counter()
    for _ in range(steps):
        timed_step(module, model, optimizer, host_x, host_y, dist, mixup)
    value = 1000 * (time.perf_counter() - started) / steps
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    del model, optimizer, host_x, host_y, dist
    torch.cuda.empty_cache()
    return value, peak


def timing():
    accepted = load_accepted()
    static_scope(accepted)
    results, peaks = {}, []
    for mixup, ri, regime in ((True, 0, "mixup"), (False, 1, "hard")):
        timing_arm(accepted, False, mixup, 37_200 + 100 * ri, 20)
        timing_arm(train, True, mixup, 37_210 + 100 * ri, 20)
        windows = {"accepted": [], "candidate": []}
        for rep in range(4):
            order = ("accepted", "candidate") if rep % 2 == 0 else ("candidate", "accepted")
            for kind in order:
                candidate = kind == "candidate"
                if candidate:
                    torch.cuda.reset_peak_memory_stats()
                value, peak = timing_arm(train if candidate else accepted, candidate, mixup, 37_300 + 100 * ri + 10 * rep, 50)
                windows[kind].append(value)
                if candidate:
                    peaks.append(peak)
        results[regime] = {
            "windows_ms": windows,
            "medians_ms": {k: statistics.median(v) for k, v in windows.items()},
            "cvs": {k: statistics.pstdev(v) / statistics.fmean(v) for k, v in windows.items()},
        }
    am, cm = results["mixup"]["medians_ms"].values()
    ah, ch = results["hard"]["medians_ms"].values()
    retention = (0.65 / cm + 0.35 / ch) / (0.65 / am + 0.35 / ah)
    payload = {"results": results, "retention": retention, "projected_passes": 130.304 * retention, "peak_mb": max(peaks)}
    print(json.dumps(payload, sort_keys=True))
    assert all(cv <= 0.05 for regime in results.values() for cv in regime["cvs"].values())
    assert retention >= 0.974644 and payload["projected_passes"] >= 127
    assert max(peaks) < 2048
    print("TIMING PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("semantics", "timing"))
    args = parser.parse_args()
    semantics() if args.mode == "semantics" else timing()
