import argparse
import copy
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
import torch.nn as nn
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


def tensor_dict_equal(left, right, label):
    assert left.keys() == right.keys(), label
    for name in left:
        assert torch.equal(left[name], right[name]), f"{label}.{name}"


def static_scope(accepted):
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_COMMIT, "--"], cwd=ROOT, text=True
    ).splitlines()
    assert changed == ["train.py"], changed
    subprocess.run(
        ["git", "diff", "--exit-code", BASE_COMMIT, "--", "prepare.py"],
        cwd=ROOT,
        check=True,
    )
    candidate = (ROOT / "train.py").read_text()
    constants = (
        "POOLED_HEAD_WIDTH = 64\n"
        "POOLED_HEAD_SCALE = 0.1\n"
        "POOLED_HEAD_INIT_SEED = 36036\n"
    )
    construction = (
        "        with torch.random.fork_rng(devices=[]):\n"
        "            torch.random.default_generator.manual_seed(POOLED_HEAD_INIT_SEED)\n"
        "            self.pooled_head = nn.Sequential(\n"
        "                nn.Linear(widths[2], POOLED_HEAD_WIDTH, bias=False),\n"
        "                nn.ReLU(),\n"
        "                nn.Linear(POOLED_HEAD_WIDTH, widths[2], bias=False),\n"
        "            )\n"
        "            init.kaiming_normal_(self.pooled_head[0].weight)\n"
        "            init.kaiming_normal_(self.pooled_head[2].weight)\n"
    )
    forward = "        out = out + POOLED_HEAD_SCALE * self.pooled_head(out)\n"
    assert candidate.count(constants) == 1
    assert candidate.count(construction) == 1
    assert candidate.count(forward) == 1
    normalized = candidate.replace(constants, "").replace(construction, "").replace(
        forward, ""
    )
    assert normalized == accepted.__source__
    assert "torch.manual_seed(POOLED_HEAD_INIT_SEED)" not in candidate


def construct(module):
    return module.WideResNet(
        module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES
    )


def common_construction_checks(accepted):
    torch.empty(1, device=DEVICE)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    accepted_model = construct(accepted)
    accepted_cpu = torch.random.get_rng_state().clone()
    accepted_cuda = torch.cuda.get_rng_state().clone()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate_model = construct(train)
    assert torch.equal(torch.random.get_rng_state(), accepted_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), accepted_cuda)
    candidate_state = candidate_model.state_dict()
    common_state = {
        name: value for name, value in candidate_state.items() if not name.startswith("pooled_head.")
    }
    tensor_dict_equal(accepted_model.state_dict(), common_state, "common_state")
    assert sum(parameter.numel() for parameter in accepted_model.parameters()) == 987_098
    assert sum(parameter.numel() for parameter in candidate_model.parameters()) == 1_003_482

    head = candidate_model.pooled_head
    assert len(head) == 3
    assert isinstance(head[0], nn.Linear) and isinstance(head[1], nn.ReLU)
    assert isinstance(head[2], nn.Linear)
    assert head[0].in_features == 128 and head[0].out_features == 64
    assert head[2].in_features == 64 and head[2].out_features == 128
    assert head[0].bias is None and head[2].bias is None
    assert sum(parameter.numel() for parameter in head.parameters()) == 16_384

    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(36_036)
        reference_head = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 128, bias=False),
        )
        nn.init.kaiming_normal_(reference_head[0].weight)
        nn.init.kaiming_normal_(reference_head[2].weight)
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_before)
    tensor_dict_equal(head.state_dict(), reference_head.state_dict(), "head_seed")
    return accepted_model, candidate_model


def pooled_features(model, inputs):
    out = model.conv1(inputs)
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = F.relu(model.bn(out))
    return F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)


def group_norm(named_parameters, prefix=None, exclude=()):
    squares = []
    for name, parameter in named_parameters:
        if prefix is not None and not name.startswith(prefix):
            continue
        if any(name.startswith(item) for item in exclude):
            continue
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            squares.append(parameter.grad.detach().float().square().sum())
    assert squares
    return torch.stack(squares).sum().sqrt().item()


def semantic_checks():
    assert torch.cuda.is_available()
    accepted = load_accepted()
    assert GuardEval.constructions == 2
    static_scope(accepted)
    accepted_cpu_model, candidate_cpu_model = common_construction_checks(accepted)

    accepted_model = accepted_cpu_model.to(DEVICE).train()
    candidate_model = candidate_cpu_model.to(DEVICE).train()
    generator = torch.Generator().manual_seed(36_100)
    inputs = torch.randn(32, 3, 32, 32, generator=generator).to(DEVICE)
    targets = (torch.arange(32, device=DEVICE) % train.NUM_CLASSES)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    accepted_pooled = pooled_features(accepted_model, inputs)
    candidate_pooled = pooled_features(candidate_model, inputs)
    assert torch.equal(accepted_pooled, candidate_pooled)
    direct_logits = candidate_model.fc(candidate_pooled)
    accepted_logits = accepted_model(inputs)
    assert torch.equal(direct_logits, accepted_logits)
    branch = candidate_model.pooled_head(candidate_pooled)
    classifier_input = candidate_pooled + train.POOLED_HEAD_SCALE * branch
    expected_logits = candidate_model.fc(classifier_input)
    candidate_logits = candidate_model(inputs)
    assert torch.equal(candidate_logits, expected_logits)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    branch_ratio = (
        (train.POOLED_HEAD_SCALE * branch).norm() / candidate_pooled.norm()
    ).item()
    logit_delta = candidate_logits - direct_logits
    diagnostics = {
        "branch_direct_norm_ratio": branch_ratio,
        "logit_delta_max_abs": logit_delta.abs().max().item(),
        "logit_delta_rms": logit_delta.square().mean().sqrt().item(),
    }
    assert all(math.isfinite(value) and value > 0 for value in diagnostics.values())

    candidate_optimizer = optimizer_for(train, candidate_model)
    names = {id(parameter): name for name, parameter in candidate_model.named_parameters()}
    grouped = [
        names[id(parameter)]
        for group in candidate_optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(grouped) == len(set(grouped)) == len(list(candidate_model.parameters()))
    decay_names = {
        names[id(parameter)] for parameter in candidate_optimizer.param_groups[0]["params"]
    }
    assert {"pooled_head.0.weight", "pooled_head.2.weight"} <= decay_names
    assert candidate_optimizer.param_groups[0]["weight_decay"] == train.WEIGHT_DECAY
    assert candidate_optimizer.param_groups[1]["weight_decay"] == 0.0

    distribution = torch.distributions.Beta(
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
    )
    torch.cuda.manual_seed(36_101)
    mixed, target_a, target_b, coefficient = train.mixup_batch(
        inputs, targets, distribution
    )
    assert coefficient.ndim == 0
    candidate_optimizer.zero_grad(set_to_none=True)
    outputs = candidate_model(mixed)
    loss = coefficient * F.cross_entropy(outputs, target_a) + (
        1.0 - coefficient
    ) * F.cross_entropy(outputs, target_b)
    assert torch.isfinite(loss)
    loss.backward()
    named = list(candidate_model.named_parameters())
    gradient_norms = {
        "backbone": group_norm(named, exclude=("fc.", "pooled_head.")),
        "classifier": group_norm(named, prefix="fc."),
        "head_first": candidate_model.pooled_head[0].weight.grad.norm().item(),
        "head_second": candidate_model.pooled_head[2].weight.grad.norm().item(),
    }
    assert all(math.isfinite(value) and value > 0 for value in gradient_norms.values())
    before_head = {
        name: parameter.detach().clone()
        for name, parameter in candidate_model.named_parameters()
        if name.startswith("pooled_head.")
    }
    candidate_optimizer.step()
    for name, parameter in candidate_model.named_parameters():
        if name in before_head:
            assert not torch.equal(before_head[name], parameter)

    candidate_optimizer.zero_grad(set_to_none=True)
    hard_loss = F.cross_entropy(candidate_model(inputs), targets)
    assert torch.isfinite(hard_loss)
    hard_loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in candidate_model.parameters()
    )
    candidate_optimizer.step()

    for progress in (0.65 - 1e-12, 0.65, 0.65 + 1e-12):
        assert (progress < train.MIXUP_END_FRACTION) == (progress < 0.65)
        seconds = progress * prepare.TIME_BUDGET_S
        assert train.learning_rate(seconds) == accepted.learning_rate(seconds)
    main_source = inspect.getsource(train.main)
    assert main_source.count("evaluator.evaluate(model, device)") == 1
    assert main_source.count("randaugment_active.value = 0") == 1
    print(json.dumps({**diagnostics, "gradient_norms": gradient_norms}, sort_keys=True))
    print(json.dumps({"params": 1_003_482, "common_params": 987_098}, sort_keys=True))
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


def timing_arm(module, mixup, seed, steps, measure):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = construct(module).to(DEVICE).train()
    optimizer = optimizer_for(module, model)
    generator = torch.Generator().manual_seed(seed + 1)
    host_inputs = torch.randn(
        module.BATCH_SIZE, 3, 32, 32, generator=generator
    ).pin_memory()
    host_targets = (torch.arange(module.BATCH_SIZE) % module.NUM_CLASSES).pin_memory()
    distribution = torch.distributions.Beta(
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
    )
    torch.cuda.manual_seed(seed + 2)
    started = time.perf_counter()
    for _ in range(steps):
        timed_step(
            module, model, optimizer, host_inputs, host_targets, distribution, mixup
        )
    elapsed = 1000.0 * (time.perf_counter() - started) / steps
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if measure else 0.0
    del model, optimizer, host_inputs, host_targets, distribution
    torch.cuda.empty_cache()
    return elapsed, peak


def population_cv(values):
    return statistics.pstdev(values) / statistics.fmean(values)


def timing_checks():
    accepted = load_accepted()
    static_scope(accepted)
    results = {}
    candidate_peaks = []
    for mixup, regime_index, regime in ((True, 0, "mixup"), (False, 1, "hard")):
        timing_arm(accepted, mixup, 36_200 + regime_index * 100, 20, False)
        timing_arm(train, mixup, 36_210 + regime_index * 100, 20, False)
        windows = {"accepted": [], "candidate": []}
        for replicate in range(4):
            order = (
                ("accepted", "candidate")
                if replicate % 2 == 0
                else ("candidate", "accepted")
            )
            pair_seed = 36_300 + regime_index * 100 + replicate * 10
            for kind in order:
                module = accepted if kind == "accepted" else train
                if kind == "candidate":
                    torch.cuda.reset_peak_memory_stats()
                value, peak = timing_arm(
                    module, mixup, pair_seed, 50, kind == "candidate"
                )
                windows[kind].append(value)
                if kind == "candidate":
                    candidate_peaks.append(peak)
        results[regime] = {
            "windows_ms": windows,
            "medians_ms": {
                kind: statistics.median(values) for kind, values in windows.items()
            },
            "cvs": {kind: population_cv(values) for kind, values in windows.items()},
        }
    am = results["mixup"]["medians_ms"]["accepted"]
    cm = results["mixup"]["medians_ms"]["candidate"]
    ah = results["hard"]["medians_ms"]["accepted"]
    ch = results["hard"]["medians_ms"]["candidate"]
    retention = (0.65 / cm + 0.35 / ch) / (0.65 / am + 0.35 / ah)
    projected = 133.00736 * retention
    payload = {
        "results": results,
        "retention": retention,
        "projected_passes": projected,
        "candidate_peak_vram_mb": max(candidate_peaks),
    }
    print(json.dumps(payload, sort_keys=True))
    assert all(
        math.isfinite(value)
        for regime in results.values()
        for values in regime["windows_ms"].values()
        for value in values
    )
    assert all(
        cv <= 0.05 for regime in results.values() for cv in regime["cvs"].values()
    )
    assert retention >= 0.9774
    assert projected >= 130.0
    assert max(candidate_peaks) < 2_048
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
