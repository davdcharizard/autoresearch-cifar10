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
    old = '''        out = out + POOLED_HEAD_SCALE * self.pooled_head(out)
        return self.fc(out)
'''
    new = '''        out = out + POOLED_HEAD_SCALE * self.pooled_head(out)
        classifier_weight = self.fc.weight
        row_norms = torch.linalg.vector_norm(classifier_weight, dim=1, keepdim=True)
        rms_row_norm = torch.linalg.vector_norm(classifier_weight) / math.sqrt(
            classifier_weight.size(0)
        )
        effective_weight = classifier_weight * (rms_row_norm / row_norms)
        return F.linear(out, effective_weight, self.fc.bias)
'''
    candidate = (ROOT / "train.py").read_text()
    assert accepted.__source__.count(old) == 1 and candidate.count(new) == 1
    assert candidate.replace(new, old) == accepted.__source__
    assert "iterator_exhausted=true" in candidate


def independent_effective(weight):
    row_norms = weight.square().sum(dim=1, keepdim=True).sqrt()
    rms_row_norm = weight.square().sum().div(weight.size(0)).sqrt()
    return weight / row_norms * rms_row_norm


def pooled_features(model, inputs):
    out = model.conv1(inputs)
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = F.relu(model.bn(out))
    out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
    return out + train.POOLED_HEAD_SCALE * model.pooled_head(out)


def geometry_checks(accepted, model_a, model_c):
    weight = model_c.fc.weight
    effective = independent_effective(weight)
    row_norms = weight.norm(dim=1)
    rms = weight.norm() / math.sqrt(weight.size(0))
    effective_rows = effective.norm(dim=1)
    diagnostics = {
        "raw_min": row_norms.min().item(),
        "raw_max": row_norms.max().item(),
        "raw_cv": (row_norms.std(unbiased=False) / row_norms.mean()).item(),
        "raw_ratio": (row_norms.max() / row_norms.min()).item(),
        "rms": rms.item(),
        "raw_frobenius": weight.norm().item(),
        "effective_frobenius": effective.norm().item(),
        "relative_weight_delta": ((effective - weight).norm() / weight.norm()).item(),
        "effective_row_min": effective_rows.min().item(),
        "effective_row_max": effective_rows.max().item(),
    }
    print(json.dumps({"geometry_preassert": diagnostics}, sort_keys=True))
    assert diagnostics["raw_min"] > 1e-6
    assert abs(diagnostics["raw_cv"] - 0.0696447) <= 1e-6
    assert abs(diagnostics["raw_ratio"] - 1.2724986) <= 1e-6
    assert abs(diagnostics["relative_weight_delta"] - 0.0695184) <= 1e-6
    torch.testing.assert_close(
        effective_rows, rms.expand_as(effective_rows), rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(effective.norm(), weight.norm(), rtol=1e-6, atol=1e-7)
    cosines = F.cosine_similarity(effective, weight, dim=1)
    torch.testing.assert_close(cosines, torch.ones_like(cosines), rtol=1e-6, atol=1e-7)

    for dtype, device, rtol, atol in (
        (torch.float64, torch.device("cpu"), 1e-10, 1e-12),
        (torch.float32, torch.device("cpu"), 1e-6, 1e-7),
        (torch.float32, DEVICE, 1e-6, 1e-7),
    ):
        generator = torch.Generator().manual_seed(40_010)
        fixture_weight = torch.randn(10, 17, generator=generator, dtype=dtype).to(device)
        production = fixture_weight * (
            torch.linalg.vector_norm(fixture_weight)
            / math.sqrt(fixture_weight.size(0))
            / torch.linalg.vector_norm(fixture_weight, dim=1, keepdim=True)
        )
        reference = independent_effective(fixture_weight)
        torch.testing.assert_close(production, reference, rtol=rtol, atol=atol)
        torch.testing.assert_close(production.norm(), fixture_weight.norm(), rtol=rtol, atol=atol)

    generator = torch.Generator().manual_seed(40_020)
    inputs = torch.randn(32, 3, 32, 32, generator=generator).to(DEVICE)
    model_a = model_a.to(DEVICE).eval()
    model_c = model_c.to(DEVICE).eval()
    with torch.no_grad():
        features_a = pooled_features(model_a, inputs)
        features_c = pooled_features(model_c, inputs)
        accepted_logits = model_a.fc(features_a)
        candidate_logits = model_c(inputs)
        reference_logits = F.linear(
            features_a, independent_effective(model_a.fc.weight), model_a.fc.bias
        )
    assert torch.equal(features_a, features_c)
    independent_logit_error = (candidate_logits - reference_logits).abs().max().item()
    print(
        json.dumps(
            {"independent_full_logit_max_abs_error": independent_logit_error},
            sort_keys=True,
        )
    )
    torch.testing.assert_close(
        candidate_logits, reference_logits, rtol=2e-5, atol=2e-7
    )
    assert not torch.equal(candidate_logits, accepted_logits)
    diagnostics.update(
        {
            "accepted_logit_rms": accepted_logits.square().mean().sqrt().item(),
            "candidate_logit_rms": candidate_logits.square().mean().sqrt().item(),
            "logit_delta_rms": (candidate_logits - accepted_logits).square().mean().sqrt().item(),
            "argmax_changes": int((candidate_logits.argmax(1) != accepted_logits.argmax(1)).sum()),
            "independent_logit_max_abs_error": independent_logit_error,
        }
    )

    base = weight.detach().double()
    scaled = independent_effective(2.3 * base)
    torch.testing.assert_close(scaled, 2.3 * independent_effective(base), rtol=1e-10, atol=1e-12)
    isolated = base.clone()
    isolated[3] *= 1.7
    before_directions = F.normalize(independent_effective(base), dim=1)
    after_directions = F.normalize(independent_effective(isolated), dim=1)
    torch.testing.assert_close(before_directions, after_directions, rtol=1e-10, atol=1e-12)

    check_generator = torch.Generator().manual_seed(40_030)
    feature = torch.randn(
        4, 7, generator=check_generator, dtype=torch.float64, requires_grad=True
    )
    grad_weight = torch.randn(
        5, 7, generator=check_generator, dtype=torch.float64, requires_grad=True
    )
    bias = torch.randn(
        5, generator=check_generator, dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        lambda x, w, b: F.linear(x, independent_effective(w), b),
        (feature, grad_weight, bias),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )

    analytic_weight = torch.randn(
        6, 9, generator=check_generator, dtype=torch.float64, requires_grad=True
    )
    upstream = torch.randn(6, 9, generator=check_generator, dtype=torch.float64)
    analytic_effective = independent_effective(analytic_weight)
    analytic_loss = (analytic_effective * upstream).sum()
    actual_gradient = torch.autograd.grad(analytic_loss, analytic_weight)[0]
    raw = analytic_weight.detach()
    row = raw.norm(dim=1, keepdim=True)
    direction = raw / row
    scale = raw.norm() / math.sqrt(raw.size(0))
    radial_sum = (upstream * direction).sum()
    tangent = (scale / row) * (
        upstream - direction * (upstream * direction).sum(dim=1, keepdim=True)
    )
    radial = raw * radial_sum / (raw.size(0) * scale)
    expected_gradient = tangent + radial
    gradient_error = (actual_gradient - expected_gradient).abs().max().item()
    print(json.dumps({"analytic_gradient_max_error": gradient_error}, sort_keys=True))
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12)
    diagnostics["analytic_gradient_max_error"] = gradient_error
    return diagnostics


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
    seed = 40_100 + int(100 * progress) + 10 * int(preseeded) + int(mixup)
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
    assert torch.equal(end_rng_a, end_rng_c)
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
        direction_a = grads_a[name] + wd * before_a[name]
        direction_c = grads_c[name] + wd * before_c[name]
        if preseeded:
            buffer_a = accepted.MOMENTUM * previous_a[name] + direction_a
            buffer_c = train.MOMENTUM * previous_c[name] + direction_c
        else:
            buffer_a = direction_a
            buffer_c = direction_c
        expected_a = before_a[name] - lr_a * (
            direction_a + accepted.MOMENTUM * buffer_a
        )
        expected_c = before_c[name] - lr_c * (
            direction_c + train.MOMENTUM * buffer_c
        )
        torch.testing.assert_close(names_a[name], expected_a, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(names_c[name], expected_c, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            opt_a.state[names_a[name]]["momentum_buffer"], buffer_a, rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            opt_c.state[names_c[name]]["momentum_buffer"], buffer_c, rtol=1e-6, atol=1e-7
        )
        max_parameter_delta = max(
            max_parameter_delta,
            (names_a[name] - names_c[name]).abs().max().item(),
        )
    assert lr_a == lr_c and max_parameter_delta > 0.0
    assert names_c["fc.weight"].norm(dim=1).min().item() > 1e-6
    return {
        "progress": progress,
        "accepted_lr": lr_a,
        "candidate_lr": lr_c,
        "accepted_loss": loss_a.item(),
        "candidate_loss": loss_c.item(),
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
    optimizer_a = optimizer_for(accepted, model_a)
    optimizer_c = optimizer_for(train, model_c)
    names_a = {id(parameter): name for name, parameter in model_a.named_parameters()}
    names_c = {id(parameter): name for name, parameter in model_c.named_parameters()}
    groups_a = [
        [names_a[id(parameter)] for parameter in group["params"]]
        for group in optimizer_a.param_groups
    ]
    groups_c = [
        [names_c[id(parameter)] for parameter in group["params"]]
        for group in optimizer_c.param_groups
    ]
    assert groups_a == groups_c
    for key in ("lr", "momentum", "weight_decay", "dampening", "nesterov"):
        assert [group[key] for group in optimizer_a.param_groups] == [
            group[key] for group in optimizer_c.param_groups
        ]
    assert not optimizer_a.state and not optimizer_c.state
    geometry_cpu = torch.random.get_rng_state().clone()
    geometry_cuda = torch.cuda.get_rng_state().clone()
    geometry = geometry_checks(accepted, model_a, model_c)
    assert torch.equal(torch.random.get_rng_state(), geometry_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), geometry_cuda)
    updates = [
        oracle_fixture(accepted, 0.50, True, False),
        oracle_fixture(accepted, 0.75, False, False),
        oracle_fixture(accepted, 0.75, False, True),
    ]
    print(json.dumps({"geometry": geometry, "updates": updates}, sort_keys=True))
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
