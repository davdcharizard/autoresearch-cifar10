import argparse
import json
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
GRAD_RTOL = 3e-3
GRAD_ATOL = 2e-4


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
    old_forward = '''    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = out + POOLED_HEAD_SCALE * self.pooled_head(out)
        return self.fc(out)
'''
    new_forward = '''    def forward(self, x, return_direct_logits=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        pooled = out.view(out.size(0), -1)
        refined = pooled + POOLED_HEAD_SCALE * self.pooled_head(pooled)
        main_logits = self.fc(refined)
        if return_direct_logits:
            return main_logits, self.fc(pooled)
        return main_logits
'''
    old_loss = '''            optimizer.zero_grad(set_to_none=True)
            if use_mixup:
                mixed_inputs, targets_a, targets_b, mix = mixup_batch(
                    inputs, targets, mixup_distribution
                )
                outputs = model(mixed_inputs)
                loss = mix * F.cross_entropy(outputs, targets_a) + (
                    1.0 - mix
                ) * F.cross_entropy(outputs, targets_b)
            else:
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
'''
    new_loss = '''            optimizer.zero_grad(set_to_none=True)
            if use_mixup:
                mixed_inputs, targets_a, targets_b, mix = mixup_batch(
                    inputs, targets, mixup_distribution
                )
                outputs, direct_outputs = model(
                    mixed_inputs, return_direct_logits=True
                )
                main_loss = mix * F.cross_entropy(outputs, targets_a) + (
                    1.0 - mix
                ) * F.cross_entropy(outputs, targets_b)
                direct_loss = mix * F.cross_entropy(direct_outputs, targets_a) + (
                    1.0 - mix
                ) * F.cross_entropy(direct_outputs, targets_b)
            else:
                outputs, direct_outputs = model(inputs, return_direct_logits=True)
                main_loss = F.cross_entropy(outputs, targets)
                direct_loss = F.cross_entropy(direct_outputs, targets)
            loss = (1.0 - POOLED_HEAD_SCALE) * main_loss + (
                POOLED_HEAD_SCALE * direct_loss
            )
'''
    candidate = (ROOT / "train.py").read_text()
    assert accepted.__source__.count(old_forward) == 1
    assert accepted.__source__.count(old_loss) == 1
    assert candidate.count(new_forward) == 1
    assert candidate.count(new_loss) == 1
    restored = candidate.replace(new_forward, old_forward).replace(new_loss, old_loss)
    assert restored == accepted.__source__
    print(json.dumps({"source_whitelist": "exact", "changed": changed}))


def raw_pooled(model, inputs):
    out = model.conv1(inputs)
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = F.relu(model.bn(out))
    return F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)


def models_from_same_seed(accepted, seed, device=DEVICE, training=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_a = construct(accepted).to(device)
    cpu_after = torch.random.get_rng_state().clone()
    cuda_after = torch.cuda.get_rng_state().clone()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_c = construct(train).to(device)
    assert torch.equal(torch.random.get_rng_state(), cpu_after)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_after)
    assert_state_equal(model_a.state_dict(), model_c.state_dict(), "construction")
    model_a.train(training)
    model_c.train(training)
    return model_a, model_c


def forward_checks(accepted):
    diagnostics = {}
    for device, batch, seed in (
        (torch.device("cpu"), 4, 41_010),
        (DEVICE, 16, 41_020),
    ):
        model_a, model_c = models_from_same_seed(
            accepted, seed, device=device, training=False
        )
        generator = torch.Generator().manual_seed(seed + 1)
        inputs = torch.randn(batch, 3, 32, 32, generator=generator).to(device)
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        accepted_default = model_a(inputs)
        default_inputs = []
        hook = model_c.fc.register_forward_pre_hook(
            lambda _module, args: default_inputs.append(args[0].detach().clone())
        )
        candidate_default = model_c(inputs)
        hook.remove()
        assert len(default_inputs) == 1
        assert torch.equal(candidate_default, accepted_default)
        assert torch.equal(torch.random.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)

        pooled = raw_pooled(model_c, inputs)
        refined = pooled + train.POOLED_HEAD_SCALE * model_c.pooled_head(pooled)
        dual_inputs = []
        hook = model_c.fc.register_forward_pre_hook(
            lambda _module, args: dual_inputs.append(args[0].detach().clone())
        )
        main_logits, direct_logits = model_c(inputs, return_direct_logits=True)
        hook.remove()
        assert len(dual_inputs) == 2
        assert torch.equal(dual_inputs[0], refined)
        assert torch.equal(dual_inputs[1], pooled)
        assert torch.equal(main_logits, candidate_default)
        expected_main = F.linear(refined, model_c.fc.weight, model_c.fc.bias)
        expected_direct = F.linear(pooled, model_c.fc.weight, model_c.fc.bias)
        assert torch.equal(main_logits, expected_main)
        assert torch.equal(direct_logits, expected_direct)
        diagnostics[str(device)] = {
            "main_direct_rms": (main_logits - direct_logits)
            .square()
            .mean()
            .sqrt()
            .item(),
            "argmax_agreement": (main_logits.argmax(1) == direct_logits.argmax(1))
            .float()
            .mean()
            .item(),
        }
        assert diagnostics[str(device)]["main_direct_rms"] > 0

    model_a, model_c = models_from_same_seed(accepted, 41_030, training=True)
    generator = torch.Generator().manual_seed(41_031)
    inputs = torch.randn(16, 3, 32, 32, generator=generator).to(DEVICE)
    output_a = model_a(inputs)
    output_c = model_c(inputs)
    assert torch.equal(output_a, output_c)
    assert_state_equal(model_a.state_dict(), model_c.state_dict(), "train_default")
    print(json.dumps({"forward": diagnostics}, sort_keys=True))
    return diagnostics


def loss_terms(model, inputs, targets_a, targets_b=None, coefficient=None):
    main_logits, direct_logits = model(inputs, return_direct_logits=True)
    if targets_b is None:
        main_loss = F.cross_entropy(main_logits, targets_a)
        direct_loss = F.cross_entropy(direct_logits, targets_a)
    else:
        main_loss = coefficient * F.cross_entropy(main_logits, targets_a) + (
            1.0 - coefficient
        ) * F.cross_entropy(main_logits, targets_b)
        direct_loss = coefficient * F.cross_entropy(direct_logits, targets_a) + (
            1.0 - coefficient
        ) * F.cross_entropy(direct_logits, targets_b)
    combined = (1.0 - train.POOLED_HEAD_SCALE) * main_loss + (
        train.POOLED_HEAD_SCALE * direct_loss
    )
    return main_logits, direct_logits, main_loss, direct_loss, combined


def fixed_batch(seed, batch=16):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(batch, 3, 32, 32, generator=generator).to(DEVICE)
    targets = torch.arange(batch, device=DEVICE) % train.NUM_CLASSES
    return inputs, targets


def mixup_rng_check(accepted):
    inputs, targets = fixed_batch(41_100)
    distribution_a = torch.distributions.Beta(
        torch.tensor(accepted.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(accepted.MIXUP_ALPHA, device=DEVICE),
    )
    distribution_c = torch.distributions.Beta(
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
    )
    torch.cuda.manual_seed(41_101)
    initial = torch.cuda.get_rng_state().clone()
    result_a = accepted.mixup_batch(inputs, targets, distribution_a)
    end_a = torch.cuda.get_rng_state().clone()
    torch.cuda.set_rng_state(initial)
    result_c = train.mixup_batch(inputs, targets, distribution_c)
    end_c = torch.cuda.get_rng_state().clone()
    assert torch.equal(end_a, end_c)
    for left, right in zip(result_a, result_c):
        assert torch.equal(left, right)
    return result_c


def gradient_fixture(seed, mixup):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    base = construct(train).to(DEVICE).train()
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    inputs, targets = fixed_batch(seed + 1)
    targets_b = targets.roll(3) if mixup else None
    coefficient = torch.tensor(0.37, device=DEVICE) if mixup else None
    results = {}
    for kind in ("main", "direct", "combined"):
        model = construct(train).to(DEVICE).train()
        model.load_state_dict(base_state)
        _, _, main_loss, direct_loss, combined = loss_terms(
            model, inputs, targets, targets_b, coefficient
        )
        loss = {"main": main_loss, "direct": direct_loss, "combined": combined}[kind]
        loss.backward()
        results[kind] = {
            "loss": loss.detach(),
            "grads": {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
            },
        }

    max_error = 0.0
    max_relative_l2_error = 0.0
    cosine_samples = {}
    comparisons = []
    for name, combined in results["combined"]["grads"].items():
        main = results["main"]["grads"][name]
        direct = results["direct"]["grads"][name]
        assert main is not None and combined is not None
        if name.startswith("pooled_head."):
            assert direct is None or torch.count_nonzero(direct).item() == 0
            expected = 0.9 * main
        else:
            assert direct is not None
            expected = 0.9 * main + 0.1 * direct
        error = (combined - expected).abs().max().item()
        relative_l2_error = (
            (combined - expected).norm() / expected.norm().clamp_min(1e-12)
        ).item()
        max_error = max(max_error, error)
        max_relative_l2_error = max(max_relative_l2_error, relative_l2_error)
        comparisons.append((name, combined, expected))
        if name in ("conv1.weight", "layer3.2.conv2.weight", "fc.weight"):
            assert direct is not None
            cosine_samples[name] = F.cosine_similarity(
                main.flatten(), direct.flatten(), dim=0
            ).item()
    payload = {
        "regime": "mixup" if mixup else "hard",
        "main_loss": results["main"]["loss"].item(),
        "direct_loss": results["direct"]["loss"].item(),
        "combined_loss": results["combined"]["loss"].item(),
        "max_gradient_error": max_error,
        "max_relative_l2_gradient_error": max_relative_l2_error,
        "grad_rtol": GRAD_RTOL,
        "grad_atol": GRAD_ATOL,
        "gradient_cosines": cosine_samples,
    }
    print(json.dumps({"gradient_fixture": payload}, sort_keys=True))
    for name, combined, expected in comparisons:
        torch.testing.assert_close(
            combined,
            expected,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            msg=lambda msg, name=name: f"{name}: {msg}",
        )
    return payload


def update_once(seed, mixup, preseeded):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = construct(train).to(DEVICE).train()
    optimizer = optimizer_for(train, model)
    inputs, targets = fixed_batch(seed + 1)
    targets_b = targets.roll(3) if mixup else None
    coefficient = torch.tensor(0.37, device=DEVICE) if mixup else None
    progress = 0.50 if mixup else 0.75
    lr = train.learning_rate(progress * train.TIME_BUDGET_S)
    for group in optimizer.param_groups:
        group["lr"] = lr
    if preseeded:
        for index, parameter in enumerate(model.parameters()):
            optimizer.state[parameter]["momentum_buffer"] = torch.full_like(
                parameter, 0.00001 * (index + 1)
            )
    optimizer.zero_grad(set_to_none=True)
    _, _, _, _, loss = loss_terms(model, inputs, targets, targets_b, coefficient)
    loss.backward()
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    previous = {
        name: optimizer.state[parameter]
        .get("momentum_buffer", torch.zeros_like(parameter))
        .clone()
        for name, parameter in model.named_parameters()
    }
    optimizer.step()
    max_parameter_error = 0.0
    max_buffer_error = 0.0
    for name, parameter in model.named_parameters():
        wd = train.WEIGHT_DECAY if before[name].ndim >= 2 else 0.0
        direction = grads[name] + wd * before[name]
        buffer = (
            train.MOMENTUM * previous[name] + direction if preseeded else direction
        )
        expected = before[name] - lr * (direction + train.MOMENTUM * buffer)
        actual_buffer = optimizer.state[parameter]["momentum_buffer"]
        max_parameter_error = max(
            max_parameter_error, (parameter - expected).abs().max().item()
        )
        max_buffer_error = max(
            max_buffer_error, (actual_buffer - buffer).abs().max().item()
        )
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual_buffer, buffer, rtol=1e-6, atol=1e-7)
    return {
        "regime": "mixup" if mixup else "hard",
        "preseeded": preseeded,
        "loss": loss.item(),
        "lr": lr,
        "max_parameter_error": max_parameter_error,
        "max_buffer_error": max_buffer_error,
        "state": {name: value.detach().clone() for name, value in model.state_dict().items()},
    }


def update_checks():
    payloads = []
    for index, (mixup, preseeded) in enumerate(
        ((True, False), (True, True), (False, False), (False, True))
    ):
        first = update_once(41_300 + 10 * index, mixup, preseeded)
        second = update_once(41_300 + 10 * index, mixup, preseeded)
        assert_state_equal(first.pop("state"), second.pop("state"), "update_replay")
        assert first == second
        payloads.append(first)
    print(json.dumps({"updates": payloads}, sort_keys=True))
    return payloads


def semantics():
    accepted = load_accepted()
    static_scope(accepted)
    assert GuardEval.constructions == 2
    constants = (
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
        "MIXUP_ALPHA",
        "MIXUP_END_FRACTION",
        "RANDAUGMENT_END_FRACTION",
        "POOLED_HEAD_WIDTH",
        "POOLED_HEAD_SCALE",
        "POOLED_HEAD_INIT_SEED",
    )
    assert all(getattr(accepted, name) == getattr(train, name) for name in constants)
    torch.empty(1, device=DEVICE)
    model_a, model_c = models_from_same_seed(accepted, 42, device=torch.device("cpu"))
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
    mixed, target_a, target_b, coefficient = mixup_rng_check(accepted)
    assert mixed.shape[0] == target_a.shape[0] == target_b.shape[0]
    assert coefficient.ndim == 0
    forward = forward_checks(accepted)
    gradients = [gradient_fixture(41_200, True), gradient_fixture(41_210, False)]
    updates = update_checks()
    print(
        json.dumps(
            {
                "summary": {
                    "forward": forward,
                    "gradients": gradients,
                    "updates": updates,
                    "parameter_count": 1_003_482,
                }
            },
            sort_keys=True,
        )
    )
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
        if module is train:
            main, direct = model(mixed, return_direct_logits=True)
            main_loss = coefficient * F.cross_entropy(main, target_a) + (
                1.0 - coefficient
            ) * F.cross_entropy(main, target_b)
            direct_loss = coefficient * F.cross_entropy(direct, target_a) + (
                1.0 - coefficient
            ) * F.cross_entropy(direct, target_b)
            loss = 0.9 * main_loss + 0.1 * direct_loss
        else:
            output = model(mixed)
            loss = coefficient * F.cross_entropy(output, target_a) + (
                1.0 - coefficient
            ) * F.cross_entropy(output, target_b)
    elif module is train:
        main, direct = model(inputs, return_direct_logits=True)
        loss = 0.9 * F.cross_entropy(main, targets) + 0.1 * F.cross_entropy(
            direct, targets
        )
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
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
    )
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        timed_step(
            module, model, optimizer, host_x, host_y, distribution, mixup, progress
        )
    elapsed_ms = 1000 * (time.perf_counter() - started) / steps
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    del model, optimizer, host_x, host_y, distribution
    torch.cuda.empty_cache()
    return elapsed_ms, peak_mb


def timing():
    accepted = load_accepted()
    static_scope(accepted)
    results, peaks = {}, []
    for mixup, progress, regime, regime_index in (
        (True, 0.50, "mixup", 0),
        (False, 0.75, "hard", 1),
    ):
        timing_arm(accepted, mixup, progress, 41_500 + 100 * regime_index, 20)
        timing_arm(train, mixup, progress, 41_500 + 100 * regime_index, 20)
        windows = {"accepted": [], "candidate": []}
        pair_index = 0
        for _cycle in range(2):
            for order in (("accepted", "candidate"), ("candidate", "accepted")):
                seed = 41_600 + 100 * regime_index + 10 * pair_index
                for kind in order:
                    module = accepted if kind == "accepted" else train
                    value, peak = timing_arm(
                        module, mixup, progress, seed, 50
                    )
                    windows[kind].append(value)
                    if kind == "candidate":
                        peaks.append(peak)
                pair_index += 1
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
    a_mix = results["mixup"]["medians_ms"]["accepted"]
    c_mix = results["mixup"]["medians_ms"]["candidate"]
    a_hard = results["hard"]["medians_ms"]["accepted"]
    c_hard = results["hard"]["medians_ms"]["candidate"]
    retention = (0.65 / c_mix + 0.35 / c_hard) / (
        0.65 / a_mix + 0.35 / a_hard
    )
    payload = {
        "results": results,
        "retention": retention,
        "projected_passes": 130.304 * retention,
        "peak_mb": max(peaks),
    }
    print(json.dumps(payload, sort_keys=True))
    assert all(
        cv <= 0.05 for regime in results.values() for cv in regime["cvs"].values()
    )
    assert retention >= 0.9746439096
    assert payload["projected_passes"] >= 127
    assert max(peaks) < 2048
    print("TIMING PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("semantics", "timing"))
    args = parser.parse_args()
    semantics() if args.mode == "semantics" else timing()
