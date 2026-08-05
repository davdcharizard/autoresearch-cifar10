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
COMMON_RTOL = 3e-3
COMMON_ATOL = 2e-4


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
    old_construct = '''            init.kaiming_normal_(self.pooled_head[0].weight)
            init.kaiming_normal_(self.pooled_head[2].weight)
'''
    new_construct = '''            init.kaiming_normal_(self.pooled_head[0].weight)
            init.kaiming_normal_(self.pooled_head[2].weight)
        with torch.random.fork_rng(devices=[]):
            self.pool_score = nn.Conv2d(widths[2], 1, 1, bias=False)
            init.zeros_(self.pool_score.weight)
'''
    old_pool = '''        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = out + POOLED_HEAD_SCALE * self.pooled_head(out)
        return self.fc(out)
'''
    new_pool = '''        out = F.relu(self.bn(out))
        spatial_features = out.flatten(2)
        mean_pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)
        score_logits = self.pool_score(out).flatten(1)
        attention_delta = F.softmax(score_logits, dim=1) - (
            1.0 / score_logits.size(1)
        )
        pooled_correction = torch.bmm(
            spatial_features, attention_delta.unsqueeze(2)
        ).squeeze(2)
        out = mean_pooled + pooled_correction
        out = out + POOLED_HEAD_SCALE * self.pooled_head(out)
        return self.fc(out)
'''
    candidate = (ROOT / "train.py").read_text()
    assert accepted.__source__.count(old_construct) == 1
    assert accepted.__source__.count(old_pool) == 1
    assert candidate.count(new_construct) == 1
    assert candidate.count(new_pool) == 1
    restored = candidate.replace(new_construct, old_construct).replace(new_pool, old_pool)
    assert restored == accepted.__source__
    subprocess.run(
        ["git", "diff", "--exit-code", BASE, "--", "prepare.py"], cwd=ROOT, check=True
    )
    print(json.dumps({"source_whitelist": "exact", "changed": changed}))


def common_state(model):
    return {
        name: value for name, value in model.state_dict().items()
        if name != "pool_score.weight"
    }


def feature_map(model, inputs):
    out = model.conv1(inputs)
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    return F.relu(model.bn(out))


def attention_pool(features, query):
    spatial = features.flatten(2)
    scores = F.conv2d(features, query).flatten(1)
    attention = F.softmax(scores, dim=1)
    delta = attention - 1.0 / scores.size(1)
    mean = F.adaptive_avg_pool2d(features, 1).flatten(1)
    correction = torch.bmm(spatial, delta.unsqueeze(2)).squeeze(2)
    return mean + correction, scores, attention, delta, correction


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
    assert_state_equal(model_a.state_dict(), common_state(model_c), "common_state")
    assert torch.count_nonzero(model_c.pool_score.weight).item() == 0
    model_a.train(training)
    model_c.train(training)
    return model_a, model_c


def optimizer_checks(accepted, model_a, model_c):
    opt_a = optimizer_for(accepted, model_a)
    opt_c = optimizer_for(train, model_c)
    names_a = {id(p): n for n, p in model_a.named_parameters()}
    names_c = {id(p): n for n, p in model_c.named_parameters()}
    groups_a = [[names_a[id(p)] for p in g["params"]] for g in opt_a.param_groups]
    groups_c = [[names_c[id(p)] for p in g["params"]] for g in opt_c.param_groups]
    assert groups_c[1] == groups_a[1]
    assert [name for name in groups_c[0] if name != "pool_score.weight"] == groups_a[0]
    assert groups_c[0].count("pool_score.weight") == 1
    for key in ("lr", "momentum", "weight_decay", "dampening", "nesterov"):
        assert [g[key] for g in opt_a.param_groups] == [g[key] for g in opt_c.param_groups]
    return opt_a, opt_c


def identity_checks(accepted):
    diagnostics = {}
    for device, batch, seed in (
        (torch.device("cpu"), 4, 42_010),
        (DEVICE, 16, 42_020),
    ):
        model_a, model_c = models_from_same_seed(accepted, seed, device, False)
        generator = torch.Generator().manual_seed(seed + 1)
        inputs = torch.randn(batch, 3, 32, 32, generator=generator).to(device)
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        accepted_logits = model_a(inputs)
        features = feature_map(model_c, inputs)
        pooled, scores, attention, delta, correction = attention_pool(
            features, model_c.pool_score.weight
        )
        candidate_logits = model_c(inputs)
        assert torch.count_nonzero(scores).item() == 0
        assert torch.equal(attention, torch.full_like(attention, 1.0 / 64))
        assert torch.count_nonzero(delta).item() == 0
        assert torch.count_nonzero(correction).item() == 0
        accepted_pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        assert torch.equal(pooled, accepted_pooled)
        assert torch.equal(candidate_logits, accepted_logits)
        assert torch.equal(torch.random.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        diagnostics[str(device)] = {
            "attention_mass_error": (attention.sum(1) - 1).abs().max().item(),
            "correction_nonzero": int(torch.count_nonzero(correction).item()),
        }

    generator = torch.Generator().manual_seed(42_030)
    features = torch.randn(5, 7, 4, 4, generator=generator)
    query = torch.randn(1, 7, 1, 1, generator=generator)
    pooled, scores, attention, _, _ = attention_pool(features, query)
    direct = torch.bmm(features.flatten(2), attention.unsqueeze(2)).squeeze(2)
    torch.testing.assert_close(pooled, direct, rtol=2e-5, atol=2e-7)
    rolled = features.roll((1, 2), dims=(2, 3))
    pooled_r, scores_r, attention_r, _, _ = attention_pool(rolled, query)
    assert torch.equal(scores_r, scores.view(5, 4, 4).roll((1, 2), dims=(1, 2)).flatten(1))
    assert torch.equal(
        attention_r,
        attention.view(5, 4, 4).roll((1, 2), dims=(1, 2)).flatten(1),
    )
    torch.testing.assert_close(pooled_r, pooled, rtol=2e-5, atol=2e-7)
    diagnostics["nonzero_query_delta_rms"] = (
        pooled - F.adaptive_avg_pool2d(features, 1).flatten(1)
    ).square().mean().sqrt().item()
    print(json.dumps({"identity": diagnostics}, sort_keys=True))
    return diagnostics


def analytic_checks():
    generator = torch.Generator().manual_seed(42_100)
    x = torch.randn(3, 5, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    q = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    g = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    scores = torch.einsum("bcs,c->bs", x, q)
    attention = scores.softmax(1)
    mean = x.mean(2)
    z = mean + torch.einsum("bcs,bs->bc", x, attention - 1.0 / x.size(2))
    loss = (z * g).sum()
    q_grad, x_grad = torch.autograd.grad(loss, (q, x))
    centered = x.detach() - x.detach().mean(2, keepdim=True)
    covariance = torch.einsum("bcs,bds->bcd", centered, centered) / x.size(2)
    expected_q = torch.einsum("bcd,bd->c", covariance, g)
    expected_x = g.unsqueeze(2).expand_as(x) / x.size(2)
    q_error = (q_grad - expected_q).abs().max().item()
    x_error = (x_grad - expected_x).abs().max().item()
    torch.testing.assert_close(q_grad, expected_q, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(x_grad, expected_x, rtol=1e-10, atol=1e-12)

    x2 = x.detach().clone().requires_grad_(True)
    q2 = torch.zeros_like(q, requires_grad=True)
    classifier = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2])
    scores2 = torch.einsum("bcs,c->bs", x2, q2)
    z2 = x2.mean(2) + torch.einsum(
        "bcs,bs->bc", x2, scores2.softmax(1) - 1.0 / x2.size(2)
    )
    ce = F.cross_entropy(F.linear(z2, classifier), labels)
    g2 = torch.autograd.grad(ce, z2, retain_graph=True)[0]
    q2_grad = torch.autograd.grad(ce, q2)[0]
    centered2 = x2.detach() - x2.detach().mean(2, keepdim=True)
    cov2 = torch.einsum("bcs,bds->bcd", centered2, centered2) / x2.size(2)
    expected_q2 = torch.einsum("bcd,bd->c", cov2, g2)
    ce_error = (q2_grad - expected_q2).abs().max().item()
    payload = {
        "query_gradient_error": q_error,
        "feature_gradient_error": x_error,
        "mean_ce_query_error": ce_error,
        "covariance_rank": [int(torch.linalg.matrix_rank(c).item()) for c in covariance],
        "query_gradient_norm": q_grad.norm().item(),
    }
    print(json.dumps({"analytic": payload}, sort_keys=True))
    torch.testing.assert_close(q2_grad, expected_q2, rtol=1e-10, atol=1e-12)
    return payload


def fixed_batch(seed, batch=16):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(batch, 3, 32, 32, generator=generator).to(DEVICE)
    targets = torch.arange(batch, device=DEVICE) % train.NUM_CLASSES
    return inputs, targets


def compute_loss(module, model, inputs, targets, mixup):
    if mixup:
        coefficient = torch.tensor(0.37, device=DEVICE)
        mixed = coefficient * inputs + (1 - coefficient) * inputs.roll(3, 0)
        output = model(mixed)
        return coefficient * F.cross_entropy(output, targets) + (
            1 - coefficient
        ) * F.cross_entropy(output, targets.roll(3))
    return F.cross_entropy(model(inputs), targets)


def gradient_identity(accepted, seed, mixup):
    model_a, model_c = models_from_same_seed(accepted, seed, DEVICE, True)
    inputs, targets = fixed_batch(seed + 1)
    opt_a, opt_c = optimizer_checks(accepted, model_a, model_c)
    opt_a.zero_grad(set_to_none=True)
    opt_c.zero_grad(set_to_none=True)
    loss_a = compute_loss(accepted, model_a, inputs, targets, mixup)
    loss_c = compute_loss(train, model_c, inputs, targets, mixup)
    assert torch.equal(loss_a, loss_c)
    loss_a.backward()
    loss_c.backward()
    grads_a = dict(model_a.named_parameters())
    grads_c = dict(model_c.named_parameters())
    max_abs = 0.0
    max_relative_l2 = 0.0
    comparisons = []
    for name, parameter_a in grads_a.items():
        parameter_c = grads_c[name]
        assert parameter_a.shape == parameter_c.shape and parameter_a.dtype == parameter_c.dtype
        assert (parameter_a.grad is None) == (parameter_c.grad is None)
        if parameter_a.grad is not None:
            max_abs = max(max_abs, (parameter_a.grad - parameter_c.grad).abs().max().item())
            relative = (
                (parameter_a.grad - parameter_c.grad).norm()
                / parameter_a.grad.norm().clamp_min(1e-12)
            ).item()
            max_relative_l2 = max(max_relative_l2, relative)
            comparisons.append((name, parameter_a.grad, parameter_c.grad))
    scorer_grad = grads_c["pool_score.weight"].grad
    assert scorer_grad is not None and torch.isfinite(scorer_grad).all()
    payload = {
        "regime": "mixup" if mixup else "hard",
        "loss": loss_c.item(),
        "max_common_gradient_abs_error": max_abs,
        "max_common_gradient_relative_l2_error": max_relative_l2,
        "scorer_gradient_norm": scorer_grad.norm().item(),
        "common_rtol": COMMON_RTOL,
        "common_atol": COMMON_ATOL,
    }
    print(json.dumps({"gradient_identity": payload}, sort_keys=True))
    assert payload["scorer_gradient_norm"] > 0
    for name, left, right in comparisons:
        torch.testing.assert_close(
            left, right, rtol=COMMON_RTOL, atol=COMMON_ATOL,
            msg=f"{name} initial gradient",
        )
    return payload


def update_once(seed, mixup, preseeded):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = construct(train).to(DEVICE).train()
    optimizer = optimizer_for(train, model)
    inputs, targets = fixed_batch(seed + 1)
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
    loss = compute_loss(train, model, inputs, targets, mixup)
    loss.backward()
    before_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    before_buffers = {
        n: value.detach().clone() for n, value in model.named_buffers()
    }
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    previous = {
        n: optimizer.state[p].get("momentum_buffer", torch.zeros_like(p)).clone()
        for n, p in model.named_parameters()
    }
    optimizer.step()
    max_parameter_error = 0.0
    max_buffer_error = 0.0
    for name, parameter in model.named_parameters():
        wd = train.WEIGHT_DECAY if parameter.ndim >= 2 else 0.0
        direction = grads[name] + wd * before_params[name]
        expected_buffer = (
            train.MOMENTUM * previous[name] + direction if preseeded else direction
        )
        expected_parameter = before_params[name] - lr * (
            direction + train.MOMENTUM * expected_buffer
        )
        actual_buffer = optimizer.state[parameter]["momentum_buffer"]
        max_parameter_error = max(
            max_parameter_error, (parameter - expected_parameter).abs().max().item()
        )
        max_buffer_error = max(
            max_buffer_error, (actual_buffer - expected_buffer).abs().max().item()
        )
        torch.testing.assert_close(parameter, expected_parameter, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual_buffer, expected_buffer, rtol=1e-6, atol=1e-7)
    for name, value in model.named_buffers():
        assert torch.equal(value, before_buffers[name]), f"optimizer changed buffer {name}"
    model.eval()
    with torch.no_grad():
        features = feature_map(model, inputs)
        _, scores, attention, _, _ = attention_pool(features, model.pool_score.weight)
    entropy = -(attention * attention.clamp_min(1e-30).log()).sum(1).mean()
    effective_sites = entropy.exp()
    payload = {
        "regime": "mixup" if mixup else "hard",
        "preseeded": preseeded,
        "loss": loss.item(),
        "lr": lr,
        "max_parameter_error": max_parameter_error,
        "max_buffer_error": max_buffer_error,
        "query_update_norm": model.pool_score.weight.norm().item(),
        "score_std": scores.std(unbiased=False).item(),
        "attention_entropy": entropy.item(),
        "effective_sites": effective_sites.item(),
        "max_attention": attention.max().item(),
        "state": {n: v.detach().clone() for n, v in model.state_dict().items()},
    }
    assert payload["query_update_norm"] > 0 and payload["score_std"] > 0
    assert payload["max_attention"] != 1.0 / 64
    return payload


def update_checks():
    payloads = []
    for index, (mixup, preseeded) in enumerate(
        ((True, False), (True, True), (False, False), (False, True))
    ):
        first = update_once(42_300 + 10 * index, mixup, preseeded)
        second = update_once(42_300 + 10 * index, mixup, preseeded)
        assert_state_equal(first.pop("state"), second.pop("state"), "update_replay")
        assert first == second
        payloads.append(first)
    print(json.dumps({"updates": payloads}, sort_keys=True))
    return payloads


def semantics():
    accepted = load_accepted()
    static_scope(accepted)
    assert GuardEval.constructions == 2
    torch.empty(1, device=DEVICE)
    model_a, model_c = models_from_same_seed(accepted, 42, torch.device("cpu"), True)
    assert sum(p.numel() for p in model_c.parameters()) == 1_003_610
    optimizer_checks(accepted, model_a, model_c)
    identity = identity_checks(accepted)
    analytic = analytic_checks()
    gradients = [
        gradient_identity(accepted, 42_200, True),
        gradient_identity(accepted, 42_210, False),
    ]
    updates = update_checks()
    print(json.dumps({"summary": {"identity": identity, "analytic": analytic, "gradients": gradients, "updates": updates, "parameter_count": 1_003_610}}, sort_keys=True))
    print("SEMANTICS PASS")


def timed_step(module, model, optimizer, host_x, host_y, distribution, mixup, progress):
    inputs = host_x.to(DEVICE, non_blocking=True)
    targets = host_y.to(DEVICE, non_blocking=True)
    lr = module.learning_rate(progress * module.TIME_BUDGET_S)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    if mixup:
        mixed, target_a, target_b, coefficient = module.mixup_batch(inputs, targets, distribution)
        output = model(mixed)
        loss = coefficient * F.cross_entropy(output, target_a) + (1 - coefficient) * F.cross_entropy(output, target_b)
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
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        timed_step(module, model, optimizer, host_x, host_y, distribution, mixup, progress)
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
        timing_arm(accepted, mixup, progress, 42_500 + 100 * regime_index, 20)
        timing_arm(train, mixup, progress, 42_500 + 100 * regime_index, 20)
        windows = {"accepted": [], "candidate": []}
        pair_index = 0
        for _cycle in range(2):
            for order in (("accepted", "candidate"), ("candidate", "accepted")):
                seed = 42_600 + 100 * regime_index + 10 * pair_index
                for kind in order:
                    module = accepted if kind == "accepted" else train
                    value, peak = timing_arm(module, mixup, progress, seed, 50)
                    windows[kind].append(value)
                    if kind == "candidate":
                        peaks.append(peak)
                pair_index += 1
        results[regime] = {
            "progress": progress,
            "windows_ms": windows,
            "medians_ms": {kind: statistics.median(values) for kind, values in windows.items()},
            "cvs": {kind: statistics.pstdev(values) / statistics.fmean(values) for kind, values in windows.items()},
        }
    a_mix = results["mixup"]["medians_ms"]["accepted"]
    c_mix = results["mixup"]["medians_ms"]["candidate"]
    a_hard = results["hard"]["medians_ms"]["accepted"]
    c_hard = results["hard"]["medians_ms"]["candidate"]
    retention = (0.65 / c_mix + 0.35 / c_hard) / (0.65 / a_mix + 0.35 / a_hard)
    payload = {"results": results, "retention": retention, "projected_passes": 130.304 * retention, "peak_mb": max(peaks)}
    print(json.dumps(payload, sort_keys=True))
    assert all(cv <= 0.05 for regime in results.values() for cv in regime["cvs"].values())
    assert retention >= 0.9746439096 and payload["projected_passes"] >= 127
    assert max(peaks) < 2048
    print("TIMING PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("semantics", "timing"))
    args = parser.parse_args()
    semantics() if args.mode == "semantics" else timing()
