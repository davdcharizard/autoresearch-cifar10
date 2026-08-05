import ast
import importlib.util
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


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
BASELINE = "a7c42dc"
EXPECTED_PARAMS = 1_011_674
RETENTION_FLOOR = 127.0 / 130.304


class BlockedEval:
    def evaluate(self, *_args, **_kwargs):
        raise AssertionError("evaluation forbidden in preflight")


def blocked_dataset(*_args, **_kwargs):
    raise AssertionError("dataset construction forbidden in preflight")


def load_modules():
    import prepare
    from torchvision import datasets

    prepare.Eval = BlockedEval
    datasets.CIFAR10 = blocked_dataset
    spec = importlib.util.spec_from_file_location("exp044_candidate", ROOT / "train.py")
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    accepted_source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:train.py"], cwd=ROOT, text=True
    )
    accepted = types.ModuleType("exp044_accepted")
    accepted.__file__ = f"git:{BASELINE}:train.py"
    exec(compile(accepted_source, accepted.__file__, "exec"), accepted.__dict__)
    return accepted, candidate, accepted_source, (ROOT / "train.py").read_text()


def clone_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_common_state(accepted_model, candidate_model):
    accepted = clone_state(accepted_model)
    candidate = clone_state(candidate_model)
    assert list(candidate) == list(accepted) + ["dispersion_adapter.weight"]
    for name in accepted:
        assert torch.equal(accepted[name], candidate[name]), name
    assert candidate["dispersion_adapter.weight"].shape == (64, 128)
    assert torch.count_nonzero(candidate["dispersion_adapter.weight"]).item() == 0


def rng_state():
    return torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def restore_rng(state):
    torch.random.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def optimizer(module, model, lr=0.037):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return optim.SGD(
        [
            {"params": decay, "weight_decay": module.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        momentum=module.MOMENTUM,
        nesterov=True,
    )


def optimizer_signature(model, opt):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result = []
    for group in opt.param_groups:
        options = {key: value for key, value in group.items() if key != "params"}
        result.append(([names[id(p)] for p in group["params"]], options))
    return result


def audit_source(accepted_source, candidate_source):
    diff = subprocess.check_output(
        ["git", "diff", "--unified=0", BASELINE, "--", "train.py"],
        cwd=ROOT,
        text=True,
    )
    removals = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    additions = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    print(f"source additions={len(additions)} removals={len(removals)}")
    assert len(additions) == 13 and len(removals) == 5
    required = [
        "self.dispersion_adapter = nn.Linear(",
        "init.zeros_(self.dispersion_adapter.weight)",
        "torch.var(features, dim=(-2, -1), correction=0) + 1e-5",
        "self.pooled_head[0](pooled) + self.dispersion_adapter(spatial_std)",
        "POOLED_HEAD_SCALE * self.pooled_head[2](F.relu(hidden))",
    ]
    assert all(fragment in candidate_source for fragment in required)
    assert "dispersion_adapter" not in accepted_source
    tree = ast.parse(candidate_source)
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "var" for node in ast.walk(tree)) == 1


def build_models(accepted, candidate, device):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    start = rng_state()
    restore_rng(start)
    model_a = accepted.WideResNet(
        accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES
    ).to(device)
    after_a = rng_state()
    restore_rng(start)
    model_c = candidate.WideResNet(
        candidate.STAGE_BLOCKS, candidate.WIDEN_FACTOR, candidate.NUM_CLASSES
    ).to(device)
    after_c = rng_state()
    assert torch.equal(after_a[0], after_c[0]) and torch.equal(after_a[1], after_c[1])
    assert_common_state(model_a, model_c)
    assert sum(p.numel() for p in model_c.parameters()) == EXPECTED_PARAMS
    sig_a = optimizer_signature(model_a, optimizer(accepted, model_a))
    sig_c = optimizer_signature(model_c, optimizer(candidate, model_c))
    assert sig_c[0][0][:-1] == sig_a[0][0]
    assert sig_c[0][0][-1] == "dispersion_adapter.weight"
    assert sig_c[1] == sig_a[1] and sig_c[0][1] == sig_a[0][1]
    return model_a, model_c


class Capture:
    def __init__(self, model):
        self.values = {}
        self.handles = [
            model.bn.register_forward_hook(self._output("bn")),
            model.dispersion_adapter.register_forward_pre_hook(self._input("sigma")),
            model.dispersion_adapter.register_forward_hook(self._output("adapter")),
            model.pooled_head[0].register_forward_hook(self._output("mean_linear")),
            model.pooled_head[2].register_forward_pre_hook(self._input("relu_hidden")),
        ]

    def _output(self, name):
        def hook(_module, _inputs, output):
            self.values[name] = output
        return hook

    def _input(self, name):
        def hook(_module, inputs):
            self.values[name] = inputs[0]
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def fixed_input(device, batch=8):
    inputs = torch.linspace(-1.0, 1.0, batch * 3 * 32 * 32, device=device).reshape(batch, 3, 32, 32)
    targets = torch.arange(batch, device=device) % 10
    return inputs, targets


def regime_loss(model, inputs, targets, regime):
    if regime == "early":
        permutation = torch.arange(inputs.shape[0] - 1, -1, -1, device=inputs.device)
        mix = inputs.new_tensor(0.3)
        logits = model(mix * inputs + (1 - mix) * inputs[permutation])
        loss = mix * F.cross_entropy(logits, targets) + (1 - mix) * F.cross_entropy(logits, targets[permutation])
        dense_target = mix * F.one_hot(targets, 10) + (1 - mix) * F.one_hot(targets[permutation], 10)
    else:
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        dense_target = F.one_hot(targets, 10)
    return logits, loss, dense_target.to(logits.dtype)


def independent_std(features):
    mean = features.mean(dim=(-2, -1), keepdim=True)
    return torch.sqrt(((features - mean) ** 2).mean(dim=(-2, -1)) + 1e-5)


def statistic_oracles(candidate):
    for device, dtype, rtol, atol in [
        ("cpu", torch.float64, 1e-10, 1e-12),
        ("cpu", torch.float32, 2e-5, 2e-7),
        ("cuda", torch.float32, 2e-5, 2e-7),
    ]:
        x = torch.linspace(-0.7, 1.2, 2 * 3 * 4 * 4, device=device, dtype=dtype).reshape(2, 3, 4, 4)
        x.requires_grad_(True)
        d = torch.linspace(-0.2, 0.3, 5 * 3, device=device, dtype=dtype).reshape(5, 3)
        g = torch.linspace(-0.5, 0.4, 2 * 5, device=device, dtype=dtype).reshape(2, 5)
        sigma = torch.sqrt(torch.var(x, dim=(-2, -1), correction=0) + x.new_tensor(1e-5))
        objective = (F.linear(sigma, d) * g).sum()
        grad_x, = torch.autograd.grad(objective, x)
        mu = x.detach().mean(dim=(-2, -1), keepdim=True)
        sigma_ref = independent_std(x.detach())
        r = g @ d
        grad_ref = r[:, :, None, None] * (x.detach() - mu) / (16 * sigma_ref[:, :, None, None])
        torch.testing.assert_close(sigma.detach(), sigma_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(grad_x, grad_ref, rtol=rtol, atol=atol)
        constant = torch.ones(1, 3, 4, 4, device=device, dtype=dtype, requires_grad=True)
        constant_sigma = torch.sqrt(torch.var(constant, dim=(-2, -1), correction=0) + constant.new_tensor(1e-5))
        constant_grad, = torch.autograd.grad(constant_sigma.sum(), constant)
        assert torch.count_nonzero(constant_grad).item() == 0
        print(f"stat_oracle device={device} dtype={dtype} grad_error={(grad_x-grad_ref).abs().max().item():.9g} floor={math.sqrt(1e-5):.9g}")


def semantic_regime(accepted, candidate, model_a, model_c, initial_state, regime):
    model_a.load_state_dict({k: v for k, v in initial_state.items() if k != "dispersion_adapter.weight"})
    model_c.load_state_dict(initial_state)
    model_a.train(); model_c.train()
    model_a.zero_grad(set_to_none=True); model_c.zero_grad(set_to_none=True)
    inputs, targets = fixed_input(next(model_c.parameters()).device)
    torch.manual_seed(44044); torch.cuda.manual_seed(44044)
    start = rng_state()
    restore_rng(start)
    logits_a, loss_a, _ = regime_loss(model_a, inputs, targets, regime)
    loss_a.backward()
    after_a = rng_state()
    capture = Capture(model_c)
    restore_rng(start)
    logits_c, loss_c, dense_target = regime_loss(model_c, inputs, targets, regime)
    hidden = capture.values["mean_linear"] + capture.values["adapter"]
    capture.values["bn"].retain_grad()
    loss_c.backward()
    after_c = rng_state()

    assert torch.equal(logits_a, logits_c) and torch.equal(loss_a, loss_c)
    assert torch.equal(after_a[0], after_c[0]) and torch.equal(after_a[1], after_c[1])
    sigma_ref = independent_std(F.relu(capture.values["bn"].detach()))
    torch.testing.assert_close(capture.values["sigma"], sigma_ref, rtol=2e-5, atol=2e-7)
    assert torch.equal(capture.values["relu_hidden"], F.relu(hidden))

    named_a = dict(model_a.named_parameters()); named_c = dict(model_c.named_parameters())
    worst = ("", 0.0, 0.0)
    for name, parameter_a in named_a.items():
        parameter_c = named_c[name]
        assert (parameter_a.grad is None) == (parameter_c.grad is None)
        if parameter_a.grad is not None:
            absolute = (parameter_a.grad - parameter_c.grad).abs().max().item()
            relative = (parameter_a.grad - parameter_c.grad).norm().item() / max(parameter_a.grad.norm().item(), 1e-30)
            if absolute > worst[1]: worst = (name, absolute, relative)
            assert torch.equal(parameter_a.grad, parameter_c.grad), f"common gradient {name}"

    probabilities = logits_c.detach().softmax(dim=1)
    g_logits = (probabilities - dense_target) / logits_c.shape[0]
    g_refined = g_logits @ model_c.fc.weight.detach()
    g_relu = model_c.POOLED_HEAD_SCALE if hasattr(model_c, "POOLED_HEAD_SCALE") else candidate.POOLED_HEAD_SCALE
    g_hidden = candidate.POOLED_HEAD_SCALE * (g_refined @ model_c.pooled_head[2].weight.detach()) * (hidden.detach() > 0)
    adapter_ref = g_hidden.T @ capture.values["sigma"].detach()
    adapter_grad = model_c.dispersion_adapter.weight.grad
    torch.testing.assert_close(adapter_grad, adapter_ref, rtol=2e-5, atol=2e-7)
    assert torch.isfinite(adapter_grad).all() and adapter_grad.norm().item() > 0

    floor_ratio = math.sqrt(1e-5) / capture.values["sigma"].detach().median().item()
    centered_mu = F.adaptive_avg_pool2d(F.relu(capture.values["bn"].detach()), 1).flatten(1)
    mu_flat = centered_mu.flatten(); sigma_flat = capture.values["sigma"].detach().flatten()
    correlation = torch.corrcoef(torch.stack([mu_flat, sigma_flat]))[0, 1].item()
    print(f"semantic regime={regime} worst_common={worst} adapter_grad={adapter_grad.norm().item():.9g} corr={correlation:.6f} floor_ratio={floor_ratio:.9g}")
    capture.close()
    return {name: p.grad.detach().clone() for name, p in named_c.items() if p.grad is not None}


def update_oracle(candidate, initial_state, gradients):
    model = candidate.WideResNet(candidate.STAGE_BLOCKS, candidate.WIDEN_FACTOR, candidate.NUM_CLASSES).cuda()
    model.load_state_dict(initial_state)
    named = dict(model.named_parameters())
    opt = optimizer(candidate, model)
    expected = {}
    buffers = {}
    for index, (name, parameter) in enumerate(named.items()):
        parameter.grad = gradients[name].clone()
        wd = 5e-4 if parameter.ndim >= 2 else 0.0
        d = parameter.grad + wd * parameter.detach()
        b0 = torch.full_like(parameter, 0.0001 * (index + 1))
        opt.state[parameter]["momentum_buffer"] = b0.clone()
        b1 = 0.9 * b0 + d
        expected[name] = parameter.detach().clone() - 0.037 * (d + 0.9 * b1)
        buffers[name] = b1
    opt.step()
    max_parameter = max((named[name] - expected[name]).abs().max().item() for name in named)
    max_buffer = max((opt.state[named[name]]["momentum_buffer"] - buffers[name]).abs().max().item() for name in named)
    print(f"update_oracle parameter_error={max_parameter:.9g} buffer_error={max_buffer:.9g}")
    assert max_parameter <= 2e-7 and max_buffer <= 2e-7


def semantics():
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA H20"
    accepted, candidate, source_a, source_c = load_modules()
    audit_source(source_a, source_c)
    torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = True
    model_a, model_c = build_models(accepted, candidate, "cuda")
    initial = clone_state(model_c)
    statistic_oracles(candidate)
    gradients_early = semantic_regime(accepted, candidate, model_a, model_c, initial, "early")
    gradients_hard = semantic_regime(accepted, candidate, model_a, model_c, initial, "hard")
    update_oracle(candidate, initial, gradients_early)
    assert set(gradients_early) == set(gradients_hard)
    print("SEMANTICS PASS")


def timing_step(module, model, opt, host_inputs, host_targets, distribution, regime):
    inputs = host_inputs.cuda(non_blocking=True); targets = host_targets.cuda(non_blocking=True)
    for group in opt.param_groups: group["lr"] = 0.037
    opt.zero_grad(set_to_none=True)
    if regime == "early":
        mixed, a, b, mix = module.mixup_batch(inputs, targets, distribution)
        logits = model(mixed)
        loss = mix * F.cross_entropy(logits, a) + (1 - mix) * F.cross_entropy(logits, b)
    else:
        loss = F.cross_entropy(model(inputs), targets)
    if not torch.isfinite(loss): raise RuntimeError("nonfinite timing loss")
    loss.backward(); opt.step(); torch.cuda.synchronize()


def run_window(module, model, base, rng, host_inputs, host_targets, regime, steps):
    model.load_state_dict(base); model.train(); opt = optimizer(module, model)
    distribution = torch.distributions.Beta(torch.tensor(module.MIXUP_ALPHA, device="cuda"), torch.tensor(module.MIXUP_ALPHA, device="cuda"))
    restore_rng(rng); torch.cuda.synchronize(); start = time.perf_counter()
    for _ in range(steps): timing_step(module, model, opt, host_inputs, host_targets, distribution, regime)
    torch.cuda.synchronize(); return 1000 * (time.perf_counter() - start) / steps


def timing():
    accepted, candidate, _sa, _sc = load_modules()
    torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = True
    model_a, model_c = build_models(accepted, candidate, "cuda")
    base_a = clone_state(model_a); base_c = clone_state(model_c)
    host_inputs = torch.linspace(-1, 1, 256 * 3 * 32 * 32).reshape(256, 3, 32, 32).pin_memory()
    host_targets = (torch.arange(256) % 10).pin_memory()
    torch.manual_seed(9944); torch.cuda.manual_seed(9944); window_rng = rng_state()
    arms = {"A": (accepted, model_a, base_a), "C": (candidate, model_c, base_c)}
    for label, regime in [("A", "early"), ("C", "early"), ("A", "hard"), ("C", "hard")]:
        run_window(*arms[label], window_rng, host_inputs, host_targets, regime, 20)
    schedule = [("A","early"),("C","early"),("A","hard"),("C","hard"),("C","hard"),("A","hard"),("C","early"),("A","early")]
    windows = {"early":{"A":[],"C":[]}, "hard":{"A":[],"C":[]}}
    pairs = {"early":[], "hard":[]}; candidate_peak = 0
    for cycle in range(2):
        values = []
        for label, regime in schedule:
            if label == "C": torch.cuda.reset_peak_memory_stats()
            value = run_window(*arms[label], window_rng, host_inputs, host_targets, regime, 80)
            if label == "C": candidate_peak = max(candidate_peak, torch.cuda.max_memory_allocated())
            windows[regime][label].append(value); values.append((label, regime, value))
            print(f"timing cycle={cycle} arm={label} regime={regime} ms={value:.9f}")
        pairs["early"].extend([(values[0][2],values[1][2]),(values[7][2],values[6][2])])
        pairs["hard"].extend([(values[2][2],values[3][2]),(values[5][2],values[4][2])])
    for regime in ("early","hard"):
        ratios = [c/a for a,c in pairs[regime]]
        ratio_cv = statistics.pstdev(ratios)/statistics.mean(ratios)
        for label in ("A","C"):
            cv = statistics.pstdev(windows[regime][label])/statistics.mean(windows[regime][label])
            print(f"timing_summary regime={regime} arm={label} values={windows[regime][label]} cv={cv:.9f}")
            assert cv <= .05
        print(f"ratio_summary regime={regime} ratios={ratios} cv={ratio_cv:.9f}"); assert ratio_cv <= .01
    retentions = []
    for i in range(4):
        ae,ce=pairs["early"][i]; ah,ch=pairs["hard"][i]
        retentions.append((.65/ce+.35/ch)/(.65/ae+.35/ah))
    median = statistics.median(retentions); projected = 130.304*median; peak = candidate_peak/1024/1024
    print(f"timing_gate retentions={retentions} median={median:.9f} projected={projected:.6f} peak_mb={peak:.3f}")
    assert all(r >= RETENTION_FLOOR for r in retentions); assert projected >= 127; assert peak < 2048
    print("TIMING PASS")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"semantics","timing"}: raise SystemExit("usage: preflight.py semantics|timing")
    semantics() if sys.argv[1] == "semantics" else timing()
