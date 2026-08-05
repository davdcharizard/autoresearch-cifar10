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
EXPECTED_PARAMS = 1_003_482
EXPECTED_PARAMETER_TENSORS = 52
EXPECTED_STATE = 97
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
    spec = importlib.util.spec_from_file_location("exp045_candidate", ROOT / "train.py")
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    accepted_source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:train.py"], cwd=ROOT, text=True
    )
    accepted = types.ModuleType("exp045_accepted")
    accepted.__file__ = f"git:{BASELINE}:train.py"
    exec(compile(accepted_source, accepted.__file__, "exec"), accepted.__dict__)
    return accepted, candidate, accepted_source, (ROOT / "train.py").read_text()


def rng_state():
    return torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def restore_rng(state):
    torch.random.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


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
    return [
        (
            [names[id(parameter)] for parameter in group["params"]],
            {key: value for key, value in group.items() if key != "params"},
        )
        for group in opt.param_groups
    ]


def audit_source(accepted_source, candidate_source):
    diff = subprocess.check_output(
        ["git", "diff", "--unified=0", BASELINE, "--", "train.py"],
        cwd=ROOT,
        text=True,
    )
    additions = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removals = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    print(f"source additions={len(additions)} removals={len(removals)}")
    assert "nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else None" in candidate_source
    assert "stride=1 if stride == 2 else stride" in candidate_source
    assert "shortcut = x" in candidate_source
    assert "shortcut_pool" not in accepted_source
    tree = ast.parse(candidate_source)
    pools = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "AvgPool2d"]
    assert len(pools) == 1
    assert subprocess.check_output(["git", "diff", "--name-only", BASELINE], cwd=ROOT, text=True).strip() == "train.py"
    subprocess.run(["git", "diff", "--exit-code", BASELINE, "--", "prepare.py", "pyproject.toml"], cwd=ROOT, check=True)


def build_models(accepted, candidate, device="cuda"):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    start = rng_state()
    restore_rng(start)
    model_a = accepted.WideResNet(accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES).to(device)
    after_a = rng_state()
    restore_rng(start)
    model_c = candidate.WideResNet(candidate.STAGE_BLOCKS, candidate.WIDEN_FACTOR, candidate.NUM_CLASSES).to(device)
    after_c = rng_state()
    assert torch.equal(after_a[0], after_c[0]) and torch.equal(after_a[1], after_c[1])
    state_a, state_c = model_a.state_dict(), model_c.state_dict()
    assert list(state_a) == list(state_c) and len(state_c) == EXPECTED_STATE
    for name in state_a:
        assert torch.equal(state_a[name], state_c[name]), name
    assert sum(parameter.numel() for parameter in model_c.parameters()) == EXPECTED_PARAMS
    assert len(list(model_c.parameters())) == EXPECTED_PARAMETER_TENSORS
    assert optimizer_signature(model_a, optimizer(accepted, model_a)) == optimizer_signature(model_c, optimizer(candidate, model_c))
    return model_a, model_c


def topology(model):
    model.eval()
    named_pools = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.AvgPool2d)]
    assert [name for name, _ in named_pools] == ["layer2.0.shortcut_pool", "layer3.0.shortcut_pool"]
    assert all(pool.kernel_size == 2 and pool.stride == 2 and pool.padding == 0 and not pool.ceil_mode for _, pool in named_pools)
    entries = [model.layer1[0], model.layer2[0], model.layer3[0]]
    assert [block.shortcut.stride for block in entries] == [(1, 1), (1, 1), (1, 1)]
    assert [block.conv1.stride for block in entries] == [(1, 1), (2, 2), (2, 2)]
    assert [block.shortcut_pool is None for block in entries] == [True, False, False]
    identities = [model.layer1[1], model.layer2[1], model.layer3[1], model.layer3[2]]
    assert all(block.shortcut is None and block.shortcut_pool is None for block in identities)
    assert not any(key.endswith("shortcut_pool") for key in model.state_dict())
    shapes = []
    x = torch.randn(2, 3, 32, 32, device=next(model.parameters()).device)
    with torch.no_grad():
        x = model.conv1(x)
        for layer in (model.layer1, model.layer2, model.layer3):
            x = layer(x)
            shapes.append(tuple(x.shape))
    assert shapes == [(2, 32, 32, 32), (2, 64, 16, 16), (2, 128, 8, 8)]
    print(f"topology pools={[name for name, _ in named_pools]} shapes={shapes}")


def capture_block(block, x):
    values = {}

    def save_output(name):
        def hook(_module, _inputs, output):
            values[name] = output
        return hook

    def save_input(name):
        def hook(_module, inputs):
            values[name] = inputs[0]
        return hook

    handles = [
        block.conv2.register_forward_hook(save_output("main")),
    ]
    shortcut_pool = getattr(block, "shortcut_pool", None)
    if shortcut_pool is not None:
        handles.append(shortcut_pool.register_forward_pre_hook(save_input("z")))
        handles.append(shortcut_pool.register_forward_hook(save_output("a")))
    if block.shortcut is not None:
        handles.append(block.shortcut.register_forward_pre_hook(save_input("shortcut_input")))
        handles.append(block.shortcut.register_forward_hook(save_output("shortcut")))
    output = block(x)
    for handle in handles:
        handle.remove()
    return output, values


def independent_pool(z):
    n, c, h, w = z.shape
    return z.reshape(n, c, h // 2, 2, w // 2, 2).mean(dim=(3, 5))


def phase_and_gradient_oracle(candidate):
    for dtype, rtol, atol in [(torch.float64, 1e-10, 1e-12), (torch.float32, 2e-5, 2e-7)]:
        block = candidate.PreActBlock(3, 5, stride=2).to(dtype)
        block.eval()
        z = torch.linspace(-0.8, 1.1, 2 * 3 * 8 * 8, dtype=dtype).reshape(2, 3, 8, 8).requires_grad_(True)
        a = block.shortcut_pool(z)
        projected = block.shortcut(a)
        reference_a = independent_pool(z)
        reference_y = torch.einsum("oc,ncij->noij", block.shortcut.weight[:, :, 0, 0], reference_a)
        torch.testing.assert_close(a, reference_a, rtol=rtol, atol=atol)
        torch.testing.assert_close(projected, reference_y, rtol=rtol, atol=atol)
        upstream = torch.linspace(-0.3, 0.5, projected.numel(), dtype=dtype).reshape_as(projected)
        objective = (projected * upstream).sum()
        grad_z, grad_w = torch.autograd.grad(objective, (z, block.shortcut.weight))
        grad_w_ref = torch.einsum("noij,ncij->oc", upstream, reference_a)[:, :, None, None]
        grad_a = torch.einsum("noij,oc->ncij", upstream, block.shortcut.weight[:, :, 0, 0])
        grad_z_ref = grad_a.repeat_interleave(2, 2).repeat_interleave(2, 3) / 4
        torch.testing.assert_close(grad_w, grad_w_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(grad_z, grad_z_ref, rtol=rtol, atol=atol)
        for py in range(2):
            for px in range(2):
                torch.testing.assert_close(grad_z[:, :, py::2, px::2], grad_a / 4, rtol=rtol, atol=atol)
        impulse_values = []
        for py in range(2):
            for px in range(2):
                impulse = torch.zeros(1, 3, 8, 8, dtype=dtype)
                impulse[:, :, py::2, px::2] = 1
                impulse_values.append(block.shortcut(block.shortcut_pool(impulse)))
        for value in impulse_values[1:]:
            torch.testing.assert_close(value, impulse_values[0], rtol=rtol, atol=atol)
        print(f"phase_oracle dtype={dtype} forward_error={(projected-reference_y).abs().max().item():.9g} grad_error={(grad_z-grad_z_ref).abs().max().item():.9g}")


def block_semantics(accepted, candidate, model_a, model_c):
    for index, (block_a, block_c, shape) in enumerate([
        (model_a.layer2[0], model_c.layer2[0], (2, 32, 32, 32)),
        (model_a.layer3[0], model_c.layer3[0], (2, 64, 16, 16)),
    ], start=2):
        x = torch.linspace(-1.2, 1.3, math.prod(shape), device="cuda").reshape(shape)
        block_a.eval(); block_c.eval()
        out_a, values_a = capture_block(block_a, x)
        out_c, values_c = capture_block(block_c, x)
        assert torch.equal(values_a["main"], values_c["main"])
        torch.testing.assert_close(values_c["a"], independent_pool(values_c["z"]), rtol=2e-5, atol=2e-7)
        assert values_c["shortcut"].shape == values_c["main"].shape
        assert not torch.equal(values_a["shortcut"], values_c["shortcut"])
        shortcut_ratio = values_c["shortcut"].square().mean().sqrt() / values_a["shortcut"].square().mean().sqrt()
        main_ratio = values_c["main"].square().mean().sqrt() / values_c["shortcut"].square().mean().sqrt()
        assert torch.isfinite(out_c).all()
        print(f"transition={index} shortcut_rms_ratio={shortcut_ratio.item():.9f} main_to_shortcut={main_ratio.item():.9f}")
    layer1_x = torch.randn(2, 16, 32, 32, device="cuda")
    out_a, values_a = capture_block(model_a.layer1[0].eval(), layer1_x)
    out_c, values_c = capture_block(model_c.layer1[0].eval(), layer1_x)
    assert torch.equal(out_a, out_c) and torch.equal(values_a["shortcut"], values_c["shortcut"])
    for block_a, block_c, shape in [
        (model_a.layer1[1], model_c.layer1[1], (2, 32, 32, 32)),
        (model_a.layer2[1], model_c.layer2[1], (2, 64, 16, 16)),
        (model_a.layer3[1], model_c.layer3[1], (2, 128, 8, 8)),
        (model_a.layer3[2], model_c.layer3[2], (2, 128, 8, 8)),
    ]:
        x = torch.randn(*shape, device="cuda")
        assert torch.equal(block_a.eval()(x), block_c.eval()(x))
    print("block controls exact")


def full_step_and_update(candidate, model):
    model.train()
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    buffers_before = {name: value.detach().clone() for name, value in model.named_buffers()}
    opt = optimizer(candidate, model)
    inputs = torch.linspace(-1, 1, 8 * 3 * 32 * 32, device="cuda").reshape(8, 3, 32, 32)
    targets = torch.arange(8, device="cuda") % 10
    opt.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert torch.isfinite(loss) and all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    expected = {}
    for name, parameter in model.named_parameters():
        decay = candidate.WEIGHT_DECAY if parameter.ndim >= 2 else 0.0
        direction = parameter.grad + decay * parameter.detach()
        expected[name] = parameter.detach().clone() - 0.037 * (direction + candidate.MOMENTUM * direction)
    opt.step()
    max_error = max((dict(model.named_parameters())[name] - value).abs().max().item() for name, value in expected.items())
    assert max_error <= 2e-7
    assert any(not torch.equal(buffers_before[name], value) for name, value in model.named_buffers() if name in buffers_before)
    print(f"full_step loss={loss.item():.9f} logit_rms={logits.square().mean().sqrt().item():.9f} update_error={max_error:.9g}")


def semantics():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "NVIDIA H20"
    accepted, candidate, source_a, source_c = load_modules()
    audit_source(source_a, source_c)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    model_a, model_c = build_models(accepted, candidate)
    topology(model_c)
    phase_and_gradient_oracle(candidate)
    block_semantics(accepted, candidate, model_a, model_c)
    full_step_and_update(candidate, model_c)
    print("SEMANTICS PASS")


def timing_step(module, model, opt, host_inputs, host_targets, distribution, regime):
    inputs = host_inputs.cuda(non_blocking=True)
    targets = host_targets.cuda(non_blocking=True)
    for group in opt.param_groups:
        group["lr"] = 0.037
    opt.zero_grad(set_to_none=True)
    if regime == "early":
        mixed, a, b, mix = module.mixup_batch(inputs, targets, distribution)
        logits = model(mixed)
        loss = mix * F.cross_entropy(logits, a) + (1 - mix) * F.cross_entropy(logits, b)
    else:
        loss = F.cross_entropy(model(inputs), targets)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite timing loss")
    loss.backward()
    opt.step()


def run_window(module, state, rng, host_inputs, host_targets, regime, steps, measure_peak=False):
    model = module.WideResNet(module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES)
    model.load_state_dict(state)
    model = model.cuda().train()
    opt = optimizer(module, model)
    distribution = torch.distributions.Beta(torch.tensor(module.MIXUP_ALPHA, device="cuda"), torch.tensor(module.MIXUP_ALPHA, device="cuda"))
    restore_rng(rng)
    torch.cuda.synchronize()
    if measure_peak:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        timing_step(module, model, opt, host_inputs, host_targets, distribution, regime)
    torch.cuda.synchronize()
    elapsed = 1000 * (time.perf_counter() - start) / steps
    peak = torch.cuda.max_memory_allocated() if measure_peak else 0
    del opt, model, distribution
    torch.cuda.empty_cache()
    return elapsed, peak


def timing():
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "NVIDIA H20"
    accepted, candidate, _source_a, _source_c = load_modules()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    model_a, model_c = build_models(accepted, candidate, device="cpu")
    states = {"A": clone_state(model_a), "C": clone_state(model_c)}
    del model_a, model_c
    host_inputs = torch.linspace(-1, 1, 256 * 3 * 32 * 32).reshape(256, 3, 32, 32).pin_memory()
    host_targets = (torch.arange(256) % 10).pin_memory()
    torch.manual_seed(9945)
    torch.cuda.manual_seed(9945)
    window_rng = rng_state()
    arms = {"A": accepted, "C": candidate}
    for label, regime in [("A", "early"), ("C", "early"), ("A", "hard"), ("C", "hard")]:
        run_window(arms[label], states[label], window_rng, host_inputs, host_targets, regime, 20)
    schedule = [("A", "early"), ("C", "early"), ("A", "hard"), ("C", "hard"), ("C", "hard"), ("A", "hard"), ("C", "early"), ("A", "early")]
    windows = {"early": {"A": [], "C": []}, "hard": {"A": [], "C": []}}
    pairs = {"early": [], "hard": []}
    candidate_peak = 0
    for cycle in range(2):
        values = []
        for label, regime in schedule:
            value, peak = run_window(arms[label], states[label], window_rng, host_inputs, host_targets, regime, 50, label == "C")
            candidate_peak = max(candidate_peak, peak)
            windows[regime][label].append(value)
            values.append((label, regime, value))
            print(f"timing cycle={cycle} arm={label} regime={regime} ms={value:.9f}")
        pairs["early"].extend([(values[0][2], values[1][2]), (values[7][2], values[6][2])])
        pairs["hard"].extend([(values[2][2], values[3][2]), (values[5][2], values[4][2])])
    for regime in ("early", "hard"):
        ratios = [candidate_ms / accepted_ms for accepted_ms, candidate_ms in pairs[regime]]
        ratio_cv = statistics.pstdev(ratios) / statistics.mean(ratios)
        for label in ("A", "C"):
            cv = statistics.pstdev(windows[regime][label]) / statistics.mean(windows[regime][label])
            print(f"timing_summary regime={regime} arm={label} values={windows[regime][label]} cv={cv:.9f}")
            assert cv <= 0.05
        print(f"ratio_summary regime={regime} ratios={ratios} cv={ratio_cv:.9f}")
        assert ratio_cv <= 0.01
    retentions = []
    for index in range(4):
        ae, ce = pairs["early"][index]
        ah, ch = pairs["hard"][index]
        retentions.append((0.65 / ce + 0.35 / ch) / (0.65 / ae + 0.35 / ah))
    median = statistics.median(retentions)
    projected = 130.304 * median
    peak_mb = candidate_peak / 1024 / 1024
    print(f"timing_gate retentions={retentions} median={median:.9f} projected={projected:.6f} peak_mb={peak_mb:.3f}")
    assert all(value >= RETENTION_FLOOR for value in retentions)
    assert projected >= 127
    assert peak_mb < 2048
    print("TIMING PASS")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"semantics", "timing"}:
        raise SystemExit("usage: preflight.py semantics|timing")
    semantics() if sys.argv[1] == "semantics" else timing()
