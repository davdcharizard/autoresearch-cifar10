import ast
import copy
import importlib.util
import math
import statistics
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
BASELINE = "a7c42dc"
EXPECTED_PARAMS = 1_003_482
EXPECTED_CONVOLUTIONS = 18
EXPECTED_CONV_VALUES = 983_472
EXPECTED_FILTERS = 1_392
RETENTION_FLOOR = 127.0 / 130.304


class BlockedEval:
    def evaluate(self, *_args, **_kwargs):
        raise AssertionError("evaluation is forbidden in preflight")


def fail_dataset(*_args, **_kwargs):
    raise AssertionError("dataset construction is forbidden in preflight")


def load_modules():
    import prepare
    from torchvision import datasets

    prepare.Eval = BlockedEval
    datasets.CIFAR10 = fail_dataset

    candidate_spec = importlib.util.spec_from_file_location(
        "exp043_candidate", ROOT / "train.py"
    )
    candidate = importlib.util.module_from_spec(candidate_spec)
    candidate_spec.loader.exec_module(candidate)

    accepted_source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:train.py"], cwd=ROOT, text=True
    )
    accepted = types.ModuleType("exp043_accepted")
    accepted.__file__ = f"git:{BASELINE}:train.py"
    exec(compile(accepted_source, accepted.__file__, "exec"), accepted.__dict__)
    return accepted, candidate, accepted_source, (ROOT / "train.py").read_text()


def state_clone(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_state_equal(left, right, label):
    assert list(left) == list(right), f"{label}: state keys differ"
    for name in left:
        assert left[name].dtype == right[name].dtype, f"{label}:{name}: dtype"
        assert left[name].shape == right[name].shape, f"{label}:{name}: shape"
        assert torch.equal(left[name], right[name]), f"{label}:{name}: bytes"


def make_optimizer(module, model, lr=0.037):
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


def convolution_weights(model):
    return [m.weight for m in model.modules() if isinstance(m, nn.Conv2d)]


def optimizer_signature(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        options = {key: value for key, value in group.items() if key != "params"}
        groups.append(([names[id(p)] for p in group["params"]], options))
    return groups


def rng_state():
    return torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def restore_rng(state):
    torch.random.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def audit_source(accepted_source, candidate_source):
    diff = subprocess.check_output(
        ["git", "diff", "--unified=0", BASELINE, "--", "train.py"],
        cwd=ROOT,
        text=True,
    )
    additions = [
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
    ]
    removals = [
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---") and line[1:].strip()
    ]
    expected_additions = [
        "def centralize_convolution_gradients(parameters):",
        "for parameter in parameters:",
        "gradient = parameter.grad",
        "if gradient is not None:",
        "gradient.sub_(gradient.mean(dim=(1, 2, 3), keepdim=True))",
        "convolution_weights = [",
        "module.weight for module in model.modules() if isinstance(module, nn.Conv2d)",
        "]",
        "centralize_convolution_gradients(convolution_weights)",
    ]
    print(f"source additions={additions}")
    print(f"source removals={removals}")
    assert not removals
    assert Counter(additions) == Counter(expected_additions)

    tree = ast.parse(candidate_source)
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "centralize_convolution_gradients"
    ]
    assert len(helpers) == 1
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "centralize_convolution_gradients"
    ]
    backward_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "backward"
    ]
    step_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
    ]
    cache_assignments = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "convolution_weights" for target in node.targets)
    ]
    assert len(calls) == len(backward_calls) == len(step_calls) == len(cache_assignments) == 1
    assert backward_calls[0].lineno < calls[0].lineno < step_calls[0].lineno
    assert "model.modules()" in ast.get_source_segment(candidate_source, cache_assignments[0])
    assert accepted_source != candidate_source


def build_paired_models(accepted, candidate, device):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    start_rng = rng_state()

    restore_rng(start_rng)
    accepted_model = accepted.WideResNet(
        accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES
    ).to(device)
    accepted_rng = rng_state()

    restore_rng(start_rng)
    candidate_model = candidate.WideResNet(
        candidate.STAGE_BLOCKS, candidate.WIDEN_FACTOR, candidate.NUM_CLASSES
    ).to(device)
    candidate_rng = rng_state()

    assert torch.equal(accepted_rng[0], candidate_rng[0])
    assert torch.equal(accepted_rng[1], candidate_rng[1])
    assert_state_equal(
        state_clone(accepted_model), state_clone(candidate_model), "initial model"
    )
    assert sum(p.numel() for p in candidate_model.parameters()) == EXPECTED_PARAMS

    accepted_optimizer = make_optimizer(accepted, accepted_model)
    candidate_optimizer = make_optimizer(candidate, candidate_model)
    assert optimizer_signature(accepted_model, accepted_optimizer) == optimizer_signature(
        candidate_model, candidate_optimizer
    )
    assert not accepted_optimizer.state and not candidate_optimizer.state
    assert rng_state()[0].equal(candidate_rng[0])
    assert rng_state()[1].equal(candidate_rng[1])
    return accepted_model, candidate_model


def assert_selection(model):
    selected = convolution_weights(model)
    named = dict(model.named_parameters())
    expected = [
        module.weight for module in model.modules() if isinstance(module, nn.Conv2d)
    ]
    linear = {id(module.weight) for module in model.modules() if isinstance(module, nn.Linear)}
    print(
        "selection "
        f"tensors={len(selected)} values={sum(p.numel() for p in selected)} "
        f"filters={sum(p.shape[0] for p in selected)}"
    )
    assert len(selected) == EXPECTED_CONVOLUTIONS
    assert sum(p.numel() for p in selected) == EXPECTED_CONV_VALUES
    assert sum(p.shape[0] for p in selected) == EXPECTED_FILTERS
    assert len({id(p) for p in selected}) == len(selected)
    assert all(left is right for left, right in zip(selected, expected))
    assert not ({id(p) for p in selected} & linear)
    assert all(any(p is q for q in named.values()) for p in selected)
    return selected


def fixed_fixture(device, batch=8):
    values = torch.linspace(-1.0, 1.0, batch * 3 * 32 * 32, device=device)
    inputs = values.reshape(batch, 3, 32, 32)
    targets = torch.arange(batch, device=device) % 10
    return inputs, targets


def loss_for_regime(model, inputs, targets, regime):
    if regime == "early":
        permutation = torch.arange(inputs.shape[0] - 1, -1, -1, device=inputs.device)
        mix = inputs.new_tensor(0.3)
        outputs = model(mix * inputs + (1.0 - mix) * inputs[permutation])
        loss = mix * F.cross_entropy(outputs, targets) + (1.0 - mix) * F.cross_entropy(
            outputs, targets[permutation]
        )
    else:
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
    return outputs, loss


def check_projection_regime(accepted, candidate, accepted_model, candidate_model, base_state, regime):
    accepted_model.load_state_dict(base_state)
    candidate_model.load_state_dict(base_state)
    accepted_model.train()
    candidate_model.train()
    accepted_model.zero_grad(set_to_none=True)
    candidate_model.zero_grad(set_to_none=True)
    inputs, targets = fixed_fixture(next(candidate_model.parameters()).device)

    torch.manual_seed(17043)
    torch.cuda.manual_seed(17043)
    fixture_rng = rng_state()
    restore_rng(fixture_rng)
    accepted_outputs, accepted_loss = loss_for_regime(
        accepted_model, inputs, targets, regime
    )
    accepted_loss.backward()
    accepted_after_rng = rng_state()

    restore_rng(fixture_rng)
    candidate_outputs, candidate_loss = loss_for_regime(
        candidate_model, inputs, targets, regime
    )
    candidate_loss.backward()
    candidate_after_rng = rng_state()

    assert torch.equal(accepted_outputs, candidate_outputs)
    assert torch.equal(accepted_loss, candidate_loss)
    assert torch.equal(accepted_after_rng[0], candidate_after_rng[0])
    assert torch.equal(accepted_after_rng[1], candidate_after_rng[1])
    assert_state_equal(state_clone(accepted_model), state_clone(candidate_model), f"{regime} forward")

    accepted_named = dict(accepted_model.named_parameters())
    candidate_named = dict(candidate_model.named_parameters())
    raw = {}
    for name in accepted_named:
        left = accepted_named[name].grad
        right = candidate_named[name].grad
        assert (left is None) == (right is None), f"{regime}:{name}: None"
        if left is not None:
            assert left.shape == right.shape and left.dtype == right.dtype
            assert torch.equal(left, right), f"{regime}:{name}: raw gradient"
            raw[name] = left.detach().clone()

    selected = assert_selection(candidate_model)
    before_helper_rng = rng_state()
    candidate.centralize_convolution_gradients(selected)
    after_helper_rng = rng_state()
    assert torch.equal(before_helper_rng[0], after_helper_rng[0])
    assert torch.equal(before_helper_rng[1], after_helper_rng[1])

    selected_ids = {id(parameter) for parameter in selected}
    aggregate_removed = 0.0
    maximum_mean = 0.0
    maximum_idempotence = 0.0
    maximum_fp64_error = 0.0
    for name, parameter in candidate_named.items():
        if parameter.grad is None:
            continue
        if id(parameter) not in selected_ids:
            assert torch.equal(parameter.grad, raw[name]), f"{regime}:{name}: excluded changed"
            continue

        raw_gradient = raw[name]
        projected = parameter.grad.detach()
        count = raw_gradient.shape[1] * raw_gradient.shape[2] * raw_gradient.shape[3]
        fp32_reference = raw_gradient - raw_gradient.sum(
            dim=(1, 2, 3), keepdim=True
        ) / count
        fp64_raw = raw_gradient.double()
        fp64_reference = fp64_raw - fp64_raw.sum(
            dim=(1, 2, 3), keepdim=True
        ) / count
        fp64_error = (projected.double() - fp64_reference).abs().max().item()
        torch.testing.assert_close(projected, fp32_reference, rtol=2e-5, atol=2e-7)

        probe = nn.Parameter(torch.empty_like(projected), requires_grad=True)
        probe.grad = projected.clone()
        candidate.centralize_convolution_gradients([probe])
        idempotence = (probe.grad - projected).abs().max().item()
        residual_mean = projected.mean(dim=(1, 2, 3)).abs().max().item()
        raw_norm = raw_gradient.norm().item()
        projected_norm = projected.norm().item()
        removed_norm = (raw_gradient - projected).norm().item()
        removed_fraction = removed_norm / max(raw_norm, 1e-30)
        print(
            f"projection regime={regime} name={name} raw={raw_norm:.9g} "
            f"projected={projected_norm:.9g} removed_fraction={removed_fraction:.9g} "
            f"residual_mean={residual_mean:.9g} idempotence={idempotence:.9g} "
            f"fp64_error={fp64_error:.9g}"
        )
        assert residual_mean <= 2e-6
        assert idempotence <= 2e-6
        assert projected_norm <= raw_norm + 2e-6 * max(1.0, raw_norm)
        aggregate_removed += removed_norm
        maximum_mean = max(maximum_mean, residual_mean)
        maximum_idempotence = max(maximum_idempotence, idempotence)
        maximum_fp64_error = max(maximum_fp64_error, fp64_error)

    print(
        f"projection_summary regime={regime} removed={aggregate_removed:.9g} "
        f"max_mean={maximum_mean:.9g} max_idempotence={maximum_idempotence:.9g} "
        f"max_fp64_error={maximum_fp64_error:.9g}"
    )
    assert math.isfinite(aggregate_removed) and aggregate_removed > 0.0
    return raw


def check_update(candidate, initial_state, raw_gradients, name, selected, seeded):
    p0 = initial_state[name].detach().clone()
    raw = raw_gradients[name].detach().clone()
    parameter = nn.Parameter(p0.clone())
    optimizer = optim.SGD(
        [{"params": [parameter], "weight_decay": 5e-4}],
        lr=0.037,
        momentum=0.9,
        nesterov=True,
    )
    parameter.grad = raw.clone()
    if selected:
        candidate.centralize_convolution_gradients([parameter])
    data_gradient = parameter.grad.detach().clone()

    b0 = None
    if seeded:
        b0 = torch.linspace(
            -0.003, 0.004, parameter.numel(), device=parameter.device
        ).reshape_as(parameter)
        optimizer.state[parameter]["momentum_buffer"] = b0.clone()

    decay_gradient = data_gradient + 5e-4 * p0
    expected_buffer = decay_gradient if b0 is None else 0.9 * b0 + decay_gradient
    expected_parameter = p0 - 0.037 * (decay_gradient + 0.9 * expected_buffer)
    if selected:
        mean_error = (
            decay_gradient.mean(dim=(1, 2, 3))
            - 5e-4 * p0.mean(dim=(1, 2, 3))
        ).abs().max().item()
        post_decay_projection = decay_gradient - decay_gradient.mean(
            dim=(1, 2, 3), keepdim=True
        )
        distinction = (decay_gradient - post_decay_projection).abs().max().item()
        print(
            f"decay_order name={name} seeded={seeded} mean_error={mean_error:.9g} "
            f"post_decay_distinction={distinction:.9g}"
        )
        assert mean_error <= 2e-7
        assert distinction > 0.0

    optimizer.step()
    actual_buffer = optimizer.state[parameter]["momentum_buffer"]
    parameter_error = (parameter.detach() - expected_parameter).abs().max().item()
    buffer_error = (actual_buffer - expected_buffer).abs().max().item()
    print(
        f"update name={name} selected={selected} seeded={seeded} "
        f"parameter_error={parameter_error:.9g} buffer_error={buffer_error:.9g}"
    )
    torch.testing.assert_close(parameter, expected_parameter, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(actual_buffer, expected_buffer, rtol=2e-5, atol=2e-7)


def check_controls(accepted, candidate, accepted_source, candidate_source):
    accepted_constants = {
        name: value
        for name, value in accepted.__dict__.items()
        if name.isupper() and isinstance(value, (int, float, tuple))
    }
    candidate_constants = {
        name: value
        for name, value in candidate.__dict__.items()
        if name.isupper() and isinstance(value, (int, float, tuple))
    }
    assert accepted_constants == candidate_constants
    for point in (0.0, 15.0, 195.0, 300.0):
        assert accepted.learning_rate(point) == candidate.learning_rate(point)

    inputs = torch.linspace(-1, 1, 8 * 3 * 4 * 4, device="cuda").reshape(8, 3, 4, 4)
    targets = torch.arange(8, device="cuda")
    torch.manual_seed(43043)
    torch.cuda.manual_seed(43043)
    start = rng_state()
    distribution_a = torch.distributions.Beta(
        torch.tensor(accepted.MIXUP_ALPHA, device="cuda"),
        torch.tensor(accepted.MIXUP_ALPHA, device="cuda"),
    )
    restore_rng(start)
    accepted_mix = accepted.mixup_batch(inputs, targets, distribution_a)
    accepted_after = rng_state()
    distribution_c = torch.distributions.Beta(
        torch.tensor(candidate.MIXUP_ALPHA, device="cuda"),
        torch.tensor(candidate.MIXUP_ALPHA, device="cuda"),
    )
    restore_rng(start)
    candidate_mix = candidate.mixup_batch(inputs, targets, distribution_c)
    candidate_after = rng_state()
    for left, right in zip(accepted_mix, candidate_mix):
        assert torch.equal(left, right)
    assert torch.equal(accepted_after[0], candidate_after[0])
    assert torch.equal(accepted_after[1], candidate_after[1])

    assert candidate_source.count("loss.backward()") == 1
    assert candidate_source.count("optimizer.step()") == 1
    assert candidate_source.count("torch.cuda.synchronize()") == 1
    assert candidate_source.count("evaluator.evaluate(model, device)") == 1
    assert "progress < MIXUP_END_FRACTION" in candidate_source
    assert "total_training_time >= RANDAUGMENT_END_FRACTION * TIME_BUDGET_S" in candidate_source
    assert "epoch % EVAL_EVERY == 0 or budget_exhausted" in candidate_source
    assert "torch.manual_seed(42)" in candidate_source
    assert "torch.cuda.manual_seed(42)" in candidate_source
    assert "DistributedDataParallel" not in candidate_source
    assert "DataParallel" not in candidate_source
    accepted_prepare = subprocess.check_output(
        ["git", "show", f"{BASELINE}:prepare.py"], cwd=ROOT
    )
    assert accepted_prepare == (ROOT / "prepare.py").read_bytes()
    print("controls exact constants/mixup/LR/temporal/cadence/single-device")


def semantics():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == "NVIDIA H20"
    accepted, candidate, accepted_source, candidate_source = load_modules()
    audit_source(accepted_source, candidate_source)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    accepted_model, candidate_model = build_paired_models(accepted, candidate, device)
    initial_state = state_clone(candidate_model)
    assert_selection(candidate_model)
    raw_early = check_projection_regime(
        accepted, candidate, accepted_model, candidate_model, initial_state, "early"
    )
    raw_hard = check_projection_regime(
        accepted, candidate, accepted_model, candidate_model, initial_state, "hard"
    )

    for seeded in (False, True):
        check_update(
            candidate,
            initial_state,
            raw_hard,
            "layer1.0.conv1.weight",
            selected=True,
            seeded=seeded,
        )
        check_update(
            candidate,
            initial_state,
            raw_early,
            "layer1.0.shortcut.weight",
            selected=True,
            seeded=seeded,
        )
        check_update(
            candidate,
            initial_state,
            raw_hard,
            "pooled_head.0.weight",
            selected=False,
            seeded=seeded,
        )
        check_update(
            candidate,
            initial_state,
            raw_early,
            "fc.weight",
            selected=False,
            seeded=seeded,
        )

    check_controls(accepted, candidate, accepted_source, candidate_source)
    print("SEMANTICS PASS")


def timing_models(accepted, candidate):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    start = rng_state()
    restore_rng(start)
    accepted_model = accepted.WideResNet(
        accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES
    ).cuda()
    restore_rng(start)
    candidate_model = candidate.WideResNet(
        candidate.STAGE_BLOCKS, candidate.WIDEN_FACTOR, candidate.NUM_CLASSES
    ).cuda()
    assert_state_equal(
        state_clone(accepted_model), state_clone(candidate_model), "timing initial"
    )
    base_state = state_clone(accepted_model)
    return accepted_model, candidate_model, base_state


def timing_step(
    module,
    model,
    optimizer,
    host_inputs,
    host_targets,
    distribution,
    regime,
    helper,
    selected,
):
    inputs = host_inputs.to("cuda", non_blocking=True)
    targets = host_targets.to("cuda", non_blocking=True)
    for group in optimizer.param_groups:
        group["lr"] = 0.037
    optimizer.zero_grad(set_to_none=True)
    if regime == "early":
        mixed, targets_a, targets_b, mix = module.mixup_batch(
            inputs, targets, distribution
        )
        outputs = model(mixed)
        loss = mix * F.cross_entropy(outputs, targets_a) + (
            1.0 - mix
        ) * F.cross_entropy(outputs, targets_b)
    else:
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite timing loss")
    loss.backward()
    if helper is not None:
        helper(selected)
    optimizer.step()
    torch.cuda.synchronize()


def prepare_window(module, model, base_state, rng):
    model.load_state_dict(base_state)
    model.train()
    optimizer = make_optimizer(module, model)
    distribution = torch.distributions.Beta(
        torch.tensor(module.MIXUP_ALPHA, device="cuda"),
        torch.tensor(module.MIXUP_ALPHA, device="cuda"),
    )
    restore_rng(rng)
    return optimizer, distribution


def run_window(
    module,
    model,
    base_state,
    rng,
    host_inputs,
    host_targets,
    regime,
    helper,
    selected,
    steps,
):
    optimizer, distribution = prepare_window(module, model, base_state, rng)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        timing_step(
            module,
            model,
            optimizer,
            host_inputs,
            host_targets,
            distribution,
            regime,
            helper,
            selected,
        )
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / steps


def warm(
    module,
    model,
    base_state,
    rng,
    host_inputs,
    host_targets,
    regime,
    helper,
    selected,
):
    optimizer, distribution = prepare_window(module, model, base_state, rng)
    for _ in range(20):
        timing_step(
            module,
            model,
            optimizer,
            host_inputs,
            host_targets,
            distribution,
            regime,
            helper,
            selected,
        )


def timing():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == "NVIDIA H20"
    accepted, candidate, _accepted_source, _candidate_source = load_modules()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    accepted_model, candidate_model, base_state = timing_models(accepted, candidate)
    values = torch.linspace(-1.0, 1.0, 256 * 3 * 32 * 32)
    host_inputs = values.reshape(256, 3, 32, 32).pin_memory()
    host_targets = (torch.arange(256) % 10).pin_memory()
    torch.manual_seed(99043)
    torch.cuda.manual_seed(99043)
    window_rng = rng_state()
    arms = {
        "A": (accepted, accepted_model, None, None),
        "C": (
            candidate,
            candidate_model,
            candidate.centralize_convolution_gradients,
            convolution_weights(candidate_model),
        ),
    }

    windows = {}
    pairs = {}
    candidate_peak = 0
    for regime in ("early", "hard"):
        for label, (module, model, helper, selected) in arms.items():
            warm(
                module,
                model,
                base_state,
                window_rng,
                host_inputs,
                host_targets,
                regime,
                helper,
                selected,
            )
        windows[regime] = {"A": [], "C": []}
        pairs[regime] = []
        for cycle in range(2):
            cycle_values = []
            for label in ("A", "C", "C", "A"):
                module, model, helper, selected = arms[label]
                if label == "C":
                    torch.cuda.reset_peak_memory_stats()
                value = run_window(
                    module,
                    model,
                    base_state,
                    window_rng,
                    host_inputs,
                    host_targets,
                    regime,
                    helper,
                    selected,
                    steps=100,
                )
                if label == "C":
                    candidate_peak = max(
                        candidate_peak, torch.cuda.max_memory_allocated()
                    )
                windows[regime][label].append(value)
                cycle_values.append((label, value))
                print(
                    f"timing regime={regime} cycle={cycle} arm={label} ms={value:.9f}"
                )
            assert [label for label, _ in cycle_values] == ["A", "C", "C", "A"]
            pairs[regime].append((cycle_values[0][1], cycle_values[1][1]))
            pairs[regime].append((cycle_values[3][1], cycle_values[2][1]))

    pair_ratios = {}
    for regime in ("early", "hard"):
        pair_ratios[regime] = [candidate_ms / accepted_ms for accepted_ms, candidate_ms in pairs[regime]]
        for label in ("A", "C"):
            values_for_arm = windows[regime][label]
            cv = statistics.pstdev(values_for_arm) / statistics.mean(values_for_arm)
            print(
                f"timing_summary regime={regime} arm={label} values={values_for_arm} "
                f"median={statistics.median(values_for_arm):.9f} cv={cv:.9f}"
            )
            assert cv <= 0.05
        ratio_cv = statistics.pstdev(pair_ratios[regime]) / statistics.mean(
            pair_ratios[regime]
        )
        print(
            f"pair_summary regime={regime} ratios={pair_ratios[regime]} cv={ratio_cv:.9f}"
        )
        assert ratio_cv <= 0.01

    retentions = []
    for index in range(4):
        accepted_early, candidate_early = pairs["early"][index]
        accepted_hard, candidate_hard = pairs["hard"][index]
        retention = (
            0.65 / candidate_early + 0.35 / candidate_hard
        ) / (0.65 / accepted_early + 0.35 / accepted_hard)
        retentions.append(retention)
    median_retention = statistics.median(retentions)
    projected_passes = 130.304 * median_retention
    candidate_peak_mb = candidate_peak / 1024 / 1024
    print(
        f"timing_gate retentions={retentions} median_retention={median_retention:.9f} "
        f"projected_passes={projected_passes:.6f} peak_mb={candidate_peak_mb:.3f} "
        f"floor={RETENTION_FLOOR:.9f}"
    )
    assert all(value >= RETENTION_FLOOR for value in retentions)
    assert projected_passes >= 127.0
    assert candidate_peak_mb < 2048.0
    print("TIMING PASS")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"semantics", "timing"}:
        raise SystemExit("usage: preflight.py semantics|timing")
    if sys.argv[1] == "semantics":
        semantics()
    else:
        timing()


if __name__ == "__main__":
    main()
