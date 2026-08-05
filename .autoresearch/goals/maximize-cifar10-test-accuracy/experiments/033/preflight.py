import argparse
import contextlib
import copy
import inspect
import io
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


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import prepare


class GuardEval:
    constructions = 0

    def __init__(self):
        type(self).constructions += 1

    def evaluate(self, model, device):
        raise AssertionError("preflight may not evaluate CIFAR-10")


prepare.Eval = GuardEval
import train


BASE_COMMIT = "67c8e98"
DEVICE = torch.device("cuda")
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


def clone_nested(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested(item) for item in value)
    return copy.deepcopy(value)


def assert_nested_equal(left, right, path="root"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_nested_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_nested_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


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


def optimizer_signature(optimizer, model):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        {
            key: value
            for key, value in group.items()
            if key != "params"
        }
        | {"params": [names[id(parameter)] for parameter in group["params"]]}
        for group in optimizer.param_groups
    ]


def make_model(module):
    return module.WideResNet(
        module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES
    ).to(DEVICE)


def populate_optimizer(model, optimizer):
    generator = torch.Generator(device=DEVICE).manual_seed(33_033)
    inputs = torch.randn(8, 3, 32, 32, generator=generator, device=DEVICE)
    targets = torch.arange(8, device=DEVICE) % train.NUM_CLASSES
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs), targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()


def source_and_construction_checks(accepted):
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_COMMIT, "--"], cwd=ROOT, text=True
    ).splitlines()
    assert changed == ["train.py"], changed
    subprocess.run(
        ["git", "diff", "--exit-code", BASE_COMMIT, "--", "prepare.py"],
        cwd=ROOT,
        check=True,
    )
    assert train.AVERAGE_FRACTIONS == (0.95, 0.975)
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
        "MIXUP_ALPHA",
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
    ):
        assert inspect.getsource(getattr(train, name)) in accepted.__source__, name

    torch.empty(1, device=DEVICE)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    accepted_model = make_model(accepted)
    accepted_cpu = torch.random.get_rng_state().clone()
    accepted_cuda = torch.cuda.get_rng_state().clone()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate_model = make_model(train)
    assert torch.equal(torch.random.get_rng_state(), accepted_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), accepted_cuda)
    assert_nested_equal(
        accepted_model.state_dict(), candidate_model.state_dict(), "initial_model"
    )
    assert sum(p.numel() for p in candidate_model.parameters()) == 987_098
    assert all(p.dtype == torch.float32 for p in candidate_model.parameters())

    accepted_optimizer = optimizer_for(accepted, accepted_model)
    candidate_optimizer = optimizer_for(train, candidate_model)
    assert optimizer_signature(
        accepted_optimizer, accepted_model
    ) == optimizer_signature(candidate_optimizer, candidate_model)
    assert_nested_equal(
        accepted_optimizer.state_dict(),
        candidate_optimizer.state_dict(),
        "initial_optimizer",
    )
    return candidate_model, candidate_optimizer


def threshold_checks():
    cases = (
        (0, 284.999, None),
        (0, 285.0, 0.95),
        (0, 292.5, 0.95),
        (1, 292.499, None),
        (1, 292.5, 0.975),
        (1, 299.999, 0.975),
        (2, 300.0, None),
    )
    for count, training_time, expected in cases:
        actual = train.due_snapshot_fraction(count, training_time)
        assert actual == expected, (count, training_time, actual, expected)

    captured = []
    for step, pre_time in enumerate((284.99, 285.0, 285.01, 292.49, 292.5, 292.51)):
        fraction = train.due_snapshot_fraction(len(captured), pre_time)
        if fraction is not None:
            captured.append((fraction, step + 1, pre_time))
    assert captured == [(0.95, 2, 285.0), (0.975, 5, 292.5)]
    return captured


class InspectEvaluator:
    def __init__(self, expected, expected_buffers, draw_rng=False, fail=False):
        self.expected = expected
        self.expected_buffers = expected_buffers
        self.draw_rng = draw_rng
        self.fail = fail
        self.calls = 0

    def evaluate(self, model, device):
        self.calls += 1
        for (name, parameter), expected in zip(model.named_parameters(), self.expected):
            assert torch.equal(parameter, expected), name
        for name, buffer in model.named_buffers():
            assert torch.equal(buffer, self.expected_buffers[name]), name
        if self.draw_rng:
            torch.rand(1)
            torch.rand(1, device=device)
        if self.fail:
            raise RuntimeError("injected evaluator failure")
        return 0.123, 94.5


def averaging_checks(model, optimizer):
    populate_optimizer(model, optimizer)
    parameters = list(model.parameters())
    names = [name for name, _ in model.named_parameters()]
    parameter_ids = [id(parameter) for parameter in parameters]
    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    with torch.no_grad():
        for index, parameter in enumerate(parameters):
            parameter.add_((index + 1) * 1e-5)
    first = train.clone_parameters(model)
    with torch.no_grad():
        for index, parameter in enumerate(parameters):
            parameter.add_((index + 1) * 2e-5)
    second = train.clone_parameters(model)
    with torch.no_grad():
        for index, parameter in enumerate(parameters):
            parameter.add_((index + 1) * 3e-5)
    terminal = train.clone_parameters(model)
    snapshots = [first, second]
    expected = [
        ((a + b) + c) / 3.0 for a, b, c in zip(first, second, terminal)
    ]
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    optimizer_state = clone_nested(optimizer.state_dict())
    gradients = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]

    assert len(first) == len(names)
    assert all(not value.requires_grad and value.grad_fn is None for value in first + second)
    assert all(
        value.dtype == torch.float32 and value.device.type == DEVICE.type
        for value in first + second
    )
    for snapshot in snapshots:
        for parameter, value in zip(parameters, snapshot):
            assert parameter.data_ptr() != value.data_ptr()
    assert set(model.state_dict()) == set([*names, *buffers])

    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone()
    evaluator = InspectEvaluator(expected, buffers)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        metrics = train.evaluate_parameter_average(model, snapshots, evaluator, DEVICE)
    assert metrics == (0.123, 94.5) and evaluator.calls == 1
    assert "terminal_restore_exact=true" in output.getvalue()
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_before)
    for name, parameter, value in zip(names, parameters, terminal):
        assert torch.equal(parameter, value), name
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, buffers[name]), name
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids
    assert [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ] == optimizer_parameter_ids
    assert_nested_equal(optimizer.state_dict(), optimizer_state, "optimizer_after_eval")
    for name, parameter, gradient in zip(names, parameters, gradients):
        if gradient is None:
            assert parameter.grad is None, name
        else:
            assert torch.equal(parameter.grad, gradient), name

    torch.manual_seed(33_103)
    torch.cuda.manual_seed(33_103)
    expected_cpu_before = torch.random.get_rng_state().clone()
    expected_cuda_before = torch.cuda.get_rng_state().clone()
    torch.rand(1)
    torch.rand(1, device=DEVICE)
    expected_cpu_after = torch.random.get_rng_state().clone()
    expected_cuda_after = torch.cuda.get_rng_state().clone()
    torch.random.set_rng_state(expected_cpu_before)
    torch.cuda.set_rng_state(expected_cuda_before)
    draw_evaluator = InspectEvaluator(expected, buffers, draw_rng=True)
    with contextlib.redirect_stdout(io.StringIO()):
        train.evaluate_parameter_average(model, snapshots, draw_evaluator, DEVICE)
    assert torch.equal(torch.random.get_rng_state(), expected_cpu_after)
    assert torch.equal(torch.cuda.get_rng_state(), expected_cuda_after)

    failing = InspectEvaluator(expected, buffers, fail=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            train.evaluate_parameter_average(model, snapshots, failing, DEVICE)
    except RuntimeError as error:
        assert str(error) == "injected evaluator failure"
    else:
        raise AssertionError("evaluator exception did not propagate")
    assert failing.calls == 1
    for name, parameter, value in zip(names, parameters, terminal):
        assert torch.equal(parameter, value), name

    bad_snapshots = [[value.clone() for value in first], [value.clone() for value in second]]
    bad_snapshots[0][0].view(-1)[0] = float("nan")
    never = InspectEvaluator(expected, buffers)
    try:
        train.evaluate_parameter_average(model, bad_snapshots, never, DEVICE)
    except RuntimeError as error:
        assert "Non-finite parameter" in str(error)
    else:
        raise AssertionError("non-finite snapshot was accepted")
    assert never.calls == 0
    for name, parameter, value in zip(names, parameters, terminal):
        assert torch.equal(parameter, value), name

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.tensor([1.0, 2.0], device=DEVICE))
            self.second = torch.nn.Parameter(torch.tensor([3.0, 4.0], device=DEVICE))

    toy = Toy()
    toy_terminal = train.clone_parameters(toy)
    toy_first = train.clone_parameters(toy)
    toy_second = train.clone_parameters(toy)
    toy_first[1] = toy_first[1].reshape(2, 1)
    toy_never = InspectEvaluator([], {})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            train.evaluate_parameter_average(
                toy, [toy_first, toy_second], toy_never, DEVICE
            )
    except RuntimeError:
        pass
    else:
        raise AssertionError("partial-install failure was not injected")
    assert toy_never.calls == 0
    for parameter, value in zip(toy.parameters(), toy_terminal):
        assert torch.equal(parameter, value)

    main_source = inspect.getsource(train.main)
    assert main_source.count("evaluate_parameter_average(") == 1
    assert main_source.count("evaluator.evaluate(model, device)") == 1
    assert "if budget_exhausted:" in main_source
    return {"parameter_tensors": len(parameters), "optimizer_states": len(optimizer.state)}


def semantic_checks():
    accepted = load_accepted()
    assert GuardEval.constructions == 2
    model, optimizer = source_and_construction_checks(accepted)
    thresholds = threshold_checks()
    state = averaging_checks(model, optimizer)
    payload = {
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "snapshot_fractions": list(train.AVERAGE_FRACTIONS),
        "threshold_trace": thresholds,
        **state,
        "semantics": "exact",
    }
    print(json.dumps(payload, sort_keys=True))
    print("SEMANTICS PASS")


def population_cv(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else float("inf")


def time_clone_window(parameters, repetitions, candidate):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        if candidate:
            temporary = [parameter.detach().clone() for parameter in parameters]
            torch.cuda.synchronize()
            del temporary
        else:
            for parameter in parameters:
                parameter.detach()
            torch.cuda.synchronize()
    return time.perf_counter() - start


def terminal_sequence(parameters, snapshots):
    terminal = [parameter.detach().clone() for parameter in parameters]
    averaged = [
        ((first + second) + current) / 3.0
        for first, second, current in zip(*snapshots, terminal)
    ]
    assert all(torch.isfinite(value).all() for value in averaged)
    with torch.no_grad():
        for parameter, value in zip(parameters, averaged):
            parameter.copy_(value)
        for parameter, value in zip(parameters, terminal):
            parameter.copy_(value)
    assert all(torch.equal(parameter, value) for parameter, value in zip(parameters, terminal))
    torch.cuda.synchronize()


def time_terminal_window(parameters, snapshots, repetitions):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        terminal_sequence(parameters, snapshots)
    return time.perf_counter() - start


def timing_checks():
    torch.empty(1, device=DEVICE)
    torch.manual_seed(33_200)
    torch.cuda.manual_seed(33_200)
    model = make_model(train)
    parameters = list(model.parameters())
    snapshots = [train.clone_parameters(model), train.clone_parameters(model)]
    for _ in range(20):
        temporary = train.clone_parameters(model)
        torch.cuda.synchronize()
        del temporary
        terminal_sequence(parameters, snapshots)

    repetitions = 200
    order = ["control", "candidate", "candidate", "control", "control", "candidate"]
    values = {"control": [], "candidate": []}
    torch.cuda.reset_peak_memory_stats()
    for arm in order:
        values[arm].append(
            time_clone_window(parameters, repetitions, arm == "candidate")
        )
    paired_differences = [
        values["candidate"][index] - values["control"][index]
        for index in range(3)
    ]

    terminal_repetitions = 50
    terminal_values = [
        time_terminal_window(parameters, snapshots, terminal_repetitions)
        for _ in range(3)
    ]
    per_snapshot_increment = max(
        0.0, statistics.median(paired_differences) / repetitions
    )
    retention = (300.0 - 2.0 * per_snapshot_increment) / 300.0
    projected_passes = 133.00736 * retention
    base_step_s = 300.0 / 25_978
    lost_steps_bound = math.ceil(2.0 * per_snapshot_increment / base_step_s)
    terminal_increment = statistics.median(terminal_values) / terminal_repetitions
    projected_wall = 345.3 + terminal_increment
    cumulative_timer_offset = 2.0 * per_snapshot_increment
    lr_points = [285.0, 292.5, 299.99]
    max_lr_offset = max(
        abs(
            train.learning_rate(value + cumulative_timer_offset)
            - train.learning_rate(value)
        )
        for value in lr_points
    )
    cvs = {
        "candidate": population_cv(values["candidate"]),
        "control": population_cv(values["control"]),
        "terminal": population_cv(terminal_values),
    }
    payload = {
        "clone_order": order,
        "candidate_window_s": values["candidate"],
        "control_window_s": values["control"],
        "paired_difference_s": paired_differences,
        "terminal_window_s": terminal_values,
        "repetitions": repetitions,
        "terminal_repetitions": terminal_repetitions,
        "cv": cvs,
        "per_snapshot_increment_s": per_snapshot_increment,
        "cumulative_timer_offset_s": cumulative_timer_offset,
        "max_lr_offset": max_lr_offset,
        "retention": retention,
        "projected_passes": projected_passes,
        "lost_steps_bound": lost_steps_bound,
        "terminal_increment_s": terminal_increment,
        "projected_wall_s": projected_wall,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }
    print(json.dumps(payload, sort_keys=True))
    assert all(math.isfinite(value) for values_ in values.values() for value in values_)
    assert all(math.isfinite(value) for value in terminal_values)
    assert all(value <= 0.05 for value in cvs.values()), cvs
    assert retention >= 0.99, retention
    assert projected_passes >= 131.6772864, projected_passes
    assert projected_wall < 500.0, projected_wall
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
