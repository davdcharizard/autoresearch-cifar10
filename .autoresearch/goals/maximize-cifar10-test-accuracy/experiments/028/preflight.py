import argparse
import contextlib
import copy
import inspect
import io
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
        raise AssertionError("preflight may not evaluate")


prepare.Eval = GuardEval
import train


DEVICE = torch.device("cuda")
BASE_COMMIT = "67c8e98"
FROZEN_COUNT = 33_424
REMAINING_COUNT = 953_674
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class Flag:
    def __init__(self, value):
        self.value = value


def load_accepted():
    source = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:train.py"], cwd=ROOT, text=True
    )
    module = types.ModuleType("accepted_train")
    module.__file__ = f"git:{BASE_COMMIT}:train.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    assert "freeze_training_prefix" not in source
    return module


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


def model_state(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def prefix_parameters(model):
    return tuple(model.conv1.parameters()) + tuple(model.layer1.parameters())


def prefix_bn_buffers(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith("layer1.")
        and ("running_" in name or "num_batches_tracked" in name)
    }


def run_step(module, model, optimizer, x, y, mixup, rng_state=None):
    if rng_state is not None:
        torch.cuda.set_rng_state(rng_state)
    optimizer.zero_grad(set_to_none=True)
    if mixup:
        distribution = torch.distributions.Beta(
            torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
            torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
        )
        mixed, a, b, coefficient = module.mixup_batch(x, y, distribution)
        outputs = model(mixed)
        loss = coefficient * F.cross_entropy(outputs, a) + (1 - coefficient) * F.cross_entropy(outputs, b)
    else:
        outputs = model(x)
        loss = F.cross_entropy(outputs, y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    return outputs.detach(), loss.detach(), torch.cuda.get_rng_state().clone()


def semantic_checks():
    accepted = load_accepted()
    assert GuardEval.constructions == 2
    constants = (
        "STAGE_BLOCKS",
        "WIDEN_FACTOR",
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
    for name in constants:
        assert getattr(train, name) == getattr(accepted, name), name

    torch.empty(1, device=DEVICE)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    accepted_model = accepted.WideResNet(
        accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR
    ).to(DEVICE).train()
    accepted_cpu_rng = torch.random.get_rng_state().clone()
    accepted_cuda_rng = torch.cuda.get_rng_state().clone()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate_model = train.WideResNet(train.STAGE_BLOCKS, train.WIDEN_FACTOR).to(DEVICE).train()
    assert torch.equal(torch.random.get_rng_state(), accepted_cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), accepted_cuda_rng)
    assert_nested_equal(accepted_model.state_dict(), candidate_model.state_dict(), "initial_model")
    assert sum(p.numel() for p in candidate_model.parameters()) == 987_098

    accepted_optimizer = optimizer_for(accepted, accepted_model)
    candidate_optimizer = optimizer_for(train, candidate_model)
    assert_nested_equal(
        accepted_optimizer.state_dict(), candidate_optimizer.state_dict(), "initial_optimizer"
    )
    x = torch.randn(32, 3, 32, 32, device=DEVICE)
    y = torch.arange(32, device=DEVICE) % train.NUM_CLASSES
    torch.cuda.manual_seed(28028)
    shared_rng = torch.cuda.get_rng_state().clone()
    accepted_outputs, accepted_loss, accepted_after_rng = run_step(
        accepted, accepted_model, accepted_optimizer, x, y, True, shared_rng
    )
    candidate_outputs, candidate_loss, candidate_after_rng = run_step(
        train, candidate_model, candidate_optimizer, x, y, True, shared_rng
    )
    assert torch.equal(accepted_outputs, candidate_outputs)
    assert torch.equal(accepted_loss, candidate_loss)
    assert torch.equal(accepted_after_rng, candidate_after_rng)
    assert_nested_equal(accepted_model.state_dict(), candidate_model.state_dict(), "pre_boundary_model")
    assert_nested_equal(
        accepted_optimizer.state_dict(), candidate_optimizer.state_dict(), "pre_boundary_optimizer"
    )

    before_model = model_state(candidate_model)
    before_optimizer = copy.deepcopy(candidate_optimizer.state_dict())
    before_group_ids = tuple(
        id(p) for group in candidate_optimizer.param_groups for p in group["params"]
    )
    before_bn = prefix_bn_buffers(candidate_model)
    flag = Flag(1)
    for exhausted, budget, seconds in (
        (True, False, 194.9),
        (False, False, 196.7),
        (True, True, 300.0),
    ):
        active = train.maybe_finish_early_phase(
            candidate_model,
            candidate_optimizer,
            flag,
            True,
            exhausted,
            budget,
            seconds,
            86,
            16_770,
        )
        assert active and flag.value == 1
        assert_nested_equal(before_model, candidate_model.state_dict(), "ineligible_model")

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        active = train.maybe_finish_early_phase(
            candidate_model,
            candidate_optimizer,
            flag,
            True,
            True,
            False,
            196.7,
            86,
            16_770,
        )
    assert not active and flag.value == 0
    text = output.getvalue()
    assert text.count("RandAugment disabled") == 1
    assert text.count("Prefix frozen") == 1
    assert "frozen_params=33424 remaining_trainable_params=953674" in text
    with contextlib.redirect_stdout(output):
        active = train.maybe_finish_early_phase(
            candidate_model,
            candidate_optimizer,
            flag,
            active,
            True,
            False,
            200.0,
            87,
            16_965,
        )
    assert not active and output.getvalue().count("Prefix frozen") == 1

    after_group_ids = tuple(
        id(p) for group in candidate_optimizer.param_groups for p in group["params"]
    )
    assert before_group_ids == after_group_ids
    for name, tensor in before_model.items():
        assert torch.equal(tensor, candidate_model.state_dict()[name]), name
    assert_nested_equal(before_optimizer, candidate_optimizer.state_dict(), "freeze_optimizer")
    prefix = prefix_parameters(candidate_model)
    assert sum(p.numel() for p in prefix) == FROZEN_COUNT
    assert all(not p.requires_grad and p.grad is None for p in prefix)
    assert sum(p.numel() for p in candidate_model.parameters() if p.requires_grad) == REMAINING_COUNT

    accepted_optimizer.zero_grad(set_to_none=True)
    candidate_optimizer.zero_grad(set_to_none=True)
    captured = []
    hook = candidate_model.layer1.register_forward_hook(
        lambda module, inputs, result: captured.append(result)
    )
    accepted_tail_outputs = accepted_model(x)
    candidate_tail_outputs = candidate_model(x)
    hook.remove()
    assert torch.equal(accepted_tail_outputs, candidate_tail_outputs)
    assert captured and not captured[0].requires_grad
    accepted_tail_loss = F.cross_entropy(accepted_tail_outputs, y)
    candidate_tail_loss = F.cross_entropy(candidate_tail_outputs, y)
    assert torch.equal(accepted_tail_loss, candidate_tail_loss)
    accepted_tail_loss.backward()
    candidate_tail_loss.backward()
    assert all(p.grad is None for p in prefix)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in candidate_model.parameters()
        if p.requires_grad
    )
    assert_nested_equal(prefix_bn_buffers(accepted_model), prefix_bn_buffers(candidate_model), "first_tail_bn")

    frozen_values = [p.detach().clone() for p in prefix]
    frozen_momentum = {
        id(p): candidate_optimizer.state[p]["momentum_buffer"].detach().clone()
        for p in prefix
    }
    candidate_optimizer.step()
    after_first_upper = [
        p.detach().clone() for p in candidate_model.parameters() if p.requires_grad
    ]
    for _ in range(3):
        run_step(train, candidate_model, candidate_optimizer, x, y, False)
    assert all(torch.equal(value, p) for value, p in zip(frozen_values, prefix))
    assert all(
        torch.equal(frozen_momentum[id(p)], candidate_optimizer.state[p]["momentum_buffer"])
        for p in prefix
    )
    assert any(
        not torch.equal(value, p)
        for value, p in zip(
            after_first_upper,
            (p for p in candidate_model.parameters() if p.requires_grad),
        )
    )
    after_bn = prefix_bn_buffers(candidate_model)
    assert any(not torch.equal(before_bn[name], after_bn[name]) for name in before_bn)

    main_source = inspect.getsource(train.main)
    controller_source = inspect.getsource(train.maybe_finish_early_phase)
    assert main_source.count("maybe_finish_early_phase(") == 1
    assert "iterator_exhausted = True" in main_source
    assert "iterator_exhausted = False" in main_source
    assert "not iterator_exhausted" in controller_source
    assert "or budget_exhausted" in controller_source
    assert controller_source.count("freeze_training_prefix(") == 1
    print("accepted_oracle=67c8e98 frozen=33424 remaining=953674 controller=pass")
    print("SEMANTICS PASS")


def timed_step(model, optimizer, host_x, host_y):
    start = time.perf_counter()
    x = host_x.to(DEVICE, non_blocking=True)
    y = host_y.to(DEVICE, non_blocking=True)
    for group in optimizer.param_groups:
        group["lr"] = train.MIN_LR
    optimizer.zero_grad(set_to_none=True)
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start)


def throughput_checks():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    accepted_model = train.WideResNet(train.STAGE_BLOCKS, train.WIDEN_FACTOR).to(DEVICE).train()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    frozen_model = train.WideResNet(train.STAGE_BLOCKS, train.WIDEN_FACTOR).to(DEVICE).train()
    accepted_optimizer = optimizer_for(train, accepted_model)
    frozen_optimizer = optimizer_for(train, frozen_model)
    host_x = torch.randn(train.BATCH_SIZE, 3, 32, 32, pin_memory=True)
    host_y = (torch.arange(train.BATCH_SIZE) % train.NUM_CLASSES).pin_memory()
    x = host_x.to(DEVICE)
    y = host_y.to(DEVICE)
    for _ in range(3):
        run_step(train, accepted_model, accepted_optimizer, x, y, False)
        run_step(train, frozen_model, frozen_optimizer, x, y, False)
    assert_nested_equal(accepted_model.state_dict(), frozen_model.state_dict(), "timing_boundary_model")
    assert_nested_equal(
        accepted_optimizer.state_dict(), frozen_optimizer.state_dict(), "timing_boundary_optimizer"
    )
    train.freeze_training_prefix(frozen_model, frozen_optimizer)

    models = (accepted_model, frozen_model)
    optimizers = (accepted_optimizer, frozen_optimizer)
    for index in range(2):
        for _ in range(25):
            timed_step(models[index], optimizers[index], host_x, host_y)
    windows = [[], []]
    for window in range(3):
        order = (0, 1) if window % 2 == 0 else (1, 0)
        for index in order:
            values = [
                timed_step(models[index], optimizers[index], host_x, host_y)
                for _ in range(50)
            ]
            windows[index].append(statistics.mean(values))
    medians = [statistics.median(values) for values in windows]
    cvs = [statistics.pstdev(values) / statistics.mean(values) for values in windows]
    assert all(cv <= 0.05 for cv in cvs), (cvs, windows)
    speed_ratio = medians[0] / medians[1]
    projected_steps = 16_770 + (25_978 - 16_770) * speed_ratio
    projected_passes = projected_steps * train.BATCH_SIZE / 50_000
    print(f"windows={windows} medians_ms={medians} cvs={cvs}")
    print(
        f"speed_ratio={speed_ratio:.6f} projected_steps={projected_steps:.3f} "
        f"projected_passes={projected_passes:.6f}"
    )
    assert projected_passes >= 145.0
    print("THROUGHPUT PASS")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--semantics", action="store_true")
    group.add_argument("--throughput", action="store_true")
    args = parser.parse_args()
    if args.semantics:
        semantic_checks()
    else:
        throughput_checks()


if __name__ == "__main__":
    main()
