import argparse
import copy
import inspect
import json
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
CL = torch.channels_last
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


def build_model(module, candidate, training):
    model = module.WideResNet(
        module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES
    )
    if candidate:
        model = model.to(device=DEVICE, memory_format=CL)
    else:
        model = model.to(DEVICE)
    return model.train(training)


def logical(tensor):
    return tensor.detach().cpu().contiguous()


def assert_logical_equal(left, right, path):
    assert left.keys() == right.keys(), path
    for name in left:
        assert torch.equal(logical(left[name]), logical(right[name])), f"{path}.{name}"


def optimizer_signature(optimizer, model):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                key: value
                for key, value in group.items()
                if key != "params"
            }
            | {"params": [names[id(parameter)] for parameter in group["params"]]}
        )
    return groups


def relative_l2(candidate, accepted):
    delta = (logical(candidate).double() - logical(accepted).double()).norm()
    denominator = logical(accepted).double().norm()
    if denominator == 0:
        assert logical(candidate).abs().max().item() <= 1e-6
        return 0.0
    return (delta / denominator).item()


def fixed_host_batch(batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(
        batch_size, 3, 32, 32, generator=generator, dtype=torch.float32
    ).pin_memory()
    targets = (torch.arange(batch_size) % train.NUM_CLASSES).pin_memory()
    return inputs, targets


def transfer(host_inputs, host_targets, candidate):
    if candidate:
        inputs = host_inputs.to(
            DEVICE, non_blocking=True, memory_format=CL
        )
    else:
        inputs = host_inputs.to(DEVICE, non_blocking=True)
    targets = host_targets.to(DEVICE, non_blocking=True)
    torch.cuda.synchronize()
    return inputs, targets


def hard_step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer.step()
    return outputs.detach(), loss.detach(), gradients


def clone_state(state):
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def replay_payload(model, optimizer, inputs, targets, snapshot):
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    model.train(snapshot["training"])
    optimizer.zero_grad(set_to_none=True)
    torch.random.set_rng_state(snapshot["cpu_rng"])
    torch.cuda.set_rng_state(snapshot["cuda_rng"])
    outputs, loss, gradients = hard_step(model, optimizer, inputs, targets)
    torch.cuda.synchronize()
    return {
        "outputs": outputs,
        "loss": loss,
        "gradients": gradients,
        "model": clone_state(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cpu_rng": torch.random.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state().clone(),
    }


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


def semantic_checks():
    accepted = load_accepted()
    assert GuardEval.constructions == 2
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
    assert inspect.getsource(train.EarlyRandAugment) in accepted.__source__
    assert inspect.getsource(train.make_train_transform) in accepted.__source__
    assert inspect.getsource(train.mixup_batch) in accepted.__source__

    torch.empty(1, device=DEVICE)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    accepted_model = build_model(accepted, False, True)
    accepted_cpu = torch.random.get_rng_state().clone()
    accepted_cuda = torch.cuda.get_rng_state().clone()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate_model = build_model(train, True, True)
    assert torch.equal(torch.random.get_rng_state(), accepted_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), accepted_cuda)
    assert_logical_equal(
        accepted_model.state_dict(), candidate_model.state_dict(), "initial_model"
    )
    assert sum(parameter.numel() for parameter in candidate_model.parameters()) == 987_098
    for module in candidate_model.modules():
        if isinstance(module, nn.Conv2d):
            assert module.weight.dtype == torch.float32
            assert module.weight.is_contiguous(memory_format=CL)
    for tensor in candidate_model.state_dict().values():
        if tensor.is_floating_point():
            assert tensor.dtype == torch.float32

    accepted_optimizer = optimizer_for(accepted, accepted_model)
    candidate_optimizer = optimizer_for(train, candidate_model)
    assert optimizer_signature(accepted_optimizer, accepted_model) == optimizer_signature(
        candidate_optimizer, candidate_model
    )

    host_inputs, host_targets = fixed_host_batch(train.BATCH_SIZE, 31_031)
    accepted_inputs, targets = transfer(host_inputs, host_targets, False)
    candidate_inputs, candidate_targets = transfer(host_inputs, host_targets, True)
    assert torch.equal(logical(accepted_inputs), logical(candidate_inputs))
    assert torch.equal(targets, candidate_targets)
    assert candidate_inputs.is_contiguous(memory_format=CL)
    assert candidate_inputs.dtype == torch.float32

    distribution_a = torch.distributions.Beta(
        torch.tensor(accepted.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(accepted.MIXUP_ALPHA, device=DEVICE),
    )
    distribution_c = torch.distributions.Beta(
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
    )
    torch.cuda.manual_seed(31_131)
    shared_rng = torch.cuda.get_rng_state().clone()
    mixed_a, target_a, paired_a, coefficient_a = accepted.mixup_batch(
        accepted_inputs, targets, distribution_a
    )
    accepted_after = torch.cuda.get_rng_state().clone()
    torch.cuda.set_rng_state(shared_rng)
    mixed_c, target_c, paired_c, coefficient_c = train.mixup_batch(
        candidate_inputs, candidate_targets, distribution_c
    )
    assert torch.equal(coefficient_a, coefficient_c)
    assert torch.equal(target_a, target_c) and torch.equal(paired_a, paired_c)
    assert torch.equal(logical(mixed_a), logical(mixed_c))
    assert torch.equal(torch.cuda.get_rng_state(), accepted_after)
    assert mixed_c.is_contiguous(memory_format=CL)

    torch.manual_seed(31_180)
    torch.cuda.manual_seed(31_180)
    activation_model = build_model(train, True, True)
    activation_layouts = []

    def layout_hook(module, args, output):
        tensors = list(args)
        if isinstance(output, torch.Tensor):
            tensors.append(output)
        elif isinstance(output, (tuple, list)):
            tensors.extend(item for item in output if isinstance(item, torch.Tensor))
        for tensor in tensors:
            if tensor.ndim == 4 and tensor.is_floating_point():
                activation_layouts.append(
                    (module.__class__.__name__, tensor.is_contiguous(memory_format=CL), tensor.dtype)
                )

    hooks = [module.register_forward_hook(layout_hook) for module in activation_model.modules()]
    with torch.no_grad():
        activation_model(candidate_inputs)
    for hook in hooks:
        hook.remove()
    assert activation_layouts
    assert all(is_layout and dtype == torch.float32 for _, is_layout, dtype in activation_layouts)

    eval_metrics = {}
    for batch_size in (256, 16):
        host_eval, host_y = fixed_host_batch(batch_size, 31_200 + batch_size)
        nchw = host_eval.to(DEVICE)
        channels_last = nchw.contiguous(memory_format=CL)
        accepted_model.eval()
        candidate_model.eval()
        candidate_before = clone_state(candidate_model.state_dict())
        with torch.inference_mode():
            accepted_logits = accepted_model(nchw)
            candidate_nchw_logits = candidate_model(nchw)
            candidate_cl_logits = candidate_model(channels_last)
        assert torch.equal(candidate_nchw_logits, candidate_cl_logits)
        assert_logical_equal(candidate_before, candidate_model.state_dict(), f"eval_state_{batch_size}")
        torch.testing.assert_close(
            candidate_nchw_logits, accepted_logits, rtol=2e-4, atol=2e-5
        )
        labels = host_y.to(DEVICE)
        accepted_loss = F.cross_entropy(accepted_logits, labels)
        candidate_loss = F.cross_entropy(candidate_nchw_logits, labels)
        loss_delta = abs(candidate_loss.item() - accepted_loss.item())
        assert loss_delta <= 2e-5
        eval_metrics[str(batch_size)] = {
            "max_logit_abs": (candidate_nchw_logits - accepted_logits).abs().max().item(),
            "loss_delta": loss_delta,
            "argmax_agreement": float(
                (candidate_nchw_logits.argmax(1) == accepted_logits.argmax(1)).float().mean()
            ),
        }

    accepted_model, candidate_model = None, None
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    accepted_model = build_model(accepted, False, True)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate_model = build_model(train, True, True)
    accepted_optimizer = optimizer_for(accepted, accepted_model)
    candidate_optimizer = optimizer_for(train, candidate_model)
    before_a = clone_state(accepted_model.state_dict())
    before_c = clone_state(candidate_model.state_dict())
    outputs_a, loss_a, gradients_a = hard_step(
        accepted_model, accepted_optimizer, accepted_inputs, targets
    )
    outputs_c, loss_c, gradients_c = hard_step(
        candidate_model, candidate_optimizer, candidate_inputs, targets
    )
    torch.testing.assert_close(outputs_c, outputs_a, rtol=2e-4, atol=2e-5)
    assert abs(loss_c.item() - loss_a.item()) <= 2e-5
    gradient_errors = {}
    update_errors = {}
    for name in gradients_a:
        assert torch.isfinite(gradients_a[name]).all()
        assert torch.isfinite(gradients_c[name]).all()
        gradient_errors[name] = relative_l2(gradients_c[name], gradients_a[name])
        update_a = before_a[name] - accepted_model.state_dict()[name]
        update_c = before_c[name] - candidate_model.state_dict()[name]
        update_errors[name] = relative_l2(update_c, update_a)
        assert gradient_errors[name] <= 1e-3, (name, gradient_errors[name])
        assert update_errors[name] <= 1e-3, (name, update_errors[name])
    for name, accepted_tensor in accepted_model.state_dict().items():
        if "running_" in name:
            torch.testing.assert_close(
                logical(candidate_model.state_dict()[name]),
                logical(accepted_tensor),
                rtol=2e-4,
                atol=2e-5,
            )
        elif "num_batches_tracked" in name:
            assert torch.equal(
                logical(candidate_model.state_dict()[name]), logical(accepted_tensor)
            )
    for optimizer, model in (
        (accepted_optimizer, accepted_model),
        (candidate_optimizer, candidate_model),
    ):
        assert len(optimizer.state) == len(list(model.parameters()))
        for parameter, state in optimizer.state.items():
            buffer = state["momentum_buffer"]
            assert buffer.dtype == torch.float32 and buffer.shape == parameter.shape
            if parameter.ndim == 4 and optimizer is candidate_optimizer:
                assert buffer.is_contiguous(memory_format=CL)

    replay_model = build_model(train, True, True)
    replay_optimizer = optimizer_for(train, replay_model)
    replay_inputs = candidate_inputs.detach().clone(memory_format=CL)
    for _ in range(3):
        hard_step(replay_model, replay_optimizer, replay_inputs, targets)
    torch.cuda.synchronize()
    replay_optimizer.zero_grad(set_to_none=True)
    snapshot = {
        "model": clone_state(replay_model.state_dict()),
        "optimizer": copy.deepcopy(replay_optimizer.state_dict()),
        "cpu_rng": torch.random.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state().clone(),
        "training": replay_model.training,
    }
    replay_one = replay_payload(
        replay_model, replay_optimizer, replay_inputs, targets, snapshot
    )
    replay_two = replay_payload(
        replay_model, replay_optimizer, replay_inputs, targets, snapshot
    )
    assert_nested_equal(replay_one, replay_two, "candidate_replay")

    diff = subprocess.check_output(
        ["git", "diff", "--unified=0", BASE_COMMIT, "--", "train.py"],
        cwd=ROOT,
        text=True,
    )
    assert "memory_format=torch.channels_last" in diff
    assert "Channels-last audit" in diff
    assert "autocast" not in diff and "compile" not in diff and "bfloat16" not in diff
    assert "randaugment_active.value = 0" not in diff
    assert "MIXUP_ALPHA" not in diff and "BATCH_SIZE" not in diff and "LR =" not in diff
    print(
        json.dumps(
            {
                "activation_observations": len(activation_layouts),
                "eval": eval_metrics,
                "max_gradient_relative_l2": max(gradient_errors.values()),
                "max_update_relative_l2": max(update_errors.values()),
                "params": 987098,
                "replay": "bitwise",
            },
            sort_keys=True,
        )
    )
    print("SEMANTICS PASS")


def timed_step(module, candidate, model, optimizer, host_x, host_y, distribution, mixup):
    started = time.perf_counter()
    if candidate:
        inputs = host_x.to(DEVICE, non_blocking=True, memory_format=CL)
    else:
        inputs = host_x.to(DEVICE, non_blocking=True)
    targets = host_y.to(DEVICE, non_blocking=True)
    for group in optimizer.param_groups:
        group["lr"] = module.MIN_LR
    optimizer.zero_grad(set_to_none=True)
    if mixup:
        mixed, a, b, coefficient = module.mixup_batch(
            inputs, targets, distribution
        )
        outputs = model(mixed)
        loss = coefficient * F.cross_entropy(outputs, a) + (
            1.0 - coefficient
        ) * F.cross_entropy(outputs, b)
    else:
        loss = F.cross_entropy(model(inputs), targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - started)


def timing_window(module, candidate, mixup, replicate):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = build_model(module, candidate, True)
    optimizer = optimizer_for(module, model)
    host_x, host_y = fixed_host_batch(module.BATCH_SIZE, 31_500 + replicate)
    distribution = torch.distributions.Beta(
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
    )
    torch.cuda.manual_seed(31_600 + replicate)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(25):
        timed_step(
            module, candidate, model, optimizer, host_x, host_y, distribution, mixup
        )
    values = [
        timed_step(
            module, candidate, model, optimizer, host_x, host_y, distribution, mixup
        )
        for _ in range(50)
    ]
    return statistics.mean(values), torch.cuda.max_memory_allocated() / 1024 / 1024


def throughput_checks():
    accepted = load_accepted()
    results = {
        regime: {"accepted": [], "candidate": []}
        for regime in ("early", "hard")
    }
    peaks = {"accepted": [], "candidate": []}
    for replicate in range(3):
        order = (
            ("accepted", "candidate")
            if replicate != 1
            else ("candidate", "accepted")
        )
        for kind in order:
            module = accepted if kind == "accepted" else train
            candidate = kind == "candidate"
            for mixup, regime in ((True, "early"), (False, "hard")):
                mean_ms, peak_mb = timing_window(
                    module, candidate, mixup, replicate
                )
                results[regime][kind].append(mean_ms)
                peaks[kind].append(peak_mb)

    cvs = {
        regime: {
            kind: statistics.pstdev(values) / statistics.mean(values)
            for kind, values in arms.items()
        }
        for regime, arms in results.items()
    }
    speedups = []
    for replicate in range(3):
        accepted_rate = (
            0.65 / results["early"]["accepted"][replicate]
            + 0.35 / results["hard"]["accepted"][replicate]
        )
        candidate_rate = (
            0.65 / results["early"]["candidate"][replicate]
            + 0.35 / results["hard"]["candidate"][replicate]
        )
        speedups.append(candidate_rate / accepted_rate)
    median_speedup = statistics.median(speedups)
    projected_passes = 133.00736 * median_speedup
    payload = {
        "results_ms": results,
        "cvs": cvs,
        "paired_speedups": speedups,
        "median_speedup": median_speedup,
        "projected_passes": projected_passes,
        "peak_vram_mb": {kind: max(values) for kind, values in peaks.items()},
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    assert all(cv <= 0.02 for regime in cvs.values() for cv in regime.values()), payload
    assert all(speedup >= 1.02 for speedup in speedups), payload
    assert projected_passes >= 135.667507, payload
    print("THROUGHPUT PASS")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--semantics", action="store_true")
    group.add_argument("--throughput", action="store_true")
    args = parser.parse_args()
    semantic_checks() if args.semantics else throughput_checks()


if __name__ == "__main__":
    main()
