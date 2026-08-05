import argparse
import contextlib
import copy
import gc
import hashlib
import inspect
import io
import json
import multiprocessing
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


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
MP_CONTEXT = multiprocessing.get_context()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class MarkerDataset(Dataset):
    def __init__(self, active, size=128):
        self.active = active
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index, int(self.active.value)


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


def configure_drop_path(model, probability=train.DROP_PATH_P, seed=train.DROP_PATH_SEED):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    model.layer3[2].drop_path_probability = probability
    model.layer3[2].drop_path_generator = generator
    return generator


def make_model_pair(accepted, training=True):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    accepted_model = accepted.WideResNet(
        accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES
    ).to(DEVICE)
    accepted_cpu = torch.random.get_rng_state().clone()
    accepted_cuda = torch.cuda.get_rng_state().clone()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate_model = train.WideResNet(
        train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES
    ).to(DEVICE)
    assert torch.equal(torch.random.get_rng_state(), accepted_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), accepted_cuda)
    assert_nested_equal(
        accepted_model.state_dict(), candidate_model.state_dict(), "initial_model"
    )
    accepted_model.train(training)
    candidate_model.train(training)
    return accepted_model, candidate_model


def complete_step(module, model, optimizer, x, y, mixup, rng_state=None):
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
        loss = coefficient * F.cross_entropy(outputs, a) + (
            1.0 - coefficient
        ) * F.cross_entropy(outputs, b)
    else:
        outputs = model(x)
        loss = F.cross_entropy(outputs, y)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer.step()
    return (
        outputs.detach(),
        loss.detach(),
        gradients,
        torch.random.get_rng_state().clone(),
        torch.cuda.get_rng_state().clone(),
    )


def shutdown_loader(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
    del loader
    gc.collect()


def marker_cutoff_check():
    active = MP_CONTEXT.Value("b", 1, lock=False)
    loader = DataLoader(
        MarkerDataset(active),
        batch_size=16,
        shuffle=False,
        num_workers=prepare.NUM_WORKERS,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context=MP_CONTEXT,
    )
    first = [markers for _, markers in loader]
    assert first and all(torch.all(markers == 1) for markers in first)
    active.value = 0
    second = [markers for _, markers in loader]
    assert second and all(torch.all(markers == 0) for markers in second)
    shutdown_loader(loader)


def trace_epochs(module, with_randaugment):
    torch.manual_seed(42)
    active = MP_CONTEXT.Value("b", 1, lock=False)
    dataset = datasets.CIFAR10(
        prepare.DATASET_DIR,
        train=True,
        download=False,
        transform=module.make_train_transform(active if with_randaugment else None),
    )
    loader = DataLoader(
        dataset,
        batch_size=module.BATCH_SIZE,
        shuffle=True,
        num_workers=prepare.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context=MP_CONTEXT,
    )
    module.WideResNet(module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES)
    active_hashes = []
    for index, (inputs, _) in enumerate(loader):
        if index < 4:
            active_hashes.append(hashlib.sha256(inputs.numpy().tobytes()).hexdigest())
    active.value = 0
    tail_hashes = []
    for index, (inputs, targets) in enumerate(loader):
        tail_hashes.append(
            (
                hashlib.sha256(inputs.numpy().tobytes()).hexdigest(),
                hashlib.sha256(targets.numpy().tobytes()).hexdigest(),
            )
        )
        if index == 15:
            break
    shutdown_loader(loader)
    return active_hashes, tail_hashes


def semantic_checks():
    accepted = load_accepted()
    assert GuardEval.constructions == 2
    assert MP_CONTEXT.get_start_method() == "forkserver"
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
    assert train.DROP_PATH_P == 0.05 and train.DROP_PATH_SEED == 28028
    assert inspect.getsource(train.EarlyRandAugment) in accepted.__source__
    assert inspect.getsource(train.make_train_transform) in accepted.__source__

    torch.empty(1, device=DEVICE)
    accepted_model, candidate_model = make_model_pair(accepted)
    assert sum(p.numel() for p in candidate_model.parameters()) == 987_098
    assert all(block.drop_path_probability == 0.0 for block in candidate_model.modules() if isinstance(block, train.PreActBlock))
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone()
    generator = configure_drop_path(candidate_model)
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_before)
    assert set(candidate_model.state_dict()) == set(accepted_model.state_dict())
    targeted = [
        name
        for name, block in candidate_model.named_modules()
        if isinstance(block, train.PreActBlock) and block.drop_path_probability != 0.0
    ]
    assert targeted == ["layer3.2"]
    assert all("drop_path" not in name for name in candidate_model.state_dict())

    ones = torch.ones(4000, 3, 2, 2, device=DEVICE)
    global_before = torch.cuda.get_rng_state().clone()
    direct_generator = torch.Generator(device=DEVICE).manual_seed(28028)
    direct_before = direct_generator.get_state().clone()
    masked = train.apply_drop_path(ones, 0.05, direct_generator, True)
    direct_after = direct_generator.get_state().clone()
    assert masked.shape == ones.shape and not torch.equal(direct_before, direct_after)
    assert torch.equal(torch.cuda.get_rng_state(), global_before)
    sample = masked[:, 0, 0, 0]
    drop_rate = float((sample == 0).float().mean())
    assert 0.04 <= drop_rate <= 0.06
    expected_scale = torch.tensor(1.0 / 0.95, device=DEVICE)
    assert torch.all((sample == 0) | (sample == expected_scale))
    replay_a = torch.Generator(device=DEVICE).manual_seed(28028)
    replay_b = torch.Generator(device=DEVICE).manual_seed(28028)
    assert torch.equal(
        train.apply_drop_path(ones, 0.05, replay_a, True),
        train.apply_drop_path(ones, 0.05, replay_b, True),
    )

    x_generator = torch.Generator(device=DEVICE).manual_seed(30030)
    x = torch.randn(16, 3, 32, 32, device=DEVICE, generator=x_generator)
    y = torch.arange(16, device=DEVICE) % train.NUM_CLASSES

    candidate_model.layer3[2].drop_path_probability = 0.0
    accepted_optimizer = optimizer_for(accepted, accepted_model)
    candidate_optimizer = optimizer_for(train, candidate_model)
    assert_nested_equal(
        accepted_optimizer.state_dict(), candidate_optimizer.state_dict(), "optimizer"
    )
    shared_rng = torch.cuda.get_rng_state().clone()
    accepted_result = complete_step(
        accepted, accepted_model, accepted_optimizer, x, y, True, shared_rng
    )
    private_before = generator.get_state().clone()
    candidate_result = complete_step(
        train, candidate_model, candidate_optimizer, x, y, True, shared_rng
    )
    assert_nested_equal(accepted_result, candidate_result, "p0_step")
    assert torch.equal(generator.get_state(), private_before)
    assert_nested_equal(
        accepted_model.state_dict(), candidate_model.state_dict(), "p0_model"
    )
    assert_nested_equal(
        accepted_optimizer.state_dict(), candidate_optimizer.state_dict(), "p0_optimizer"
    )

    accepted_eval, candidate_eval = make_model_pair(accepted, training=False)
    eval_generator = configure_drop_path(candidate_eval)
    accepted_eval_optimizer = optimizer_for(accepted, accepted_eval)
    candidate_eval_optimizer = optimizer_for(train, candidate_eval)
    private_before = eval_generator.get_state().clone()
    shared_rng = torch.cuda.get_rng_state().clone()
    accepted_result = complete_step(
        accepted, accepted_eval, accepted_eval_optimizer, x, y, False, shared_rng
    )
    candidate_result = complete_step(
        train, candidate_eval, candidate_eval_optimizer, x, y, False, shared_rng
    )
    assert_nested_equal(accepted_result, candidate_result, "eval_step")
    assert torch.equal(eval_generator.get_state(), private_before)
    assert_nested_equal(
        accepted_eval.state_dict(), candidate_eval.state_dict(), "eval_model"
    )

    active_model = train.WideResNet(
        train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES
    ).to(DEVICE).train()
    active_generator = configure_drop_path(active_model)
    active_optimizer = optimizer_for(train, active_model)
    global_before = torch.cuda.get_rng_state().clone()
    private_before = active_generator.get_state().clone()
    result = complete_step(train, active_model, active_optimizer, x, y, False)
    assert torch.equal(result[-1], global_before)
    assert not torch.equal(active_generator.get_state(), private_before)
    assert all(torch.isfinite(gradient).all() for gradient in result[2].values())

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        enabled = train.maybe_disable_drop_path(
            active_model.layer3[2], True, True, 85, 16_500, 194.9, 0.6497, 0.058
        )
    assert enabled and active_model.layer3[2].drop_path_probability == 0.05
    assert output.getvalue() == ""
    with contextlib.redirect_stdout(output):
        enabled = train.maybe_disable_drop_path(
            active_model.layer3[2], enabled, False, 85, 16_501, 195.0, 0.65, 0.058
        )
    assert not enabled and active_model.layer3[2].drop_path_probability == 0.0
    assert output.getvalue().count("Drop-path disabled") == 1
    private_before = active_generator.get_state().clone()
    with contextlib.redirect_stdout(output):
        enabled = train.maybe_disable_drop_path(
            active_model.layer3[2], enabled, False, 85, 16_502, 195.1, 0.6503, 0.058
        )
    active_model(x)
    assert not enabled and output.getvalue().count("Drop-path disabled") == 1
    assert torch.equal(active_generator.get_state(), private_before)

    source = inspect.getsource(train.main)
    use_index = source.index("use_mixup = progress < MIXUP_END_FRACTION")
    cutoff_index = source.index("drop_path_enabled = maybe_disable_drop_path")
    zero_index = source.index("optimizer.zero_grad", cutoff_index)
    forward_index = source.index("outputs = model", cutoff_index)
    assert use_index < cutoff_index < zero_index < forward_index
    assert source.count("drop_path_enabled = maybe_disable_drop_path") == 1
    assert source.count("randaugment_active.value = 0") == 1
    assert "and not budget_exhausted" in source and "iterator_exhausted=true" in source

    marker_cutoff_check()
    base_active, base_tail = trace_epochs(train, False)
    candidate_active, candidate_tail = trace_epochs(train, True)
    assert base_active != candidate_active
    assert base_tail == candidate_tail
    print(
        f"params=987098 target=layer3.2 mask_rate={drop_rate:.6f} "
        "p0_identity=pass eval_identity=pass private_rng=pass global_rng=pass "
        "controller=pass worker_tail_replay=pass"
    )
    print("SEMANTICS PASS")


def timed_step(module, model, optimizer, host_x, host_y, distribution, mixup):
    started = time.perf_counter()
    x = host_x.to(DEVICE, non_blocking=True)
    y = host_y.to(DEVICE, non_blocking=True)
    for group in optimizer.param_groups:
        group["lr"] = module.MIN_LR
    optimizer.zero_grad(set_to_none=True)
    if mixup:
        mixed, a, b, coefficient = module.mixup_batch(x, y, distribution)
        outputs = model(mixed)
        loss = coefficient * F.cross_entropy(outputs, a) + (
            1.0 - coefficient
        ) * F.cross_entropy(outputs, b)
    else:
        loss = F.cross_entropy(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - started)


def timing_window(module, candidate, mixup, replicate):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = module.WideResNet(
        module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES
    ).to(DEVICE).train()
    if candidate:
        configure_drop_path(model, train.DROP_PATH_P if mixup else 0.0)
    optimizer = optimizer_for(module, model)
    fixture_generator = torch.Generator().manual_seed(30_030 + replicate)
    host_x = torch.randn(
        module.BATCH_SIZE, 3, 32, 32, generator=fixture_generator
    ).pin_memory()
    host_y = (torch.arange(module.BATCH_SIZE) % module.NUM_CLASSES).pin_memory()
    distribution = torch.distributions.Beta(
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
    )
    torch.cuda.manual_seed(30_130 + replicate)
    for _ in range(20):
        timed_step(module, model, optimizer, host_x, host_y, distribution, mixup)
    values = [
        timed_step(module, model, optimizer, host_x, host_y, distribution, mixup)
        for _ in range(50)
    ]
    return statistics.mean(values)


def throughput_checks():
    accepted = load_accepted()
    results = {}
    for mixup, regime in ((True, "early"), (False, "hard")):
        windows = {"accepted": [], "candidate": []}
        for replicate in range(3):
            order = (
                ("accepted", "candidate")
                if replicate % 2 == 0
                else ("candidate", "accepted")
            )
            for kind in order:
                module = accepted if kind == "accepted" else train
                windows[kind].append(
                    timing_window(module, kind == "candidate", mixup, replicate)
                )
        medians = {
            kind: statistics.median(values) for kind, values in windows.items()
        }
        cvs = {
            kind: statistics.pstdev(values) / statistics.mean(values)
            for kind, values in windows.items()
        }
        results[regime] = {
            "windows_ms": windows,
            "medians_ms": medians,
            "cvs": cvs,
        }

    accepted_rate = (
        0.65 / results["early"]["medians_ms"]["accepted"]
        + 0.35 / results["hard"]["medians_ms"]["accepted"]
    )
    candidate_rate = (
        0.65 / results["early"]["medians_ms"]["candidate"]
        + 0.35 / results["hard"]["medians_ms"]["candidate"]
    )
    retention = candidate_rate / accepted_rate
    projected_passes = 133.00736 * retention
    payload = {
        "results": results,
        "retention": retention,
        "projected_passes": projected_passes,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    assert all(
        cv <= 0.05
        for regime in results.values()
        for cv in regime["cvs"].values()
    ), payload
    assert retention >= 0.9774, payload
    assert projected_passes >= 130.0, payload
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
