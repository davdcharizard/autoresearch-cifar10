import argparse
import gc
import hashlib
import inspect
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


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
MP_CONTEXT = multiprocessing.get_context()
BASE_COMMIT = "67c8e98"
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


def trace_batch128_epochs(with_randaugment):
    torch.manual_seed(42)
    active = MP_CONTEXT.Value("b", 1, lock=False)
    transform = train.make_train_transform(active if with_randaugment else None)
    dataset = datasets.CIFAR10(
        prepare.DATASET_DIR, train=True, download=False, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=prepare.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context=MP_CONTEXT,
    )
    train.WideResNet(train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES)
    active_hashes = []
    for index, (inputs, targets) in enumerate(loader):
        if index < 4:
            active_hashes.append(hashlib.sha256(inputs.numpy().tobytes()).hexdigest())
    active.value = 0
    tail = []
    for index, (inputs, targets) in enumerate(loader):
        tail.append(
            (
                hashlib.sha256(inputs.numpy().tobytes()).hexdigest(),
                hashlib.sha256(targets.numpy().tobytes()).hexdigest(),
            )
        )
        if index == 15:
            break
    shutdown_loader(loader)
    return active_hashes, tail


def semantic_checks():
    accepted = load_accepted()
    assert GuardEval.constructions == 2
    assert MP_CONTEXT.get_start_method() == "forkserver"
    assert accepted.BATCH_SIZE == 256 and train.BATCH_SIZE == 128
    assert accepted.LR == 0.2 and train.LR == 0.1
    assert accepted.MIN_LR == 0.002 and train.MIN_LR == 0.001
    assert accepted.MAX_STEPS == 64_000 and train.MAX_STEPS == 128_000
    unchanged = (
        "STAGE_BLOCKS",
        "WIDEN_FACTOR",
        "NUM_CLASSES",
        "WARMUP_FRACTION",
        "MOMENTUM",
        "WEIGHT_DECAY",
        "EVAL_EVERY",
        "MIXUP_ALPHA",
        "MIXUP_END_FRACTION",
        "RANDAUGMENT_END_FRACTION",
    )
    for name in unchanged:
        assert getattr(train, name) == getattr(accepted, name), name

    torch.empty(1, device=DEVICE)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    accepted_active = MP_CONTEXT.Value("b", 1, lock=False)
    accepted_transform = accepted.make_train_transform(accepted_active)
    accepted_model = accepted.WideResNet(
        accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES
    )
    accepted_cpu = torch.random.get_rng_state().clone()
    accepted_cuda = torch.cuda.get_rng_state().clone()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate_active = MP_CONTEXT.Value("b", 1, lock=False)
    candidate_transform = train.make_train_transform(candidate_active)
    candidate_model = train.WideResNet(
        train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES
    )
    assert torch.equal(torch.random.get_rng_state(), accepted_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), accepted_cuda)
    for name, tensor in accepted_model.state_dict().items():
        assert torch.equal(tensor, candidate_model.state_dict()[name]), name
    assert sum(p.numel() for p in candidate_model.parameters()) == 987_098

    accepted_optimizer = optimizer_for(accepted, accepted_model)
    candidate_optimizer = optimizer_for(train, candidate_model)
    for optimizer, model in (
        (accepted_optimizer, accepted_model),
        (candidate_optimizer, candidate_model),
    ):
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert [len(group["params"]) for group in accepted_optimizer.param_groups] == [
        len(group["params"]) for group in candidate_optimizer.param_groups
    ]

    for progress in (0.0, 0.025, 0.05, 0.5, 0.65, 1.0):
        accepted_lr = accepted.learning_rate(progress * prepare.TIME_BUDGET_S)
        candidate_lr = train.learning_rate(progress * prepare.TIME_BUDGET_S)
        assert candidate_lr == 0.5 * accepted_lr, progress

    x = torch.zeros(train.BATCH_SIZE, 3, 32, 32, device=DEVICE)
    y = torch.arange(train.BATCH_SIZE, device=DEVICE) % train.NUM_CLASSES
    distribution = torch.distributions.Beta(
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
    )
    mixed, a, b, coefficient = train.mixup_batch(x, y, distribution)
    outputs = candidate_model.to(DEVICE)(mixed)
    assert outputs.shape == (128, 10)
    loss = coefficient * F.cross_entropy(outputs, a) + (1 - coefficient) * F.cross_entropy(outputs, b)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate_model.parameters())
    assert 50_000 // train.BATCH_SIZE == 390
    assert (50_000 // train.BATCH_SIZE) * train.BATCH_SIZE == 49_920

    wrapper = candidate_transform.transforms[2]
    policy = wrapper.transform
    assert tuple(type(op).__name__ for op in candidate_transform.transforms) == tuple(
        type(op).__name__ for op in accepted_transform.transforms
    )
    assert policy.num_ops == 1 and policy.magnitude == 5
    assert policy.num_magnitude_bins == 31
    assert policy.interpolation == transforms.InterpolationMode.BILINEAR
    assert policy.fill == [125, 123, 114]
    assert len(policy._augmentation_space(31, (32, 32))) == 14
    image = Image.fromarray(
        torch.arange(3072).remainder(256).byte().reshape(32, 32, 3).numpy()
    )
    torch.manual_seed(29029)
    accepted_rng = torch.random.get_rng_state().clone()
    wrapper(image)
    assert torch.equal(torch.random.get_rng_state(), accepted_rng)

    marker_cutoff_check()
    base_active, base_tail = trace_batch128_epochs(False)
    augmented_active, augmented_tail = trace_batch128_epochs(True)
    assert base_active != augmented_active
    assert base_tail == augmented_tail
    source = inspect.getsource(train.main)
    assert source.count("randaugment_active.value = 0") == 1
    assert "and not budget_exhausted" in source
    assert "iterator_exhausted=true" in source
    print("params=987098 batches=390 examples=49920 lr_half=pass batch128_tail_replay=pass")
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
        loss = coefficient * F.cross_entropy(outputs, a) + (1 - coefficient) * F.cross_entropy(outputs, b)
    else:
        loss = F.cross_entropy(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - started)


def timing_window(module, batch_size, mixup, replicate):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = module.WideResNet(
        module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES
    ).to(DEVICE).train()
    optimizer = optimizer_for(module, model)
    generator = torch.Generator().manual_seed(29_000 + replicate)
    host_x = torch.randn(batch_size, 3, 32, 32, generator=generator).pin_memory()
    host_y = (torch.arange(batch_size) % module.NUM_CLASSES).pin_memory()
    distribution = torch.distributions.Beta(
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
    )
    torch.cuda.manual_seed(29_100 + replicate)
    for _ in range(20):
        timed_step(module, model, optimizer, host_x, host_y, distribution, mixup)
    values = [
        timed_step(module, model, optimizer, host_x, host_y, distribution, mixup)
        for _ in range(50)
    ]
    return statistics.mean(values)


def throughput_measurements():
    accepted = load_accepted()
    results = {}
    for mixup, regime in ((True, "mixup"), (False, "hard")):
        windows = {"accepted": [], "candidate": []}
        for replicate in range(3):
            order = ("accepted", "candidate") if replicate % 2 == 0 else ("candidate", "accepted")
            for kind in order:
                if kind == "accepted":
                    windows[kind].append(timing_window(accepted, 256, mixup, replicate))
                else:
                    windows[kind].append(timing_window(train, 128, mixup, replicate))
        medians = {kind: statistics.median(values) for kind, values in windows.items()}
        cvs = {
            kind: statistics.pstdev(values) / statistics.mean(values)
            for kind, values in windows.items()
        }
        assert all(value <= 0.05 for value in cvs.values()), (regime, windows, cvs)
        results[regime] = {"windows_ms": windows, "medians_ms": medians, "cvs": cvs}

    accepted_rate = (
        0.65 * 256 / results["mixup"]["medians_ms"]["accepted"]
        + 0.35 * 256 / results["hard"]["medians_ms"]["accepted"]
    )
    candidate_rate = (
        0.65 * 128 / results["mixup"]["medians_ms"]["candidate"]
        + 0.35 * 128 / results["hard"]["medians_ms"]["candidate"]
    )
    retention = candidate_rate / accepted_rate
    projected_passes = 133.00736 * retention
    projected_updates = projected_passes * 50_000 / 128
    payload = {
        "results": results,
        "retention": retention,
        "projected_passes": projected_passes,
        "projected_updates": projected_updates,
    }
    assert retention >= 0.9022
    assert projected_passes >= 120.0
    assert 46_875 <= projected_updates < 128_000
    return payload


def throughput_checks():
    payload = throughput_measurements()
    print(json.dumps(payload, sort_keys=True))
    print("THROUGHPUT PASS")


def make_real_loader(batch_size, active_value):
    active = MP_CONTEXT.Value("b", active_value, lock=False)
    dataset = datasets.CIFAR10(
        prepare.DATASET_DIR,
        train=True,
        download=False,
        transform=train.make_train_transform(active),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=prepare.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context=MP_CONTEXT,
    )
    return loader


def paced_epoch(loader, batch_size, consumer_seconds):
    started = time.perf_counter()
    batches = 0
    for inputs, targets in loader:
        assert inputs.shape == (batch_size, 3, 32, 32)
        assert targets.shape == (batch_size,)
        assert torch.isfinite(inputs).all()
        assert torch.all((targets >= 0) & (targets < train.NUM_CLASSES))
        time.sleep(consumer_seconds)
        batches += 1
    expected = 50_000 // batch_size
    assert batches == expected
    return time.perf_counter() - started


def loader_arm(batch_size, active_value, consumer_seconds):
    loader = make_real_loader(batch_size, active_value)
    paced_epoch(loader, batch_size, consumer_seconds)
    values = [paced_epoch(loader, batch_size, consumer_seconds) for _ in range(3)]
    shutdown_loader(loader)
    return values


def loader_timing_checks():
    payload = throughput_measurements()
    config = {
        "accepted": (256, payload["results"]["mixup"]["medians_ms"]["accepted"] / 1000, payload["results"]["hard"]["medians_ms"]["accepted"] / 1000),
        "candidate": (128, payload["results"]["mixup"]["medians_ms"]["candidate"] / 1000, payload["results"]["hard"]["medians_ms"]["candidate"] / 1000),
    }
    observations = {kind: {"active": [], "inactive": []} for kind in config}
    for kind in ("accepted", "candidate", "candidate", "accepted"):
        batch, mixup_seconds, hard_seconds = config[kind]
        observations[kind]["active"].extend(loader_arm(batch, 1, mixup_seconds))
        observations[kind]["inactive"].extend(loader_arm(batch, 0, hard_seconds))

    summaries = {}
    for kind, (batch, mixup_seconds, hard_seconds) in config.items():
        active_values = observations[kind]["active"]
        inactive_values = observations[kind]["inactive"]
        active_median = statistics.median(active_values)
        inactive_median = statistics.median(inactive_values)
        active_cv = statistics.pstdev(active_values) / statistics.mean(active_values)
        inactive_cv = statistics.pstdev(inactive_values) / statistics.mean(inactive_values)
        assert active_cv <= 0.05 and inactive_cv <= 0.05
        batches = 50_000 // batch
        active_stall = max(0.0, active_median - batches * mixup_seconds)
        inactive_stall = max(0.0, inactive_median - batches * hard_seconds)
        summaries[kind] = {
            "active_values": active_values,
            "inactive_values": inactive_values,
            "active_median": active_median,
            "inactive_median": inactive_median,
            "active_cv": active_cv,
            "inactive_cv": inactive_cv,
            "epoch_wall": 0.65 * active_median + 0.35 * inactive_median,
            "epoch_stall": 0.65 * active_stall + 0.35 * inactive_stall,
        }

    projected_epochs = payload["projected_passes"] * 50_000 / 49_920
    differential = 345.3 + max(
        0.0,
        summaries["candidate"]["epoch_stall"] - summaries["accepted"]["epoch_stall"],
    ) * projected_epochs
    absolute = 45.3 + summaries["candidate"]["epoch_wall"] * projected_epochs
    print(json.dumps({"summaries": summaries, "projected_epochs": projected_epochs, "differential_s": differential, "absolute_s": absolute}, sort_keys=True))
    assert differential <= 500 and absolute <= 500
    print("LOADER TIMING PASS")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--semantics", action="store_true")
    group.add_argument("--throughput", action="store_true")
    group.add_argument("--loader-timing", action="store_true")
    args = parser.parse_args()
    if args.semantics:
        semantic_checks()
    elif args.throughput:
        throughput_checks()
    else:
        loader_timing_checks()


if __name__ == "__main__":
    main()
