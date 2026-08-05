import argparse
import gc
import hashlib
import inspect
import multiprocessing
import statistics
import sys
import time
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


class MarkerDataset(Dataset):
    def __init__(self, active, size=64):
        self.active = active
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index, int(self.active.value)


MP_CONTEXT = multiprocessing.get_context()
EXPECTED_TRANSFORMS = (
    transforms.RandomCrop,
    transforms.RandomHorizontalFlip,
    train.EarlyRandAugment,
    transforms.ToTensor,
    transforms.Normalize,
)
CONSUMER_SECONDS = 0.0108


def optimizer_for(model):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return optim.SGD(
        [
            {"params": decay, "weight_decay": train.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train.MIN_LR,
        momentum=train.MOMENTUM,
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
        batch_size=8,
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


def trace_epochs(candidate):
    torch.manual_seed(42)
    active = MP_CONTEXT.Value("b", 1, lock=False)
    transform = train.make_train_transform(active if candidate else None)
    dataset = datasets.CIFAR10(
        prepare.DATASET_DIR, train=True, download=False, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=train.BATCH_SIZE,
        shuffle=True,
        num_workers=prepare.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context=MP_CONTEXT,
    )
    train.WideResNet(train.NUM_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES)
    active_hashes = []
    for index, (inputs, targets) in enumerate(loader):
        if index < 4:
            active_hashes.append(
                hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
            )
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
    assert GuardEval.constructions == 1
    assert MP_CONTEXT.get_start_method() == multiprocessing.get_start_method()
    assert MP_CONTEXT.get_start_method() == "forkserver"
    assert train.RANDAUGMENT_END_FRACTION == train.MIXUP_END_FRACTION == 0.65

    torch.empty(1, device="cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone()
    accepted_transform = train.make_train_transform(None)
    accepted_model = train.WideResNet(
        train.NUM_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES
    )
    cpu_after = torch.random.get_rng_state().clone()
    cuda_after = torch.cuda.get_rng_state().clone()

    torch.random.set_rng_state(cpu_before)
    torch.cuda.set_rng_state(cuda_before)
    active = MP_CONTEXT.Value("b", 1, lock=False)
    candidate_transform = train.make_train_transform(active)
    candidate_model = train.WideResNet(
        train.NUM_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES
    )
    assert torch.equal(torch.random.get_rng_state(), cpu_after)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_after)
    assert sum(p.numel() for p in candidate_model.parameters()) == 691_674
    for name, tensor in accepted_model.state_dict().items():
        assert torch.equal(tensor, candidate_model.state_dict()[name]), name
    with torch.inference_mode():
        probe = torch.zeros(8, 3, 32, 32)
        assert torch.equal(accepted_model(probe), candidate_model(probe))

    assert tuple(type(op) for op in candidate_transform.transforms) == EXPECTED_TRANSFORMS
    assert tuple(type(op) for op in accepted_transform.transforms) == (
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ToTensor,
        transforms.Normalize,
    )
    wrapper = candidate_transform.transforms[2]
    policy = wrapper.transform
    assert policy.num_ops == 1 and policy.magnitude == 5
    assert policy.num_magnitude_bins == 31
    assert policy.interpolation == transforms.InterpolationMode.BILINEAR
    assert policy.fill == [125, 123, 114]
    assert len(policy._augmentation_space(31, (32, 32))) == 14

    torch.manual_seed(26026)
    rng_before = torch.random.get_rng_state().clone()
    image = Image.fromarray(
        torch.arange(32 * 32 * 3, dtype=torch.int64)
        .remainder(256)
        .byte()
        .reshape(32, 32, 3)
        .numpy()
    )
    outputs = []
    for _ in range(64):
        outputs.append(hashlib.sha256(wrapper(image).tobytes()).hexdigest())
        assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert len(set(outputs)) > 1
    active.value = 0
    assert wrapper(image).tobytes() == image.tobytes()

    optimizer = optimizer_for(candidate_model)
    decay_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    assert decay_ids | no_decay_ids == {id(p) for p in candidate_model.parameters()}
    assert not decay_ids & no_decay_ids
    targets = torch.arange(8) % train.NUM_CLASSES
    backward_probe = torch.zeros(8, 3, 32, 32)
    loss = F.cross_entropy(candidate_model(backward_probe), targets)
    assert torch.isfinite(loss)
    loss.backward()

    marker_cutoff_check()
    accepted_active, accepted_tail = trace_epochs(False)
    candidate_active, candidate_tail = trace_epochs(True)
    assert accepted_active != candidate_active
    assert accepted_tail == candidate_tail

    source = inspect.getsource(train.main)
    assert source.count("randaugment_active.value = 0") == 1
    assert "and not budget_exhausted" in source
    assert "iterator_exhausted=true" in source
    print(
        "context=forkserver policy_ops=14 params=691674 "
        "worker_rng_isolation=pass cutoff_no_leak=pass"
    )
    print("SEMANTICS PASS")


def make_real_loader(candidate):
    active = MP_CONTEXT.Value("b", 1, lock=False)
    transform = train.make_train_transform(active if candidate else None)
    dataset = datasets.CIFAR10(
        prepare.DATASET_DIR, train=True, download=False, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=train.BATCH_SIZE,
        shuffle=True,
        num_workers=prepare.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context=MP_CONTEXT,
    )
    return active, loader


def paced_epoch(loader):
    started = time.perf_counter()
    batches = 0
    for inputs, targets in loader:
        assert inputs.shape == (train.BATCH_SIZE, 3, 32, 32)
        assert targets.shape == (train.BATCH_SIZE,)
        assert torch.isfinite(inputs).all()
        assert torch.all((targets >= 0) & (targets < train.NUM_CLASSES))
        time.sleep(CONSUMER_SECONDS)
        batches += 1
    assert batches == 195
    return time.perf_counter() - started


def timed_arm(candidate):
    active, loader = make_real_loader(candidate)
    paced_epoch(loader)
    measured = [paced_epoch(loader) for _ in range(3)]
    active.value = 0
    boundary = paced_epoch(loader)
    shutdown_loader(loader)
    return measured, boundary


def loader_timing_checks():
    arms = []
    for candidate in (False, True, True, False):
        arms.append((candidate, *timed_arm(candidate)))
    accepted = [value for kind, values, _ in arms if not kind for value in values]
    candidate = [value for kind, values, _ in arms if kind for value in values]
    boundaries = [boundary for kind, _, boundary in arms if kind]
    accepted_median = statistics.median(accepted)
    candidate_median = statistics.median(candidate)
    accepted_cv = statistics.pstdev(accepted) / statistics.mean(accepted)
    candidate_cv = statistics.pstdev(candidate) / statistics.mean(candidate)
    assert accepted_cv <= 0.05 and candidate_cv <= 0.05
    assert all(value <= 1.20 * accepted_median for value in boundaries)
    historical_projection = 341.2 + max(
        0.0, candidate_median - accepted_median
    ) * 143
    absolute_projection = 41.2 + 143 * candidate_median
    print(f"arms={arms}")
    print(
        f"accepted_median_s={accepted_median:.6f} candidate_median_s={candidate_median:.6f} "
        f"accepted_cv={accepted_cv:.6f} candidate_cv={candidate_cv:.6f} "
        f"boundaries={boundaries} historical_projection_s={historical_projection:.3f} "
        f"absolute_projection_s={absolute_projection:.3f}"
    )
    assert historical_projection <= 500 and absolute_projection <= 500
    print("LOADER TIMING PASS")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--semantics", action="store_true")
    group.add_argument("--loader-timing", action="store_true")
    args = parser.parse_args()
    semantic_checks() if args.semantics else loader_timing_checks()


if __name__ == "__main__":
    main()
