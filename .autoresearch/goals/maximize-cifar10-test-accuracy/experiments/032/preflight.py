import argparse
import gc
import hashlib
import json
import multiprocessing
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


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
OriginalCIFAR10 = datasets.CIFAR10


class GuardCIFAR10(OriginalCIFAR10):
    def __init__(self, *args, **kwargs):
        train_flag = kwargs.get("train", args[1] if len(args) > 1 else True)
        if not train_flag:
            raise AssertionError("preflight may not construct test data")
        super().__init__(*args, **kwargs)


datasets.CIFAR10 = GuardCIFAR10
import train


BASE_COMMIT = "67c8e98"
MP_CONTEXT = multiprocessing.get_context()
CONSUMER_SECONDS = 0.01155


class Flag:
    def __init__(self, value):
        self.value = value


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


def state_hash(state):
    digest = hashlib.sha256(state.numpy().tobytes()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def numpy_pad(image, mode):
    array = np.asarray(image)
    if mode == "constant":
        padded = np.pad(array, ((4, 4), (4, 4), (0, 0)), mode="constant")
    else:
        padded = np.pad(array, ((4, 4), (4, 4), (0, 0)), mode="reflect")
    return Image.fromarray(padded.astype(np.uint8, copy=False))


def decode_randaugment(wrapper, image):
    main_state = torch.random.get_rng_state().clone()
    effective_state = (
        main_state
        if wrapper._randaugment_rng_state is None
        else wrapper._randaugment_rng_state.clone()
    )
    torch.random.set_rng_state(effective_state)
    op_meta = wrapper.transform._augmentation_space(31, image.size)
    op_index = int(torch.randint(len(op_meta), (1,)).item())
    op_name = list(op_meta.keys())[op_index]
    _, signed = op_meta[op_name]
    sign = int(bool(torch.randint(2, (1,)).item())) if signed else -1
    predicted_after = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(main_state)
    return op_index, sign, effective_state, predicted_after


class InstrumentedTransform:
    def __init__(self, mode, active):
        self.mode = mode
        self.active = active
        self.randaugment = train.EarlyRandAugment(active)
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1))

    def __call__(self, image):
        padded = TF.pad(image, [4, 4, 4, 4], padding_mode=self.mode)
        i, j, height, width = transforms.RandomCrop.get_params(padded, (32, 32))
        image = TF.crop(padded, i, j, height, width)
        flip = int(torch.rand(1).item() < 0.5)
        if flip:
            image = TF.hflip(image)
        active = int(self.active.value)
        if active:
            op_index, sign, private_before, predicted_after = decode_randaugment(
                self.randaugment, image
            )
            main_before = torch.random.get_rng_state().clone()
            image = self.randaugment(image)
            assert torch.equal(torch.random.get_rng_state(), main_before)
            assert torch.equal(self.randaugment._randaugment_rng_state, predicted_after)
            private_after = self.randaugment._randaugment_rng_state.clone()
        else:
            op_index, sign = -1, -1
            private_before = (
                torch.random.get_rng_state().clone()
                if self.randaugment._randaugment_rng_state is None
                else self.randaugment._randaugment_rng_state.clone()
            )
            image_before = image.tobytes()
            main_before = torch.random.get_rng_state().clone()
            image = self.randaugment(image)
            assert image.tobytes() == image_before
            assert torch.equal(torch.random.get_rng_state(), main_before)
            private_after = (
                private_before.clone()
                if self.randaugment._randaugment_rng_state is None
                else self.randaugment._randaugment_rng_state.clone()
            )
            assert torch.equal(private_before, private_after)
        tensor = self.normalize(TF.to_tensor(image))
        worker = get_worker_info()
        worker_id = -1 if worker is None else worker.id
        trace = (
            worker_id,
            i,
            j,
            flip,
            active,
            op_index,
            sign,
            state_hash(private_before),
            state_hash(private_after),
        )
        return tensor, trace


class InstrumentedDataset(Dataset):
    def __init__(self, active, mode):
        self.base = GuardCIFAR10(
            prepare.DATASET_DIR, train=True, download=False, transform=None
        )
        self.transform = InstrumentedTransform(mode, active)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, target = self.base[index]
        tensor, trace = self.transform(image)
        return tensor, target, index, trace


def shutdown_loader(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
    del loader
    gc.collect()


def trace_worker_arm(mode):
    torch.manual_seed(42)
    active = MP_CONTEXT.Value("b", 1, lock=False)
    dataset = InstrumentedDataset(active, mode)
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
    train.WideResNet(train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES)
    records = []
    image_hashes = []
    for inputs, targets, indices, traces in loader:
        image_hashes.append(hashlib.sha256(inputs.numpy().tobytes()).hexdigest())
        trace_columns = [column.tolist() for column in traces]
        for row in zip(indices.tolist(), targets.tolist(), *trace_columns):
            records.append(tuple(row))
    active.value = 0
    inactive_records = []
    inactive_hashes = []
    for batch_index, (inputs, targets, indices, traces) in enumerate(loader):
        inactive_hashes.append(hashlib.sha256(inputs.numpy().tobytes()).hexdigest())
        trace_columns = [column.tolist() for column in traces]
        for row in zip(indices.tolist(), targets.tolist(), *trace_columns):
            inactive_records.append(tuple(row))
        if batch_index == 15:
            break
    terminal_rng = torch.random.get_rng_state().clone()
    shutdown_loader(loader)
    return records, inactive_records, image_hashes, inactive_hashes, terminal_rng


def make_asymmetric_image():
    values = (np.arange(32 * 32 * 3, dtype=np.uint32) % 255 + 1).astype(np.uint8)
    return Image.fromarray(values.reshape(32, 32, 3))


def crop_flip_oracle_check():
    image = make_asymmetric_image()
    accepted_crop = transforms.RandomCrop(32, padding=4)
    candidate_crop = transforms.RandomCrop(32, padding=4, padding_mode="reflect")
    flip_transform = transforms.RandomHorizontalFlip()
    torch.manual_seed(32_032)
    saved = torch.random.get_rng_state().clone()

    torch.random.set_rng_state(saved)
    accepted_output = flip_transform(accepted_crop(image))
    accepted_terminal = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(saved)
    candidate_output = flip_transform(candidate_crop(image))
    candidate_terminal = torch.random.get_rng_state().clone()

    torch.random.set_rng_state(saved)
    before_padding = torch.random.get_rng_state().clone()
    constant = numpy_pad(image, "constant")
    reflected = numpy_pad(image, "reflect")
    assert torch.equal(torch.random.get_rng_state(), before_padding)
    i, j, height, width = transforms.RandomCrop.get_params(constant, (32, 32))
    flip = bool(torch.rand(1).item() < 0.5)
    manual_terminal = torch.random.get_rng_state().clone()
    accepted_manual = TF.crop(constant, i, j, height, width)
    candidate_manual = TF.crop(reflected, i, j, height, width)
    if flip:
        accepted_manual = TF.hflip(accepted_manual)
        candidate_manual = TF.hflip(candidate_manual)
    assert accepted_output.tobytes() == accepted_manual.tobytes()
    assert candidate_output.tobytes() == candidate_manual.tobytes()
    assert torch.equal(accepted_terminal, candidate_terminal)
    assert torch.equal(accepted_terminal, manual_terminal)
    return i, j, int(flip)


def pixel_confinement_check():
    image = make_asymmetric_image()
    constant = np.asarray(numpy_pad(image, "constant"))
    reflected = np.asarray(numpy_pad(image, "reflect"))
    checked = 0
    for i in range(9):
        for j in range(9):
            y = np.arange(i, i + 32)[:, None]
            x = np.arange(j, j + 32)[None, :]
            mask = (y < 4) | (y >= 36) | (x < 4) | (x >= 36)
            accepted = constant[i : i + 32, j : j + 32]
            candidate = reflected[i : i + 32, j : j + 32]
            for flip in (False, True):
                active_mask = np.fliplr(mask) if flip else mask
                left = np.fliplr(accepted) if flip else accepted
                right = np.fliplr(candidate) if flip else candidate
                differences = np.any(left != right, axis=2)
                assert not np.any(differences & ~active_mask)
                if np.any(active_mask):
                    assert np.any(differences & active_mask)
                else:
                    assert np.array_equal(left, right)
                checked += 1
    generator = torch.Generator().manual_seed(32_132)
    coordinates = torch.randint(0, 9, (100_000, 2), generator=generator)
    contact_rate = (
        ~((coordinates[:, 0] == 4) & (coordinates[:, 1] == 4))
    ).float().mean().item()
    assert 0.985 <= contact_rate <= 0.990
    return checked, contact_rate


def direct_randaugment_check():
    active_a = Flag(1)
    active_c = Flag(1)
    wrapper_a = train.EarlyRandAugment(active_a)
    wrapper_c = train.EarlyRandAugment(active_c)
    image = make_asymmetric_image()
    constant = TF.crop(numpy_pad(image, "constant"), 0, 0, 32, 32)
    reflected = TF.crop(numpy_pad(image, "reflect"), 0, 0, 32, 32)
    decisions = []
    torch.manual_seed(32_232)
    evolving_main = torch.random.get_rng_state().clone()
    for _ in range(64):
        torch.random.set_rng_state(evolving_main)
        decision_a = decode_randaugment(wrapper_a, constant)
        before_a = torch.random.get_rng_state().clone()
        wrapper_a(constant)
        after_main_a = torch.random.get_rng_state().clone()
        after_private_a = wrapper_a._randaugment_rng_state.clone()

        torch.random.set_rng_state(evolving_main)
        decision_c = decode_randaugment(wrapper_c, reflected)
        before_c = torch.random.get_rng_state().clone()
        wrapper_c(reflected)
        after_main_c = torch.random.get_rng_state().clone()
        after_private_c = wrapper_c._randaugment_rng_state.clone()
        assert decision_a[:2] == decision_c[:2]
        assert torch.equal(before_a, before_c)
        assert torch.equal(after_main_a, after_main_c)
        assert torch.equal(after_private_a, after_private_c)
        decisions.append(decision_a[:2])
        torch.random.set_rng_state(evolving_main)
        torch.rand(3)
        evolving_main = torch.random.get_rng_state().clone()
    assert len(set(decisions)) > 1
    active_a.value = 0
    active_c.value = 0
    for wrapper, probe in ((wrapper_a, constant), (wrapper_c, reflected)):
        private_before = wrapper._randaugment_rng_state.clone()
        main_before = torch.random.get_rng_state().clone()
        assert wrapper(probe).tobytes() == probe.tobytes()
        assert torch.equal(wrapper._randaugment_rng_state, private_before)
        assert torch.equal(torch.random.get_rng_state(), main_before)
    return len(set(decisions))


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

    diff = subprocess.check_output(
        ["git", "diff", "--unified=0", BASE_COMMIT, "--", "train.py"],
        cwd=ROOT,
        text=True,
    )
    additions = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removals = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    assert additions == ['+        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),']
    assert removals == ["-        transforms.RandomCrop(32, padding=4),"]

    torch.empty(1, device="cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    accepted_model = accepted.WideResNet(
        accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES
    )
    accepted_cpu = torch.random.get_rng_state().clone()
    accepted_cuda = torch.cuda.get_rng_state().clone()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate_model = train.WideResNet(
        train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES
    )
    assert torch.equal(torch.random.get_rng_state(), accepted_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), accepted_cuda)
    for name, tensor in accepted_model.state_dict().items():
        assert torch.equal(tensor, candidate_model.state_dict()[name]), name
    assert sum(parameter.numel() for parameter in candidate_model.parameters()) == 987_098
    accepted_optimizer = optimizer_for(accepted, accepted_model)
    candidate_optimizer = optimizer_for(train, candidate_model)
    assert [
        (group["weight_decay"], group["momentum"], group["nesterov"], len(group["params"]))
        for group in accepted_optimizer.param_groups
    ] == [
        (group["weight_decay"], group["momentum"], group["nesterov"], len(group["params"]))
        for group in candidate_optimizer.param_groups
    ]

    candidate_transform = train.make_train_transform(Flag(1))
    crop = candidate_transform.transforms[0]
    assert isinstance(crop, transforms.RandomCrop)
    assert crop.size == (32, 32) and crop.padding == 4 and crop.padding_mode == "reflect"
    assert tuple(type(op).__name__ for op in candidate_transform.transforms) == (
        "RandomCrop",
        "RandomHorizontalFlip",
        "EarlyRandAugment",
        "ToTensor",
        "Normalize",
    )
    oracle_decision = crop_flip_oracle_check()
    exhaustive, contact_rate = pixel_confinement_check()
    distinct_decisions = direct_randaugment_check()

    accepted_records, accepted_tail, accepted_hashes, accepted_tail_hashes, accepted_rng = trace_worker_arm("constant")
    candidate_records, candidate_tail, candidate_hashes, candidate_tail_hashes, candidate_rng = trace_worker_arm("reflect")
    assert accepted_records == candidate_records
    assert accepted_tail == candidate_tail
    assert accepted_hashes != candidate_hashes
    assert accepted_tail_hashes != candidate_tail_hashes
    assert torch.equal(accepted_rng, candidate_rng)
    assert all(record[6] == 1 for record in candidate_records)
    assert all(record[6] == 0 and record[7] == -1 for record in candidate_tail)

    print(
        json.dumps(
            {
                "crop_flip_oracle": oracle_decision,
                "exhaustive_cases": exhaustive,
                "padding_contact_rate": contact_rate,
                "randaugment_decisions": distinct_decisions,
                "active_trace_samples": len(candidate_records),
                "inactive_trace_samples": len(candidate_tail),
                "worker_trace": "exact",
                "params": 987098,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("SEMANTICS PASS")


def make_real_loader(mode, active_value):
    torch.manual_seed(32_500)
    active = MP_CONTEXT.Value("b", active_value, lock=False)
    if mode == "reflect":
        transform = train.make_train_transform(active)
    else:
        operations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            train.EarlyRandAugment(active),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
        ]
        transform = transforms.Compose(operations)
    dataset = GuardCIFAR10(
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
    train.WideResNet(train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES)
    return loader


def paced_epoch(loader):
    started = time.perf_counter()
    batches = 0
    examples = 0
    for inputs, targets in loader:
        assert inputs.shape == (train.BATCH_SIZE, 3, 32, 32)
        assert targets.shape == (train.BATCH_SIZE,)
        assert torch.isfinite(inputs).all()
        time.sleep(CONSUMER_SECONDS)
        batches += 1
        examples += len(targets)
    assert batches == 195 and examples == 49_920
    return time.perf_counter() - started


def loader_arm(mode, active_value):
    loader = make_real_loader(mode, active_value)
    paced_epoch(loader)
    values = [paced_epoch(loader) for _ in range(3)]
    shutdown_loader(loader)
    return values


def loader_timing_checks():
    observations = {
        mode: {phase: [] for phase in ("active", "inactive")}
        for mode in ("constant", "reflect")
    }
    order = ("constant", "reflect", "reflect", "constant")
    for mode in order:
        observations[mode]["active"].extend(loader_arm(mode, 1))
        observations[mode]["inactive"].extend(loader_arm(mode, 0))
    summaries = {}
    for mode, phases in observations.items():
        summaries[mode] = {}
        for phase, values in phases.items():
            median = statistics.median(values)
            cv = statistics.pstdev(values) / statistics.mean(values)
            summaries[mode][phase] = {
                "values": values,
                "median": median,
                "cv": cv,
            }
        summaries[mode]["weighted"] = (
            0.65 * summaries[mode]["active"]["median"]
            + 0.35 * summaries[mode]["inactive"]["median"]
        )
    projected_epochs = 133.00736 * 50_000 / 49_920
    differential = 345.3 + max(
        0.0, summaries["reflect"]["weighted"] - summaries["constant"]["weighted"]
    ) * projected_epochs
    absolute = 45.3 + summaries["reflect"]["weighted"] * projected_epochs
    payload = {
        "order": order,
        "summaries": summaries,
        "projected_passes": 133.00736,
        "projected_epochs": projected_epochs,
        "differential_wall_s": differential,
        "absolute_wall_s": absolute,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    assert all(
        phase["cv"] <= 0.05
        for mode in summaries.values()
        for name, phase in mode.items()
        if name in ("active", "inactive")
    ), payload
    for phase in ("active", "inactive"):
        assert summaries["reflect"][phase]["median"] <= 1.10 * summaries["constant"][phase]["median"], payload
    assert differential < 500 and absolute < 500, payload
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
