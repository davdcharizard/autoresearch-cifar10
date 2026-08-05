import argparse
import ast
import gc
import hashlib
import json
import math
import multiprocessing
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import prepare


BASELINE = "a7c42dc"
ACCEPTED_FILL = (0, 0, 0)
CANDIDATE_FILL = (125, 123, 114)
EXPECTED_PARAMS = 1_003_482
EXPECTED_PARAMETER_TENSORS = 52
EXPECTED_STATE = 97
CONSUMER_SECONDS = 0.011
RETENTION_FLOOR = 127.0 / 130.304
MP_CONTEXT = multiprocessing.get_context()


class GuardEval:
    constructions = 0

    def __init__(self):
        type(self).constructions += 1

    def evaluate(self, *_args, **_kwargs):
        raise AssertionError("preflight may not evaluate")


prepare.Eval = GuardEval
OriginalCIFAR10 = datasets.CIFAR10


class GuardCIFAR10(OriginalCIFAR10):
    def __init__(self, *args, **kwargs):
        train_flag = kwargs.get("train", args[1] if len(args) > 1 else True)
        if not train_flag:
            raise AssertionError("preflight may not construct test data")
        kwargs["download"] = False
        super().__init__(*args, **kwargs)


datasets.CIFAR10 = GuardCIFAR10
import train


class Flag:
    def __init__(self, value):
        self.value = value


def load_accepted():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:train.py"], cwd=ROOT, text=True
    )
    module = types.ModuleType("exp046_accepted")
    module.__file__ = f"git:{BASELINE}:train.py"
    module.__source__ = source
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def optimizer_for(module, model, lr=None):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return optim.SGD(
        [
            {"params": decay, "weight_decay": module.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=module.MIN_LR if lr is None else lr,
        momentum=module.MOMENTUM,
        nesterov=True,
    )


def optimizer_signature(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        (
            [names[id(parameter)] for parameter in group["params"]],
            {key: value for key, value in group.items() if key != "params"},
        )
        for group in optimizer.param_groups
    ]


def rng_state():
    return torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def restore_rng(state):
    torch.random.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_hash(state):
    digest = hashlib.sha256(state.numpy().tobytes()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def make_asymmetric_image():
    values = (np.arange(32 * 32 * 3, dtype=np.uint32) % 253 + 1).astype(np.uint8)
    return Image.fromarray(values.reshape(32, 32, 3), mode="RGB")


def numpy_padded(image, fill):
    source = np.asarray(image)
    result = np.empty((40, 40, 3), dtype=np.uint8)
    result[...] = np.asarray(fill, dtype=np.uint8)
    result[4:36, 4:36] = source
    return result


def numpy_oracle(image, fill, i, j, flip):
    padded = numpy_padded(image, fill)
    output = padded[i : i + 32, j : j + 32].copy()
    source_mask = np.zeros((40, 40), dtype=bool)
    source_mask[4:36, 4:36] = True
    pad_mask = ~source_mask[i : i + 32, j : j + 32]
    if flip:
        output = output[:, ::-1].copy()
        pad_mask = pad_mask[:, ::-1].copy()
    return output, pad_mask


def installed_forced(image, fill, i, j, flip):
    padded = TF.pad(image, [4, 4, 4, 4], fill=fill, padding_mode="constant")
    output = TF.crop(padded, i, j, 32, 32)
    return TF.hflip(output) if flip else output


def exhaustive_pixel_oracle():
    image = make_asymmetric_image()
    checked = 0
    touching = 0
    for i in range(9):
        for j in range(9):
            for flip in (False, True):
                accepted = np.asarray(installed_forced(image, ACCEPTED_FILL, i, j, flip))
                candidate = np.asarray(installed_forced(image, CANDIDATE_FILL, i, j, flip))
                accepted_ref, mask = numpy_oracle(image, ACCEPTED_FILL, i, j, flip)
                candidate_ref, candidate_mask = numpy_oracle(image, CANDIDATE_FILL, i, j, flip)
                assert np.array_equal(mask, candidate_mask)
                assert np.array_equal(accepted, accepted_ref)
                assert np.array_equal(candidate, candidate_ref)
                differences = np.any(accepted != candidate, axis=2)
                assert not np.any(differences & ~mask)
                if mask.any():
                    assert np.all(accepted[mask] == np.asarray(ACCEPTED_FILL))
                    assert np.all(candidate[mask] == np.asarray(CANDIDATE_FILL))
                    assert np.array_equal(differences, mask)
                    touching += 1
                else:
                    assert i == 4 and j == 4 and np.array_equal(accepted, candidate)
                checked += 1
    expected_candidate = np.asarray(CANDIDATE_FILL, dtype=np.float32) / 255 - np.asarray(
        (0.4914, 0.4822, 0.4465), dtype=np.float32
    )
    expected_accepted = -np.asarray((0.4914, 0.4822, 0.4465), dtype=np.float32)
    stated = np.asarray((-0.001204, 0.000153, 0.000559), dtype=np.float32)
    np.testing.assert_allclose(expected_candidate, stated, rtol=0, atol=8e-7)
    print(
        f"pixel_oracle cases={checked} touching={touching} "
        f"accepted_norm={expected_accepted.tolist()} candidate_norm={expected_candidate.tolist()}"
    )


def sampled_crop_flip(fill, image):
    padded = TF.pad(image, [4, 4, 4, 4], fill=fill, padding_mode="constant")
    i, j, h, w = transforms.RandomCrop.get_params(padded, (32, 32))
    output = TF.crop(padded, i, j, h, w)
    flip = bool(torch.rand(1).item() < 0.5)
    if flip:
        output = TF.hflip(output)
    return output, (i, j, flip)


def sampled_crop_oracle():
    image = make_asymmetric_image()
    generator = torch.Generator().manual_seed(46_105)
    starts = [torch.randint(0, 2**31 - 1, (1,), generator=generator).item() for _ in range(256)]
    for seed in starts:
        torch.manual_seed(seed)
        accepted, decision_a = sampled_crop_flip(ACCEPTED_FILL, image)
        terminal_a = torch.random.get_rng_state().clone()
        torch.manual_seed(seed)
        candidate, decision_c = sampled_crop_flip(CANDIDATE_FILL, image)
        terminal_c = torch.random.get_rng_state().clone()
        assert decision_a == decision_c and torch.equal(terminal_a, terminal_c)
        for output, fill in ((accepted, ACCEPTED_FILL), (candidate, CANDIDATE_FILL)):
            reference, _mask = numpy_oracle(image, fill, *decision_a)
            assert np.array_equal(np.asarray(output), reference)
    coordinates = torch.randint(0, 9, (100_000, 2), generator=torch.Generator().manual_seed(46_205))
    contact = (~((coordinates[:, 0] == 4) & (coordinates[:, 1] == 4))).float().mean().item()
    retained_h = 32 - (coordinates[:, 0] - 4).abs()
    retained_w = 32 - (coordinates[:, 1] - 4).abs()
    synthetic = (1 - retained_h.float() * retained_w.float() / (32 * 32)).mean().item()
    print(f"crop_distribution contact={contact:.9f} synthetic_share={synthetic:.9f}")
    assert 0.9865 <= contact <= 0.9888
    assert 0.132 <= synthetic <= 0.136


def decode_randaugment(wrapper, image):
    main_state = torch.random.get_rng_state().clone()
    effective = main_state if wrapper._randaugment_rng_state is None else wrapper._randaugment_rng_state.clone()
    torch.random.set_rng_state(effective)
    metadata = wrapper.transform._augmentation_space(31, image.size)
    op_index = int(torch.randint(len(metadata), (1,)).item())
    op_name = list(metadata.keys())[op_index]
    _magnitudes, signed = metadata[op_name]
    sign = int(bool(torch.randint(2, (1,)).item())) if signed else -1
    predicted_after = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(main_state)
    return op_index, sign, effective, predicted_after


def direct_randaugment_oracle(accepted):
    active_a, active_c = Flag(1), Flag(1)
    wrapper_a = accepted.EarlyRandAugment(active_a)
    wrapper_c = train.EarlyRandAugment(active_c)
    image = make_asymmetric_image()
    probe_a = installed_forced(image, ACCEPTED_FILL, 0, 8, True)
    probe_c = installed_forced(image, CANDIDATE_FILL, 0, 8, True)
    evolving = torch.Generator().manual_seed(46_305)
    decisions = []
    for _ in range(64):
        seed = torch.randint(0, 2**31 - 1, (1,), generator=evolving).item()
        torch.manual_seed(seed)
        decision_a = decode_randaugment(wrapper_a, probe_a)
        before_a = torch.random.get_rng_state().clone()
        wrapper_a(probe_a)
        after_main_a = torch.random.get_rng_state().clone()
        after_private_a = wrapper_a._randaugment_rng_state.clone()
        torch.manual_seed(seed)
        decision_c = decode_randaugment(wrapper_c, probe_c)
        before_c = torch.random.get_rng_state().clone()
        wrapper_c(probe_c)
        after_main_c = torch.random.get_rng_state().clone()
        after_private_c = wrapper_c._randaugment_rng_state.clone()
        assert decision_a[:2] == decision_c[:2]
        assert torch.equal(before_a, before_c)
        assert torch.equal(after_main_a, after_main_c)
        assert torch.equal(after_private_a, after_private_c)
        decisions.append(decision_a[:2])
    active_a.value = active_c.value = 0
    for wrapper, probe in ((wrapper_a, probe_a), (wrapper_c, probe_c)):
        main_before = torch.random.get_rng_state().clone()
        private_before = wrapper._randaugment_rng_state.clone()
        assert wrapper(probe).tobytes() == probe.tobytes()
        assert torch.equal(torch.random.get_rng_state(), main_before)
        assert torch.equal(wrapper._randaugment_rng_state, private_before)
    print(f"randaugment_oracle calls=64 distinct_decisions={len(set(decisions))}")
    assert len(set(decisions)) > 1


class InstrumentedTransform:
    def __init__(self, fill, active):
        self.fill = fill
        self.active = active
        self.randaugment = train.EarlyRandAugment(active)
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1))

    def __call__(self, image):
        padded = TF.pad(image, [4, 4, 4, 4], fill=self.fill, padding_mode="constant")
        i, j, h, w = transforms.RandomCrop.get_params(padded, (32, 32))
        image = TF.crop(padded, i, j, h, w)
        flip = int(torch.rand(1).item() < 0.5)
        if flip:
            image = TF.hflip(image)
        active = int(self.active.value)
        if active:
            op_index, sign, private_before, predicted_after = decode_randaugment(self.randaugment, image)
            main_before = torch.random.get_rng_state().clone()
            image = self.randaugment(image)
            assert torch.equal(torch.random.get_rng_state(), main_before)
            assert torch.equal(self.randaugment._randaugment_rng_state, predicted_after)
            private_after = self.randaugment._randaugment_rng_state.clone()
        else:
            op_index = sign = -1
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
        return tensor, (
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


class InstrumentedDataset(Dataset):
    def __init__(self, fill, active):
        self.base = GuardCIFAR10(prepare.DATASET_DIR, train=True, transform=None)
        self.transform = InstrumentedTransform(fill, active)

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


def worker_transition_arm(fill):
    torch.manual_seed(46_405)
    active = MP_CONTEXT.Value("b", 1, lock=False)
    loader = DataLoader(
        InstrumentedDataset(fill, active),
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
    phase_records, phase_hashes = [], []
    for phase in (1, 0):
        if phase == 0:
            active.value = 0
        records, hashes = [], []
        for inputs, targets, indices, traces in loader:
            assert inputs.shape == (256, 3, 32, 32) and torch.isfinite(inputs).all()
            hashes.append(hashlib.sha256(inputs.numpy().tobytes()).hexdigest())
            columns = [column.tolist() for column in traces]
            records.extend(tuple(row) for row in zip(indices.tolist(), targets.tolist(), *columns))
        assert len(records) == 49_920
        phase_records.append(records)
        phase_hashes.append(hashes)
    terminal = torch.random.get_rng_state().clone()
    shutdown_loader(loader)
    return phase_records, phase_hashes, terminal


def worker_transition_oracle():
    records_a, hashes_a, terminal_a = worker_transition_arm(ACCEPTED_FILL)
    records_c, hashes_c, terminal_c = worker_transition_arm(CANDIDATE_FILL)
    assert records_a == records_c
    assert hashes_a[0] != hashes_c[0] and hashes_a[1] != hashes_c[1]
    assert torch.equal(terminal_a, terminal_c)
    active_records, inactive_records = records_c
    assert all(row[6] == 1 and row[7] >= 0 for row in active_records)
    assert all(row[6] == 0 and row[7] == -1 and row[8] == -1 and row[9] == row[10] for row in inactive_records)
    print(f"worker_transition active={len(active_records)} inactive={len(inactive_records)} trace=exact")


def source_and_state_oracle(accepted):
    diff = subprocess.check_output(
        ["git", "diff", "--unified=0", BASELINE, "--", "train.py"], cwd=ROOT, text=True
    )
    additions = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removals = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    print(f"source additions={additions} removals={removals}")
    assert additions == ['+        transforms.RandomCrop(32, padding=4, fill=(125, 123, 114)),']
    assert removals == ['-        transforms.RandomCrop(32, padding=4),']
    assert subprocess.check_output(["git", "diff", "--name-only", BASELINE], cwd=ROOT, text=True).strip() == "train.py"
    subprocess.run(["git", "diff", "--exit-code", BASELINE, "--", "prepare.py", "pyproject.toml"], cwd=ROOT, check=True)
    tree = ast.parse((ROOT / "train.py").read_text())
    crops = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "RandomCrop"]
    assert len(crops) == 1
    candidate_transform = train.make_train_transform(Flag(1))
    crop, flip, randaugment, to_tensor, normalize = candidate_transform.transforms
    assert isinstance(crop, transforms.RandomCrop) and crop.size == (32, 32)
    assert crop.padding == 4 and crop.fill == CANDIDATE_FILL and crop.padding_mode == "constant" and not crop.pad_if_needed
    assert isinstance(flip, transforms.RandomHorizontalFlip)
    assert isinstance(randaugment, train.EarlyRandAugment)
    assert isinstance(to_tensor, transforms.ToTensor) and isinstance(normalize, transforms.Normalize)
    assert tuple(normalize.mean) == (0.4914, 0.4822, 0.4465) and tuple(normalize.std) == (1, 1, 1)
    assert randaugment.transform.fill == [125, 123, 114]
    unchanged = (
        "STAGE_BLOCKS", "WIDEN_FACTOR", "NUM_CLASSES", "BATCH_SIZE", "LR", "MIN_LR",
        "WARMUP_FRACTION", "MOMENTUM", "WEIGHT_DECAY", "MAX_STEPS", "EVAL_EVERY",
        "MIXUP_ALPHA", "MIXUP_END_FRACTION", "RANDAUGMENT_END_FRACTION",
        "POOLED_HEAD_WIDTH", "POOLED_HEAD_SCALE", "POOLED_HEAD_INIT_SEED",
    )
    for name in unchanged:
        assert getattr(train, name) == getattr(accepted, name), name
    torch.empty(1, device="cuda")
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    start = rng_state()
    restore_rng(start)
    model_a = accepted.WideResNet(accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES)
    after_a = rng_state()
    restore_rng(start)
    model_c = train.WideResNet(train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES)
    after_c = rng_state()
    assert torch.equal(after_a[0], after_c[0]) and torch.equal(after_a[1], after_c[1])
    assert len(model_c.state_dict()) == EXPECTED_STATE and len(list(model_c.parameters())) == EXPECTED_PARAMETER_TENSORS
    assert sum(parameter.numel() for parameter in model_c.parameters()) == EXPECTED_PARAMS
    assert list(model_a.state_dict()) == list(model_c.state_dict())
    for name, value in model_a.state_dict().items():
        assert torch.equal(value, model_c.state_dict()[name]), name
    assert optimizer_signature(model_a, optimizer_for(accepted, model_a)) == optimizer_signature(model_c, optimizer_for(train, model_c))
    for seconds in (0.0, 15.0, 195.0, 300.0):
        assert accepted.learning_rate(seconds) == train.learning_rate(seconds)
    assert train.MIXUP_END_FRACTION == train.RANDAUGMENT_END_FRACTION == 0.65
    print(f"state params={EXPECTED_PARAMS} parameter_tensors={EXPECTED_PARAMETER_TENSORS} state_entries={EXPECTED_STATE}")


def semantics():
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "NVIDIA H20"
    assert MP_CONTEXT.get_start_method() == "forkserver"
    accepted = load_accepted()
    assert GuardEval.constructions == 2
    source_and_state_oracle(accepted)
    exhaustive_pixel_oracle()
    sampled_crop_oracle()
    direct_randaugment_oracle(accepted)
    worker_transition_oracle()
    print("SEMANTICS PASS")


def make_real_loader(module, fill, active_value, seed):
    torch.manual_seed(seed)
    active = MP_CONTEXT.Value("b", active_value, lock=False)
    if fill == CANDIDATE_FILL:
        transform = train.make_train_transform(active)
    else:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                train.EarlyRandAugment(active),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
            ]
        )
    loader = DataLoader(
        GuardCIFAR10(prepare.DATASET_DIR, train=True, transform=transform),
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
    return loader


def loader_epoch(loader, delay):
    started = time.perf_counter()
    batches = examples = 0
    for inputs, targets in loader:
        assert inputs.shape == (256, 3, 32, 32) and targets.shape == (256,)
        assert torch.isfinite(inputs).all()
        if delay:
            time.sleep(CONSUMER_SECONDS)
        batches += 1
        examples += len(targets)
    elapsed = time.perf_counter() - started
    assert batches == 195 and examples == 49_920
    return elapsed


def loader_occurrence(module, fill, active, seed):
    loader = make_real_loader(module, fill, active, seed)
    loader_epoch(loader, True)
    service = [loader_epoch(loader, False) for _ in range(3)]
    overlap = [loader_epoch(loader, True) for _ in range(3)]
    shutdown_loader(loader)
    return service, overlap


def summarize(values):
    return {
        "values": values,
        "median": statistics.median(values),
        "cv": statistics.pstdev(values) / statistics.mean(values),
    }


def loader_timing(accepted):
    schedule = [("A", 1), ("C", 1), ("C", 0), ("A", 0), ("A", 0), ("C", 0), ("C", 1), ("A", 1)]
    seeds = [46_000, 46_000, 46_001, 46_001, 46_002, 46_002, 46_003, 46_003]
    values = {mode: {phase: {kind: [] for kind in ("service", "overlap")} for phase in ("active", "inactive")} for mode in ("A", "C")}
    for (mode, active), seed in zip(schedule, seeds):
        module = accepted if mode == "A" else train
        fill = ACCEPTED_FILL if mode == "A" else CANDIDATE_FILL
        service, overlap = loader_occurrence(module, fill, active, seed)
        phase = "active" if active else "inactive"
        values[mode][phase]["service"].extend(service)
        values[mode][phase]["overlap"].extend(overlap)
        print(f"loader occurrence={mode}{phase[0].upper()} seed={seed} service={service} overlap={overlap}")
    summaries = {
        mode: {
            phase: {kind: summarize(kind_values) for kind, kind_values in kinds.items()}
            for phase, kinds in phases.items()
        }
        for mode, phases in values.items()
    }
    for mode in ("A", "C"):
        summaries[mode]["weighted_overlap"] = (
            0.65 * summaries[mode]["active"]["overlap"]["median"]
            + 0.35 * summaries[mode]["inactive"]["overlap"]["median"]
        )
    projected_wall = 343.9 + max(0.0, summaries["C"]["weighted_overlap"] - summaries["A"]["weighted_overlap"]) * 130
    print(json.dumps({"loader_summaries": summaries, "projected_wall": projected_wall}, sort_keys=True), flush=True)
    for mode in ("A", "C"):
        for phase in ("active", "inactive"):
            for kind in ("service", "overlap"):
                assert summaries[mode][phase][kind]["cv"] <= 0.05
    for phase in ("active", "inactive"):
        for kind in ("service", "overlap"):
            candidate = summaries["C"][phase][kind]
            accepted_summary = summaries["A"][phase][kind]
            assert candidate["median"] <= 1.05 * accepted_summary["median"]
            assert max(candidate["values"]) <= 1.10 * accepted_summary["median"]
    assert projected_wall < 500
    print("LOADER TIMING PASS")


def timing_step(module, model, optimizer, host_inputs, host_targets, distribution, regime):
    inputs = host_inputs.cuda(non_blocking=True)
    targets = host_targets.cuda(non_blocking=True)
    for group in optimizer.param_groups:
        group["lr"] = 0.037
    optimizer.zero_grad(set_to_none=True)
    if regime == "early":
        mixed, targets_a, targets_b, mix = module.mixup_batch(inputs, targets, distribution)
        logits = model(mixed)
        loss = mix * F.cross_entropy(logits, targets_a) + (1 - mix) * F.cross_entropy(logits, targets_b)
    else:
        loss = F.cross_entropy(model(inputs), targets)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite timing loss")
    loss.backward()
    optimizer.step()


def run_window(module, state, window_rng, host_inputs, host_targets, regime, steps, peak=False):
    model = module.WideResNet(module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES)
    model.load_state_dict(state)
    model = model.cuda().train()
    optimizer = optimizer_for(module, model, lr=0.037)
    distribution = torch.distributions.Beta(torch.tensor(module.MIXUP_ALPHA, device="cuda"), torch.tensor(module.MIXUP_ALPHA, device="cuda"))
    restore_rng(window_rng)
    torch.cuda.synchronize()
    if peak:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        timing_step(module, model, optimizer, host_inputs, host_targets, distribution, regime)
    torch.cuda.synchronize()
    elapsed = 1000 * (time.perf_counter() - started) / steps
    allocated = torch.cuda.max_memory_allocated() if peak else 0
    del distribution, optimizer, model
    torch.cuda.empty_cache()
    return elapsed, allocated


def gpu_timing(accepted):
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    start = rng_state()
    restore_rng(start)
    model_a = accepted.WideResNet(accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES)
    after_a = rng_state()
    restore_rng(start)
    model_c = train.WideResNet(train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES)
    after_c = rng_state()
    assert torch.equal(after_a[0], after_c[0]) and torch.equal(after_a[1], after_c[1])
    states = {"A": clone_state(model_a), "C": clone_state(model_c)}
    del model_a, model_c
    host_inputs = torch.linspace(-1, 1, 256 * 3 * 32 * 32).reshape(256, 3, 32, 32).pin_memory()
    host_targets = (torch.arange(256) % 10).pin_memory()
    torch.manual_seed(46_505); torch.cuda.manual_seed(46_505)
    window_rng = rng_state()
    modules = {"A": accepted, "C": train}
    for mode, regime in (("A", "early"), ("C", "early"), ("A", "hard"), ("C", "hard")):
        run_window(modules[mode], states[mode], window_rng, host_inputs, host_targets, regime, 20)
    schedule = [("A", "early"), ("C", "early"), ("A", "hard"), ("C", "hard"), ("C", "hard"), ("A", "hard"), ("C", "early"), ("A", "early")]
    windows = {regime: {mode: [] for mode in ("A", "C")} for regime in ("early", "hard")}
    pairs = {regime: [] for regime in ("early", "hard")}
    candidate_peak = 0
    for cycle in range(2):
        block = []
        for mode, regime in schedule:
            value, peak = run_window(modules[mode], states[mode], window_rng, host_inputs, host_targets, regime, 50, mode == "C")
            candidate_peak = max(candidate_peak, peak)
            windows[regime][mode].append(value)
            block.append((mode, regime, value))
            print(f"gpu cycle={cycle} arm={mode} regime={regime} ms={value:.9f}")
        pairs["early"].extend([(block[0][2], block[1][2]), (block[7][2], block[6][2])])
        pairs["hard"].extend([(block[2][2], block[3][2]), (block[5][2], block[4][2])])
    for regime in ("early", "hard"):
        ratios = [candidate / accepted_ms for accepted_ms, candidate in pairs[regime]]
        ratio_cv = statistics.pstdev(ratios) / statistics.mean(ratios)
        for mode in ("A", "C"):
            cv = statistics.pstdev(windows[regime][mode]) / statistics.mean(windows[regime][mode])
            print(f"gpu_summary regime={regime} arm={mode} values={windows[regime][mode]} cv={cv:.9f}")
            assert cv <= 0.05
        print(f"gpu_ratio regime={regime} values={ratios} cv={ratio_cv:.9f}")
        assert ratio_cv <= 0.01
    retentions = []
    for index in range(4):
        ae, ce = pairs["early"][index]
        ah, ch = pairs["hard"][index]
        retentions.append((0.65 / ce + 0.35 / ch) / (0.65 / ae + 0.35 / ah))
    median = statistics.median(retentions)
    projected = 130.304 * median
    peak_mb = candidate_peak / 1024 / 1024
    print(f"gpu_gate retentions={retentions} median={median:.9f} projected={projected:.6f} peak_mb={peak_mb:.3f}")
    assert all(value >= RETENTION_FLOOR for value in retentions)
    assert projected >= 127 and peak_mb < 2048
    print("GPU TIMING PASS")


def timing():
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "NVIDIA H20"
    accepted = load_accepted()
    loader_timing(accepted)
    gpu_timing(accepted)
    print("TIMING PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("semantics", "timing"))
    args = parser.parse_args()
    semantics() if args.mode == "semantics" else timing()
