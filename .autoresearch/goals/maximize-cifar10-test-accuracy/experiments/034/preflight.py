import argparse
import gc
import hashlib
import inspect
import json
import math
import multiprocessing
import os
import secrets
import statistics
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[5]
ARTIFACT_DIR = Path(__file__).resolve().parent
THROUGHPUT_PATH = ARTIFACT_DIR / "throughput.json"
BASE_COMMIT = "67c8e98"
DEVICE = torch.device("cuda")
MP_CONTEXT = multiprocessing.get_context()
ORIGINAL_CIFAR10 = datasets.CIFAR10


class GuardEval:
    constructions = 0

    def __init__(self):
        type(self).constructions += 1

    def evaluate(self, model, device):
        raise AssertionError("preflight may not evaluate")


class GuardedCIFAR10(ORIGINAL_CIFAR10):
    def __init__(self, *args, **kwargs):
        train_split = kwargs.get("train", args[1] if len(args) > 1 else True)
        if not train_split:
            raise AssertionError("preflight may not construct test data")
        super().__init__(*args, **kwargs)


sys.path.insert(0, str(ROOT))
import prepare


prepare.Eval = GuardEval
datasets.CIFAR10 = GuardedCIFAR10
import train


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
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return optim.SGD(
        [
            {"params": decay, "weight_decay": module.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=module.MIN_LR,
        momentum=module.MOMENTUM,
        nesterov=True,
    )


def optimizer_signature(optimizer, model, omit_lr=False):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    signature = []
    for group in optimizer.param_groups:
        values = {
            key: value
            for key, value in group.items()
            if key != "params" and (not omit_lr or key != "lr")
        }
        values["params"] = [names[id(parameter)] for parameter in group["params"]]
        signature.append(values)
    return signature


def assert_state_equal(left, right, label):
    assert left.keys() == right.keys(), label
    for name in left:
        assert torch.equal(left[name], right[name]), f"{label}.{name}"


def shutdown_loader(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
    del loader
    gc.collect()


def current_provenance():
    diff = subprocess.check_output(
        ["git", "diff", "--binary", BASE_COMMIT, "--", "train.py"],
        cwd=ROOT,
    )
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    return {
        "accepted_commit": BASE_COMMIT,
        "candidate_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "constants": {
            "batch_size": train.BATCH_SIZE,
            "lr": train.LR,
            "min_lr": train.MIN_LR,
            "max_steps": train.MAX_STEPS,
        },
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": gpu_uuid,
    }


def static_scope_checks(accepted):
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_COMMIT, "--"], cwd=ROOT, text=True
    ).splitlines()
    assert changed == ["train.py"], changed
    subprocess.run(
        ["git", "diff", "--exit-code", BASE_COMMIT, "--", "prepare.py"],
        cwd=ROOT,
        check=True,
    )
    diff = subprocess.check_output(
        ["git", "diff", "--unified=0", BASE_COMMIT, "--", "train.py"],
        cwd=ROOT,
        text=True,
    )
    removed = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    assert removed == [
        "-BATCH_SIZE = 256",
        "-LR = 0.2",
        "-MIN_LR = 0.002",
        "-MAX_STEPS = 64000",
    ], removed
    assert added == [
        "+BATCH_SIZE = 512",
        "+LR = 0.4",
        "+MIN_LR = 0.004",
        "+MAX_STEPS = 32000",
    ], added
    assert train.BATCH_SIZE == 512 and accepted.BATCH_SIZE == 256
    assert train.LR == 0.4 and accepted.LR == 0.2
    assert train.MIN_LR == 0.004 and accepted.MIN_LR == 0.002
    assert train.MAX_STEPS == 32_000 and accepted.MAX_STEPS == 64_000
    assert train.BATCH_SIZE * train.MAX_STEPS == accepted.BATCH_SIZE * accepted.MAX_STEPS
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
    for name in (
        "EarlyRandAugment",
        "make_train_transform",
        "PreActBlock",
        "WideResNet",
        "learning_rate",
        "mixup_batch",
    ):
        assert inspect.getsource(getattr(train, name)) in accepted.__source__, name


def construction_checks(accepted):
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
    assert_state_equal(accepted_model.state_dict(), candidate_model.state_dict(), "model")
    assert sum(parameter.numel() for parameter in candidate_model.parameters()) == 987_098
    assert all(parameter.dtype == torch.float32 for parameter in candidate_model.parameters())

    accepted_optimizer = optimizer_for(accepted, accepted_model)
    candidate_optimizer = optimizer_for(train, candidate_model)
    assert optimizer_signature(
        accepted_optimizer, accepted_model, omit_lr=True
    ) == optimizer_signature(candidate_optimizer, candidate_model, omit_lr=True)
    assert all(
        candidate_group["lr"] == 2.0 * accepted_group["lr"]
        for accepted_group, candidate_group in zip(
            accepted_optimizer.param_groups, candidate_optimizer.param_groups
        )
    )
    for progress in (0.0, 0.025, 0.05, 0.5, 0.65, 1.0):
        accepted_lr = accepted.learning_rate(progress * prepare.TIME_BUDGET_S)
        candidate_lr = train.learning_rate(progress * prepare.TIME_BUDGET_S)
        assert candidate_lr == 2.0 * accepted_lr, progress
    assert tuple(type(op).__name__ for op in accepted_transform.transforms) == tuple(
        type(op).__name__ for op in candidate_transform.transforms
    )


def make_real_loader(module, batch_size, active_value, shuffle=True):
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
        shuffle=shuffle,
        num_workers=prepare.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context=MP_CONTEXT,
    )
    return loader, active


def loader_shape_and_update_checks():
    loader, _ = make_real_loader(train, train.BATCH_SIZE, 1)
    assert len(loader) == 97
    shutdown_loader(loader)
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(34_010)
    host_inputs = torch.randn(512, 3, 32, 32, generator=generator).pin_memory()
    host_targets = (torch.arange(512) % train.NUM_CLASSES).pin_memory()
    model = train.WideResNet(
        train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES
    ).to(DEVICE).train()
    optimizer = optimizer_for(train, model)
    distribution = torch.distributions.Beta(
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
    )
    inputs = host_inputs.to(DEVICE, non_blocking=True)
    targets = host_targets.to(DEVICE, non_blocking=True)
    for mixup in (True, False):
        optimizer.zero_grad(set_to_none=True)
        if mixup:
            mixed, target_a, target_b, coefficient = train.mixup_batch(
                inputs, targets, distribution
            )
            assert coefficient.ndim == 0 and torch.isfinite(coefficient)
            assert target_a.shape == target_b.shape == targets.shape
            outputs = model(mixed)
            loss = coefficient * F.cross_entropy(outputs, target_a) + (
                1.0 - coefficient
            ) * F.cross_entropy(outputs, target_b)
        else:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
        assert outputs.shape == (512, 10) and torch.isfinite(outputs).all()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    assert peak < 90_000, peak
    return peak


def state_digest(state):
    return int.from_bytes(hashlib.sha256(state.numpy().tobytes()).digest()[:4], "little")


class TraceDataset(Dataset):
    def __init__(self, active, with_randaugment):
        self.dataset = ORIGINAL_CIFAR10(
            prepare.DATASET_DIR, train=True, download=False, transform=None
        )
        self.active = active
        self.randaugment = train.EarlyRandAugment(active) if with_randaugment else None
        self.mean = (0.4914, 0.4822, 0.4465)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = TF.pad(image, [4], fill=0, padding_mode="constant")
        i, j, height, width = transforms.RandomCrop.get_params(image, (32, 32))
        image = TF.crop(image, i, j, height, width)
        flipped = bool(torch.rand(1) < 0.5)
        if flipped:
            image = TF.hflip(image)
        active = int(self.active.value)
        private_before = 0
        private_after = 0
        main_restored = 1
        if self.randaugment is not None:
            if self.randaugment._randaugment_rng_state is not None:
                private_before = state_digest(self.randaugment._randaugment_rng_state)
            main_before = torch.random.get_rng_state().clone()
            image = self.randaugment(image)
            main_restored = int(torch.equal(torch.random.get_rng_state(), main_before))
            if self.randaugment._randaugment_rng_state is not None:
                private_after = state_digest(self.randaugment._randaugment_rng_state)
        image = TF.normalize(TF.to_tensor(image), self.mean, (1, 1, 1))
        worker = get_worker_info()
        return (
            image,
            target,
            index,
            worker.id if worker is not None else -1,
            i,
            j,
            int(flipped),
            active,
            private_before,
            private_after,
            main_restored,
        )


def trace_arm(with_randaugment):
    torch.manual_seed(42)
    active = MP_CONTEXT.Value("b", 1, lock=False)
    dataset = TraceDataset(active, with_randaugment)
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=prepare.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context=MP_CONTEXT,
    )
    train.WideResNet(train.STAGE_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES)
    active_trace = []
    for batch in loader:
        fields = batch[1:]
        for row in zip(*(field.tolist() for field in fields)):
            active_trace.append(tuple(row))
    active.value = 0
    tail_trace = []
    for batch in loader:
        images = batch[0]
        fields = batch[1:]
        for position, row in enumerate(zip(*(field.tolist() for field in fields))):
            digest = hashlib.sha256(images[position].numpy().tobytes()).hexdigest()
            tail_trace.append((tuple(row), digest))
    shutdown_loader(loader)
    return active_trace, tail_trace


def worker_replay_checks():
    base_active, base_tail = trace_arm(False)
    augmented_active, augmented_tail = trace_arm(True)
    assert len(base_active) == len(augmented_active) == 49_664
    assert len(base_tail) == len(augmented_tail) == 49_664
    decision_positions = (0, 1, 2, 3, 4, 5)
    for base, augmented in zip(base_active, augmented_active):
        assert tuple(base[index] for index in decision_positions) == tuple(
            augmented[index] for index in decision_positions
        )
        assert base[6] == augmented[6] == 1
        assert augmented[9] == 1
    for (base, base_hash), (augmented, augmented_hash) in zip(base_tail, augmented_tail):
        assert tuple(base[index] for index in (*decision_positions, 6, 9)) == tuple(
            augmented[index] for index in (*decision_positions, 6, 9)
        )
        assert base_hash == augmented_hash
        assert augmented[6] == 0
        assert augmented[7] == augmented[8]
        assert augmented[9] == 1
    return {"active_samples": len(augmented_active), "tail_samples": len(augmented_tail)}


def semantic_checks():
    accepted = load_accepted()
    assert GuardEval.constructions == 2
    assert MP_CONTEXT.get_start_method() == "forkserver"
    static_scope_checks(accepted)
    construction_checks(accepted)
    peak = loader_shape_and_update_checks()
    traces = worker_replay_checks()
    main_source = inspect.getsource(train.main)
    assert main_source.count("randaugment_active.value = 0") == 1
    assert "and not budget_exhausted" in main_source
    assert main_source.count("evaluator.evaluate(model, device)") == 1
    payload = {
        "params": 987_098,
        "batch_size": 512,
        "batches": 97,
        "examples": 49_664,
        "peak_vram_mb": peak,
        **traces,
    }
    print(json.dumps(payload, sort_keys=True))
    print("SEMANTICS PASS")


def timed_step(module, model, optimizer, host_inputs, host_targets, distribution, mixup):
    started = time.perf_counter()
    inputs = host_inputs.to(DEVICE, non_blocking=True)
    targets = host_targets.to(DEVICE, non_blocking=True)
    for group in optimizer.param_groups:
        group["lr"] = module.MIN_LR
    optimizer.zero_grad(set_to_none=True)
    if mixup:
        mixed, target_a, target_b, coefficient = module.mixup_batch(
            inputs, targets, distribution
        )
        outputs = model(mixed)
        loss = coefficient * F.cross_entropy(outputs, target_a) + (
            1.0 - coefficient
        ) * F.cross_entropy(outputs, target_b)
    else:
        loss = F.cross_entropy(model(inputs), targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def timing_window(module, batch_size, mixup, replicate):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = module.WideResNet(
        module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES
    ).to(DEVICE).train()
    optimizer = optimizer_for(module, model)
    generator = torch.Generator().manual_seed(34_100 + replicate)
    host_inputs = torch.randn(
        batch_size, 3, 32, 32, generator=generator
    ).pin_memory()
    host_targets = (torch.arange(batch_size) % module.NUM_CLASSES).pin_memory()
    distribution = torch.distributions.Beta(
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(module.MIXUP_ALPHA, device=DEVICE),
    )
    torch.cuda.manual_seed(34_200 + replicate)
    for _ in range(20):
        timed_step(
            module, model, optimizer, host_inputs, host_targets, distribution, mixup
        )
    started = time.perf_counter()
    for _ in range(50):
        timed_step(
            module, model, optimizer, host_inputs, host_targets, distribution, mixup
        )
    elapsed = time.perf_counter() - started
    del model, optimizer, host_inputs, host_targets, distribution
    torch.cuda.empty_cache()
    return elapsed / 50


def population_cv(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else float("inf")


def publish_payload(payload):
    if THROUGHPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite {THROUGHPUT_PATH}")
    nonce = payload["provenance"]["session_nonce"]
    temporary = THROUGHPUT_PATH.with_name(f".throughput-{nonce}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, THROUGHPUT_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def throughput_checks():
    if THROUGHPUT_PATH.exists():
        raise FileExistsError(f"Refusing to remeasure with {THROUGHPUT_PATH} present")
    accepted = load_accepted()
    torch.cuda.reset_peak_memory_stats()
    results = {}
    for mixup, regime in ((True, "mixup"), (False, "hard")):
        windows = {"accepted": [], "candidate": []}
        for replicate in range(3):
            order = (
                ("accepted", "candidate")
                if replicate % 2 == 0
                else ("candidate", "accepted")
            )
            for kind in order:
                module = accepted if kind == "accepted" else train
                batch_size = 256 if kind == "accepted" else 512
                windows[kind].append(
                    timing_window(module, batch_size, mixup, replicate)
                )
        medians = {
            kind: statistics.median(values) for kind, values in windows.items()
        }
        cvs = {kind: population_cv(values) for kind, values in windows.items()}
        results[regime] = {
            "windows_s": windows,
            "medians_s": medians,
            "cvs": cvs,
            "window_count": {kind: len(values) for kind, values in windows.items()},
        }

    accepted_rate = (
        0.65 * 256 / results["mixup"]["medians_s"]["accepted"]
        + 0.35 * 256 / results["hard"]["medians_s"]["accepted"]
    )
    candidate_rate = (
        0.65 * 512 / results["mixup"]["medians_s"]["candidate"]
        + 0.35 * 512 / results["hard"]["medians_s"]["candidate"]
    )
    retention = candidate_rate / accepted_rate
    projected_passes = 133.00736 * retention
    projected_steps = projected_passes * 50_000 / 512
    provenance = current_provenance() | {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_nonce": secrets.token_hex(16),
    }
    payload = {
        "provenance": provenance,
        "results": results,
        "accepted_rate_images_s": accepted_rate,
        "candidate_rate_images_s": candidate_rate,
        "retention": retention,
        "projected_passes": projected_passes,
        "projected_steps": projected_steps,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "pass_status": False,
    }
    print(json.dumps(payload, sort_keys=True))
    assert all(
        math.isfinite(value)
        for regime in results.values()
        for values in regime["windows_s"].values()
        for value in values
    )
    assert all(
        value <= 0.05
        for regime in results.values()
        for value in regime["cvs"].values()
    ), results
    assert retention >= 1.10, retention
    assert projected_passes >= 146.308096, projected_passes
    assert projected_steps >= 14_287, projected_steps
    assert payload["peak_vram_mb"] < 90_000
    payload["pass_status"] = True
    publish_payload(payload)
    print("THROUGHPUT PASS")


def validate_saved_payload():
    if not THROUGHPUT_PATH.is_file():
        raise FileNotFoundError(THROUGHPUT_PATH)
    payload = json.loads(THROUGHPUT_PATH.read_text())
    assert payload["pass_status"] is True
    assert payload["provenance"].keys() == (
        current_provenance()
        | {"created_utc": "", "session_nonce": ""}
    ).keys()
    for key, value in current_provenance().items():
        assert payload["provenance"][key] == value, key
    assert payload["provenance"]["created_utc"]
    assert len(payload["provenance"]["session_nonce"]) == 32
    for regime in ("mixup", "hard"):
        for kind in ("accepted", "candidate"):
            assert payload["results"][regime]["window_count"][kind] == 3
            assert len(payload["results"][regime]["windows_s"][kind]) == 3
    assert payload["retention"] >= 1.10
    assert payload["projected_passes"] >= 146.308096
    return payload


def paced_epoch(loader, batch_size, consumer_seconds):
    started = time.perf_counter()
    batches = 0
    examples = 0
    for inputs, targets in loader:
        assert inputs.shape == (batch_size, 3, 32, 32)
        assert targets.shape == (batch_size,)
        assert torch.isfinite(inputs).all()
        assert torch.all((targets >= 0) & (targets < train.NUM_CLASSES))
        time.sleep(consumer_seconds)
        batches += 1
        examples += len(targets)
    assert batches == 50_000 // batch_size
    assert examples == batches * batch_size
    return time.perf_counter() - started


def loader_arm(module, batch_size, active_value, consumer_seconds):
    loader, active = make_real_loader(module, batch_size, active_value)
    paced_epoch(loader, batch_size, consumer_seconds)
    values = [paced_epoch(loader, batch_size, consumer_seconds) for _ in range(3)]
    initial = active.value
    active.value = 0
    assert initial == active_value and active.value == 0
    shutdown_loader(loader)
    return values


def loader_timing_checks():
    payload = validate_saved_payload()
    accepted = load_accepted()
    config = {
        "accepted": {
            "module": accepted,
            "batch": 256,
            "active_s": payload["results"]["mixup"]["medians_s"]["accepted"],
            "hard_s": payload["results"]["hard"]["medians_s"]["accepted"],
        },
        "candidate": {
            "module": train,
            "batch": 512,
            "active_s": payload["results"]["mixup"]["medians_s"]["candidate"],
            "hard_s": payload["results"]["hard"]["medians_s"]["candidate"],
        },
    }
    observations = {
        kind: {"active": [], "inactive": []} for kind in config
    }
    order = ["accepted", "candidate", "candidate", "accepted"]
    for kind in order:
        item = config[kind]
        observations[kind]["active"].extend(
            loader_arm(item["module"], item["batch"], 1, item["active_s"])
        )
        observations[kind]["inactive"].extend(
            loader_arm(item["module"], item["batch"], 0, item["hard_s"])
        )

    summaries = {}
    for kind, item in config.items():
        batches = 50_000 // item["batch"]
        active_values = observations[kind]["active"]
        inactive_values = observations[kind]["inactive"]
        active_median = statistics.median(active_values)
        inactive_median = statistics.median(inactive_values)
        summaries[kind] = {
            "active_values_s": active_values,
            "inactive_values_s": inactive_values,
            "active_median_s": active_median,
            "inactive_median_s": inactive_median,
            "active_cv": population_cv(active_values),
            "inactive_cv": population_cv(inactive_values),
            "active_stall_s": max(0.0, active_median - batches * item["active_s"]),
            "inactive_stall_s": max(0.0, inactive_median - batches * item["hard_s"]),
        }

    accepted_active_epochs = 195.0 / (195 * config["accepted"]["active_s"])
    accepted_hard_epochs = 105.0 / (195 * config["accepted"]["hard_s"])
    candidate_active_epochs = 195.0 / (97 * config["candidate"]["active_s"])
    candidate_hard_epochs = 105.0 / (97 * config["candidate"]["hard_s"])
    candidate_epochs = candidate_active_epochs + candidate_hard_epochs
    candidate_eval_count = math.floor(candidate_epochs / 5) + 1
    accepted_stall_total = (
        accepted_active_epochs * summaries["accepted"]["active_stall_s"]
        + accepted_hard_epochs * summaries["accepted"]["inactive_stall_s"]
    )
    candidate_stall_total = (
        candidate_active_epochs * summaries["candidate"]["active_stall_s"]
        + candidate_hard_epochs * summaries["candidate"]["inactive_stall_s"]
    )
    stall_delta = candidate_stall_total - accepted_stall_total
    evaluation_delta = max(0.0, 44.2 * (candidate_eval_count / 27 - 1))
    differential = 345.3 + max(0.0, stall_delta) + evaluation_delta
    absolute = (
        1.1
        + 300.0
        + 44.2 * candidate_eval_count / 27
        + candidate_stall_total
    )
    result = {
        "order": order,
        "summaries": summaries,
        "accepted_active_epochs": accepted_active_epochs,
        "accepted_hard_epochs": accepted_hard_epochs,
        "candidate_active_epochs": candidate_active_epochs,
        "candidate_hard_epochs": candidate_hard_epochs,
        "candidate_epochs": candidate_epochs,
        "candidate_eval_count": candidate_eval_count,
        "accepted_stall_total_s": accepted_stall_total,
        "candidate_stall_total_s": candidate_stall_total,
        "stall_delta_s": stall_delta,
        "differential_wall_s": differential,
        "absolute_wall_s": absolute,
    }
    print(json.dumps(result, sort_keys=True))
    assert all(
        math.isfinite(value)
        for item in summaries.values()
        for phase in ("active_values_s", "inactive_values_s")
        for value in item[phase]
    )
    assert all(
        item[metric] <= 0.05
        for item in summaries.values()
        for metric in ("active_cv", "inactive_cv")
    ), summaries
    assert differential < 500 and absolute < 500
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
