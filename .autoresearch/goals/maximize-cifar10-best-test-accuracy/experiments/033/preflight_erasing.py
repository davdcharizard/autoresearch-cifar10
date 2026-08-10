import argparse
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate, get_worker_info
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = EXPERIMENT_DIR / "exact-corpus.pt"
CORPUS_DIR = EXPERIMENT_DIR / "exact-corpus"
REPORT_PATH = EXPERIMENT_DIR / "preflight-report.json"
MEAN = (0.4914, 0.4822, 0.4465)
SCALE = (0.02, 0.10)
RATIO = (0.3, 3.3)
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


THRESHOLDS = {
    "erased_fraction": [0.23, 0.27],
    "live_erased_fraction": [0.235, 0.265],
    "placement_success": 0.995,
    "achieved_area": [0.010, 0.110],
    "conditional_mean_area": [0.045, 0.075],
    "unconditional_mean_area": [0.011, 0.019],
    "effective_area_max": 0.22,
    "candidate_class_share": 0.95,
    "per_step_ratio": 1.5,
    "terminal_ema_ratio": 1.25,
    "live_cutmix_fraction": [0.48, 0.52],
    "live_throughput_ratio": 0.80,
    "live_batches_per_second": 140.0,
    "live_wait_median_ms": 0.5,
    "live_wait_p95_ms": 1.5,
    "weak_rebuild_seconds": 5.0,
}


def sha256_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


class CaptureRNG:
    def __call__(self, image):
        return image, torch.get_rng_state().clone()


def source_collate(batch):
    (inputs, image_states), targets = default_collate(batch)
    worker = get_worker_info()
    return (
        inputs,
        targets,
        image_states,
        torch.get_rng_state().clone(),
        worker.id,
        torch.initial_seed(),
    )


def instrumented_collate(batch):
    inputs, targets = default_collate(batch)
    mask = (inputs == 0).all(dim=1)
    worker = get_worker_info()
    erased_examples = int(mask.flatten(1).any(1).sum())
    erased_pixels = int(mask.sum())
    achieved = [
        int(value) / (32 * 32) for value in mask.flatten(1).sum(1).tolist() if value > 0
    ]
    with torch.random.fork_rng(devices=[]):
        mixed = torch.rand(()).item() < train.CUTMIX_PROBABILITY
        if mixed:
            inputs, targets = train.cutmix(inputs, targets)
    return (
        inputs,
        targets,
        erased_examples,
        erased_pixels,
        achieved,
        worker.id,
        mixed,
    )


def control_collate(batch):
    inputs, targets = default_collate(batch)
    worker = get_worker_info()
    with torch.random.fork_rng(devices=[]):
        mixed = torch.rand(()).item() < train.CUTMIX_PROBABILITY
        if mixed:
            inputs, targets = train.cutmix(inputs, targets)
    return inputs, targets, worker.id, mixed


def strong_source_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            CaptureRNG(),
        ]
    )


def strong_control_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, (1, 1, 1)),
        ]
    )


def strong_candidate_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            train.RNGNeutralRandomErasing(
                p=0.25,
                scale=SCALE,
                ratio=RATIO,
                value=MEAN,
                inplace=False,
            ),
            transforms.Normalize(MEAN, (1, 1, 1)),
        ]
    )


def weak_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, (1, 1, 1)),
        ]
    )


def stop_loader(loader):
    workers = train.shutdown_train_loader(loader)
    if len(workers) != train.NUM_WORKERS:
        raise RuntimeError(f"stopped {len(workers)} workers")
    return workers


def semantic_gate():
    eraser = train.RNGNeutralRandomErasing(
        p=1.0, scale=SCALE, ratio=RATIO, value=MEAN, inplace=False
    )
    no_op = train.RNGNeutralRandomErasing(
        p=0.0, scale=SCALE, ratio=RATIO, value=MEAN, inplace=False
    )
    image = torch.linspace(0.001, 0.999, 3 * 32 * 32).reshape(3, 32, 32)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone()
    no_op_output = no_op(image)
    if not torch.equal(no_op_output, image):
        raise RuntimeError("forced no-op changed the image")
    state = torch.get_rng_state().clone()
    first = eraser(image)
    torch.set_rng_state(state)
    second = eraser(image)
    if not torch.equal(first, second):
        raise RuntimeError("saved state did not reproduce erasing")
    mask = (first == torch.tensor(MEAN)[:, None, None]).all(0)
    if not mask.any() or not torch.equal(first[:, ~mask], image[:, ~mask]):
        raise RuntimeError("mask fill/outside semantics failed")
    normalized = transforms.Normalize(MEAN, (1, 1, 1))(first)
    if not torch.equal(normalized[:, mask], torch.zeros_like(normalized[:, mask])):
        raise RuntimeError("mean fill did not normalize to exact zero")
    torch.set_rng_state(before_cpu)
    if not torch.equal(torch.get_rng_state(), before_cpu):
        raise RuntimeError("CPU state restoration failed")
    if not torch.equal(torch.cuda.get_rng_state(), before_cuda):
        raise RuntimeError("CUDA RNG changed")
    torch.set_rng_state(state)
    _ = eraser(image)
    downstream_candidate = torch.rand(16)
    torch.set_rng_state(state)
    downstream_control = torch.rand(16)
    if not torch.equal(downstream_candidate, downstream_control):
        raise RuntimeError("eraser advanced downstream accepted RNG")
    pickle.loads(pickle.dumps(eraser))
    production = train.RNGNeutralRandomErasing(
        p=0.25, scale=SCALE, ratio=RATIO, value=MEAN, inplace=False
    ).transform
    if (
        production.p != 0.25
        or production.scale != SCALE
        or production.ratio != RATIO
        or tuple(production.value) != MEAN
        or production.inplace
    ):
        raise RuntimeError("production policy configuration mismatch")
    model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER)
    params = sum(parameter.numel() for parameter in model.parameters())
    if params != 1_073_962:
        raise RuntimeError(f"parameter count mismatch: {params}")
    return {
        "forced_mask_pixels": int(mask.sum()),
        "forced_area": float(mask.float().mean()),
        "cpu_rng_neutral": True,
        "cuda_rng_neutral": True,
        "downstream_draws_equal": True,
        "picklable": True,
        "num_params": params,
    }


def erase_batch(raw_inputs, image_states):
    eraser = train.RNGNeutralRandomErasing(
        p=0.25, scale=SCALE, ratio=RATIO, value=MEAN, inplace=False
    )
    outputs = []
    masks = []
    selected = []
    placements = []
    achieved = []
    for image_index, (image, state) in enumerate(
        zip(raw_inputs, image_states, strict=True)
    ):
        state = state.contiguous().clone()
        with torch.random.fork_rng(devices=[]):
            try:
                torch.set_rng_state(state)
            except RuntimeError as error:
                raise RuntimeError(
                    f"invalid image state {image_index}: shape={tuple(state.shape)} "
                    f"dtype={state.dtype} bytes={state[:24].tolist()}"
                ) from error
            gate = float(torch.rand(1))
            params = None
            if gate < 0.25:
                params = transforms.RandomErasing.get_params(
                    image, SCALE, RATIO, list(MEAN)
                )
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(state)
            output = eraser(image)
        mask = (output == torch.tensor(MEAN)[:, None, None]).all(0)
        was_selected = gate < 0.25
        selected.append(was_selected)
        if was_selected:
            i, j, h, w, _value = params
            expected = torch.zeros((32, 32), dtype=torch.bool)
            if h < 32 and w < 32:
                expected[i : i + h, j : j + w] = True
            success = bool(expected.any())
            placements.append(success)
            if not torch.equal(mask, expected):
                raise RuntimeError("production mask differs from registered params")
            if success:
                area = float(mask.float().mean())
                achieved.append(area)
                if (
                    not THRESHOLDS["achieved_area"][0]
                    <= area
                    <= THRESHOLDS["achieved_area"][1]
                ):
                    raise RuntimeError(f"achieved area out of bounds: {area}")
        elif mask.any():
            raise RuntimeError("failed gate produced an erased mask")
        if not torch.equal(output[:, ~mask], image[:, ~mask]):
            raise RuntimeError("erasing changed pixels outside mask")
        outputs.append(output)
        masks.append(mask)
    return torch.stack(outputs), torch.stack(masks), selected, placements, achieved


def policy_kind(policy_state, inputs):
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(policy_state)
        gate = float(torch.rand(()))
        params = train.cutmix.make_params([inputs]) if gate < 0.5 else None
    return gate < 0.5, params


def apply_policy(inputs, targets, policy_state):
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(policy_state)
        if torch.rand(()).item() < train.CUTMIX_PROBABILITY:
            return (*train.cutmix(inputs, targets), "cutmix")
    return inputs, targets, "hard"


def create_corpus():
    CORPUS_DIR.mkdir(exist_ok=True)
    for stale in CORPUS_DIR.glob("*.pt"):
        stale.unlink()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    loader = train.make_train_loader(strong_source_transform(), source_collate)
    mirror = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER)
    mirror_optimizer = torch.optim.SGD(
        mirror.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    del mirror, mirror_optimizer
    iterator = iter(loader)
    records = []
    kind_counts = {"hard": 0, "cutmix": 0}
    erased_selected = 0
    placements = []
    achieved = []
    erased_pixels = 0
    final_max = 0.0
    worker_ids = set()
    source_ordinal = 0
    while min(kind_counts.values()) < 100 and source_ordinal < 260:
        raw, targets, image_states, policy_state, worker_id, worker_seed = next(
            iterator
        )
        # CPUGeneratorImpl cannot safely consume pinned shared-memory views here.
        raw = raw.contiguous().clone()
        targets = targets.contiguous().clone()
        image_states = image_states.contiguous().clone()
        policy_state = policy_state.contiguous().clone()
        source_ordinal += 1
        mixed, registered_params = policy_kind(policy_state, raw)
        kind = "cutmix" if mixed else "hard"
        if kind_counts[kind] >= 100:
            continue
        kind_counts[kind] += 1
        worker_ids.add(int(worker_id))
        candidate_raw, source_masks, selected, batch_placements, batch_achieved = (
            erase_batch(raw, image_states)
        )
        control_inputs = transforms.Normalize(MEAN, (1, 1, 1))(raw)
        candidate_inputs = transforms.Normalize(MEAN, (1, 1, 1))(candidate_raw)
        control_inputs, control_targets, control_kind = apply_policy(
            control_inputs, targets, policy_state
        )
        candidate_inputs, candidate_targets, candidate_kind = apply_policy(
            candidate_inputs, targets, policy_state
        )
        if control_kind != kind or candidate_kind != kind:
            raise RuntimeError("registered policy kind mismatch")
        if not torch.equal(control_targets, candidate_targets):
            raise RuntimeError("candidate changed labels or CutMix targets")
        final_masks = (candidate_inputs == 0).all(1)
        batch_final_max = float(final_masks.flatten(1).float().mean(1).max())
        final_max = max(final_max, batch_final_max)
        if batch_final_max > THRESHOLDS["effective_area_max"]:
            raise RuntimeError(f"effective erased area too large: {batch_final_max}")
        erased_selected += sum(selected)
        placements.extend(batch_placements)
        achieved.extend(batch_achieved)
        erased_pixels += int(source_masks.sum())
        ordinal = len(records)
        payload = {
            "ordinal": ordinal,
            "source_ordinal": source_ordinal,
            "source": raw.clone(),
            "image_states": image_states.clone(),
            "policy_state": policy_state.clone(),
            "control": control_inputs.clone(),
            "candidate": candidate_inputs.clone(),
            "targets": control_targets.clone(),
            "hard_targets": targets.clone(),
            "source_masks": source_masks.clone(),
            "kind": kind,
            "box": list(registered_params["box"]) if mixed else None,
            "lambda": registered_params["lam_adjusted"] if mixed else None,
            "worker_id": int(worker_id),
            "worker_seed": int(worker_seed),
            "hashes": {
                "source": sha256_tensor(raw),
                "control": sha256_tensor(control_inputs),
                "candidate": sha256_tensor(candidate_inputs),
                "targets": sha256_tensor(control_targets),
                "mask": sha256_tensor(source_masks),
            },
        }
        record_path = CORPUS_DIR / f"strong-{ordinal:03d}.pt"
        torch.save(payload, record_path)
        records.append(
            {
                "ordinal": ordinal,
                "source_ordinal": source_ordinal,
                "path": str(record_path),
                "file_sha256": sha256_file(record_path),
                "kind": kind,
                "box": list(registered_params["box"]) if mixed else None,
                "lambda": registered_params["lam_adjusted"] if mixed else None,
                "worker_id": int(worker_id),
                "worker_seed": int(worker_seed),
                "hashes": payload["hashes"],
            }
        )
    iterator = None
    strong_workers = stop_loader(loader)
    if kind_counts != {"hard": 100, "cutmix": 100}:
        raise RuntimeError(f"could not build balanced corpus: {kind_counts}")

    weak_loader = train.make_train_loader(weak_transform())
    weak_iterator = iter(weak_loader)
    weak_records = []
    for ordinal in range(64):
        inputs, targets = next(weak_iterator)
        payload = {
            "ordinal": ordinal,
            "inputs": inputs.clone(),
            "targets": targets.clone(),
            "hashes": {
                "inputs": sha256_tensor(inputs),
                "targets": sha256_tensor(targets),
            },
        }
        record_path = CORPUS_DIR / f"weak-{ordinal:03d}.pt"
        torch.save(payload, record_path)
        weak_records.append(
            {
                "ordinal": ordinal,
                "path": str(record_path),
                "file_sha256": sha256_file(record_path),
                "hashes": payload["hashes"],
            }
        )
    weak_iterator = None
    weak_workers = stop_loader(weak_loader)
    total_examples = len(records) * train.BATCH_SIZE
    erased_fraction = erased_selected / total_examples
    placement_success = sum(placements) / len(placements)
    conditional_mean = statistics.mean(achieved)
    unconditional_mean = erased_pixels / (total_examples * 32 * 32)
    geometry = {
        "source_batches_considered": source_ordinal,
        "selected_batches": len(records),
        "kind_counts": kind_counts,
        "erased_examples": erased_selected,
        "total_examples": total_examples,
        "erased_fraction": erased_fraction,
        "placement_success": placement_success,
        "achieved_min": min(achieved),
        "achieved_max": max(achieved),
        "conditional_mean_area": conditional_mean,
        "unconditional_mean_area": unconditional_mean,
        "final_effective_max": final_max,
        "worker_ids": sorted(worker_ids),
        "strong_workers_stopped": len(strong_workers),
        "weak_workers_stopped": len(weak_workers),
    }
    failures = []
    if (
        not THRESHOLDS["erased_fraction"][0]
        <= erased_fraction
        <= THRESHOLDS["erased_fraction"][1]
    ):
        failures.append(f"erased fraction {erased_fraction:.6f}")
    if placement_success < THRESHOLDS["placement_success"]:
        failures.append(f"placement success {placement_success:.6f}")
    if (
        not THRESHOLDS["conditional_mean_area"][0]
        <= conditional_mean
        <= THRESHOLDS["conditional_mean_area"][1]
    ):
        failures.append(f"conditional area {conditional_mean:.6f}")
    if (
        not THRESHOLDS["unconditional_mean_area"][0]
        <= unconditional_mean
        <= THRESHOLDS["unconditional_mean_area"][1]
    ):
        failures.append(f"unconditional area {unconditional_mean:.6f}")
    if worker_ids != set(range(train.NUM_WORKERS)):
        failures.append(f"incomplete source workers {sorted(worker_ids)}")
    if failures:
        raise RuntimeError("; ".join(failures))
    corpus = {
        "metadata": {
            "seed": 42,
            "policy": "p=.25 scale=.02-.10 ratio=.3-3.3 mean fill before normalize",
            "strong_transform": repr(strong_source_transform()),
            "weak_transform": repr(weak_transform()),
            "geometry": geometry,
        },
        "strong": records,
        "weak": weak_records,
    }
    temporary = CORPUS_PATH.with_suffix(".tmp")
    torch.save(corpus, temporary)
    temporary.replace(CORPUS_PATH)
    with CORPUS_PATH.open("rb") as handle:
        os.fsync(handle.fileno())
    return corpus


def all_finite(model, optimizer):
    values = list(model.parameters()) + list(model.buffers())
    for state in optimizer.state.values():
        values.extend(value for value in state.values() if torch.is_tensor(value))
    return all(torch.isfinite(value).all().item() for value in values)


def tensor_norm(tensors):
    return math.sqrt(sum(float(tensor.float().square().sum()) for tensor in tensors))


def run_trajectory(arm):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    corpus = torch.load(CORPUS_PATH, map_location="cpu", weights_only=False)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = (
        train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER)
        .cuda()
        .train()
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    metrics = []
    ema = {"strong": 0.0, "weak": 0.0}
    ema_steps = {"strong": 0, "weak": 0}
    beta = 0.95
    records = [("strong", metadata) for metadata in corpus["strong"]] + [
        ("weak", metadata) for metadata in corpus["weak"]
    ]
    for step, (phase, metadata) in enumerate(records, start=1):
        record = torch.load(metadata["path"], map_location="cpu", weights_only=False)
        if sha256_file(Path(metadata["path"])) != metadata["file_sha256"]:
            raise RuntimeError(f"corpus file hash mismatch at step {step}")
        if phase == "strong":
            cpu_inputs = record[arm]
            cpu_targets = record["targets"]
        else:
            cpu_inputs = record["inputs"]
            cpu_targets = record["targets"]
        inputs = cpu_inputs.cuda(non_blocking=True)
        targets = cpu_targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        grad_norm = tensor_norm([parameter.grad for parameter in model.parameters()])
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.step()
        torch.cuda.synchronize()
        update_norm = tensor_norm(
            [
                parameter.detach() - old
                for parameter, old in zip(model.parameters(), before, strict=True)
            ]
        )
        value = float(loss)
        ema[phase] = beta * ema[phase] + (1 - beta) * value
        ema_steps[phase] += 1
        metrics.append(
            {
                "step": step,
                "phase": phase,
                "loss": value,
                "logit_rms": float(outputs.float().square().mean().sqrt()),
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "class_share": float(outputs.argmax(1).bincount(minlength=10).max())
                / outputs.shape[0],
            }
        )
        if not math.isfinite(value) or not all_finite(model, optimizer):
            raise RuntimeError(f"{arm} nonfinite at step {step}")
    bn_counts = sorted(
        {
            int(module.num_batches_tracked)
            for module in model.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        }
    )
    min_running_var = min(
        float(module.running_var.min())
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )
    result = {
        "arm": arm,
        "metrics": metrics,
        "terminal_ema": {
            phase: ema[phase] / (1 - beta ** ema_steps[phase]) for phase in ema
        },
        "bn_counts": bn_counts,
        "min_running_var": min_running_var,
        "momentum_buffers": sum(
            "momentum_buffer" in state for state in optimizer.state.values()
        ),
    }
    print(json.dumps(result))


def run_trajectory_child(arm):
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--trajectory", arm],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return json.loads(
        [line for line in completed.stdout.splitlines() if line.strip()][-1]
    )


def validate_trajectories(control, candidate):
    failures = []
    max_ratios = {
        name: 0.0 for name in ("loss", "logit_rms", "grad_norm", "update_norm")
    }
    concentration = []
    for control_metric, candidate_metric in zip(
        control["metrics"], candidate["metrics"], strict=True
    ):
        for name in max_ratios:
            ratio = candidate_metric[name] / max(control_metric[name], 1e-12)
            max_ratios[name] = max(max_ratios[name], ratio)
            if ratio > THRESHOLDS["per_step_ratio"]:
                failures.append(
                    f"{name} ratio {ratio:.6f} at step {control_metric['step']}"
                )
        if (
            candidate_metric["class_share"] > 0.95
            and control_metric["class_share"] <= 0.95
        ):
            concentration.append(control_metric["step"])
    ema_ratios = {
        phase: candidate["terminal_ema"][phase] / control["terminal_ema"][phase]
        for phase in ("strong", "weak")
    }
    if any(value > THRESHOLDS["terminal_ema_ratio"] for value in ema_ratios.values()):
        failures.append(f"terminal EMA ratios {ema_ratios}")
    if concentration:
        failures.append(f"candidate-only concentration {concentration}")
    for arm in (control, candidate):
        if (
            arm["bn_counts"] != [264]
            or arm["min_running_var"] <= 0
            or arm["momentum_buffers"] != 59
        ):
            failures.append(f"incomplete state for {arm['arm']}")
    return {
        "status": "failed" if failures else "pass",
        "max_ratios": max_ratios,
        "terminal_ema_ratios": ema_ratios,
        "candidate_only_concentration": concentration,
        "control_terminal_ema": control["terminal_ema"],
        "candidate_terminal_ema": candidate["terminal_ema"],
        "failures": failures,
    }


def consume_live(arm, batches=5_000):
    transform = (
        strong_candidate_transform()
        if arm == "candidate"
        else strong_control_transform()
    )
    collate = instrumented_collate if arm == "candidate" else control_collate
    torch.manual_seed(42)
    loader = train.make_train_loader(transform, collate)
    iterator = iter(loader)
    waits = []
    worker_ids = set()
    cutmix_count = 0
    erased_examples = 0
    erased_pixels = 0
    achieved = []
    started = time.perf_counter()
    for _index in range(batches):
        wait_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        waits.append(time.perf_counter() - wait_started)
        if arm == "candidate":
            inputs, targets, examples, pixels, areas, worker_id, mixed = batch
            erased_examples += int(examples)
            erased_pixels += int(pixels)
            achieved.extend(float(value) for value in areas)
        else:
            inputs, targets, worker_id, mixed = batch
        worker_ids.add(int(worker_id))
        cutmix_count += int(mixed)
        if inputs.shape != (128, 3, 32, 32):
            raise RuntimeError("live input contract mismatch")
        if (mixed and targets.shape != (128, 10)) or (
            not mixed and targets.shape != (128,)
        ):
            raise RuntimeError("live target contract mismatch")
    elapsed = time.perf_counter() - started
    iterator = None
    workers = stop_loader(loader)
    result = {
        "arm": arm,
        "batches": batches,
        "elapsed_seconds": elapsed,
        "throughput": batches / elapsed,
        "wait_median_ms": 1_000 * statistics.median(waits),
        "wait_p95_ms": 1_000 * percentile(waits, 0.95),
        "worker_ids": sorted(worker_ids),
        "workers_stopped": len(workers),
        "cutmix_fraction": cutmix_count / batches,
    }
    if arm == "candidate":
        result.update(
            {
                "erased_examples": erased_examples,
                "erased_fraction": erased_examples / (batches * train.BATCH_SIZE),
                "conditional_mean_area": statistics.mean(achieved),
                "unconditional_mean_area": erased_pixels
                / (batches * train.BATCH_SIZE * 32 * 32),
                "achieved_min": min(achieved),
                "achieved_max": max(achieved),
            }
        )
    return result


def lifecycle_gate():
    candidate = consume_live("candidate")
    control = consume_live("control")
    rebuild_started = time.perf_counter()
    weak_loader = train.make_train_loader(weak_transform())
    weak_iterator = iter(weak_loader)
    weak_inputs, weak_targets = next(weak_iterator)
    weak_rebuild = time.perf_counter() - rebuild_started
    weak_zero_pixels = int((weak_inputs == 0).all(1).sum())
    weak_iterator = None
    weak_workers = stop_loader(weak_loader)
    live_children = [
        child.pid for child in multiprocessing.active_children() if child.is_alive()
    ]
    failures = []
    throughput_ratio = candidate["throughput"] / control["throughput"]
    if (
        not THRESHOLDS["live_erased_fraction"][0]
        <= candidate["erased_fraction"]
        <= THRESHOLDS["live_erased_fraction"][1]
    ):
        failures.append(f"live erased fraction {candidate['erased_fraction']:.6f}")
    if (
        not THRESHOLDS["conditional_mean_area"][0]
        <= candidate["conditional_mean_area"]
        <= THRESHOLDS["conditional_mean_area"][1]
    ):
        failures.append("live conditional area")
    if (
        not THRESHOLDS["unconditional_mean_area"][0]
        <= candidate["unconditional_mean_area"]
        <= THRESHOLDS["unconditional_mean_area"][1]
    ):
        failures.append("live unconditional area")
    if (
        not THRESHOLDS["live_cutmix_fraction"][0]
        <= candidate["cutmix_fraction"]
        <= THRESHOLDS["live_cutmix_fraction"][1]
    ):
        failures.append("live CutMix fraction")
    if (
        throughput_ratio < THRESHOLDS["live_throughput_ratio"]
        or candidate["throughput"] < THRESHOLDS["live_batches_per_second"]
    ):
        failures.append(
            f"live throughput {candidate['throughput']:.2f}, ratio {throughput_ratio:.4f}"
        )
    if (
        candidate["wait_median_ms"] > THRESHOLDS["live_wait_median_ms"]
        or candidate["wait_p95_ms"] > THRESHOLDS["live_wait_p95_ms"]
    ):
        failures.append(
            f"live waits {candidate['wait_median_ms']:.3f}/{candidate['wait_p95_ms']:.3f}ms"
        )
    if candidate["worker_ids"] != list(range(train.NUM_WORKERS)) or control[
        "worker_ids"
    ] != list(range(train.NUM_WORKERS)):
        failures.append("incomplete live worker coverage")
    if (
        weak_rebuild >= THRESHOLDS["weak_rebuild_seconds"]
        or weak_zero_pixels
        or weak_targets.ndim != 1
    ):
        failures.append("weak lifecycle/policy mismatch")
    if len(weak_workers) != train.NUM_WORKERS or live_children:
        failures.append("worker lifecycle leak")
    return {
        "status": "failed" if failures else "pass",
        "candidate": candidate,
        "control": control,
        "throughput_ratio": throughput_ratio,
        "weak_rebuild_seconds": weak_rebuild,
        "weak_zero_pixels": weak_zero_pixels,
        "live_children": live_children,
        "failures": failures,
    }


def parent():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    semantic = semantic_gate()
    print(json.dumps({"stage": "semantic", "status": "pass", **semantic}), flush=True)
    corpus = (
        torch.load(CORPUS_PATH, map_location="cpu", weights_only=False)
        if CORPUS_PATH.exists()
        else create_corpus()
    )
    corpus_sha = sha256_file(CORPUS_PATH)
    geometry = corpus["metadata"]["geometry"]
    del corpus
    print(
        json.dumps(
            {"stage": "corpus", "status": "pass", "sha256": corpus_sha, **geometry}
        ),
        flush=True,
    )
    control = run_trajectory_child("control")
    print(
        json.dumps({"stage": "trajectory", "arm": "control", "status": "complete"}),
        flush=True,
    )
    candidate = run_trajectory_child("candidate")
    print(
        json.dumps({"stage": "trajectory", "arm": "candidate", "status": "complete"}),
        flush=True,
    )
    trajectory = validate_trajectories(control, candidate)
    lifecycle = lifecycle_gate()
    failures = trajectory["failures"] + lifecycle["failures"]
    report = {
        "status": "failed" if failures else "pass",
        "thresholds": THRESHOLDS,
        "semantic": semantic,
        "corpus_path": str(CORPUS_PATH),
        "corpus_sha256": corpus_sha,
        "geometry": geometry,
        "trajectory": trajectory,
        "lifecycle": lifecycle,
        "failures": failures,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    with REPORT_PATH.open("rb") as handle:
        os.fsync(handle.fileno())
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", choices=("control", "candidate"))
    parser.add_argument("--corpus-only", action="store_true")
    args = parser.parse_args()
    if args.trajectory:
        run_trajectory(args.trajectory)
    elif args.corpus_only:
        print(json.dumps(create_corpus()["metadata"]))
    else:
        parent()


if __name__ == "__main__":
    main()
