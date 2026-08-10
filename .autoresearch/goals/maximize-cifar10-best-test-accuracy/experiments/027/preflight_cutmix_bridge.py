import argparse
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate, get_worker_info
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = EXPERIMENT_DIR / "pre-policy-corpus.pt"
REPORT_PATH = EXPERIMENT_DIR / "preflight-report.json"
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


def sha256_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def strong_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
        ]
    )


def weak_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
        ]
    )


def source_collate(batch):
    inputs, targets = default_collate(batch)
    worker = get_worker_info()
    return (
        inputs,
        targets,
        torch.get_rng_state().clone(),
        worker.id,
        torch.initial_seed(),
    )


class TaggedPhaseCollator(train.PhaseCutMixCollator):
    def __call__(self, batch):
        inputs, targets, policy = super().__call__(batch)
        return inputs, targets, policy, get_worker_info().id


def stop_loader(loader):
    workers = train.shutdown_train_loader(loader)
    if len(workers) != train.NUM_WORKERS:
        raise RuntimeError(
            f"stopped {len(workers)} workers, expected {train.NUM_WORKERS}"
        )
    return workers


def create_corpus():
    if CORPUS_PATH.exists():
        raise RuntimeError(
            "EXP027 corpus already exists; rematerialization is forbidden"
        )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    loader = train.make_train_loader(strong_transform(), collate_fn=source_collate)
    mirror_model = train.ResNet(
        train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER
    )
    mirror_optimizer = torch.optim.SGD(
        mirror_model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    del mirror_optimizer, mirror_model
    iterator = iter(loader)
    records = []
    for ordinal in range(200):
        inputs, targets, state, worker_id, worker_seed = next(iterator)
        inputs, targets, state = inputs.clone(), targets.clone(), state.clone()
        records.append(
            {
                "ordinal": ordinal,
                "inputs": inputs,
                "targets": targets,
                "rng_state": state,
                "worker_id": int(worker_id),
                "worker_seed": int(worker_seed),
                "input_sha256": sha256_tensor(inputs),
                "target_sha256": sha256_tensor(targets),
                "state_sha256": sha256_tensor(state),
            }
        )
    iterator = None
    stopped = stop_loader(loader)
    corpus = {
        "records": records,
        "metadata": {
            "seed": 42,
            "ordering": "loader, model, optimizer, iterator",
            "source": "first 200 unfiltered post-N1/M7 pre-policy batches",
            "workers_stopped": len(stopped),
            "torch_version": torch.__version__,
        },
    }
    temporary = CORPUS_PATH.with_suffix(".tmp")
    torch.save(corpus, temporary)
    temporary.replace(CORPUS_PATH)
    with CORPUS_PATH.open("rb") as handle:
        os.fsync(handle.fileno())
    return corpus


def validate_corpus(corpus):
    records = corpus["records"]
    if len(records) != 200 or [record["ordinal"] for record in records] != list(
        range(200)
    ):
        raise RuntimeError("corpus is incomplete")
    worker_ids = set()
    for record in records:
        inputs, targets, state = (
            record["inputs"],
            record["targets"],
            record["rng_state"],
        )
        if inputs.shape != (128, 3, 32, 32) or inputs.dtype != torch.float32:
            raise RuntimeError("input contract mismatch")
        if targets.shape != (128,) or targets.dtype != torch.int64:
            raise RuntimeError("target contract mismatch")
        if not 0 <= targets.min() <= targets.max() < train.NUM_CLASSES:
            raise RuntimeError("target range mismatch")
        for tensor, key in (
            (inputs, "input_sha256"),
            (targets, "target_sha256"),
            (state, "state_sha256"),
        ):
            if sha256_tensor(tensor) != record[key]:
                raise RuntimeError(f"digest mismatch at {record['ordinal']}:{key}")
        worker_ids.add(record["worker_id"])
    if worker_ids != set(range(train.NUM_WORKERS)):
        raise RuntimeError(f"worker coverage mismatch: {worker_ids}")


def apply_record(record, policy_on):
    surrounding = torch.get_rng_state().clone()
    inputs, targets = record["inputs"].clone(), record["targets"].clone()
    draw = None
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(record["rng_state"])
        if policy_on:
            draw = torch.rand(()).item()
            if draw < train.CUTMIX_PROBABILITY:
                inputs, targets = train.cutmix(inputs, targets)
    if not torch.equal(torch.get_rng_state(), surrounding):
        raise RuntimeError("surrounding RNG changed")
    return inputs, targets, draw


def validate_policy(inputs, targets, policy_on):
    if inputs.shape != (128, 3, 32, 32) or not torch.isfinite(inputs).all():
        raise RuntimeError("invalid policy inputs")
    if targets.ndim == 1:
        if targets.dtype != torch.int64:
            raise RuntimeError("invalid hard targets")
    elif (
        not policy_on
        or targets.shape != (128, 10)
        or not torch.isfinite(targets).all()
        or targets.min() < 0
        or not torch.allclose(targets.sum(1), torch.ones(128), atol=1e-6, rtol=0)
    ):
        raise RuntimeError("invalid soft targets")


def semantic_gate(corpus):
    cutmix_count = 0
    entries = []
    for record in corpus["records"]:
        control_inputs, control_targets, draw = apply_record(record, True)
        off_inputs, off_targets, off_draw = apply_record(record, False)
        validate_policy(control_inputs, control_targets, True)
        validate_policy(off_inputs, off_targets, False)
        if off_draw is not None or not torch.equal(off_inputs, record["inputs"]):
            raise RuntimeError("policy-off branch changed source or drew RNG")
        if not torch.equal(off_targets, record["targets"]):
            raise RuntimeError("policy-off targets changed")
        mixed = control_targets.ndim == 2
        cutmix_count += int(mixed)
        if not mixed and (
            not torch.equal(control_inputs, off_inputs)
            or not torch.equal(control_targets, off_targets)
        ):
            raise RuntimeError("shared hard branch differs")
        entries.append(
            {
                "ordinal": record["ordinal"],
                "draw": draw,
                "control_mixed": mixed,
                "source_sha256": record["input_sha256"],
                "control_sha256": sha256_tensor(control_inputs),
                "off_sha256": sha256_tensor(off_inputs),
            }
        )
    if not 80 <= cutmix_count <= 120:
        raise RuntimeError(f"natural CutMix count {cutmix_count} outside [80,120]")
    return cutmix_count, entries


def class_share(outputs):
    return outputs.argmax(1).bincount(minlength=10).max().item() / outputs.shape[0]


def train_records(model, optimizer, records, policy_on):
    losses, shares = [], []
    beta, ema = 0.95, 0.0
    for record in records:
        cpu_inputs, cpu_targets, _draw = apply_record(record, policy_on)
        inputs = cpu_inputs.cuda(non_blocking=True)
        targets = cpu_targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError("non-finite loss")
        losses.append(value)
        shares.append(class_share(outputs))
        ema = beta * ema + (1 - beta) * value
    tensors = list(model.parameters()) + list(model.buffers())
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise RuntimeError("non-finite model/optimizer state")
    return losses, shares, ema / (1 - beta ** len(records))


def continuation_gate(corpus):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = train.ResNet(3, 10, 2).cuda().train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    prefix_losses, _prefix_shares, _prefix_ema = train_records(
        model, optimizer, corpus["records"][:100], True
    )
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    arms = {}
    for name, policy_on in (("control", True), ("candidate", False)):
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        losses, shares, ema = train_records(
            model, optimizer, corpus["records"][100:], policy_on
        )
        bn_counts = sorted(
            {
                int(module.num_batches_tracked.item())
                for module in model.modules()
                if isinstance(module, torch.nn.BatchNorm2d)
            }
        )
        momentum_count = sum(
            "momentum_buffer" in state for state in optimizer.state.values()
        )
        if bn_counts != [200] or momentum_count != len(list(model.parameters())):
            raise RuntimeError(f"{name} incomplete state")
        arms[name] = {
            "losses": losses,
            "class_shares": shares,
            "terminal_loss_ema": ema,
            "bn_counts": bn_counts,
            "momentum_count": momentum_count,
        }
    concentration = [
        index + 1
        for index, (control, candidate) in enumerate(
            zip(
                arms["control"]["class_shares"],
                arms["candidate"]["class_shares"],
                strict=True,
            )
        )
        if candidate > 0.95 and control <= 0.95
    ]
    ratio = (
        arms["candidate"]["terminal_loss_ema"] / arms["control"]["terminal_loss_ema"]
    )
    if concentration or ratio > 1.5:
        raise RuntimeError(
            f"continuation veto: concentration={concentration[:5]} ratio={ratio}"
        )
    return {
        "prefix_losses": prefix_losses,
        "arms": arms,
        "loss_ema_ratio": ratio,
        "concentration": concentration,
    }


def lifecycle_gate():
    context = multiprocessing.get_context("forkserver")
    flag = context.Value("b", True, lock=True)
    torch.manual_seed(42)
    loader = train.make_train_loader(
        strong_transform(), collate_fn=TaggedPhaseCollator(flag)
    )
    iterator = iter(loader)
    worker_pids = [worker.pid for worker in loader._iterator._workers]
    policy_on, cutmix_count = 0, 0
    post_request = []
    request_at = 15_000
    for delivered in range(20_000):
        try:
            inputs, targets, policy, worker_id = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets, policy, worker_id = next(iterator)
        validate_policy(inputs, targets, bool(policy))
        if delivered < request_at:
            if policy != train.POLICY_ON:
                raise RuntimeError("policy disabled before request")
            policy_on += 1
            cutmix_count += int(targets.ndim == 2)
            if delivered + 1 == request_at:
                with flag.get_lock():
                    flag.value = False
        else:
            post_request.append(
                {
                    "offset": delivered - request_at + 1,
                    "policy": policy,
                    "worker_id": worker_id,
                }
            )
        if (delivered + 1) % 2_500 == 0:
            print(
                json.dumps({"stage": "lifecycle", "delivered": delivered + 1}),
                flush=True,
            )
    fraction = cutmix_count / policy_on
    last_on = max(
        (item["offset"] for item in post_request if item["policy"]), default=0
    )
    off_workers = {item["worker_id"] for item in post_request if not item["policy"]}
    if not 0.475 <= fraction <= 0.525:
        raise RuntimeError(f"CutMix fraction {fraction} outside interval")
    if last_on > 24 or off_workers != set(range(train.NUM_WORKERS)):
        raise RuntimeError(
            f"propagation failed: last_on={last_on} workers={off_workers}"
        )
    if any(item["policy"] for item in post_request[24:]):
        raise RuntimeError("policy-on delivery appeared after drain")
    iterator = None
    stopped = stop_loader(loader)
    if stopped != worker_pids:
        raise RuntimeError("worker PID set changed")
    started = time.perf_counter()
    weak_loader = train.make_train_loader(weak_transform())
    weak_iterator = iter(weak_loader)
    weak_inputs, weak_targets = next(weak_iterator)
    weak_rebuild = time.perf_counter() - started
    validate_policy(weak_inputs, weak_targets, False)
    weak_iterator = None
    weak_stopped = stop_loader(weak_loader)
    live = [
        child.pid for child in multiprocessing.active_children() if child.is_alive()
    ]
    if weak_rebuild >= 5 or live:
        raise RuntimeError(f"weak lifecycle failed: rebuild={weak_rebuild} live={live}")
    return {
        "policy_on": policy_on,
        "cutmix_count": cutmix_count,
        "cutmix_fraction": fraction,
        "last_policy_on_offset": last_on,
        "off_worker_ids": sorted(off_workers),
        "post_request_count": len(post_request),
        "strong_workers_stopped": len(stopped),
        "weak_workers_stopped": len(weak_stopped),
        "weak_rebuild_seconds": weak_rebuild,
        "live_children": live,
    }


def main():
    argparse.ArgumentParser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    corpus = (
        torch.load(CORPUS_PATH, map_location="cpu", weights_only=False)
        if CORPUS_PATH.exists()
        else create_corpus()
    )
    validate_corpus(corpus)
    corpus_sha = sha256_file(CORPUS_PATH)
    cutmix_count, semantics = semantic_gate(corpus)
    print(json.dumps({"stage": "semantics", "cutmix_count": cutmix_count}), flush=True)
    continuation = continuation_gate(corpus)
    print(json.dumps({"stage": "continuation", "status": "pass"}), flush=True)
    lifecycle = lifecycle_gate()
    report = {
        "status": "pass",
        "corpus_sha256": corpus_sha,
        "corpus_metadata": corpus["metadata"],
        "semantic_cutmix_count": cutmix_count,
        "semantic_entries": semantics,
        "continuation": continuation,
        "lifecycle": lifecycle,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    with REPORT_PATH.open("rb") as handle:
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("semantic_entries", "continuation")
            }
        )
    )


if __name__ == "__main__":
    main()
