import argparse
import hashlib
import json
import math
import multiprocessing
import os
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
CORPUS_PATH = EXPERIMENT_DIR / "pre-policy-corpus.pt"
REPORT_PATH = EXPERIMENT_DIR / "preflight-report.json"
THRESHOLDS = {
    "natural_kind_floor": 35,
    "candidate_class_share": 0.95,
    "control_class_share_ceiling": 0.95,
    "loss_ema_ratio": 1.5,
    "hard_fraction_min": 0.485,
    "hard_fraction_max": 0.515,
    "cutmix_fraction_min": 0.235,
    "cutmix_fraction_max": 0.265,
    "mixup_fraction_min": 0.235,
    "mixup_fraction_max": 0.265,
    "weak_rebuild_seconds": 5.0,
}
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


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


def sha256_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def strong_transform():
    mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def weak_transform():
    mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def stop_loader(loader):
    stopped = train.shutdown_train_loader(loader)
    if len(stopped) != train.NUM_WORKERS:
        raise RuntimeError(
            f"expected {train.NUM_WORKERS} stopped workers, got {len(stopped)}"
        )
    return stopped


def create_corpus():
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
        inputs, targets, rng_state, worker_id, worker_seed = next(iterator)
        inputs = inputs.contiguous().clone()
        targets = targets.contiguous().clone()
        rng_state = rng_state.contiguous().clone()
        records.append(
            {
                "ordinal": ordinal,
                "inputs": inputs,
                "targets": targets,
                "rng_state": rng_state,
                "worker_id": int(worker_id),
                "worker_seed": int(worker_seed),
                "input_sha256": sha256_tensor(inputs),
                "target_sha256": sha256_tensor(targets),
                "state_sha256": sha256_tensor(rng_state),
            }
        )
    iterator = None
    workers = stop_loader(loader)
    corpus = {
        "records": records,
        "metadata": {
            "seed": 42,
            "ordering": "loader, model, optimizer, iterator",
            "source_collate": "default_collate then capture CPU RNG; no policy draw",
            "transform": repr(strong_transform()),
            "loader": {
                "batch_size": train.BATCH_SIZE,
                "shuffle": True,
                "num_workers": train.NUM_WORKERS,
                "pin_memory": True,
                "drop_last": True,
                "persistent_workers": True,
                "multiprocessing_context": "forkserver",
            },
            "workers_stopped": len(workers),
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
        raise RuntimeError("corpus ordinals are incomplete")
    worker_ids = set()
    for record in records:
        inputs = record["inputs"]
        targets = record["targets"]
        state = record["rng_state"]
        if inputs.shape != (128, 3, 32, 32) or inputs.dtype != torch.float32:
            raise RuntimeError("source input contract mismatch")
        if targets.shape != (128,) or targets.dtype != torch.int64:
            raise RuntimeError("source target contract mismatch")
        if targets.min() < 0 or targets.max() >= train.NUM_CLASSES:
            raise RuntimeError("source label range mismatch")
        if sha256_tensor(inputs) != record["input_sha256"]:
            raise RuntimeError("source input digest mismatch")
        if sha256_tensor(targets) != record["target_sha256"]:
            raise RuntimeError("source target digest mismatch")
        if sha256_tensor(state) != record["state_sha256"]:
            raise RuntimeError("source RNG digest mismatch")
        worker_ids.add(record["worker_id"])
    if worker_ids != set(range(train.NUM_WORKERS)):
        raise RuntimeError(f"source worker ids incomplete: {worker_ids}")


def accepted_policy(inputs, targets):
    draw = torch.rand(()).item()
    if draw < 0.5:
        inputs, targets = train.cutmix(inputs, targets)
        kind = train.CUTMIX
    else:
        kind = train.HARD
    return inputs, targets, kind, draw


def apply_record(record, arm):
    surrounding = torch.get_rng_state().clone()
    cuda_surrounding = (
        torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    )
    inputs = record["inputs"].clone()
    targets = record["targets"].clone()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(record["rng_state"])
        if arm == "control":
            outputs = accepted_policy(inputs, targets)
        else:
            policy_inputs, policy_targets, kind = train.apply_strong_policy(
                inputs, targets
            )
            torch.set_rng_state(record["rng_state"])
            draw = torch.rand(()).item()
            outputs = policy_inputs, policy_targets, kind, draw
    if not torch.equal(torch.get_rng_state(), surrounding):
        raise RuntimeError("policy changed surrounding CPU RNG")
    if cuda_surrounding is not None and not torch.equal(
        torch.cuda.get_rng_state(), cuda_surrounding
    ):
        raise RuntimeError("policy changed CUDA RNG")
    if sha256_tensor(record["inputs"]) != record["input_sha256"]:
        raise RuntimeError("policy mutated source inputs")
    if sha256_tensor(record["targets"]) != record["target_sha256"]:
        raise RuntimeError("policy mutated source targets")
    return outputs


def validate_policy_output(inputs, targets, kind):
    if inputs.shape != (128, 3, 32, 32) or not torch.isfinite(inputs).all():
        raise RuntimeError("invalid policy inputs")
    if kind == train.HARD:
        if targets.shape != (128,) or targets.dtype != torch.int64:
            raise RuntimeError("invalid hard targets")
    else:
        if targets.shape != (128, 10) or not torch.is_floating_point(targets):
            raise RuntimeError("invalid mixed targets")
        if not torch.isfinite(targets).all() or targets.min() < 0:
            raise RuntimeError("invalid mixed target values")
        if not torch.allclose(targets.sum(1), torch.ones(128), atol=1e-6, rtol=0):
            raise RuntimeError("mixed target rows do not sum to one")


def semantic_report(corpus):
    counts = {"control": [0, 0, 0], "candidate": [0, 0, 0]}
    entries = []
    for record in corpus["records"]:
        control_inputs, control_targets, control_kind, control_draw = apply_record(
            record, "control"
        )
        candidate_inputs, candidate_targets, candidate_kind, candidate_draw = (
            apply_record(record, "candidate")
        )
        validate_policy_output(control_inputs, control_targets, control_kind)
        validate_policy_output(candidate_inputs, candidate_targets, candidate_kind)
        if control_draw != candidate_draw:
            raise RuntimeError("categorical draw mismatch")
        if (control_kind != train.HARD) != (candidate_kind != train.HARD):
            raise RuntimeError("total mixed decision mismatch")
        shared = control_draw < 0.25 or control_draw >= 0.5
        if shared and (
            not torch.equal(control_inputs, candidate_inputs)
            or not torch.equal(control_targets, candidate_targets)
        ):
            raise RuntimeError("shared policy branch is not bitwise equal")
        if 0.25 <= control_draw < 0.5 and candidate_kind != train.MIXUP:
            raise RuntimeError("middle-quarter candidate is not Mixup")
        counts["control"][control_kind] += 1
        counts["candidate"][candidate_kind] += 1
        entries.append(
            {
                "ordinal": record["ordinal"],
                "worker_id": record["worker_id"],
                "worker_seed": record["worker_seed"],
                "draw": control_draw,
                "control_kind": control_kind,
                "candidate_kind": candidate_kind,
                "source_input_sha256": record["input_sha256"],
                "control_input_sha256": sha256_tensor(control_inputs),
                "candidate_input_sha256": sha256_tensor(candidate_inputs),
                "control_target_sha256": sha256_tensor(control_targets),
                "candidate_target_sha256": sha256_tensor(candidate_targets),
                "shared_branch_equal": shared,
            }
        )
    return counts, entries


def backend_flags():
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def configure_backend():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def all_finite(model, optimizer):
    tensors = list(model.parameters()) + list(model.buffers())
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def class_share(outputs):
    return outputs.argmax(1).bincount(minlength=10).max().item() / outputs.shape[0]


def run_arm(arm):
    configure_backend()
    corpus = torch.load(CORPUS_PATH, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = train.ResNet(3, 10, 2).to(device).train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    beta = 0.95
    loss_ema = 0.0
    losses = []
    shares = []
    kinds = []
    for step, record in enumerate(corpus["records"], start=1):
        cpu_inputs, cpu_targets, kind, _draw = apply_record(record, arm)
        inputs = cpu_inputs.to(device, non_blocking=True)
        targets = cpu_targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        value = loss.item()
        losses.append(value)
        shares.append(class_share(outputs))
        kinds.append(kind)
        loss_ema = beta * loss_ema + (1 - beta) * value
        if not math.isfinite(value) or not all_finite(model, optimizer):
            raise RuntimeError(f"{arm} non-finite state at step {step}")
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
        raise RuntimeError(f"{arm} incomplete BN/momentum state")
    print(
        json.dumps(
            {
                "arm": arm,
                "policy": "accepted 50/50 hard/CutMix"
                if arm == "control"
                else "candidate 50/25/25 hard/CutMix/Mixup",
                "backend": backend_flags(),
                "losses": losses,
                "class_shares": shares,
                "kinds": kinds,
                "terminal_loss_ema": loss_ema / (1 - beta**200),
                "bn_counts": bn_counts,
                "momentum_count": momentum_count,
            }
        )
    )


def run_child(arm):
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", arm],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(
        [line for line in completed.stdout.splitlines() if line.strip()][-1]
    )


def lifecycle_gate():
    started = time.perf_counter()
    torch.manual_seed(42)
    loader = train.make_train_loader(strong_transform(), collate_fn=train.mixed_collate)
    counts = [0, 0, 0]
    iterator = None
    for index in range(20_000):
        if iterator is None:
            iterator = iter(loader)
        try:
            _inputs, targets, kind = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            _inputs, targets, kind = next(iterator)
        validate_policy_output(_inputs, targets, kind)
        counts[kind] += 1
        if (index + 1) % 2_500 == 0:
            print(
                json.dumps(
                    {
                        "stage": "lifecycle",
                        "batches": index + 1,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    iterator = None
    strong_workers = stop_loader(loader)
    started = time.perf_counter()
    weak_loader = train.make_train_loader(weak_transform())
    weak_iterator = iter(weak_loader)
    weak_inputs, weak_targets = next(weak_iterator)
    rebuild_seconds = time.perf_counter() - started
    if (
        weak_inputs.shape != (128, 3, 32, 32)
        or weak_targets.shape != (128,)
        or weak_targets.dtype != torch.int64
    ):
        raise RuntimeError("weak loader contract mismatch")
    weak_iterator = None
    weak_workers = stop_loader(weak_loader)
    live_children = [
        child.pid for child in multiprocessing.active_children() if child.is_alive()
    ]
    fractions = [count / 20_000 for count in counts]
    return {
        "counts": counts,
        "fractions": fractions,
        "strong_workers_stopped": len(strong_workers),
        "weak_workers_stopped": len(weak_workers),
        "weak_rebuild_seconds": rebuild_seconds,
        "live_children": live_children,
    }


def parent():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_backend()
    corpus = (
        torch.load(CORPUS_PATH, map_location="cpu", weights_only=False)
        if CORPUS_PATH.exists()
        else create_corpus()
    )
    print(json.dumps({"stage": "corpus", "status": "ready"}), flush=True)
    validate_corpus(corpus)
    corpus_sha = sha256_file(CORPUS_PATH)
    counts, semantics = semantic_report(corpus)
    print(json.dumps({"stage": "semantics", "counts": counts}), flush=True)
    failures = []
    for kind, name in (
        (train.HARD, "hard"),
        (train.CUTMIX, "cutmix"),
        (train.MIXUP, "mixup"),
    ):
        if counts["candidate"][kind] < THRESHOLDS["natural_kind_floor"]:
            failures.append(
                f"natural candidate {name} count {counts['candidate'][kind]} < 35"
            )
    semantic_path = EXPERIMENT_DIR / "semantic-report.json"
    semantic_path.write_text(
        json.dumps(
            {
                "thresholds": THRESHOLDS,
                "corpus_sha256": corpus_sha,
                "counts": counts,
                "entries": semantics,
            },
            indent=2,
        )
        + "\n"
    )
    with semantic_path.open("rb") as handle:
        os.fsync(handle.fileno())
    if failures:
        raise RuntimeError("; ".join(failures))
    del corpus
    control = run_child("control")
    print(json.dumps({"stage": "control", "status": "complete"}), flush=True)
    candidate = run_child("candidate")
    print(json.dumps({"stage": "candidate", "status": "complete"}), flush=True)
    if control["backend"] != candidate["backend"]:
        failures.append("control/candidate backend flags differ")
    concentration = []
    for step, (control_share, candidate_share) in enumerate(
        zip(control["class_shares"], candidate["class_shares"], strict=True), start=1
    ):
        if (
            candidate_share > THRESHOLDS["candidate_class_share"]
            and control_share <= THRESHOLDS["control_class_share_ceiling"]
        ):
            concentration.append(
                {"step": step, "control": control_share, "candidate": candidate_share}
            )
    ratio = candidate["terminal_loss_ema"] / control["terminal_loss_ema"]
    if concentration:
        failures.append(
            f"candidate-only concentration at steps {[item['step'] for item in concentration[:5]]}"
        )
    if ratio > THRESHOLDS["loss_ema_ratio"]:
        failures.append(f"loss EMA ratio {ratio:.6f} > 1.5")
    lifecycle = lifecycle_gate()
    hard, cutmix, mixup = lifecycle["fractions"]
    if not THRESHOLDS["hard_fraction_min"] <= hard <= THRESHOLDS["hard_fraction_max"]:
        failures.append(f"hard fraction {hard:.6f} outside interval")
    if (
        not THRESHOLDS["cutmix_fraction_min"]
        <= cutmix
        <= THRESHOLDS["cutmix_fraction_max"]
    ):
        failures.append(f"CutMix fraction {cutmix:.6f} outside interval")
    if (
        not THRESHOLDS["mixup_fraction_min"]
        <= mixup
        <= THRESHOLDS["mixup_fraction_max"]
    ):
        failures.append(f"Mixup fraction {mixup:.6f} outside interval")
    if lifecycle["weak_rebuild_seconds"] >= THRESHOLDS["weak_rebuild_seconds"]:
        failures.append("weak rebuild exceeded 5 seconds")
    if lifecycle["live_children"]:
        failures.append(f"live worker children remain: {lifecycle['live_children']}")
    report = {
        "status": "failed" if failures else "pass",
        "thresholds": THRESHOLDS,
        "corpus_path": str(CORPUS_PATH),
        "corpus_sha256": corpus_sha,
        "corpus_metadata": torch.load(
            CORPUS_PATH, map_location="cpu", weights_only=False
        )["metadata"],
        "semantic_counts": counts,
        "control": control,
        "candidate": candidate,
        "candidate_control_loss_ema_ratio": ratio,
        "candidate_only_concentration_events": concentration,
        "lifecycle": lifecycle,
        "failures": failures,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    with REPORT_PATH.open("rb") as handle:
        os.fsync(handle.fileno())
    if failures:
        raise RuntimeError("; ".join(failures))
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("control", "candidate")
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("control", "candidate"))
    args = parser.parse_args()
    if args.child:
        run_arm(args.child)
    else:
        parent()


if __name__ == "__main__":
    main()
