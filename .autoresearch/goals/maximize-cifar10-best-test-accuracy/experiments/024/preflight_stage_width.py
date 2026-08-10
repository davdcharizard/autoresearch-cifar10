import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = EXPERIMENT_DIR / "preflight-corpus.pt"
REPORT_PATH = EXPERIMENT_DIR / "preflight-report.json"
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def clone_batch(batch):
    return tuple(tensor.contiguous().clone() for tensor in batch)


def stop_loader(loader):
    stopped = train.shutdown_train_loader(loader)
    if len(stopped) != train.NUM_WORKERS:
        raise RuntimeError(
            f"expected {train.NUM_WORKERS} stopped workers, got {len(stopped)}"
        )
    return stopped


def materialize_corpus():
    mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)
    strong_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    weak_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    torch.manual_seed(42)
    strong_loader = train.make_train_loader(
        strong_transform, collate_fn=train.cutmix_collate
    )
    strong_hard = []
    strong_soft = []
    strong_iterator = iter(strong_loader)
    draws = 0
    while len(strong_hard) < 100 or len(strong_soft) < 100:
        batch = next(strong_iterator)
        draws += 1
        target_bucket = strong_soft if batch[1].ndim == 2 else strong_hard
        if len(target_bucket) < 100:
            target_bucket.append(clone_batch(batch))
        if draws > 1000:
            raise RuntimeError("failed to collect balanced strong hard/soft buckets")
    strong_iterator = None
    strong_workers = stop_loader(strong_loader)

    torch.manual_seed(4242)
    weak_loader = train.make_train_loader(weak_transform)
    weak_iterator = iter(weak_loader)
    weak_hard = [clone_batch(next(weak_iterator)) for _ in range(100)]
    weak_iterator = None
    weak_workers = stop_loader(weak_loader)

    corpus = {
        "strong_hard": strong_hard,
        "strong_soft": strong_soft,
        "weak_hard": weak_hard,
        "metadata": {
            "strong_seed": 42,
            "weak_seed": 4242,
            "strong_draws": draws,
            "strong_workers_stopped": len(strong_workers),
            "weak_workers_stopped": len(weak_workers),
        },
    }
    torch.save(corpus, CORPUS_PATH)
    torch.save(
        {name: batches[0] for name, batches in corpus.items() if name != "metadata"},
        EXPERIMENT_DIR / "timing-batches.pt",
    )
    return corpus


def tensor_norm(tensors):
    return math.sqrt(
        sum(tensor.detach().float().square().sum().item() for tensor in tensors)
    )


def grouped_parameters(model):
    groups = {name: [] for name in ("stem", "layer1", "layer2", "layer3", "fc")}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        if group in ("conv1", "bn1"):
            group = "stem"
        groups[group].append((name, parameter))
    return groups


def group_diagnostics(groups, starts):
    result = {}
    for group, named_parameters in groups.items():
        parameters = [parameter for _, parameter in named_parameters]
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        updates = [
            parameter.detach() - starts[name] for name, parameter in named_parameters
        ]
        scale = math.sqrt(sum(parameter.numel() for parameter in parameters))
        result[group] = {
            "gradient_rms": tensor_norm(gradients) / scale,
            "update_rms": tensor_norm(updates) / scale,
        }
    return result


def all_finite(model, optimizer):
    tensors = list(model.parameters()) + list(model.buffers())
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def class_share(outputs):
    counts = outputs.argmax(1).bincount(minlength=train.NUM_CLASSES)
    return counts.max().item() / outputs.shape[0]


def build_model(arm, device):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cpu_rng_before = sha256_tensor(torch.random.get_rng_state())
    cuda_rng_before = sha256_tensor(torch.cuda.get_rng_state())
    if arm == "control":
        model = train.ResNet(3, train.NUM_CLASSES, 2).to(device)
        constructor = "ResNet(3, 10, 2)"
    else:
        model = train.ResNet(3, train.NUM_CLASSES, 2, 160).to(device)
        constructor = "ResNet(3, 10, 2, 160)"
    rng = {
        "seed": 42,
        "cpu_before": cpu_rng_before,
        "cuda_before": cuda_rng_before,
        "cpu_after": sha256_tensor(torch.random.get_rng_state()),
        "cuda_after": sha256_tensor(torch.cuda.get_rng_state()),
    }
    return model, constructor, rng


def structural_checks(model, arm, sample):
    expected_params = 1_073_962 if arm == "control" else 1_507_818
    expected_shapes = {
        "layer1": (sample.shape[0], 32, 32, 32),
        "layer2": (sample.shape[0], 64, 16, 16),
        "layer3": (
            sample.shape[0],
            128 if arm == "control" else 160,
            8,
            8,
        ),
    }
    shapes = {}

    def record(name):
        def hook(_module, _inputs, output):
            shapes[name] = tuple(output.shape)

        return hook

    handles = [
        getattr(model, name).register_forward_hook(record(name))
        for name in ("layer1", "layer2", "layer3")
    ]
    model.eval()
    buffers_before = [buffer.detach().clone() for buffer in model.buffers()]
    with torch.inference_mode():
        output = model(sample)
    for handle in handles:
        handle.remove()
    if shapes != expected_shapes or output.shape != (sample.shape[0], 10):
        raise RuntimeError(f"{arm} shape mismatch: {shapes}")
    if not torch.isfinite(output).all():
        raise RuntimeError(f"{arm} evaluation output is non-finite")
    if not all(
        torch.equal(before, after)
        for before, after in zip(buffers_before, model.buffers(), strict=True)
    ):
        raise RuntimeError(f"{arm} evaluation mutated buffers")
    facts = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "block_count": sum(
            isinstance(module, train.BasicBlock) for module in model.modules()
        ),
        "conv2d_count": sum(
            isinstance(module, torch.nn.Conv2d) for module in model.modules()
        ),
        "stage_shapes": shapes,
        "option_a_pads": [model.layer2[0].pad_channels, model.layer3[0].pad_channels],
        "fc_in_features": model.fc.in_features,
        "dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
    }
    expected_pad = 64 if arm == "control" else 96
    if facts != {
        "parameter_count": expected_params,
        "block_count": 9,
        "conv2d_count": 19,
        "stage_shapes": expected_shapes,
        "option_a_pads": [32, expected_pad],
        "fc_in_features": 128 if arm == "control" else 160,
        "dtypes": ["torch.float32"],
    }:
        raise RuntimeError(f"{arm} structural facts mismatch: {facts}")
    return facts


def run_arm(arm):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.use_deterministic_algorithms(True)
    corpus = torch.load(CORPUS_PATH, map_location="cpu", weights_only=False)
    sequence = [
        batch
        for pair in zip(corpus["strong_hard"], corpus["strong_soft"], strict=True)
        for batch in pair
    ]
    device = torch.device("cuda")
    model, constructor, rng = build_model(arm, device)
    facts = structural_checks(model, arm, sequence[0][0][:2].to(device))
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    groups = grouped_parameters(model)
    beta = 0.95
    loss_ema = 0.0
    losses = []
    class_shares = []
    diagnostics = []
    for step, (cpu_inputs, cpu_targets) in enumerate(sequence, start=1):
        inputs = cpu_inputs.to(device, non_blocking=True)
        targets = cpu_targets.to(device, non_blocking=True)
        starts = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        value = loss.item()
        loss_ema = beta * loss_ema + (1 - beta) * value
        losses.append(value)
        class_shares.append(class_share(outputs))
        if step <= 20 or step % 20 == 0:
            diagnostics.append(
                {
                    "step": step,
                    "loss": value,
                    "class_share": class_shares[-1],
                    "groups": group_diagnostics(groups, starts),
                }
            )
        if not math.isfinite(value) or not all_finite(model, optimizer):
            raise RuntimeError(f"{arm} non-finite state at step {step}")

    bn_counts = sorted(
        {
            int(module.num_batches_tracked.item())
            for module in model.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        }
    )
    if bn_counts != [len(sequence)]:
        raise RuntimeError(f"{arm} BN counters mismatch: {bn_counts}")
    momentum_tensors = [
        state["momentum_buffer"]
        for state in optimizer.state.values()
        if "momentum_buffer" in state
    ]
    if len(momentum_tensors) != len(list(model.parameters())):
        raise RuntimeError(f"{arm} momentum state is incomplete")
    result = {
        "arm": arm,
        "constructor": constructor,
        "rng": rng,
        "facts": facts,
        "num_steps": len(sequence),
        "hard_steps": 100,
        "soft_steps": 100,
        "terminal_loss_ema": loss_ema / (1 - beta ** len(sequence)),
        "losses": losses,
        "class_shares": class_shares,
        "bn_batch_counts": bn_counts,
        "momentum_tensor_count": len(momentum_tensors),
        "diagnostics": diagnostics,
    }
    print(json.dumps(result))


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
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def parent():
    corpus = materialize_corpus()
    control = run_child("control")
    candidate = run_child("candidate")
    concentration_events = []
    for step, (control_share, candidate_share) in enumerate(
        zip(control["class_shares"], candidate["class_shares"], strict=True),
        start=1,
    ):
        if candidate_share > 0.95 and control_share <= 0.90:
            concentration_events.append(
                {
                    "step": step,
                    "candidate_share": candidate_share,
                    "control_share": control_share,
                }
            )
    ratio = candidate["terminal_loss_ema"] / control["terminal_loss_ema"]
    report = {
        "status": "pass",
        "corpus_path": str(CORPUS_PATH),
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "bucket_counts": {
            name: len(corpus[name])
            for name in ("strong_hard", "strong_soft", "weak_hard")
        },
        "corpus_metadata": corpus["metadata"],
        "control": control,
        "candidate": candidate,
        "candidate_only_concentration_events": concentration_events,
        "candidate_control_loss_ema_ratio": ratio,
    }
    failures = []
    if concentration_events:
        failures.append(
            f"candidate-only concentration at steps {[event['step'] for event in concentration_events[:5]]}"
        )
    if ratio > 1.5:
        failures.append(f"candidate/control loss EMA ratio {ratio:.6f} > 1.5")
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise RuntimeError("; ".join(failures))
    print(
        json.dumps(
            {
                "status": report["status"],
                "corpus_sha256": report["corpus_sha256"],
                "bucket_counts": report["bucket_counts"],
                "control_constructor": control["constructor"],
                "candidate_constructor": candidate["constructor"],
                "control_terminal_loss_ema": control["terminal_loss_ema"],
                "candidate_terminal_loss_ema": candidate["terminal_loss_ema"],
                "candidate_control_loss_ema_ratio": ratio,
                "candidate_only_concentration_events": concentration_events,
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
