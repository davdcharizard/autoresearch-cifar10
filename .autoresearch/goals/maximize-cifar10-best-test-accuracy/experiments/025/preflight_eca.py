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


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = EXPERIMENT_DIR.parent / "024" / "preflight-corpus.pt"
ORACLE_PATH = EXPERIMENT_DIR / "baseline-identity-state.pt"
ORACLE_META_PATH = EXPERIMENT_DIR / "baseline-identity-meta.json"
REPORT_PATH = EXPERIMENT_DIR / "preflight-report.json"
EXPECTED_CORPUS_SHA = "d4294f5adb2e58e0847366231458b21901c6f01f270d4cd1c9eae14a05b64565"
THRESHOLDS = {
    "first_max_abs_weight": 0.25,
    "first_gate_min": 0.75,
    "first_gate_max": 1.25,
    "first_mean_min": 0.95,
    "first_mean_max": 1.05,
    "trajectory_gate_min": 0.5,
    "trajectory_gate_max": 1.5,
    "trajectory_mean_min": 0.85,
    "trajectory_mean_max": 1.15,
    "candidate_class_share": 0.95,
    "control_class_share_ceiling": 0.90,
    "loss_ema_ratio": 1.5,
    "shared_gradient_max_abs": 1e-7,
    "shared_gradient_relative_norm": 1e-6,
}
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


def validate_corpus(corpus):
    counts = {
        name: len(corpus[name]) for name in ("strong_hard", "strong_soft", "weak_hard")
    }
    if counts != {"strong_hard": 100, "strong_soft": 100, "weak_hard": 100}:
        raise RuntimeError(f"corpus bucket mismatch: {counts}")
    metadata = {}
    for name, batches in corpus.items():
        if name == "metadata":
            continue
        target_ndim = 2 if name == "strong_soft" else 1
        for inputs, targets in batches:
            if inputs.shape != (train.BATCH_SIZE, 3, 32, 32):
                raise RuntimeError(f"{name} input shape mismatch: {inputs.shape}")
            if targets.ndim != target_ndim:
                raise RuntimeError(f"{name} target rank mismatch: {targets.ndim}")
            if target_ndim == 2 and not torch.allclose(
                targets.sum(1), torch.ones(targets.shape[0]), atol=1e-6, rtol=0
            ):
                raise RuntimeError(f"{name} probability rows do not sum to one")
        metadata[name] = {
            "input_shape": list(batches[0][0].shape),
            "input_dtype": str(batches[0][0].dtype),
            "target_shape": list(batches[0][1].shape),
            "target_dtype": str(batches[0][1].dtype),
            "target_row_sum_min": (
                min(batch[1].sum(1).min().item() for batch in batches)
                if target_ndim == 2
                else None
            ),
            "target_row_sum_max": (
                max(batch[1].sum(1).max().item() for batch in batches)
                if target_ndim == 2
                else None
            ),
        }
    return counts, metadata


def build_model(arm, device):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = train.ResNet(3, train.NUM_CLASSES, 2, use_eca=(arm == "candidate")).to(
        device
    )
    return model


def eca_modules(model):
    return [module for module in model.modules() if isinstance(module, train.ECAGate)]


def gate_values(module, residual):
    descriptor = F.adaptive_avg_pool2d(residual, 1).flatten(2).transpose(1, 2)
    logits = module.channel_conv(descriptor)
    return (2.0 * torch.sigmoid(logits)).transpose(1, 2).unsqueeze(-1)


def gate_snapshot(model, inputs):
    captured = []

    def hook(module, args, _output):
        captured.append(gate_values(module, args[0]).detach().float().flatten().cpu())

    handles = [module.register_forward_hook(hook) for module in eca_modules(model)]
    was_training = model.training
    model.eval()
    buffers = [buffer.detach().clone() for buffer in model.buffers()]
    with torch.inference_mode():
        model(inputs)
    for handle in handles:
        handle.remove()
    if was_training:
        model.train()
    if not all(
        torch.equal(before, after)
        for before, after in zip(buffers, model.buffers(), strict=True)
    ):
        raise RuntimeError("gate snapshot mutated model buffers")
    result = []
    for values in captured:
        result.append(
            {
                "min": values.min().item(),
                "p01": torch.quantile(values, 0.01).item(),
                "p50": torch.quantile(values, 0.50).item(),
                "p99": torch.quantile(values, 0.99).item(),
                "max": values.max().item(),
                "mean": values.mean().item(),
            }
        )
    return result


def class_share(outputs):
    counts = outputs.argmax(1).bincount(minlength=train.NUM_CLASSES)
    return counts.max().item() / outputs.shape[0]


def all_finite(model, optimizer):
    tensors = list(model.parameters()) + list(model.buffers())
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def identity_checks(device, hard_batch, soft_batch):
    oracle = torch.load(ORACLE_PATH, map_location="cpu", weights_only=True)
    oracle_meta = json.loads(ORACLE_META_PATH.read_text())
    candidate = build_model("candidate", device)
    shared = {
        name: value.detach().cpu()
        for name, value in candidate.state_dict().items()
        if ".eca." not in name
    }
    if shared.keys() != oracle.keys() or not all(
        torch.equal(shared[name], oracle[name]) for name in oracle
    ):
        raise RuntimeError("candidate shared state differs from pre-edit oracle")
    if sha256_tensor(torch.random.get_rng_state()) != oracle_meta["cpu_rng_sha256"]:
        raise RuntimeError("candidate CPU RNG differs from pre-edit oracle")
    if sha256_tensor(torch.cuda.get_rng_state()) != oracle_meta["cuda_rng_sha256"]:
        raise RuntimeError("candidate CUDA RNG differs from pre-edit oracle")
    facts = {
        "parameter_count": sum(
            parameter.numel() for parameter in candidate.parameters()
        ),
        "eca_module_count": len(eca_modules(candidate)),
        "eca_parameter_count": sum(
            parameter.numel()
            for module in eca_modules(candidate)
            for parameter in module.parameters()
        ),
        "conv1d_count": sum(
            isinstance(module, torch.nn.Conv1d) for module in candidate.modules()
        ),
        "conv2d_count": sum(
            isinstance(module, torch.nn.Conv2d) for module in candidate.modules()
        ),
        "block_count": sum(
            isinstance(module, train.BasicBlock) for module in candidate.modules()
        ),
        "option_a_pads": [
            candidate.layer2[0].pad_channels,
            candidate.layer3[0].pad_channels,
        ],
    }
    expected = {
        "parameter_count": 1_073_977,
        "eca_module_count": 3,
        "eca_parameter_count": 15,
        "conv1d_count": 3,
        "conv2d_count": 19,
        "block_count": 9,
        "option_a_pads": [32, 64],
    }
    if facts != expected:
        raise RuntimeError(f"candidate structural mismatch: {facts}")
    if any(
        torch.count_nonzero(module.channel_conv.weight)
        for module in eca_modules(candidate)
    ):
        raise RuntimeError("ECA weights are not zero")

    results = []
    for label, (cpu_inputs, cpu_targets) in (
        ("hard", hard_batch),
        ("soft", soft_batch),
    ):
        control = build_model("control", device)
        candidate = build_model("candidate", device)
        control.train()
        candidate.train()
        inputs = cpu_inputs.to(device)
        targets = cpu_targets.to(device)
        control.zero_grad()
        candidate.zero_grad()
        control_outputs = control(inputs)
        candidate_outputs = candidate(inputs)
        if not torch.equal(control_outputs, candidate_outputs):
            raise RuntimeError(f"{label} initial logits are not bitwise equal")
        candidate_buffers = dict(candidate.named_buffers())
        for name, buffer in control.named_buffers():
            if not torch.equal(buffer, candidate_buffers[name]):
                raise RuntimeError(f"{label} shared BN buffer mismatch: {name}")
        control_loss = F.cross_entropy(control_outputs, targets)
        candidate_loss = F.cross_entropy(candidate_outputs, targets)
        control_loss.backward()
        candidate_loss.backward()
        control_grads = dict(control.named_parameters())
        candidate_grads = dict(candidate.named_parameters())
        gradient_differences = []
        for name, parameter in control_grads.items():
            difference = (parameter.grad - candidate_grads[name].grad).float()
            max_abs = difference.abs().max().item()
            relative_norm = (
                difference.norm() / parameter.grad.float().norm().clamp_min(1e-30)
            ).item()
            gradient_differences.append(
                {"name": name, "max_abs": max_abs, "relative_norm": relative_norm}
            )
            if (
                max_abs > THRESHOLDS["shared_gradient_max_abs"]
                or relative_norm > THRESHOLDS["shared_gradient_relative_norm"]
            ):
                raise RuntimeError(
                    f"{label} shared gradient mismatch: {name}; "
                    f"max_abs={max_abs:.9g}; relative_norm={relative_norm:.9g}"
                )
        gate_grad_norms = [
            module.channel_conv.weight.grad.float().norm().item()
            for module in eca_modules(candidate)
        ]
        if any(not math.isfinite(value) or value == 0 for value in gate_grad_norms):
            raise RuntimeError(f"{label} invalid initial ECA gradients")
        results.append(
            {
                "target_type": label,
                "loss": candidate_loss.item(),
                "gate_gradient_norms": gate_grad_norms,
                "initial_gate_stats": gate_snapshot(candidate, inputs),
                "worst_shared_gradient_max_abs": max(
                    gradient_differences, key=lambda item: item["max_abs"]
                ),
                "worst_shared_gradient_relative_norm": max(
                    gradient_differences, key=lambda item: item["relative_norm"]
                ),
            }
        )
    return facts, results


def first_update_check(cpu_batch, label, device):
    model = build_model("candidate", device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    model.train()
    inputs = cpu_batch[0].to(device)
    targets = cpu_batch[1].to(device)
    optimizer.zero_grad()
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    gradient_norms = [
        module.channel_conv.weight.grad.float().norm().item()
        for module in eca_modules(model)
    ]
    optimizer.step()
    weights = [
        module.channel_conv.weight.detach().float() for module in eca_modules(model)
    ]
    stats = gate_snapshot(model, inputs)
    result = {
        "target_type": label,
        "loss": loss.item(),
        "gradient_norms": gradient_norms,
        "weight_norms": [weight.norm().item() for weight in weights],
        "max_abs_weight": max(weight.abs().max().item() for weight in weights),
        "gate_stats": stats,
    }
    failures = []
    if any(value == 0 or not math.isfinite(value) for value in result["weight_norms"]):
        failures.append(f"{label} ECA weight did not move")
    if result["max_abs_weight"] > THRESHOLDS["first_max_abs_weight"]:
        failures.append(f"{label} first ECA weight exceeded bound")
    for index, block in enumerate(stats):
        if (
            block["min"] < THRESHOLDS["first_gate_min"]
            or block["max"] > THRESHOLDS["first_gate_max"]
        ):
            failures.append(f"{label} block {index} first gate range failed")
        if (
            block["mean"] < THRESHOLDS["first_mean_min"]
            or block["mean"] > THRESHOLDS["first_mean_max"]
        ):
            failures.append(f"{label} block {index} first gate mean failed")
    return result, failures


def run_arm(arm):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    corpus = torch.load(CORPUS_PATH, map_location="cpu", weights_only=False)
    sequence = [
        (label, batch)
        for hard, soft in zip(corpus["strong_hard"], corpus["strong_soft"], strict=True)
        for label, batch in (("hard", hard), ("soft", soft))
    ]
    model = build_model(arm, device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    model.train()
    beta = 0.95
    loss_ema = 0.0
    losses = []
    shares = []
    gate_records = {"hard": [], "soft": []}
    gradient_sums = {"hard": [0.0, 0.0, 0.0], "soft": [0.0, 0.0, 0.0]}
    for step, (label, (cpu_inputs, cpu_targets)) in enumerate(sequence, start=1):
        inputs = cpu_inputs.to(device, non_blocking=True)
        targets = cpu_targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        if arm == "candidate":
            for index, module in enumerate(eca_modules(model)):
                gradient_sums[label][index] += (
                    module.channel_conv.weight.grad.float().norm().item()
                )
        optimizer.step()
        torch.cuda.synchronize()
        value = loss.item()
        loss_ema = beta * loss_ema + (1 - beta) * value
        losses.append(value)
        shares.append(class_share(outputs))
        if arm == "candidate" and (step <= 20 or step % 20 == 0):
            gate_records[label].append(
                {"step": step, "blocks": gate_snapshot(model, inputs)}
            )
        if not math.isfinite(value) or not all_finite(model, optimizer):
            raise RuntimeError(f"{arm} non-finite state at step {step}")
    result = {
        "arm": arm,
        "constructor": f"ResNet(3, 10, 2, use_eca={arm == 'candidate'})",
        "num_steps": len(sequence),
        "losses": losses,
        "class_shares": shares,
        "terminal_loss_ema": loss_ema / (1 - beta ** len(sequence)),
        "gate_records": gate_records,
        "gate_gradient_sums": gradient_sums,
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
    return json.loads(
        [line for line in completed.stdout.splitlines() if line.strip()][-1]
    )


def parent():
    if sha256_file(CORPUS_PATH) != EXPECTED_CORPUS_SHA:
        raise RuntimeError("immutable corpus SHA mismatch")
    corpus = torch.load(CORPUS_PATH, map_location="cpu", weights_only=False)
    counts, corpus_metadata = validate_corpus(corpus)
    device = torch.device("cuda")
    facts, identity = identity_checks(
        device, corpus["strong_hard"][0], corpus["strong_soft"][0]
    )
    first_updates = []
    failures = []
    for label, batch in (
        ("hard", corpus["strong_hard"][0]),
        ("soft", corpus["strong_soft"][0]),
    ):
        result, update_failures = first_update_check(batch, label, device)
        first_updates.append(result)
        failures.extend(update_failures)
    del corpus
    torch.cuda.empty_cache()
    control = run_child("control")
    candidate = run_child("candidate")
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
        failures.append(f"loss EMA ratio {ratio:.6f} > {THRESHOLDS['loss_ema_ratio']}")
    for label in ("hard", "soft"):
        if any(
            value == 0 or not math.isfinite(value)
            for value in candidate["gate_gradient_sums"][label]
        ):
            failures.append(f"missing {label} ECA gradients")
        for record in candidate["gate_records"][label]:
            for index, block in enumerate(record["blocks"]):
                if (
                    block["min"] < THRESHOLDS["trajectory_gate_min"]
                    or block["max"] > THRESHOLDS["trajectory_gate_max"]
                ):
                    failures.append(
                        f"{label} step {record['step']} block {index} gate range failed"
                    )
                if (
                    block["mean"] < THRESHOLDS["trajectory_mean_min"]
                    or block["mean"] > THRESHOLDS["trajectory_mean_max"]
                ):
                    failures.append(
                        f"{label} step {record['step']} block {index} gate mean failed"
                    )
    report = {
        "status": "failed" if failures else "pass",
        "thresholds": THRESHOLDS,
        "corpus_path": str(CORPUS_PATH),
        "corpus_sha256": EXPECTED_CORPUS_SHA,
        "bucket_counts": counts,
        "corpus_metadata": corpus_metadata,
        "structural_facts": facts,
        "identity_checks": identity,
        "first_updates": first_updates,
        "control": control,
        "candidate": candidate,
        "candidate_only_concentration_events": concentration,
        "candidate_control_loss_ema_ratio": ratio,
        "failures": failures,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
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
