import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
STRONG_PATH = (
    PROJECT_ROOT
    / ".autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight-corpus.pt"
)
WEAK_PATH = (
    PROJECT_ROOT
    / ".autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/028/weak-corpus.pt"
)
REPORT_PATH = EXPERIMENT_DIR / "preflight-report.json"
EXPECTED_STRONG_SHA = "e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946"
EXPECTED_WEAK_SHA = "ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032"
CHANGED = {
    "conv1": math.sqrt(3 / 32),
    "layer2.0.conv1": 1 / math.sqrt(2),
    "layer3.0.conv1": 1 / math.sqrt(2),
}
PAIRS = {
    "conv1": "bn1",
    "layer2.0.conv1": "layer2.0.bn1",
    "layer3.0.conv1": "layer3.0.bn1",
}
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_digest(batches):
    digest = hashlib.sha256()
    for inputs, targets in batches:
        digest.update(inputs.contiguous().numpy().tobytes())
        digest.update(targets.contiguous().numpy().tobytes())
    return digest.hexdigest()


def backend_flags():
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def control_init(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(module.weight)


def make_model(arm):
    if arm == "candidate":
        return train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER)
    original = train.ResNet._weights_init
    try:
        train.ResNet._weights_init = staticmethod(control_init)
        return train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER)
    finally:
        train.ResNet._weights_init = staticmethod(original)


def seeded_model(arm, device="cpu"):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = make_model(arm)
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state().clone()
    return model.to(device), cpu_state, cuda_state


def construction_gate():
    control, control_cpu_rng, control_cuda_rng = seeded_model("control")
    candidate, candidate_cpu_rng, candidate_cuda_rng = seeded_model("candidate")
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    if list(control_state) != list(candidate_state):
        raise RuntimeError("state ordering mismatch")
    modules = dict(candidate.named_modules())
    conv_names = [
        name for name, module in modules.items() if isinstance(module, torch.nn.Conv2d)
    ]
    bn_names = [
        name
        for name, module in modules.items()
        if isinstance(module, torch.nn.BatchNorm2d)
    ]
    linear_names = [
        name for name, module in modules.items() if isinstance(module, torch.nn.Linear)
    ]
    unequal = {
        name
        for name in conv_names
        if modules[name].in_channels != modules[name].out_channels
    }
    if (
        len(conv_names) != 19
        or len(bn_names) != 19
        or linear_names != ["fc"]
        or unequal != set(CHANGED)
    ):
        raise RuntimeError("module inventory mismatch")
    if sum(parameter.numel() for parameter in candidate.parameters()) != 1_073_962:
        raise RuntimeError("parameter count mismatch")
    changed_report = {}
    unequal_keys = {f"{name}.weight" for name in CHANGED}
    for key, control_tensor in control_state.items():
        candidate_tensor = candidate_state[key]
        if key not in unequal_keys:
            if not torch.equal(control_tensor, candidate_tensor):
                raise RuntimeError(f"unaffected tensor differs: {key}")
            continue
        name = key.removesuffix(".weight")
        scale = CHANGED[name]
        maximum_error = float((candidate_tensor - control_tensor * scale).abs().max())
        if not torch.allclose(
            candidate_tensor, control_tensor * scale, atol=1e-7, rtol=1e-6
        ):
            raise RuntimeError(f"analytic scaling mismatch: {name}")
        changed_report[name] = {
            "shape": list(control_tensor.shape),
            "analytic_scale": scale,
            "norm_ratio": float(candidate_tensor.norm() / control_tensor.norm()),
            "std_ratio": float(candidate_tensor.std() / control_tensor.std()),
            "max_abs_scale_error": maximum_error,
        }
    if not torch.equal(control_cpu_rng, candidate_cpu_rng) or not torch.equal(
        control_cuda_rng, candidate_cuda_rng
    ):
        raise RuntimeError("post-construction RNG mismatch")
    return {
        "conv_count": len(conv_names),
        "bn_count": len(bn_names),
        "linear_names": linear_names,
        "unequal_fan_convs": sorted(unequal),
        "parameter_count": sum(p.numel() for p in candidate.parameters()),
        "cpu_rng_equal": True,
        "cuda_rng_equal": True,
        "changed": changed_report,
    }


def capture_pair(arm, inputs, targets):
    model, _cpu_rng, _cuda_rng = seeded_model(arm, "cuda")
    model.train()
    captures = {}
    handles = []
    modules = dict(model.named_modules())
    for conv_name, bn_name in PAIRS.items():
        handles.append(
            modules[conv_name].register_forward_hook(
                lambda _module, _args, output, name=conv_name: captures.__setitem__(
                    f"conv:{name}", output.detach().clone()
                )
            )
        )
        handles.append(
            modules[bn_name].register_forward_hook(
                lambda _module, _args, output, name=conv_name: captures.__setitem__(
                    f"bn:{name}", output.detach().clone()
                )
            )
        )
    cuda_inputs = inputs.cuda(non_blocking=True)
    cuda_targets = targets.cuda(non_blocking=True)
    outputs = model(cuda_inputs)
    loss = F.cross_entropy(outputs, cuda_targets)
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    bn_counts = sorted(
        {
            int(module.num_batches_tracked)
            for module in model.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        }
    )
    min_var = min(
        float(module.running_var.min())
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )
    return {
        "outputs": outputs.detach().cpu(),
        "loss": float(loss),
        "captures": {key: value.cpu() for key, value in captures.items()},
        "bn_counts": bn_counts,
        "min_running_var": min_var,
    }


def initial_function_gate(strong_batches):
    selected = {
        "hard": next(batch for batch in strong_batches if batch[1].ndim == 1),
        "soft": next(batch for batch in strong_batches if batch[1].ndim == 2),
    }
    cases = {}
    failures = []
    for kind, (inputs, targets) in selected.items():
        control = capture_pair("control", inputs, targets)
        candidate = capture_pair("candidate", inputs, targets)
        output_delta = candidate["outputs"] - control["outputs"]
        relative_logit_l2 = float(output_delta.norm() / control["outputs"].norm())
        loss_ratio = candidate["loss"] / control["loss"]
        layer_report = {}
        for name, analytic in CHANGED.items():
            control_pre = control["captures"][f"conv:{name}"].float()
            candidate_pre = candidate["captures"][f"conv:{name}"].float()
            control_post = control["captures"][f"bn:{name}"].float()
            candidate_post = candidate["captures"][f"bn:{name}"].float()
            pre_ratio = float(
                candidate_pre.square().mean().sqrt()
                / control_pre.square().mean().sqrt()
            )
            post_ratio = float(
                candidate_post.square().mean().sqrt()
                / control_post.square().mean().sqrt()
            )
            layer_report[name] = {
                "analytic_scale": analytic,
                "pre_bn_rms_ratio": pre_ratio,
                "post_bn_rms_ratio": post_ratio,
            }
            if abs(pre_ratio / analytic - 1) > 0.02:
                failures.append(f"{kind} {name} pre-BN ratio {pre_ratio:.6f}")
            if not 0.98 <= post_ratio <= 1.02:
                failures.append(f"{kind} {name} post-BN ratio {post_ratio:.6f}")
        if relative_logit_l2 > 0.02 or not 0.95 <= loss_ratio <= 1.05:
            failures.append(
                f"{kind} initial output/loss {relative_logit_l2:.6f}/{loss_ratio:.6f}"
            )
        if control["bn_counts"] != [1] or candidate["bn_counts"] != [1]:
            failures.append(f"{kind} BN counter mismatch")
        if control["min_running_var"] <= 0 or candidate["min_running_var"] <= 0:
            failures.append(f"{kind} nonpositive BN variance")
        cases[kind] = {
            "relative_logit_l2": relative_logit_l2,
            "loss_ratio": loss_ratio,
            "layers": layer_report,
            "control_min_running_var": control["min_running_var"],
            "candidate_min_running_var": candidate["min_running_var"],
        }
    return {
        "status": "failed" if failures else "pass",
        "cases": cases,
        "failures": failures,
    }


def tensor_norm(tensors):
    return math.sqrt(sum(float(tensor.float().square().sum()) for tensor in tensors))


def class_share(outputs):
    counts = outputs.argmax(1).bincount(minlength=train.NUM_CLASSES)
    return float(counts.max()) / outputs.shape[0]


def all_finite(model, optimizer):
    tensors = list(model.parameters()) + list(model.buffers())
    tensors += [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def weak_lr(index):
    progress = 0.8 + 0.2 * index / 63
    cosine_progress = (progress - train.LR_HOLD_FRACTION) / (
        1.0 - train.LR_HOLD_FRACTION
    )
    return train.MIN_LR + 0.5 * (train.ANNEAL_START_LR - train.MIN_LR) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )


def run_trajectory(arm):
    strong = torch.load(STRONG_PATH, map_location="cpu", weights_only=False)
    weak = torch.load(WEAK_PATH, map_location="cpu", weights_only=False)
    model, _cpu_rng, _cuda_rng = seeded_model(arm, "cuda")
    model.train()
    parameters = list(model.parameters())
    named_parameters = dict(model.named_parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    records = []
    phase_ema = {"strong": 0.0, "weak": 0.0}
    phase_count = {"strong": 0, "weak": 0}
    update_history = []
    beta = 0.95
    batches = [("strong", index, batch) for index, batch in enumerate(strong)] + [
        ("weak", index, batch) for index, batch in enumerate(weak)
    ]
    for step, (phase, phase_index, (cpu_inputs, cpu_targets)) in enumerate(
        batches, start=1
    ):
        lr = train.LR if phase == "strong" else weak_lr(phase_index)
        optimizer.param_groups[0]["lr"] = lr
        inputs = cpu_inputs.cuda(non_blocking=True)
        targets = cpu_targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        gradient_norm = tensor_norm(
            [parameter.grad for parameter in parameters if parameter.grad is not None]
        )
        starts = [parameter.detach().clone() for parameter in parameters]
        parameter_norm = tensor_norm(starts)
        changed_starts = {
            name: named_parameters[f"{name}.weight"].detach().clone()
            for name in CHANGED
        }
        optimizer.step()
        torch.cuda.synchronize()
        update_norm = tensor_norm(
            [
                parameter.detach() - start
                for parameter, start in zip(parameters, starts, strict=True)
            ]
        )
        changed_updates = {}
        for name, start in changed_starts.items():
            parameter = named_parameters[f"{name}.weight"].detach()
            changed_updates[name] = float((parameter - start).norm() / start.norm())
        preceding_ratio = (
            update_norm / statistics.median(update_history[-16:])
            if len(update_history) >= 16
            else None
        )
        update_history.append(update_norm)
        value = float(loss)
        phase_ema[phase] = beta * phase_ema[phase] + (1 - beta) * value
        phase_count[phase] += 1
        record = {
            "step": step,
            "phase": phase,
            "lr": lr,
            "loss": value,
            "class_share": class_share(outputs),
            "logit_rms": float(outputs.float().square().mean().sqrt()),
            "gradient_norm": gradient_norm,
            "update_norm": update_norm,
            "parameter_norm": parameter_norm,
            "update_parameter_ratio": update_norm / parameter_norm,
            "preceding_update_median_ratio": preceding_ratio,
            "changed_relative_updates": changed_updates,
        }
        records.append(record)
        if not math.isfinite(value) or not all_finite(model, optimizer):
            raise RuntimeError(f"{arm} nonfinite state at step {step}")
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
        "backend": backend_flags(),
        "records": records,
        "terminal_ema": {
            phase: phase_ema[phase] / (1 - beta ** phase_count[phase])
            for phase in phase_ema
        },
        "bn_counts": bn_counts,
        "min_running_var": min_running_var,
        "momentum_buffers": sum(
            "momentum_buffer" in state for state in optimizer.state.values()
        ),
    }
    print(json.dumps(result))


def trajectory_child(arm):
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--trajectory", arm],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return json.loads(
        [line for line in completed.stdout.splitlines() if line.strip()][-1]
    )


def compare_trajectories(control, candidate):
    failures = []
    concentration = []
    maxima = {
        "logit_rms_ratio": 0.0,
        "gradient_norm_ratio": 0.0,
        "update_norm_ratio": 0.0,
        "update_parameter_ratio": 0.0,
        "preceding_update_median_ratio": 0.0,
        "changed_relative_updates": {name: 0.0 for name in CHANGED},
    }
    if control["backend"] != candidate["backend"]:
        failures.append("backend flags differ")
    for control_record, candidate_record in zip(
        control["records"], candidate["records"], strict=True
    ):
        step = control_record["step"]
        if (
            candidate_record["class_share"] > 0.95
            and control_record["class_share"] <= 0.95
        ):
            concentration.append(
                {
                    "step": step,
                    "control": control_record["class_share"],
                    "candidate": candidate_record["class_share"],
                }
            )
        for metric, source in (
            ("logit_rms_ratio", "logit_rms"),
            ("gradient_norm_ratio", "gradient_norm"),
            ("update_norm_ratio", "update_norm"),
        ):
            ratio = candidate_record[source] / max(control_record[source], 1e-30)
            maxima[metric] = max(maxima[metric], ratio)
            if ratio > 5:
                failures.append(f"{metric} {ratio:.6f} at step {step}")
        maxima["update_parameter_ratio"] = max(
            maxima["update_parameter_ratio"],
            candidate_record["update_parameter_ratio"],
        )
        if candidate_record["update_parameter_ratio"] > 0.25:
            failures.append(f"whole update/parameter >.25 at step {step}")
        preceding = candidate_record["preceding_update_median_ratio"]
        if preceding is not None:
            maxima["preceding_update_median_ratio"] = max(
                maxima["preceding_update_median_ratio"], preceding
            )
            if preceding > 5:
                failures.append(f"update/median {preceding:.6f} at step {step}")
        for name, ratio in candidate_record["changed_relative_updates"].items():
            maxima["changed_relative_updates"][name] = max(
                maxima["changed_relative_updates"][name], ratio
            )
            if ratio > 0.5:
                failures.append(f"{name} relative update {ratio:.6f} at step {step}")
    if concentration:
        failures.append(
            f"candidate-only concentration at {[item['step'] for item in concentration]}"
        )
    ema_ratios = {
        phase: candidate["terminal_ema"][phase] / control["terminal_ema"][phase]
        for phase in ("strong", "weak")
    }
    if any(value > 1.5 for value in ema_ratios.values()):
        failures.append(f"terminal EMA ratios {ema_ratios}")
    for arm in (control, candidate):
        if arm["bn_counts"] != [264] or arm["min_running_var"] <= 0:
            failures.append(f"{arm['arm']} BN state incomplete")
        if arm["momentum_buffers"] != len(list(make_model("control").parameters())):
            failures.append(f"{arm['arm']} momentum state incomplete")
    return {
        "status": "failed" if failures else "pass",
        "maxima": maxima,
        "candidate_only_concentration": concentration,
        "control_terminal_ema": control["terminal_ema"],
        "candidate_terminal_ema": candidate["terminal_ema"],
        "terminal_ema_ratios": ema_ratios,
        "control_bn_counts": control["bn_counts"],
        "candidate_bn_counts": candidate["bn_counts"],
        "control_min_running_var": control["min_running_var"],
        "candidate_min_running_var": candidate["min_running_var"],
        "failures": failures,
    }


def load_and_validate_corpora():
    strong_sha = file_sha256(STRONG_PATH)
    weak_sha = file_sha256(WEAK_PATH)
    if strong_sha != EXPECTED_STRONG_SHA or weak_sha != EXPECTED_WEAK_SHA:
        raise RuntimeError("corpus file SHA mismatch")
    strong = torch.load(STRONG_PATH, map_location="cpu", weights_only=False)
    weak = torch.load(WEAK_PATH, map_location="cpu", weights_only=False)
    if len(strong) != 200 or len(weak) != 64:
        raise RuntimeError("corpus batch count mismatch")
    if (
        sum(targets.ndim == 1 for _, targets in strong) != 94
        or sum(targets.ndim == 2 for _, targets in strong) != 106
    ):
        raise RuntimeError("strong target-rank coverage mismatch")
    for inputs, targets in strong + weak:
        if inputs.shape != (128, 3, 32, 32) or inputs.dtype != torch.float32:
            raise RuntimeError("corpus input contract mismatch")
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise RuntimeError("nonfinite corpus tensor")
    if any(
        targets.shape != (128,) or targets.dtype != torch.int64 for _, targets in weak
    ):
        raise RuntimeError("weak target contract mismatch")
    return (
        strong,
        weak,
        {
            "strong_file_sha256": strong_sha,
            "weak_file_sha256": weak_sha,
            "strong_tensor_sha256": corpus_digest(strong),
            "weak_tensor_sha256": corpus_digest(weak),
            "strong_batches": len(strong),
            "strong_hard_batches": 94,
            "strong_soft_batches": 106,
            "weak_batches": len(weak),
        },
    )


def parent():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    construction = construction_gate()
    print(json.dumps({"stage": "construction", "status": "pass"}), flush=True)
    strong, _weak, corpus = load_and_validate_corpora()
    initial_function = initial_function_gate(strong)
    print(
        json.dumps({"stage": "initial_function", "status": initial_function["status"]}),
        flush=True,
    )
    control = trajectory_child("control")
    print(
        json.dumps({"stage": "trajectory", "arm": "control", "status": "complete"}),
        flush=True,
    )
    candidate = trajectory_child("candidate")
    print(
        json.dumps({"stage": "trajectory", "arm": "candidate", "status": "complete"}),
        flush=True,
    )
    trajectory = compare_trajectories(control, candidate)
    strong_after, weak_after, corpus_after = load_and_validate_corpora()
    del strong_after, weak_after
    if corpus != corpus_after:
        raise RuntimeError("corpus declaration changed after replay")
    failures = initial_function["failures"] + trajectory["failures"]
    report = {
        "status": "failed" if failures else "pass",
        "construction": construction,
        "initial_function": initial_function,
        "corpus": corpus,
        "trajectory": trajectory,
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
    args = parser.parse_args()
    if args.trajectory:
        run_trajectory(args.trajectory)
    else:
        parent()


if __name__ == "__main__":
    main()
