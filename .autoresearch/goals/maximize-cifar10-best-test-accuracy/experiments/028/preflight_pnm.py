import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
STRONG_CORPUS_PATH = (
    PROJECT_ROOT
    / ".autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight-corpus.pt"
)
WEAK_CORPUS_PATH = EXPERIMENT_DIR / "weak-corpus.pt"
REPORT_PATH = EXPERIMENT_DIR / "preflight-report.json"
EXPECTED_STRONG_SHA256 = (
    "e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946"
)
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


def tensor_list_norm(tensors):
    return math.sqrt(sum(tensor.float().square().sum().item() for tensor in tensors))


def parameter_delta_norm(parameters, starts):
    return tensor_list_norm(
        [
            parameter.detach() - start
            for parameter, start in zip(parameters, starts, strict=True)
        ]
    )


def all_finite(tensors):
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def class_share(outputs):
    counts = outputs.argmax(1).bincount(minlength=train.NUM_CLASSES)
    return counts.max().item() / outputs.shape[0]


def pnm_buffers(optimizer, name):
    return [
        optimizer.state[parameter][name]
        for parameter in optimizer.param_groups[0]["params"]
    ]


def materialize_or_load_weak_corpus():
    if WEAK_CORPUS_PATH.exists():
        batches = torch.load(WEAK_CORPUS_PATH, map_location="cpu", weights_only=False)
        if len(batches) != 64 or any(targets.ndim != 1 for _, targets in batches):
            raise RuntimeError("existing weak corpus has an invalid structure")
        return batches

    torch.manual_seed(20260806)
    mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)
    weak_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    loader = train.make_train_loader(weak_transform)
    iterator = iter(loader)
    batches = []
    for _ in range(64):
        inputs, targets = next(iterator)
        if targets.ndim != 1:
            raise RuntimeError("weak corpus unexpectedly contains soft targets")
        batches.append((inputs.contiguous().clone(), targets.contiguous().clone()))
    iterator = None
    stopped = train.shutdown_train_loader(loader)
    if len(stopped) != train.NUM_WORKERS:
        raise RuntimeError(f"expected {train.NUM_WORKERS} stopped workers, got {len(stopped)}")
    torch.save(batches, WEAK_CORPUS_PATH)
    return batches


def constant_direction_oracle():
    initial = torch.linspace(-0.75, 0.75, 257)
    control_parameter = torch.nn.Parameter(initial.clone())
    candidate_parameter = torch.nn.Parameter(initial.clone())
    gradient = torch.linspace(0.6, -0.4, 257)
    control = torch.optim.SGD(
        [control_parameter], lr=0.1, momentum=train.MOMENTUM, weight_decay=0.0
    )
    candidate = train.ScaleMatchedPNM(
        [candidate_parameter], lr=0.1, momentum=train.MOMENTUM, weight_decay=0.0
    )
    max_abs_direction_error = 0.0
    max_relative_direction_error = 0.0
    first_parameter_error = None
    roundtrips = {}

    for step in range(1, 65):
        lr = 0.1 if step <= 32 else 0.01
        control.param_groups[0]["lr"] = lr
        candidate.param_groups[0]["lr"] = lr
        control_before = control_parameter.detach().clone()
        candidate_before = candidate_parameter.detach().clone()
        control_parameter.grad = gradient.clone()
        candidate_parameter.grad = gradient.clone()
        control.step()
        candidate.step()
        control_delta = control_parameter.detach() - control_before
        candidate_delta = candidate_parameter.detach() - candidate_before
        error = (candidate_delta - control_delta).abs().max().item()
        relative = (candidate_delta - control_delta).norm().item() / max(
            control_delta.norm().item(), 1e-30
        )
        max_abs_direction_error = max(max_abs_direction_error, error)
        max_relative_direction_error = max(max_relative_direction_error, relative)
        if step == 1:
            first_parameter_error = (
                candidate_parameter.detach() - control_parameter.detach()
            ).abs().max().item()
        if step in (1, 2, 63, 64):
            restored_parameter = torch.nn.Parameter(candidate_parameter.detach().clone())
            restored = train.ScaleMatchedPNM(
                [restored_parameter],
                lr=lr,
                momentum=train.MOMENTUM,
                weight_decay=0.0,
            )
            restored.load_state_dict(copy.deepcopy(candidate.state_dict()))
            state_equal = restored.param_groups[0]["pnm_step"] == step and all(
                torch.equal(
                    restored.state[restored_parameter][name],
                    candidate.state[candidate_parameter][name],
                )
                for name in ("positive_buffer", "negative_buffer")
            )
            roundtrips[str(step)] = state_equal

    return {
        "steps": 64,
        "max_abs_direction_error": max_abs_direction_error,
        "max_relative_direction_error": max_relative_direction_error,
        "first_parameter_max_abs_error": first_parameter_error,
        "state_roundtrips": roundtrips,
    }


def changing_direction_oracle():
    candidate_parameter = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 193))
    manual_parameter = candidate_parameter.detach().clone()
    candidate = train.ScaleMatchedPNM(
        [candidate_parameter],
        lr=0.1,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    positive = torch.zeros_like(manual_parameter)
    negative = torch.zeros_like(manual_parameter)
    max_parameter_error = 0.0
    max_buffer_error = 0.0
    inactive_preserved = True
    gradients_unchanged = True
    rng_unchanged = True
    rho = train.MOMENTUM**2

    for step in range(1, 202):
        lr = 0.1 if step <= 100 else 0.013
        candidate.param_groups[0]["lr"] = lr
        index = torch.arange(manual_parameter.numel(), dtype=torch.float32)
        gradient = torch.sin(index * 0.17 + step * 0.31) * (0.1 + step / 1000)
        gradient_before = gradient.clone()
        candidate_parameter.grad = gradient
        inactive_name = "negative_buffer" if step % 2 == 1 else "positive_buffer"
        if candidate.state[candidate_parameter]:
            inactive_before = candidate.state[candidate_parameter][inactive_name].clone()
        else:
            inactive_before = torch.zeros_like(candidate_parameter)
        cpu_rng_before = torch.get_rng_state().clone()
        candidate.step()
        rng_unchanged = rng_unchanged and torch.equal(cpu_rng_before, torch.get_rng_state())
        gradients_unchanged = gradients_unchanged and torch.equal(
            candidate_parameter.grad, gradient_before
        )

        direction = gradient_before + train.WEIGHT_DECAY * manual_parameter
        if step % 2 == 1:
            positive.mul_(rho).add_(direction, alpha=1.0 - rho)
            raw = 2.0 * positive - negative
        else:
            negative.mul_(rho).add_(direction, alpha=1.0 - rho)
            raw = 2.0 * negative - positive
        scale = train.ScaleMatchedPNM.signal_scale(step, train.MOMENTUM)
        manual_parameter.add_(raw, alpha=-lr * scale / math.sqrt(5.0))

        candidate_state = candidate.state[candidate_parameter]
        inactive_preserved = inactive_preserved and torch.equal(
            candidate_state[inactive_name], inactive_before
        )
        max_parameter_error = max(
            max_parameter_error,
            (candidate_parameter.detach() - manual_parameter).abs().max().item(),
        )
        max_buffer_error = max(
            max_buffer_error,
            (candidate_state["positive_buffer"] - positive).abs().max().item(),
            (candidate_state["negative_buffer"] - negative).abs().max().item(),
        )

    return {
        "steps": 201,
        "max_parameter_abs_error": max_parameter_error,
        "max_buffer_abs_error": max_buffer_error,
        "inactive_buffer_preserved": inactive_preserved,
        "gradients_unchanged": gradients_unchanged,
        "rng_unchanged": rng_unchanged,
        "pnm_step": candidate.param_groups[0]["pnm_step"],
    }


def run_model_trajectory(device, strong_batches, weak_batches):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    template = train.ResNet(
        train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER
    )
    control_model = copy.deepcopy(template).to(device)
    candidate_model = copy.deepcopy(template).to(device)
    control_parameters = list(control_model.parameters())
    candidate_parameters = list(candidate_model.parameters())
    if sum(parameter.numel() for parameter in control_parameters) != 1_073_962:
        raise RuntimeError("unexpected parameter count")
    if not all(
        torch.equal(control, candidate)
        for control, candidate in zip(
            control_parameters, candidate_parameters, strict=True
        )
    ):
        raise RuntimeError("aligned models did not start byte-identical")

    control_optimizer = torch.optim.SGD(
        control_parameters,
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    candidate_optimizer = train.ScaleMatchedPNM(
        candidate_parameters,
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    records = []
    concentration_failures = []
    update_ratios = []
    candidate_update_norms = []
    recent_median_ratios = []
    inactive_preserved = True
    gradients_unchanged = True
    rng_unchanged = True
    phase_emas = {}
    all_batches = [("strong", batch) for batch in strong_batches] + [
        ("weak", batch) for batch in weak_batches
    ]
    phase_accumulators = {
        "strong": {"control": 0.0, "candidate": 0.0, "count": 0},
        "weak": {"control": 0.0, "candidate": 0.0, "count": 0},
    }
    beta = 0.95

    for batch_index, (phase, (cpu_inputs, cpu_targets)) in enumerate(
        all_batches, start=1
    ):
        lr = train.LR if phase == "strong" else train.ANNEAL_START_LR
        control_optimizer.param_groups[0]["lr"] = lr
        candidate_optimizer.param_groups[0]["lr"] = lr
        inputs = cpu_inputs.to(device, non_blocking=True)
        targets = cpu_targets.to(device, non_blocking=True)

        control_model.train()
        control_optimizer.zero_grad()
        control_outputs = control_model(inputs)
        control_loss = F.cross_entropy(control_outputs, targets)
        control_loss.backward()
        control_starts = [parameter.detach().clone() for parameter in control_parameters]
        control_gradient_norm = tensor_list_norm(
            [parameter.grad for parameter in control_parameters]
        )
        control_optimizer.step()

        candidate_model.train()
        candidate_optimizer.zero_grad()
        candidate_outputs = candidate_model(inputs)
        candidate_loss = F.cross_entropy(candidate_outputs, targets)
        candidate_loss.backward()
        candidate_starts = [
            parameter.detach().clone() for parameter in candidate_parameters
        ]
        candidate_gradients = [
            parameter.grad.detach().clone() for parameter in candidate_parameters
        ]
        candidate_gradient_norm = tensor_list_norm(candidate_gradients)
        inactive_name = (
            "negative_buffer" if batch_index % 2 == 1 else "positive_buffer"
        )
        if batch_index == 1:
            inactive_before = [
                torch.zeros_like(parameter) for parameter in candidate_parameters
            ]
        else:
            inactive_before = [
                tensor.clone()
                for tensor in pnm_buffers(candidate_optimizer, inactive_name)
            ]
        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state().clone()
        candidate_optimizer.step()
        torch.cuda.synchronize()
        rng_unchanged = (
            rng_unchanged
            and torch.equal(cpu_rng_before, torch.get_rng_state())
            and torch.equal(cuda_rng_before, torch.cuda.get_rng_state())
        )
        gradients_unchanged = gradients_unchanged and all(
            torch.equal(parameter.grad, before)
            for parameter, before in zip(
                candidate_parameters, candidate_gradients, strict=True
            )
        )
        inactive_preserved = inactive_preserved and all(
            torch.equal(after, before)
            for after, before in zip(
                pnm_buffers(candidate_optimizer, inactive_name),
                inactive_before,
                strict=True,
            )
        )

        control_update = parameter_delta_norm(control_parameters, control_starts)
        candidate_update = parameter_delta_norm(candidate_parameters, candidate_starts)
        update_ratio = candidate_update / max(control_update, 1e-30)
        update_ratios.append(update_ratio)
        if len(candidate_update_norms) >= 16:
            previous_median = statistics.median(candidate_update_norms[-16:])
            recent_median_ratios.append(candidate_update / max(previous_median, 1e-30))
        candidate_update_norms.append(candidate_update)

        control_share = class_share(control_outputs)
        candidate_share = class_share(candidate_outputs)
        if candidate_share > 0.95 and control_share <= 0.95:
            concentration_failures.append(
                {
                    "step": batch_index,
                    "phase": phase,
                    "candidate_share": candidate_share,
                    "control_share": control_share,
                }
            )

        accumulator = phase_accumulators[phase]
        accumulator["count"] += 1
        accumulator["control"] = (
            beta * accumulator["control"] + (1.0 - beta) * control_loss.item()
        )
        accumulator["candidate"] = (
            beta * accumulator["candidate"] + (1.0 - beta) * candidate_loss.item()
        )

        tensors_to_check = (
            control_parameters
            + candidate_parameters
            + list(control_model.buffers())
            + list(candidate_model.buffers())
            + [
                state["momentum_buffer"]
                for parameter in control_parameters
                if (state := control_optimizer.state[parameter])
            ]
            + pnm_buffers(candidate_optimizer, "positive_buffer")
            + pnm_buffers(candidate_optimizer, "negative_buffer")
        )
        finite = (
            math.isfinite(control_loss.item())
            and math.isfinite(candidate_loss.item())
            and all_finite(tensors_to_check)
        )
        bn_positive = all(
            (module.running_var > 0).all().item()
            for model in (control_model, candidate_model)
            for module in model.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        )
        if not finite or not bn_positive:
            raise RuntimeError(f"invalid finite/BN state at step {batch_index}")

        records.append(
            {
                "step": batch_index,
                "phase": phase,
                "target_ndim": targets.ndim,
                "control_loss": control_loss.item(),
                "candidate_loss": candidate_loss.item(),
                "control_class_share": control_share,
                "candidate_class_share": candidate_share,
                "control_gradient_norm": control_gradient_norm,
                "candidate_gradient_norm": candidate_gradient_norm,
                "control_update_norm": control_update,
                "candidate_update_norm": candidate_update,
                "candidate_control_update_ratio": update_ratio,
                "positive_state_norm": tensor_list_norm(
                    pnm_buffers(candidate_optimizer, "positive_buffer")
                ),
                "negative_state_norm": tensor_list_norm(
                    pnm_buffers(candidate_optimizer, "negative_buffer")
                ),
                "scale": train.ScaleMatchedPNM.signal_scale(
                    batch_index, train.MOMENTUM
                ),
            }
        )

    for phase, accumulator in phase_accumulators.items():
        debias = 1.0 - beta ** accumulator["count"]
        control_ema = accumulator["control"] / debias
        candidate_ema = accumulator["candidate"] / debias
        phase_emas[phase] = {
            "control": control_ema,
            "candidate": candidate_ema,
            "ratio": candidate_ema / control_ema,
        }

    expected_batches = len(all_batches)
    bn_counters_valid = all(
        module.num_batches_tracked.item() == expected_batches
        for model in (control_model, candidate_model)
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )
    return {
        "steps": expected_batches,
        "parameter_count": sum(
            parameter.numel() for parameter in candidate_parameters
        ),
        "concentration_failures": concentration_failures,
        "update_ratio_median": statistics.median(update_ratios),
        "update_ratio_p95": sorted(update_ratios)[
            math.ceil(0.95 * len(update_ratios)) - 1
        ],
        "update_ratio_max": max(update_ratios),
        "recent_median_ratio_max": max(recent_median_ratios),
        "phase_loss_emas": phase_emas,
        "inactive_buffer_preserved": inactive_preserved,
        "gradients_unchanged": gradients_unchanged,
        "rng_unchanged": rng_unchanged,
        "bn_counters_valid": bn_counters_valid,
        "pnm_step": candidate_optimizer.param_groups[0]["pnm_step"],
        "records": records,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.use_deterministic_algorithms(True)
    strong_sha256 = file_sha256(STRONG_CORPUS_PATH)
    if strong_sha256 != EXPECTED_STRONG_SHA256:
        raise RuntimeError(f"strong corpus SHA-256 mismatch: {strong_sha256}")
    strong_batches = torch.load(
        STRONG_CORPUS_PATH, map_location="cpu", weights_only=False
    )
    if len(strong_batches) != 200:
        raise RuntimeError("strong corpus does not contain 200 batches")
    weak_batches = materialize_or_load_weak_corpus()
    strong_tensor_digest_before = corpus_digest(strong_batches)
    weak_tensor_digest_before = corpus_digest(weak_batches)

    constant = constant_direction_oracle()
    changing = changing_direction_oracle()
    trajectory = run_model_trajectory(
        torch.device("cuda"), strong_batches, weak_batches
    )
    strong_tensor_digest_after = corpus_digest(strong_batches)
    weak_tensor_digest_after = corpus_digest(weak_batches)
    report = {
        "status": "pass",
        "strong_corpus_path": str(STRONG_CORPUS_PATH),
        "strong_file_sha256": strong_sha256,
        "strong_tensor_digest": strong_tensor_digest_before,
        "strong_batches": len(strong_batches),
        "strong_hard_batches": sum(
            targets.ndim == 1 for _, targets in strong_batches
        ),
        "strong_soft_batches": sum(
            targets.ndim == 2 for _, targets in strong_batches
        ),
        "weak_corpus_path": str(WEAK_CORPUS_PATH),
        "weak_file_sha256": file_sha256(WEAK_CORPUS_PATH),
        "weak_tensor_digest": weak_tensor_digest_before,
        "weak_batches": len(weak_batches),
        "corpora_unchanged": (
            strong_tensor_digest_before == strong_tensor_digest_after
            and weak_tensor_digest_before == weak_tensor_digest_after
        ),
        "constant_direction_oracle": constant,
        "changing_direction_oracle": changing,
        "trajectory": trajectory,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    failures = []
    if constant["max_abs_direction_error"] > 1e-6:
        failures.append("constant-direction absolute error")
    if constant["max_relative_direction_error"] > 1e-6:
        failures.append("constant-direction relative error")
    if constant["first_parameter_max_abs_error"] > 1e-7:
        failures.append("first-step parameter error")
    if not all(constant["state_roundtrips"].values()):
        failures.append("state_dict roundtrip")
    if changing["max_parameter_abs_error"] > 1e-6:
        failures.append("changing-direction parameter recurrence")
    if changing["max_buffer_abs_error"] > 1e-6:
        failures.append("changing-direction buffer recurrence")
    for key in (
        "inactive_buffer_preserved",
        "gradients_unchanged",
        "rng_unchanged",
    ):
        if not changing[key] or not trajectory[key]:
            failures.append(key)
    if not report["corpora_unchanged"] or not trajectory["bn_counters_valid"]:
        failures.append("corpus or BN integrity")
    if trajectory["pnm_step"] != trajectory["steps"]:
        failures.append("global PNM step")
    if trajectory["concentration_failures"]:
        failures.append("candidate-only class concentration")
    if trajectory["update_ratio_median"] > 1.30:
        failures.append("median total update ratio")
    if trajectory["update_ratio_max"] > 5.0:
        failures.append("paired update spike")
    if trajectory["recent_median_ratio_max"] > 10.0:
        failures.append("recent-median update spike")
    if any(values["ratio"] > 1.5 for values in trajectory["phase_loss_emas"].values()):
        failures.append("phase terminal loss EMA")
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(f"preflight gates failed: {failures}")

    compact = {key: value for key, value in report.items() if key != "trajectory"}
    compact["trajectory"] = {
        key: value for key, value in trajectory.items() if key != "records"
    }
    print(json.dumps(compact))


if __name__ == "__main__":
    main()
