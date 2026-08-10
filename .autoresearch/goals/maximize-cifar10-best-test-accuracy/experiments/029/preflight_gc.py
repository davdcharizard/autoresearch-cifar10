import copy
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
EXPECTED_STRONG_FILE = (
    "e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946"
)
EXPECTED_WEAK_FILE = (
    "ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032"
)
EXPECTED_STRONG_TENSORS = (
    "4242043f3a4cbc04c3de0c2ffbc9f78c5c01c8314ae695c2ba8e94a3e992ad40"
)
EXPECTED_WEAK_TENSORS = (
    "df97b02a24ff5f4ca17fe0697c31b2da1cbdb6e0b3c9ca14107cfe2444408eae"
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


def write_report(report):
    with REPORT_PATH.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def momentum_buffers(optimizer):
    return [
        optimizer.state[parameter]["momentum_buffer"]
        for parameter in optimizer.param_groups[0]["params"]
    ]


def projection_fixture():
    torch.manual_seed(42)
    model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER)
    conv_weights = [
        module.weight for module in model.modules() if isinstance(module, nn.Conv2d)
    ]
    if len(conv_weights) != 19 or len({id(weight) for weight in conv_weights}) != 19:
        raise RuntimeError("unexpected Conv2d eligibility")

    for index, parameter in enumerate(model.parameters()):
        values = torch.arange(parameter.numel(), dtype=torch.float32).reshape_as(parameter)
        parameter.grad = 0.05 * torch.sin(values * 0.013 + index) + 0.001 * (
            index + 1
        )
    raw = {id(parameter): parameter.grad.clone() for parameter in model.parameters()}
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    buffers_before = [buffer.detach().clone() for buffer in model.buffers()]
    cpu_rng_before = torch.get_rng_state().clone()
    train.centralize_conv_weight_gradients(model)

    max_reference_error = 0.0
    max_mean = 0.0
    norm_nonincreasing = True
    for weight in conv_weights:
        source = raw[id(weight)]
        reference = source.double() - source.double().mean(
            dim=(1, 2, 3), keepdim=True
        )
        max_reference_error = max(
            max_reference_error,
            (weight.grad.double() - reference).abs().max().item(),
        )
        max_mean = max(
            max_mean,
            weight.grad.mean(dim=(1, 2, 3)).abs().max().item(),
        )
        norm_nonincreasing = (
            norm_nonincreasing
            and weight.grad.norm().item() <= source.norm().item() + 1e-6
        )

    nonconv_unchanged = all(
        torch.equal(parameter.grad, raw[id(parameter)])
        for parameter in model.parameters()
        if id(parameter) not in {id(weight) for weight in conv_weights}
    )
    parameters_unchanged = all(
        torch.equal(parameter, before)
        for parameter, before in zip(model.parameters(), parameters_before, strict=True)
    )
    buffers_unchanged = all(
        torch.equal(buffer, before)
        for buffer, before in zip(model.buffers(), buffers_before, strict=True)
    )
    once = [weight.grad.clone() for weight in conv_weights]
    train.centralize_conv_weight_gradients(model)
    idempotence_error = max(
        (weight.grad - prior).abs().max().item()
        for weight, prior in zip(conv_weights, once, strict=True)
    )
    return {
        "eligible_conv_weights": len(conv_weights),
        "max_fp64_reference_abs_error": max_reference_error,
        "max_post_projection_filter_mean": max_mean,
        "idempotence_max_abs_error": idempotence_error,
        "norm_nonincreasing": norm_nonincreasing,
        "nonconv_gradients_unchanged": nonconv_unchanged,
        "parameters_unchanged": parameters_unchanged,
        "buffers_unchanged": buffers_unchanged,
        "cpu_rng_unchanged": torch.equal(cpu_rng_before, torch.get_rng_state()),
    }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 3, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 2)


def recurrence_fixture():
    torch.manual_seed(7)
    model = TinyModel()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=train.MOMENTUM, weight_decay=train.WEIGHT_DECAY
    )
    parameters = list(model.parameters())
    manual_parameters = [parameter.detach().clone() for parameter in parameters]
    manual_buffers = [torch.zeros_like(parameter) for parameter in parameters]
    max_parameter_error = 0.0
    max_buffer_error = 0.0
    for step in range(1, 6):
        lr = 0.1 if step <= 3 else 0.017
        optimizer.param_groups[0]["lr"] = lr
        raw_gradients = []
        for index, parameter in enumerate(parameters):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape_as(parameter)
            gradient = torch.cos(values * 0.021 + step * 0.37 + index) + 0.1 * index
            parameter.grad = gradient.clone()
            raw_gradients.append(gradient)
        train.centralize_conv_weight_gradients(model)
        optimizer.step()

        for index, (manual_parameter, raw_gradient) in enumerate(
            zip(manual_parameters, raw_gradients, strict=True)
        ):
            if index == 0:
                direction = raw_gradient - raw_gradient.mean(
                    dim=(1, 2, 3), keepdim=True
                )
            else:
                direction = raw_gradient
            direction = direction + train.WEIGHT_DECAY * manual_parameter
            if step == 1:
                manual_buffers[index].copy_(direction)
            else:
                manual_buffers[index].mul_(train.MOMENTUM).add_(direction)
            manual_parameter.add_(manual_buffers[index], alpha=-lr)

        installed_buffers = momentum_buffers(optimizer)
        max_parameter_error = max(
            max_parameter_error,
            max(
                (actual.detach() - expected).abs().max().item()
                for actual, expected in zip(parameters, manual_parameters, strict=True)
            ),
        )
        max_buffer_error = max(
            max_buffer_error,
            max(
                (actual - expected).abs().max().item()
                for actual, expected in zip(
                    installed_buffers, manual_buffers, strict=True
                )
            ),
        )
    return {
        "steps": 5,
        "max_parameter_abs_error": max_parameter_error,
        "max_momentum_abs_error": max_buffer_error,
    }


def conv_stage(name):
    if name == "conv1":
        return "stem"
    if name.startswith("layer1"):
        return "stage1"
    if name.startswith("layer2"):
        return "stage2"
    if name.startswith("layer3"):
        return "stage3"
    raise RuntimeError(f"unexpected Conv2d name: {name}")


def run_trajectory(device, strong_batches, weak_batches):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    template = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER)
    control_model = copy.deepcopy(template).to(device)
    candidate_model = copy.deepcopy(template).to(device)
    control_parameters = list(control_model.parameters())
    candidate_parameters = list(candidate_model.parameters())
    if sum(parameter.numel() for parameter in candidate_parameters) != 1_073_962:
        raise RuntimeError("unexpected parameter count")
    if not all(
        torch.equal(left, right)
        for left, right in zip(control_parameters, candidate_parameters, strict=True)
    ):
        raise RuntimeError("models do not start byte-identical")
    control_optimizer = torch.optim.SGD(
        control_parameters,
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    candidate_optimizer = torch.optim.SGD(
        candidate_parameters,
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    candidate_convs = [
        (name, module.weight)
        for name, module in candidate_model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    candidate_conv_ids = {id(weight) for _, weight in candidate_convs}
    records = []
    concentration_failures = []
    update_ratios = []
    candidate_update_norms = []
    recent_median_ratios = []
    max_conv_update_parameter_ratio = 0.0
    max_filter_mean = 0.0
    nonconv_unchanged = True
    projection_norm_nonincreasing = True
    rng_unchanged = True
    phase_accumulators = {
        phase: {"control": 0.0, "candidate": 0.0, "count": 0}
        for phase in ("strong", "weak")
    }
    stage_totals = {
        phase: {
            stage: {"raw_sq": 0.0, "projected_sq": 0.0, "removed_sq": 0.0}
            for stage in ("stem", "stage1", "stage2", "stage3")
        }
        for phase in ("strong", "weak")
    }
    beta = 0.95
    all_batches = [("strong", batch) for batch in strong_batches] + [
        ("weak", batch) for batch in weak_batches
    ]

    for step, (phase, (cpu_inputs, cpu_targets)) in enumerate(all_batches, start=1):
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
        control_optimizer.step()

        candidate_model.train()
        candidate_optimizer.zero_grad()
        candidate_outputs = candidate_model(inputs)
        candidate_loss = F.cross_entropy(candidate_outputs, targets)
        candidate_loss.backward()
        candidate_starts = [parameter.detach().clone() for parameter in candidate_parameters]
        raw_gradients = {
            id(parameter): parameter.grad.detach().clone()
            for parameter in candidate_parameters
        }
        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state().clone()
        train.centralize_conv_weight_gradients(candidate_model)
        rng_unchanged = (
            rng_unchanged
            and torch.equal(cpu_rng_before, torch.get_rng_state())
            and torch.equal(cuda_rng_before, torch.cuda.get_rng_state())
        )
        for name, weight in candidate_convs:
            raw_gradient = raw_gradients[id(weight)]
            projected = weight.grad
            removed = raw_gradient - projected
            raw_norm = raw_gradient.norm().item()
            projected_norm = projected.norm().item()
            projection_norm_nonincreasing = (
                projection_norm_nonincreasing
                and projected_norm <= raw_norm + max(1e-7, 1e-6 * raw_norm)
            )
            max_filter_mean = max(
                max_filter_mean,
                projected.mean(dim=(1, 2, 3)).abs().max().item(),
            )
            totals = stage_totals[phase][conv_stage(name)]
            totals["raw_sq"] += raw_gradient.float().square().sum().item()
            totals["projected_sq"] += projected.float().square().sum().item()
            totals["removed_sq"] += removed.float().square().sum().item()
        nonconv_unchanged = nonconv_unchanged and all(
            torch.equal(parameter.grad, raw_gradients[id(parameter)])
            for parameter in candidate_parameters
            if id(parameter) not in candidate_conv_ids
        )
        candidate_optimizer.step()
        torch.cuda.synchronize()

        control_update = parameter_delta_norm(control_parameters, control_starts)
        candidate_update = parameter_delta_norm(candidate_parameters, candidate_starts)
        update_ratio = candidate_update / max(control_update, 1e-30)
        update_ratios.append(update_ratio)
        if len(candidate_update_norms) >= 16:
            recent_median = statistics.median(candidate_update_norms[-16:])
            recent_median_ratios.append(candidate_update / max(recent_median, 1e-30))
        candidate_update_norms.append(candidate_update)
        for (_, weight), start in zip(
            candidate_convs,
            [
                start
                for parameter, start in zip(
                    candidate_parameters, candidate_starts, strict=True
                )
                if id(parameter) in candidate_conv_ids
            ],
            strict=True,
        ):
            ratio = (weight.detach() - start).norm().item() / max(start.norm().item(), 1e-30)
            max_conv_update_parameter_ratio = max(
                max_conv_update_parameter_ratio, ratio
            )

        control_share = class_share(control_outputs)
        candidate_share = class_share(candidate_outputs)
        if candidate_share > 0.95 and control_share <= 0.95:
            concentration_failures.append(
                {
                    "step": step,
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

        tensors = (
            control_parameters
            + candidate_parameters
            + list(control_model.buffers())
            + list(candidate_model.buffers())
            + momentum_buffers(control_optimizer)
            + momentum_buffers(candidate_optimizer)
        )
        bn_positive = all(
            (module.running_var > 0).all().item()
            for model in (control_model, candidate_model)
            for module in model.modules()
            if isinstance(module, nn.BatchNorm2d)
        )
        if (
            not math.isfinite(control_loss.item())
            or not math.isfinite(candidate_loss.item())
            or not all_finite(tensors)
            or not bn_positive
        ):
            raise RuntimeError(f"invalid finite/BN state at step {step}")
        records.append(
            {
                "step": step,
                "phase": phase,
                "target_ndim": targets.ndim,
                "control_loss": control_loss.item(),
                "candidate_loss": candidate_loss.item(),
                "control_class_share": control_share,
                "candidate_class_share": candidate_share,
                "control_update_norm": control_update,
                "candidate_update_norm": candidate_update,
                "candidate_control_update_ratio": update_ratio,
            }
        )

    phase_loss_emas = {}
    for phase, accumulator in phase_accumulators.items():
        debias = 1.0 - beta ** accumulator["count"]
        control = accumulator["control"] / debias
        candidate = accumulator["candidate"] / debias
        phase_loss_emas[phase] = {
            "control": control,
            "candidate": candidate,
            "ratio": candidate / control,
        }
    stage_fractions = {}
    for phase, stages in stage_totals.items():
        stage_fractions[phase] = {}
        for stage, values in stages.items():
            raw_norm = math.sqrt(values["raw_sq"])
            projected_norm = math.sqrt(values["projected_sq"])
            removed_norm = math.sqrt(values["removed_sq"])
            stage_fractions[phase][stage] = {
                "raw_norm": raw_norm,
                "projected_norm": projected_norm,
                "removed_norm": removed_norm,
                "removed_raw_fraction": removed_norm / max(raw_norm, 1e-30),
            }
    expected_steps = len(all_batches)
    bn_counters_valid = all(
        module.num_batches_tracked.item() == expected_steps
        for model in (control_model, candidate_model)
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
    )
    return {
        "steps": expected_steps,
        "parameter_count": sum(parameter.numel() for parameter in candidate_parameters),
        "concentration_failures": concentration_failures,
        "update_ratio_median": statistics.median(update_ratios),
        "update_ratio_p95": sorted(update_ratios)[math.ceil(0.95 * len(update_ratios)) - 1],
        "update_ratio_max": max(update_ratios),
        "recent_median_ratio_max": max(recent_median_ratios),
        "max_conv_update_parameter_ratio": max_conv_update_parameter_ratio,
        "max_post_projection_filter_mean": max_filter_mean,
        "nonconv_gradients_unchanged": nonconv_unchanged,
        "projection_norm_nonincreasing": projection_norm_nonincreasing,
        "rng_unchanged": rng_unchanged,
        "bn_counters_valid": bn_counters_valid,
        "phase_loss_emas": phase_loss_emas,
        "stage_gradient_fractions": stage_fractions,
        "records": records,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.use_deterministic_algorithms(True)
    strong_file = file_sha256(STRONG_PATH)
    weak_file = file_sha256(WEAK_PATH)
    if strong_file != EXPECTED_STRONG_FILE or weak_file != EXPECTED_WEAK_FILE:
        raise RuntimeError("immutable corpus file digest mismatch")
    strong_batches = torch.load(STRONG_PATH, map_location="cpu", weights_only=False)
    weak_batches = torch.load(WEAK_PATH, map_location="cpu", weights_only=False)
    if len(strong_batches) != 200 or len(weak_batches) != 64:
        raise RuntimeError("immutable corpus count mismatch")
    if any(targets.ndim != 1 for _, targets in weak_batches):
        raise RuntimeError("weak corpus contains non-hard targets")
    strong_tensors = corpus_digest(strong_batches)
    weak_tensors = corpus_digest(weak_batches)
    if strong_tensors != EXPECTED_STRONG_TENSORS or weak_tensors != EXPECTED_WEAK_TENSORS:
        raise RuntimeError("immutable corpus tensor digest mismatch")

    projection = projection_fixture()
    recurrence = recurrence_fixture()
    trajectory = run_trajectory(torch.device("cuda"), strong_batches, weak_batches)
    corpora_unchanged = (
        corpus_digest(strong_batches) == strong_tensors
        and corpus_digest(weak_batches) == weak_tensors
    )
    report = {
        "status": "pass",
        "strong_file_sha256": strong_file,
        "weak_file_sha256": weak_file,
        "strong_tensor_digest": strong_tensors,
        "weak_tensor_digest": weak_tensors,
        "strong_batches": len(strong_batches),
        "strong_hard_batches": sum(targets.ndim == 1 for _, targets in strong_batches),
        "strong_soft_batches": sum(targets.ndim == 2 for _, targets in strong_batches),
        "weak_batches": len(weak_batches),
        "corpora_unchanged": corpora_unchanged,
        "projection_fixture": projection,
        "recurrence_fixture": recurrence,
        "trajectory": trajectory,
    }
    write_report(report)

    failures = []
    if projection["eligible_conv_weights"] != 19:
        failures.append("Conv eligibility")
    if projection["max_fp64_reference_abs_error"] > 1e-7:
        failures.append("FP64 projection reference")
    if projection["idempotence_max_abs_error"] > 1e-7:
        failures.append("projection idempotence")
    for key in (
        "norm_nonincreasing",
        "nonconv_gradients_unchanged",
        "parameters_unchanged",
        "buffers_unchanged",
        "cpu_rng_unchanged",
    ):
        if not projection[key]:
            failures.append(key)
    if recurrence["max_parameter_abs_error"] > 1e-7:
        failures.append("parameter recurrence")
    if recurrence["max_momentum_abs_error"] > 1e-7:
        failures.append("momentum recurrence")
    for key in (
        "nonconv_gradients_unchanged",
        "projection_norm_nonincreasing",
        "rng_unchanged",
        "bn_counters_valid",
    ):
        if not trajectory[key]:
            failures.append(key)
    if not corpora_unchanged:
        failures.append("corpus mutation")
    if trajectory["concentration_failures"]:
        failures.append("candidate-only class concentration")
    if any(values["ratio"] > 1.5 for values in trajectory["phase_loss_emas"].values()):
        failures.append("phase loss EMA")
    if trajectory["update_ratio_max"] > 2.0:
        failures.append("paired update ratio")
    if trajectory["recent_median_ratio_max"] > 5.0:
        failures.append("recent-median update ratio")
    if trajectory["max_conv_update_parameter_ratio"] > 0.25:
        failures.append("Conv update/parameter ratio")
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
        write_report(report)
        raise RuntimeError(f"preflight gates failed: {failures}")

    compact = {key: value for key, value in report.items() if key != "trajectory"}
    compact["trajectory"] = {
        key: value for key, value in trajectory.items() if key != "records"
    }
    print(json.dumps(compact))


if __name__ == "__main__":
    main()
