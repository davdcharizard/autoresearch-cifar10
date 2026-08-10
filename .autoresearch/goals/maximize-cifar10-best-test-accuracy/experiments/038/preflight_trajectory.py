import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parent
STRONG = (
    ROOT
    / ".autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight-corpus.pt"
)
WEAK = (
    ROOT
    / ".autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/028/weak-corpus.pt"
)
STRONG_SHA = "e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946"
WEAK_SHA = "ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032"
sys.path.insert(0, str(ROOT))
import train  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_module():
    source = subprocess.check_output(
        ["git", "show", "7c1e7d8:train.py"], cwd=ROOT, text=True
    )
    module = types.ModuleType("control_train_038_trajectory")
    module.__file__ = str(ROOT / "train.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def seeded(module):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = module.ResNet(
        module.NUM_BLOCKS, module.NUM_CLASSES, module.WIDTH_MULTIPLIER
    ).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=module.LR,
        momentum=module.MOMENTUM,
        weight_decay=module.WEIGHT_DECAY,
    )
    return model, optimizer


def rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt())


def norm(tensors):
    squares = sum(float(tensor.detach().float().square().sum()) for tensor in tensors)
    return math.sqrt(squares)


def cosine_lr(index, count):
    progress = 0.8 + 0.2 * index / max(count - 1, 1)
    cosine_progress = (progress - train.LR_HOLD_FRACTION) / (1 - train.LR_HOLD_FRACTION)
    return train.MIN_LR + 0.5 * (train.ANNEAL_START_LR - train.MIN_LR) * (
        1 + math.cos(math.pi * cosine_progress)
    )


def validate_corpora(strong, weak):
    failures = []
    if sha(STRONG) != STRONG_SHA or sha(WEAK) != WEAK_SHA:
        failures.append("corpus hash")
    if len(strong) != 200 or len(weak) != 64:
        failures.append(f"corpus lengths {len(strong)}/{len(weak)}")
    if sum(targets.ndim == 1 for _inputs, targets in strong) != 94:
        failures.append("strong hard count")
    for inputs, targets in strong + weak:
        if inputs.shape != (128, 3, 32, 32) or not torch.isfinite(inputs).all():
            failures.append("input schema")
            break
        if targets.ndim == 2 and (
            targets.shape != (128, 10)
            or not torch.isfinite(targets).all()
            or not torch.allclose(targets.sum(1), torch.ones(128), atol=1e-6)
        ):
            failures.append("soft target schema")
            break
        if targets.ndim == 1 and targets.shape != (128,):
            failures.append("hard target schema")
            break
    if any(targets.ndim != 1 for _inputs, targets in weak):
        failures.append("weak target rank")
    return failures


def self_test():
    controls = [2.0, 3.0]
    floor = 1e-8
    denominator = max(max(abs(value) for value in controls), floor)
    return {
        "denominator": denominator,
        "ratio_known": 12.0 / denominator,
        "envelope_min": min(controls),
        "envelope_max": max(controls),
        "passes": denominator == 3.0 and 12.0 / denominator == 4.0,
    }


def run_arm(name, module, batches, snapshot_every=1, short=False):
    model, optimizer = seeded(module)
    initial_bias = model.fc.bias.detach().clone()
    metrics = []
    loss_ema = {"strong": 0.0, "weak": 0.0}
    loss_count = {"strong": 0, "weak": 0}
    hard_tail_correct = 0
    hard_tail_total = 0
    hard_tail_start = 8192 - 512 if len(batches) == 10240 else -1
    max_row_update_fraction = 0.0
    max_update_fraction = 0.0
    spike_failures = []
    update_history = []

    pooled = {}

    def save_pooled(_module, _args, output):
        pooled["value"] = F.adaptive_avg_pool2d(output.detach(), 1).flatten(1)

    hook = model.layer3.register_forward_hook(save_pooled)
    model.train()
    for step, (inputs, targets, lr, phase) in enumerate(batches):
        for group in optimizer.param_groups:
            group["lr"] = lr
        before = (
            [parameter.detach().clone() for parameter in model.parameters()]
            if short
            else None
        )
        row_before = model.fc.weight.detach().clone()
        optimizer.zero_grad()
        logits = model(inputs.cuda(non_blocking=True))
        loss = F.cross_entropy(logits, targets.cuda(non_blocking=True))
        loss.backward()

        weight = model.fc.weight.detach()
        gradient = model.fc.weight.grad.detach()
        unit_weight = F.normalize(weight, dim=1, eps=1e-30)
        radial = (gradient * unit_weight).sum(1, keepdim=True) * unit_weight
        tangent = gradient - radial
        whole_grad_norm = norm(
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        parameter_norm = norm(model.parameters())
        optimizer.step()
        torch.cuda.synchronize()

        row_after = model.fc.weight.detach()
        row_delta = row_after - row_before
        row_updates = row_delta.norm(dim=1)
        row_norms_before = row_before.norm(dim=1)
        row_update_fraction = float(
            (row_updates / row_norms_before.clamp_min(1e-30)).max()
        )
        update_radial = (row_delta * unit_weight).sum(1, keepdim=True) * unit_weight
        update_tangent = row_delta - update_radial
        max_row_update_fraction = max(max_row_update_fraction, row_update_fraction)
        if short:
            updates = [
                parameter.detach() - prior
                for parameter, prior in zip(model.parameters(), before)
            ]
            whole_update_norm = norm(updates)
            update_fraction = whole_update_norm / max(parameter_norm, 1e-30)
            max_update_fraction = max(max_update_fraction, update_fraction)
            update_history.append(whole_update_norm)
            if step >= 16:
                median = float(torch.tensor(update_history[-16:]).median())
                if whole_update_norm > 5 * max(median, 1e-8):
                    spike_failures.append(step)
        else:
            whole_update_norm = None
            update_fraction = None

        loss_count[phase] += 1
        beta = 0.95
        loss_ema[phase] = beta * loss_ema[phase] + (1 - beta) * float(loss.detach())
        debiased_loss = loss_ema[phase] / (1 - beta ** loss_count[phase])
        predictions = logits.detach().argmax(1)
        class_share = (
            float(predictions.bincount(minlength=10).max()) / predictions.numel()
        )
        if step >= hard_tail_start and phase == "strong" and targets.ndim == 1:
            hard_tail_correct += int(predictions.cpu().eq(targets).sum())
            hard_tail_total += targets.numel()

        if step % snapshot_every == 0 or step == len(batches) - 1:
            row_norms = model.fc.weight.detach().norm(dim=1).cpu()
            feature_norms = pooled["value"].norm(dim=1).cpu()
            metrics.append(
                {
                    "step": step + 1,
                    "phase": phase,
                    "lr": lr,
                    "loss": float(loss.detach()),
                    "loss_ema": debiased_loss,
                    "logit_rms": rms(logits),
                    "max_logit": float(logits.detach().abs().max()),
                    "class_share": class_share,
                    "whole_grad_norm": whole_grad_norm,
                    "whole_update_norm": whole_update_norm,
                    "update_fraction": update_fraction,
                    "row_norms": row_norms.tolist(),
                    "row_min": float(row_norms.min()),
                    "row_max_min": float(row_norms.max() / row_norms.min()),
                    "feature_norm_median": float(feature_norms.median()),
                    "radial_grad_rms": rms(radial),
                    "tangent_grad_rms": rms(tangent),
                    "row_update_rms": rms(row_delta),
                    "tangent_update_rms": rms(update_tangent),
                    "row_update_fraction": row_update_fraction,
                }
            )

    hook.remove()
    bn_counts = sorted(
        {
            int(module.num_batches_tracked)
            for module in model.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        }
    )
    all_finite = all(
        torch.isfinite(parameter).all() for parameter in model.parameters()
    ) and all(torch.isfinite(buffer).all() for buffer in model.buffers())
    return {
        "name": name,
        "metrics": metrics,
        "bn_counts": bn_counts,
        "all_finite": bool(all_finite),
        "bias_grad_none": model.fc.bias.grad is None,
        "bias_unchanged": torch.equal(initial_bias, model.fc.bias.detach()),
        "max_row_update_fraction": max_row_update_fraction,
        "max_update_fraction": max_update_fraction,
        "spike_failures": spike_failures,
        "final_loss_ema": {
            phase: metrics_for_phase(metrics, phase)[-1]["loss_ema"]
            for phase in ("strong", "weak")
        },
        "hard_tail_top1": 100 * hard_tail_correct / hard_tail_total
        if hard_tail_total
        else None,
    }


def metrics_for_phase(metrics, phase):
    return [metric for metric in metrics if metric["phase"] == phase]


def short_batches(strong, weak):
    result = [(inputs, targets, train.LR, "strong") for inputs, targets in strong]
    result.extend(
        (inputs, targets, cosine_lr(index, len(weak)), "weak")
        for index, (inputs, targets) in enumerate(weak)
    )
    return result


def long_batches(strong, weak):
    result = [
        (*strong[index % len(strong)], train.LR, "strong") for index in range(8192)
    ]
    result.extend(
        (*weak[index % len(weak)], cosine_lr(index, 2048), "weak")
        for index in range(2048)
    )
    return result


def max_metric(arm, key, phase=None):
    metrics = (
        arm["metrics"] if phase is None else metrics_for_phase(arm["metrics"], phase)
    )
    return max(metric[key] for metric in metrics)


def analyze_short(arms):
    failures = []
    c1, c2, candidate = arms
    for arm in arms:
        if not arm["all_finite"] or arm["bn_counts"] != [264]:
            failures.append(f"{arm['name']} finite/BN {arm['bn_counts']}")
    for control in (c1, c2):
        if control["max_update_fraction"] > 0.25 or control["spike_failures"]:
            failures.append(f"accepted control failed update gate {control['name']}")
    if not candidate["bias_grad_none"] or not candidate["bias_unchanged"]:
        failures.append("candidate bias changed")
    if candidate["max_update_fraction"] > 0.25 or candidate["spike_failures"]:
        failures.append("candidate update fraction/spike")

    ratio_keys = ("logit_rms", "whole_grad_norm", "whole_update_norm")
    ratios = {}
    for key in ratio_keys:
        denominator = max(max_metric(c1, key), max_metric(c2, key), 1e-8)
        ratios[key] = max_metric(candidate, key) / denominator
        if ratios[key] > 5:
            failures.append(f"candidate {key} ratio {ratios[key]}")
    for phase in ("strong", "weak"):
        denominator = max(
            c1["final_loss_ema"][phase], c2["final_loss_ema"][phase], 1e-6
        )
        ratio = candidate["final_loss_ema"][phase] / denominator
        ratios[f"{phase}_loss_ema"] = ratio
        if ratio > 1.5:
            failures.append(f"candidate {phase} loss EMA {ratio}")

    concentration_steps = []
    candidate_metrics = candidate["metrics"]
    for index, metric in enumerate(candidate_metrics):
        if (
            metric["class_share"] > 0.95
            and c1["metrics"][index]["class_share"] <= 0.95
            and c2["metrics"][index]["class_share"] <= 0.95
        ):
            concentration_steps.append(metric["step"])
    consecutive = any(
        right == left + 1
        for left, right in zip(concentration_steps, concentration_steps[1:])
    )
    if consecutive or len(concentration_steps) >= 3:
        failures.append(f"candidate concentration {concentration_steps}")
    if max_metric(candidate, "max_logit") > train.COSINE_SCALE * (1 + 1e-5):
        failures.append("candidate logit bound")
    return failures, {
        "ratios": ratios,
        "candidate_concentration_steps": concentration_steps,
    }


def window_losses(metrics, phase, window):
    phase_metrics = metrics_for_phase(metrics, phase)
    endpoints = []
    for end in range(window, len(phase_metrics) + 1, window):
        start_rows = torch.tensor(phase_metrics[end - window]["row_norms"])
        end_rows = torch.tensor(phase_metrics[end - 1]["row_norms"])
        endpoints.append((1 - end_rows / start_rows).tolist())
    return endpoints


def analyze_long(control, candidate):
    failures = []
    for arm in (control, candidate):
        if not arm["all_finite"] or arm["bn_counts"] != [10240]:
            failures.append(f"{arm['name']} finite/BN {arm['bn_counts']}")
    candidate_metrics = candidate["metrics"]
    if min(metric["row_min"] for metric in candidate_metrics) < 0.50:
        failures.append("row norm floor")
    if max(metric["row_max_min"] for metric in candidate_metrics) > 3:
        failures.append("row max/min")
    if min(metric["feature_norm_median"] for metric in candidate_metrics) < 1.0:
        failures.append("feature norm floor")
    if candidate["max_row_update_fraction"] > 0.50:
        failures.append("row update fraction")

    early = [metric for metric in candidate_metrics if metric["step"] <= 256]
    late = candidate_metrics[-4:]
    early_reciprocal = float(
        torch.tensor([1 / metric["row_min"] for metric in early]).median()
    )
    late_reciprocal = float(
        torch.tensor([1 / metric["row_min"] for metric in late]).median()
    )
    early_tangent = float(
        torch.tensor([metric["tangent_update_rms"] for metric in early]).median()
    )
    late_tangent = float(
        torch.tensor([metric["tangent_update_rms"] for metric in late]).median()
    )
    reciprocal_ratio = late_reciprocal / max(early_reciprocal, 1e-8)
    tangent_ratio = late_tangent / max(early_tangent, 1e-8)
    if reciprocal_ratio > 2 or tangent_ratio > 2:
        failures.append(f"terminal drift {reciprocal_ratio}/{tangent_ratio}")

    # Snapshots occur every 64 steps, hence 16 snapshots per 1,024-step window.
    strong_losses = window_losses(candidate_metrics, "strong", 16)[-4:]
    weak_losses = window_losses(candidate_metrics, "weak", 16)[-2:]
    if any(max(losses) > 0.10 for losses in strong_losses + weak_losses):
        failures.append("consecutive-window row loss")
    if candidate["hard_tail_top1"] is None or control["hard_tail_top1"] is None:
        failures.append("hard-tail fit missing")
    elif candidate["hard_tail_top1"] < control["hard_tail_top1"] - 10:
        failures.append(
            f"hard-tail fit {candidate['hard_tail_top1']}/{control['hard_tail_top1']}"
        )
    concentration_steps = [
        candidate_metric["step"]
        for control_metric, candidate_metric in zip(
            control["metrics"], candidate_metrics
        )
        if candidate_metric["class_share"] > 0.95
        and control_metric["class_share"] <= 0.95
    ]
    if len(concentration_steps) >= 2:
        failures.append(f"long candidate concentration {concentration_steps}")
    loss_ratios = {}
    for phase in ("strong", "weak"):
        ratio = candidate["final_loss_ema"][phase] / max(
            control["final_loss_ema"][phase], 1e-6
        )
        loss_ratios[phase] = ratio
        if ratio > 1.5:
            failures.append(f"long {phase} loss EMA {ratio}")
    return failures, {
        "early_reciprocal": early_reciprocal,
        "late_reciprocal": late_reciprocal,
        "reciprocal_ratio": reciprocal_ratio,
        "early_tangent": early_tangent,
        "late_tangent": late_tangent,
        "tangent_ratio": tangent_ratio,
        "strong_window_losses": strong_losses,
        "weak_window_losses": weak_losses,
        "loss_ratios": loss_ratios,
        "candidate_concentration_steps": concentration_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("short", "long"), required=True)
    args = parser.parse_args()
    report_path = EXP / f"trajectory-{args.mode}-report.json"
    log_self_test = self_test()
    failures = [] if log_self_test["passes"] else ["self-test"]
    strong = torch.load(STRONG, map_location="cpu", weights_only=False)
    weak = torch.load(WEAK, map_location="cpu", weights_only=False)
    failures.extend(validate_corpora(strong, weak))
    control = control_module()
    if args.mode == "short":
        batches = short_batches(strong, weak)
        arms = [
            run_arm("control-1", control, batches, short=True),
            run_arm("control-2", control, batches, short=True),
            run_arm("candidate", train, batches, short=True),
        ]
        analysis_failures, analysis = analyze_short(arms)
    else:
        batches = long_batches(strong, weak)
        arms = [
            run_arm("control", control, batches, snapshot_every=64),
            run_arm("candidate", train, batches, snapshot_every=64),
        ]
        analysis_failures, analysis = analyze_long(*arms)
    failures.extend(analysis_failures)
    report = {
        "status": "failed" if failures else "pass",
        "failures": failures,
        "mode": args.mode,
        "controller_sha256": sha(Path(__file__)),
        "train_sha256": sha(ROOT / "train.py"),
        "strong_sha256": sha(STRONG),
        "weak_sha256": sha(WEAK),
        "self_test": log_self_test,
        "analysis": analysis,
        "arms": arms,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    with report_path.open("rb") as handle:
        os.fsync(handle.fileno())
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps({"status": "pass", "report": str(report_path)}))


if __name__ == "__main__":
    main()
