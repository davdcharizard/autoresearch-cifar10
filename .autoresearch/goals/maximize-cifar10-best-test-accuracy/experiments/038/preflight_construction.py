import hashlib
import json
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
REPORT = EXP / "construction-report.json"
EXPECTED_CORPUS_SHA = "e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946"
EXPECTED_SCALE = 22.786916732788086
sys.path.insert(0, str(ROOT))
import train  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_module():
    source = subprocess.check_output(
        ["git", "show", "7c1e7d8:train.py"], cwd=ROOT, text=True
    )
    module = types.ModuleType("control_train_038")
    module.__file__ = str(ROOT / "train.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def seeded(module):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = module.ResNet(
        module.NUM_BLOCKS, module.NUM_CLASSES, module.WIDTH_MULTIPLIER
    )
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state().clone()
    return model.cuda(), cpu, cuda


def rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt())


def formula(features, weights, eps, scale):
    feature_norm = torch.linalg.vector_norm(features, dim=1, keepdim=True).clamp_min(
        eps
    )
    weight_norm = torch.linalg.vector_norm(weights, dim=1, keepdim=True).clamp_min(eps)
    return scale * (features / feature_norm) @ (weights / weight_norm).T


def oracle_case(features, weights, probe, eps=1e-6):
    actual_features = features.detach().clone().requires_grad_(True)
    actual_weights = weights.detach().clone().requires_grad_(True)
    expected_features = features.detach().clone().requires_grad_(True)
    expected_weights = weights.detach().clone().requires_grad_(True)
    actual = EXPECTED_SCALE * F.linear(
        F.normalize(actual_features, dim=1, eps=eps),
        F.normalize(actual_weights, dim=1, eps=eps),
    )
    expected = formula(expected_features, expected_weights, eps, EXPECTED_SCALE)
    actual.backward(probe)
    expected.backward(probe)
    return {
        "output_error": float((actual.detach() - expected.detach()).abs().max()),
        "feature_grad_error": float(
            (actual_features.grad.detach() - expected_features.grad.detach())
            .abs()
            .max()
        ),
        "weight_grad_error": float(
            (actual_weights.grad.detach() - expected_weights.grad.detach()).abs().max()
        ),
        "finite": bool(
            torch.isfinite(actual).all()
            and torch.isfinite(actual_features.grad).all()
            and torch.isfinite(actual_weights.grad).all()
        ),
        "max_logit": float(actual.abs().max()),
    }


def capture(model, inputs):
    pooled = {}

    def save_pooled(_module, _args, output):
        pooled["value"] = F.adaptive_avg_pool2d(output.detach(), 1).flatten(1)

    handle = model.layer3.register_forward_hook(save_pooled)
    model.train()
    with torch.no_grad():
        logits = model(inputs.cuda())
    handle.remove()
    return logits.detach(), pooled["value"]


def main():
    failures = []
    corpus_sha = sha(STRONG)
    if corpus_sha != EXPECTED_CORPUS_SHA:
        failures.append(f"corpus sha {corpus_sha}")
    batches = torch.load(STRONG, map_location="cpu", weights_only=False)
    if len(batches) != 200:
        failures.append(f"corpus length {len(batches)}")

    control = control_module()
    accepted, accepted_cpu, accepted_cuda = seeded(control)
    candidate, candidate_cpu, candidate_cuda = seeded(train)
    accepted_state, candidate_state = accepted.state_dict(), candidate.state_dict()
    state_equal = list(accepted_state) == list(candidate_state) and all(
        torch.equal(accepted_state[key], candidate_state[key]) for key in accepted_state
    )
    rng_equal = torch.equal(accepted_cpu, candidate_cpu) and torch.equal(
        accepted_cuda, candidate_cuda
    )
    modules = dict(candidate.named_modules())
    inventory = {
        "conv": sum(isinstance(module, torch.nn.Conv2d) for module in modules.values()),
        "bn": sum(
            isinstance(module, torch.nn.BatchNorm2d) for module in modules.values()
        ),
        "linear": sum(
            isinstance(module, torch.nn.Linear) for module in modules.values()
        ),
        "params": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if not state_equal or not rng_equal:
        failures.append("construction state/RNG mismatch")
    if inventory != {"conv": 19, "bn": 19, "linear": 1, "params": 1_073_962}:
        failures.append(f"inventory {inventory}")
    if train.COSINE_SCALE != EXPECTED_SCALE:
        failures.append(f"scale {train.COSINE_SCALE}")

    torch.manual_seed(380)
    cases = {
        "random": (
            torch.randn(7, 128, dtype=torch.float64),
            torch.randn(10, 128, dtype=torch.float64),
        ),
        "tiny": (
            torch.randn(7, 128, dtype=torch.float64) * 1e-12,
            torch.randn(10, 128, dtype=torch.float64) * 1e-12,
        ),
        "zero": (
            torch.zeros(7, 128, dtype=torch.float64),
            torch.zeros(10, 128, dtype=torch.float64),
        ),
    }
    oracle = {}
    for name, (features, weights) in cases.items():
        probe = torch.randn(7, 10, dtype=torch.float64)
        oracle[name] = oracle_case(features, weights, probe)
        errors = torch.tensor(
            [
                oracle[name]["output_error"],
                oracle[name]["feature_grad_error"],
                oracle[name]["weight_grad_error"],
            ]
        )
        if (
            not oracle[name]["finite"]
            or not torch.isfinite(errors).all()
            or errors.max() > 1e-9
        ):
            failures.append(f"oracle {name}")
        if oracle[name]["max_logit"] > EXPECTED_SCALE * (1 + 1e-9):
            failures.append(f"logit bound {name}")

    first = batches[0]
    hard = next(batch for batch in batches if batch[1].ndim == 1)
    accepted, *_ = seeded(control)
    candidate, *_ = seeded(train)
    accepted_logits, _ = capture(accepted, first[0])
    candidate_logits, candidate_pooled = capture(candidate, first[0])
    unit_cosine = F.linear(
        F.normalize(candidate_pooled, dim=1, eps=1e-6),
        F.normalize(candidate.fc.weight, dim=1, eps=1e-6),
    )
    derived_scale = rms(accepted_logits) / rms(unit_cosine)
    first_calibration = {
        "target_rank": first[1].ndim,
        "accepted_rms": rms(accepted_logits),
        "unit_cosine_rms": rms(unit_cosine),
        "derived_scale": derived_scale,
        "candidate_rms": rms(candidate_logits),
        "accepted_loss": float(F.cross_entropy(accepted_logits, first[1].cuda())),
        "candidate_loss": float(F.cross_entropy(candidate_logits, first[1].cuda())),
    }
    if abs(derived_scale / EXPECTED_SCALE - 1) > 5e-5:
        failures.append(f"calibration scale {derived_scale}")
    if (
        abs(first_calibration["candidate_rms"] / first_calibration["accepted_rms"] - 1)
        > 2e-5
    ):
        failures.append("calibration RMS mismatch")

    accepted, *_ = seeded(control)
    candidate, *_ = seeded(train)
    accepted_hard, _ = capture(accepted, hard[0])
    candidate_hard, _ = capture(candidate, hard[0])
    hard_calibration = {
        "accepted_rms": rms(accepted_hard),
        "candidate_rms": rms(candidate_hard),
        "ratio": rms(candidate_hard) / rms(accepted_hard),
    }
    if not 0.98 <= hard_calibration["ratio"] <= 1.02:
        failures.append(f"hard calibration {hard_calibration['ratio']}")

    candidate, *_ = seeded(train)
    optimizer = torch.optim.SGD(
        candidate.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    initial_bias = candidate.fc.bias.detach().clone()
    inputs, targets = first
    optimizer.zero_grad()
    logits = candidate(inputs.cuda())
    loss = F.cross_entropy(logits, targets.cuda())
    loss.backward()
    bias_grad_none = candidate.fc.bias.grad is None
    optimizer.step()
    bias_unchanged = torch.equal(initial_bias, candidate.fc.bias.detach())
    if not bias_grad_none or not bias_unchanged:
        failures.append("production SGD changed unused bias")

    report = {
        "status": "failed" if failures else "pass",
        "failures": failures,
        "controller_sha256": sha(Path(__file__)),
        "train_sha256": sha(ROOT / "train.py"),
        "corpus_sha256": corpus_sha,
        "inventory": inventory,
        "state_equal": state_equal,
        "rng_equal": rng_equal,
        "oracle": oracle,
        "first_calibration": first_calibration,
        "hard_calibration": hard_calibration,
        "bias_grad_none": bias_grad_none,
        "bias_unchanged": bias_unchanged,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    with REPORT.open("rb") as handle:
        os.fsync(handle.fileno())
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps({"status": "pass", "report": str(REPORT)}))


if __name__ == "__main__":
    main()
