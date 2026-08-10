import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as TORCH_F


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
STRONG_PATH = PROJECT_ROOT / ".autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight-corpus.pt"
WEAK_PATH = PROJECT_ROOT / ".autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/028/weak-corpus.pt"
REPORT_PATH = EXPERIMENT_DIR / "preflight-report.json"
EXPECTED_STRONG_SHA = "e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946"
EXPECTED_WEAK_SHA = "ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032"
BASELINE_COMMIT = "7c1e7d8"
sys.path.insert(0, str(PROJECT_ROOT))

import train as candidate_train  # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(batches):
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


def load_control_module():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:train.py"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    module = types.ModuleType("accepted_train_exp035")
    module.__file__ = str(PROJECT_ROOT / "train.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


class ActivationProxy:
    def __init__(self, base, operation, capture=False):
        self.base = base
        self.operation = operation
        self.capture = capture
        self.outputs = []

    def __getattr__(self, name):
        if name == self.operation:
            return self.activate
        return getattr(self.base, name)

    def activate(self, value, *args, **kwargs):
        output = getattr(self.base, self.operation)(value, *args, **kwargs)
        if self.capture:
            output.retain_grad()
            self.outputs.append(output)
        return output

    def reset(self):
        self.outputs = []


def module_for_arm(arm, capture=False):
    if arm == "candidate":
        module = candidate_train
        operation = "silu"
    else:
        module, _source_hash = load_control_module()
        operation = "relu"
    proxy = ActivationProxy(TORCH_F, operation, capture=capture)
    module.F = proxy
    return module, proxy


def seeded_model(arm, capture=False, device="cpu"):
    module, proxy = module_for_arm(arm, capture=capture)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = module.ResNet(
        module.NUM_BLOCKS, module.NUM_CLASSES, module.WIDTH_MULTIPLIER
    )
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    return model.to(device), module, proxy, cpu_rng, cuda_rng


def load_and_validate_corpora():
    strong_sha = file_sha256(STRONG_PATH)
    weak_sha = file_sha256(WEAK_PATH)
    if strong_sha != EXPECTED_STRONG_SHA or weak_sha != EXPECTED_WEAK_SHA:
        raise RuntimeError("corpus file SHA mismatch")
    strong = torch.load(STRONG_PATH, map_location="cpu", weights_only=False)
    weak = torch.load(WEAK_PATH, map_location="cpu", weights_only=False)
    if len(strong) != 200 or len(weak) != 64:
        raise RuntimeError("corpus batch count mismatch")
    hard = sum(targets.ndim == 1 for _, targets in strong)
    soft = sum(targets.ndim == 2 for _, targets in strong)
    if (hard, soft) != (94, 106):
        raise RuntimeError("strong target-rank coverage mismatch")
    for inputs, targets in strong + weak:
        if inputs.shape != (128, 3, 32, 32) or inputs.dtype != torch.float32:
            raise RuntimeError("corpus input contract mismatch")
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise RuntimeError("nonfinite corpus tensor")
    if any(
        targets.shape != (128,) or targets.dtype != torch.int64
        for _, targets in weak
    ):
        raise RuntimeError("weak target contract mismatch")
    report = {
        "strong_file_sha256": strong_sha,
        "weak_file_sha256": weak_sha,
        "strong_tensor_sha256": tensor_digest(strong),
        "weak_tensor_sha256": tensor_digest(weak),
        "strong_batches": len(strong),
        "strong_hard_batches": hard,
        "strong_soft_batches": soft,
        "weak_batches": len(weak),
    }
    return strong, weak, report


def construction_gate():
    control, _cm, _cp, control_cpu, control_cuda = seeded_model("control")
    candidate, _xm, _xp, candidate_cpu, candidate_cuda = seeded_model("candidate")
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    if list(control_state) != list(candidate_state):
        raise RuntimeError("state ordering mismatch")
    unequal = [
        key for key in control_state if not torch.equal(control_state[key], candidate_state[key])
    ]
    modules = list(candidate.modules())
    inventory = {
        "conv": sum(isinstance(module, torch.nn.Conv2d) for module in modules),
        "bn": sum(isinstance(module, torch.nn.BatchNorm2d) for module in modules),
        "linear": sum(isinstance(module, torch.nn.Linear) for module in modules),
        "parameters": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if inventory != {"conv": 19, "bn": 19, "linear": 1, "parameters": 1_073_962}:
        raise RuntimeError(f"module inventory mismatch: {inventory}")
    if unequal:
        raise RuntimeError(f"initial tensors differ: {unequal}")
    if not torch.equal(control_cpu, candidate_cpu) or not torch.equal(
        control_cuda, candidate_cuda
    ):
        raise RuntimeError("post-construction RNG mismatch")
    return {
        **inventory,
        "state_key_count": len(control_state),
        "initial_tensors_equal": True,
        "cpu_rng_equal": True,
        "cuda_rng_equal": True,
    }


def gate_math_self_test():
    values = torch.tensor([-9.0, -2.0, -0.1, 0.0, 0.1, 2.0, 9.0], device="cuda")
    values.requires_grad_(True)
    actual = TORCH_F.silu(values)
    expected = values * torch.sigmoid(values)
    actual.sum().backward()
    sigmoid = torch.sigmoid(values.detach())
    expected_grad = sigmoid + values.detach() * sigmoid * (1 - sigmoid)
    value_error = float((actual.detach() - expected.detach()).abs().max())
    grad_error = float((values.grad - expected_grad).abs().max())
    known_control = {"x": 2.0, "y": 4.0}
    known_candidate = {"x": 4.0, "y": 1.0}
    ratios = {
        key: known_candidate[key] / known_control[key] for key in known_control
    }
    identity_ratios = {key: value / value for key, value in known_control.items()}
    identity_vetoes = [key for key, ratio in identity_ratios.items() if ratio > 5]
    if value_error > 1e-6 or grad_error > 1e-6:
        raise RuntimeError("SiLU oracle mismatch")
    if ratios != {"x": 2.0, "y": 0.25} or identity_vetoes:
        raise RuntimeError("gate ratio self-test mismatch")
    return {
        "value_max_error": value_error,
        "gradient_max_error": grad_error,
        "zero_value": float(actual[3]),
        "zero_gradient": float(values.grad[3]),
        "known_ratios": ratios,
        "identity_ratios": identity_ratios,
        "identity_vetoes": identity_vetoes,
    }


def rms(tensor):
    return float(tensor.float().square().mean().sqrt())


def tensor_norm(tensors):
    return math.sqrt(sum(float(tensor.float().square().sum()) for tensor in tensors))


def class_share(outputs, classes=10):
    counts = outputs.argmax(1).bincount(minlength=classes)
    return float(counts.max()) / outputs.shape[0]


def activation_stats(outputs):
    return [
        {
            "output_rms": rms(output.detach()),
            "gradient_rms": rms(output.grad),
            "negative_fraction": float((output.detach() < 0).float().mean()),
            "zero_fraction": float((output.detach() == 0).float().mean()),
        }
        for output in outputs
    ]


def one_batch_capture(arm, inputs, targets):
    model, _module, proxy, _cpu_rng, _cuda_rng = seeded_model(
        arm, capture=True, device="cuda"
    )
    model.train()
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone()
    cuda_inputs = inputs.cuda(non_blocking=True)
    cuda_targets = targets.cuda(non_blocking=True)
    proxy.reset()
    outputs = model(cuda_inputs)
    loss = TORCH_F.cross_entropy(outputs, cuda_targets)
    loss.backward()
    torch.cuda.synchronize()
    after_cpu = torch.get_rng_state().clone()
    after_cuda = torch.cuda.get_rng_state().clone()
    if len(proxy.outputs) != 19:
        raise RuntimeError(f"{arm} dynamic activation count {len(proxy.outputs)}")
    pooled = TORCH_F.adaptive_avg_pool2d(proxy.outputs[-1], 1).flatten(1)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return {
        "loss": float(loss),
        "logit_rms": rms(outputs),
        "pooled_rms": rms(pooled),
        "pooled_negative_fraction": float((pooled < 0).float().mean()),
        "class_share": class_share(outputs),
        "gradient_norm": tensor_norm(gradients),
        "activation_sites": activation_stats(proxy.outputs),
        "cpu_rng_unchanged": torch.equal(before_cpu, after_cpu),
        "cuda_rng_unchanged": torch.equal(before_cuda, after_cuda),
    }


def initial_function_gate(strong):
    selected = {
        "hard": next(batch for batch in strong if batch[1].ndim == 1),
        "soft": next(batch for batch in strong if batch[1].ndim == 2),
    }
    cases = {}
    failures = []
    for name, (inputs, targets) in selected.items():
        control = one_batch_capture("control", inputs, targets)
        candidate = one_batch_capture("candidate", inputs, targets)
        ratios = {
            metric: candidate[metric] / max(control[metric], 1e-30)
            for metric in ("loss", "logit_rms", "pooled_rms", "gradient_norm")
        }
        if any(not 0.25 <= value <= 4.0 for value in ratios.values()):
            failures.append(f"{name} initial ratios {ratios}")
        if candidate["class_share"] > 0.95 and control["class_share"] <= 0.95:
            failures.append(f"{name} initial candidate-only concentration")
        if not all(
            control[key] and candidate[key]
            for key in ("cpu_rng_unchanged", "cuda_rng_unchanged")
        ):
            failures.append(f"{name} forward RNG changed")
        site_ratios = []
        for c_site, x_site in zip(
            control["activation_sites"], candidate["activation_sites"], strict=True
        ):
            site_ratios.append(
                {
                    "output_rms": x_site["output_rms"] / max(c_site["output_rms"], 1e-30),
                    "gradient_rms": x_site["gradient_rms"] / max(c_site["gradient_rms"], 1e-30),
                }
            )
        cases[name] = {"control": control, "candidate": candidate, "ratios": ratios, "site_ratios": site_ratios}
    return {"status": "failed" if failures else "pass", "cases": cases, "failures": failures}


def weak_lr(index, module):
    progress = 0.8 + 0.2 * index / 63
    cosine = (progress - module.LR_HOLD_FRACTION) / (1 - module.LR_HOLD_FRACTION)
    return module.MIN_LR + 0.5 * (module.ANNEAL_START_LR - module.MIN_LR) * (
        1 + math.cos(math.pi * cosine)
    )


def all_finite(model, optimizer):
    tensors = list(model.parameters()) + list(model.buffers())
    tensors += [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def run_trajectory(arm):
    strong, weak, _corpus = load_and_validate_corpora()
    model, module, proxy, _cpu_rng, _cuda_rng = seeded_model(
        "candidate" if arm == "candidate" else "control", capture=True, device="cuda"
    )
    model.train()
    parameters = list(model.parameters())
    optimizer = torch.optim.SGD(
        parameters, lr=module.LR, momentum=module.MOMENTUM, weight_decay=module.WEIGHT_DECAY
    )
    phase_ema = {"strong": 0.0, "weak": 0.0}
    phase_count = {"strong": 0, "weak": 0}
    update_history = []
    records = []
    beta = 0.95
    batches = [("strong", index, batch) for index, batch in enumerate(strong)] + [
        ("weak", index, batch) for index, batch in enumerate(weak)
    ]
    for step, (phase, index, (cpu_inputs, cpu_targets)) in enumerate(batches, 1):
        lr = module.LR if phase == "strong" else weak_lr(index, module)
        optimizer.param_groups[0]["lr"] = lr
        inputs = cpu_inputs.cuda(non_blocking=True)
        targets = cpu_targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        proxy.reset()
        outputs = model(inputs)
        loss = TORCH_F.cross_entropy(outputs, targets)
        loss.backward()
        if len(proxy.outputs) != 19:
            raise RuntimeError(f"{arm} activation count mismatch at {step}")
        starts = [parameter.detach().clone() for parameter in parameters]
        gradient_norm = tensor_norm(
            [parameter.grad for parameter in parameters if parameter.grad is not None]
        )
        parameter_norm = tensor_norm(starts)
        sites = activation_stats(proxy.outputs)
        pooled = TORCH_F.adaptive_avg_pool2d(proxy.outputs[-1], 1).flatten(1)
        optimizer.step()
        torch.cuda.synchronize()
        updates = [
            parameter.detach() - start
            for parameter, start in zip(parameters, starts, strict=True)
        ]
        update_norm = tensor_norm(updates)
        relative_updates = [
            float(update.norm() / max(float(start.norm()), 1e-30))
            for update, start in zip(updates, starts, strict=True)
        ]
        preceding = (
            update_norm / statistics.median(update_history[-16:])
            if len(update_history) >= 16
            else None
        )
        update_history.append(update_norm)
        value = float(loss)
        phase_ema[phase] = beta * phase_ema[phase] + (1 - beta) * value
        phase_count[phase] += 1
        records.append(
            {
                "step": step,
                "phase": phase,
                "loss": value,
                "class_share": class_share(outputs),
                "logit_rms": rms(outputs),
                "pooled_rms": rms(pooled),
                "gradient_norm": gradient_norm,
                "update_norm": update_norm,
                "parameter_norm": parameter_norm,
                "update_parameter_ratio": update_norm / parameter_norm,
                "preceding_update_median_ratio": preceding,
                "max_tensor_relative_update": max(relative_updates),
                "site_output_rms": [site["output_rms"] for site in sites],
                "site_gradient_rms": [site["gradient_rms"] for site in sites],
            }
        )
        if not math.isfinite(value) or not all_finite(model, optimizer):
            raise RuntimeError(f"{arm} nonfinite at step {step}")
    bn_modules = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    result = {
        "arm": arm,
        "backend": backend_flags(),
        "records": records,
        "terminal_ema": {
            phase: phase_ema[phase] / (1 - beta ** phase_count[phase])
            for phase in phase_ema
        },
        "bn_counts": sorted({int(module.num_batches_tracked) for module in bn_modules}),
        "min_running_var": min(float(module.running_var.min()) for module in bn_modules),
        "momentum_buffers": sum("momentum_buffer" in state for state in optimizer.state.values()),
        "parameter_tensors": len(parameters),
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
        timeout=360,
    )
    return json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1])


def compare(control, candidate, gate):
    failures = []
    concentration = []
    maxima = {
        "logit_rms_ratio": 0.0,
        "pooled_rms_ratio": 0.0,
        "gradient_norm_ratio": 0.0,
        "update_norm_ratio": 0.0,
        "update_parameter_ratio": 0.0,
        "preceding_update_median_ratio": 0.0,
        "max_tensor_relative_update": 0.0,
        "site_output_ratio": 0.0,
        "site_gradient_ratio": 0.0,
    }
    if control["backend"] != candidate["backend"]:
        failures.append("backend flags differ")
    for c_record, x_record in zip(control["records"], candidate["records"], strict=True):
        step = c_record["step"]
        if x_record["class_share"] > 0.95 and c_record["class_share"] <= 0.95:
            concentration.append({"step": step, "control": c_record["class_share"], "candidate": x_record["class_share"]})
        for metric in ("logit_rms", "pooled_rms", "gradient_norm", "update_norm"):
            ratio = x_record[metric] / max(c_record[metric], 1e-30)
            maxima[f"{metric}_ratio"] = max(maxima[f"{metric}_ratio"], ratio)
            if gate and ratio > 5:
                failures.append(f"{metric} ratio {ratio:.6f} at step {step}")
        maxima["update_parameter_ratio"] = max(maxima["update_parameter_ratio"], x_record["update_parameter_ratio"])
        maxima["max_tensor_relative_update"] = max(maxima["max_tensor_relative_update"], x_record["max_tensor_relative_update"])
        if gate and x_record["update_parameter_ratio"] > 0.25:
            failures.append(f"whole update/parameter at step {step}")
        if gate and x_record["max_tensor_relative_update"] > 0.5:
            failures.append(f"tensor relative update at step {step}")
        preceding = x_record["preceding_update_median_ratio"]
        if preceding is not None:
            maxima["preceding_update_median_ratio"] = max(maxima["preceding_update_median_ratio"], preceding)
            if gate and preceding > 5:
                failures.append(f"update/median at step {step}")
        for kind in ("output", "gradient"):
            ratios = [
                x / max(c, 1e-30)
                for c, x in zip(c_record[f"site_{kind}_rms"], x_record[f"site_{kind}_rms"], strict=True)
            ]
            maxima[f"site_{kind}_ratio"] = max(maxima[f"site_{kind}_ratio"], max(ratios))
            if gate and any(not 0.2 <= ratio <= 5.0 for ratio in ratios):
                failures.append(f"site {kind} ratio at step {step}")
    ema_ratios = {
        phase: candidate["terminal_ema"][phase] / control["terminal_ema"][phase]
        for phase in ("strong", "weak")
    }
    if gate and concentration:
        failures.append(f"candidate-only concentration at {[item['step'] for item in concentration]}")
    if gate and any(value > 1.5 for value in ema_ratios.values()):
        failures.append(f"terminal EMA ratios {ema_ratios}")
    for arm in (control, candidate):
        if arm["bn_counts"] != [264] or arm["min_running_var"] <= 0:
            failures.append(f"{arm['arm']} BN state incomplete")
        if arm["momentum_buffers"] != arm["parameter_tensors"]:
            failures.append(f"{arm['arm']} momentum state incomplete")
    return {
        "status": "failed" if failures else "pass",
        "gate_enabled": gate,
        "maxima": maxima,
        "one_sided_concentration": concentration,
        "terminal_ema_ratios": ema_ratios,
        "control_min_running_var": control["min_running_var"],
        "candidate_min_running_var": candidate["min_running_var"],
        "failures": failures,
    }


def parent():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    controller_hash = file_sha256(Path(__file__).resolve())
    construction = construction_gate()
    self_test = gate_math_self_test()
    strong, _weak, corpus = load_and_validate_corpora()
    initial = initial_function_gate(strong)
    print(json.dumps({"stage": "semantic", "status": initial["status"]}), flush=True)
    controls = []
    for label in ("control-a", "control-b", "control-c", "control-d"):
        controls.append(trajectory_child(label))
        print(json.dumps({"stage": "trajectory", "arm": label, "status": "complete"}), flush=True)
    candidate = trajectory_child("candidate")
    print(json.dumps({"stage": "trajectory", "arm": "candidate", "status": "complete"}), flush=True)
    calibrations = [compare(controls[0], controls[1], False), compare(controls[2], controls[3], False)]
    trajectory = compare(controls[0], candidate, True)
    _strong_after, _weak_after, corpus_after = load_and_validate_corpora()
    if corpus != corpus_after:
        raise RuntimeError("corpus declaration changed after replay")
    failures = initial["failures"] + trajectory["failures"]
    report = {
        "status": "failed" if failures else "pass",
        "controller_sha256": controller_hash,
        "candidate_train_sha256": file_sha256(PROJECT_ROOT / "train.py"),
        "baseline_commit": BASELINE_COMMIT,
        "construction": construction,
        "self_test": self_test,
        "corpus": corpus,
        "initial_function": initial,
        "control_control_calibrations": calibrations,
        "trajectory": trajectory,
        "failures": failures,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    with REPORT_PATH.open("rb") as handle:
        os.fsync(handle.fileno())
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps({"status": "pass", "report": str(REPORT_PATH)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory")
    args = parser.parse_args()
    if args.trajectory:
        run_trajectory(args.trajectory)
    else:
        parent()


if __name__ == "__main__":
    main()
