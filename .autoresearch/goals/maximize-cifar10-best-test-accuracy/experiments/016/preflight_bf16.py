import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import train
from prepare import Eval


HERE = Path(__file__).resolve().parent
NUMERICAL_RESULT = HERE / "preflight-numerical.json"
LOADER_RESULT = HERE / "preflight-loader.json"
EVAL_RESULT = HERE / "preflight-eval-wall.json"
TIMING_RESULT = HERE / "timing-bf16.json"


def backend_state():
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


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


def model_and_optimizer(state):
    model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, 3).cuda()
    model.load_state_dict(state)
    optimizer = optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    return model, optimizer


def run_step(model, optimizer, inputs, targets, bf16):
    optimizer.zero_grad()
    if bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
    else:
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_vec = torch.cat([g.detach().float().reshape(-1) for g in grads])
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    update_vec = torch.cat(
        [(p.detach() - old).float().reshape(-1) for p, old in zip(model.parameters(), before)]
    )
    return logits.detach().float(), loss.detach().float(), grad_vec, update_vec


def cosine(a, b):
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def relative_norm(a, b):
    denom = max(a.float().norm().item(), b.float().norm().item(), 1e-6)
    return (a.float() - b.float()).norm().item() / denom


def bn_differences(control, candidate):
    control_bn = [m for m in control.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    candidate_bn = [m for m in candidate.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    counters_match = all(
        torch.equal(a.num_batches_tracked, b.num_batches_tracked)
        for a, b in zip(control_bn, candidate_bn)
    )
    mean_diff = max(
        relative_norm(a.running_mean, b.running_mean)
        for a, b in zip(control_bn, candidate_bn)
    )
    var_diff = max(
        relative_norm(a.running_var, b.running_var)
        for a, b in zip(control_bn, candidate_bn)
    )
    return counters_match, mean_diff, var_diff


def all_finite(model, optimizer):
    tensors = list(model.parameters()) + list(model.buffers())
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def persistent_fp32(model, optimizer):
    floating = [p for p in model.parameters()]
    floating.extend(b for b in model.buffers() if b.is_floating_point())
    for state in optimizer.state.values():
        floating.extend(v for v in state.values() if torch.is_tensor(v) and v.is_floating_point())
    return all(t.dtype == torch.float32 for t in floating)


def materialize_batches(count=20):
    torch.manual_seed(42)
    loader = train.make_train_loader(strong_transform(), collate_fn=train.cutmix_collate)
    iterator = iter(loader)
    batches = []
    for _ in range(count):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets = next(iterator)
        batches.append((inputs.contiguous(), targets.contiguous()))
    iterator = None
    stopped = train.shutdown_train_loader(loader)
    assert len(stopped) == train.NUM_WORKERS
    return batches


def numerical_gate():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert train.WIDTH_MULTIPLIER == 3
    initial_backends = backend_state()
    batches = materialize_batches()
    hard_count = sum(targets.ndim == 1 for _, targets in batches[:20])
    soft_count = sum(targets.ndim == 2 for _, targets in batches[:20])
    assert hard_count and soft_count

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    base = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, 3).cuda()
    assert sum(p.numel() for p in base.parameters()) == 2_412_730
    base_state = copy.deepcopy(base.state_dict())
    del base

    one_step = []
    observed_dtypes = {"conv_linear": set(), "batchnorm": set(), "loss": set()}
    for batch_index, (cpu_inputs, cpu_targets) in enumerate(batches[:20]):
        control, control_opt = model_and_optimizer(base_state)
        candidate, candidate_opt = model_and_optimizer(base_state)
        control.train()
        candidate.train()
        hooks = []

        def record(kind):
            def hook(_module, _args, output):
                observed_dtypes[kind].add(str(output.dtype))
            return hook

        for module in candidate.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(record("conv_linear")))
            elif isinstance(module, torch.nn.BatchNorm2d):
                hooks.append(module.register_forward_hook(record("batchnorm")))

        inputs = cpu_inputs.cuda(non_blocking=True)
        targets = cpu_targets.cuda(non_blocking=True)
        control_logits, control_loss, control_grad, control_update = run_step(
            control, control_opt, inputs, targets, False
        )
        candidate_logits, candidate_loss, candidate_grad, candidate_update = run_step(
            candidate, candidate_opt, inputs, targets, True
        )
        observed_dtypes["loss"].add(str(candidate_loss.dtype))
        for hook in hooks:
            hook.remove()

        counters, mean_diff, var_diff = bn_differences(control, candidate)
        grad_ratio = candidate_grad.norm().item() / control_grad.norm().item()
        update_ratio = candidate_update.norm().item() / control_update.norm().item()
        control_zero = (control_grad == 0).float().mean().item()
        candidate_zero = (candidate_grad == 0).float().mean().item()
        metrics = {
            "batch": batch_index,
            "target_ndim": cpu_targets.ndim,
            "loss_relative_error": abs(candidate_loss.item() - control_loss.item())
            / max(abs(control_loss.item()), 1e-8),
            "logit_cosine": cosine(control_logits, candidate_logits),
            "gradient_cosine": cosine(control_grad, candidate_grad),
            "gradient_norm_ratio": grad_ratio,
            "update_cosine": cosine(control_update, candidate_update),
            "update_norm_ratio": update_ratio,
            "zero_gradient_increase_pp": 100 * (candidate_zero - control_zero),
            "bn_counters_match": counters,
            "bn_running_mean_relative": mean_diff,
            "bn_running_var_relative": var_diff,
        }
        one_step.append(metrics)
        assert metrics["loss_relative_error"] <= 0.02
        assert metrics["logit_cosine"] >= 0.995
        assert metrics["gradient_cosine"] >= 0.99
        assert 0.90 <= grad_ratio <= 1.10
        assert metrics["update_cosine"] >= 0.99
        assert 0.90 <= update_ratio <= 1.10
        assert metrics["zero_gradient_increase_pp"] <= 1.0
        assert counters and mean_diff <= 0.02 and var_diff <= 0.02
        assert all_finite(control, control_opt) and all_finite(candidate, candidate_opt)
        assert persistent_fp32(control, control_opt) and persistent_fp32(candidate, candidate_opt)
        del control, candidate, control_opt, candidate_opt, inputs, targets

    assert observed_dtypes["conv_linear"] == {"torch.bfloat16"}
    assert observed_dtypes["loss"] == {"torch.float32"}

    control, control_opt = model_and_optimizer(base_state)
    candidate, candidate_opt = model_and_optimizer(base_state)
    trajectory = []
    trajectory_alignment = []
    torch.manual_seed(42016)
    trajectory_loader = train.make_train_loader(
        strong_transform(), collate_fn=train.cutmix_collate
    )
    trajectory_iterator = iter(trajectory_loader)
    for step in range(200):
        cpu_inputs, cpu_targets = next(trajectory_iterator)
        inputs = cpu_inputs.cuda(non_blocking=True)
        targets = cpu_targets.cuda(non_blocking=True)
        control_logits, control_loss, control_grad, control_update = run_step(
            control, control_opt, inputs, targets, False
        )
        candidate_logits, candidate_loss, candidate_grad, candidate_update = run_step(
            candidate, candidate_opt, inputs, targets, True
        )
        control_concentration = torch.bincount(
            control_logits.argmax(1), minlength=train.NUM_CLASSES
        ).max().item() / train.BATCH_SIZE
        candidate_concentration = torch.bincount(
            candidate_logits.argmax(1), minlength=train.NUM_CLASSES
        ).max().item() / train.BATCH_SIZE
        assert torch.isfinite(control_loss) and torch.isfinite(candidate_loss)
        assert candidate_loss.item() <= 2.0 * control_loss.item()
        assert not (candidate_concentration > 0.95 and control_concentration <= 0.95)
        assert all_finite(control, control_opt) and all_finite(candidate, candidate_opt)
        assert persistent_fp32(control, control_opt) and persistent_fp32(candidate, candidate_opt)
        if step + 1 in {25, 50, 100, 150, 200}:
            alignment = {
                "step": step + 1,
                "logit_cosine": cosine(control_logits, candidate_logits),
                "gradient_cosine": cosine(control_grad, candidate_grad),
                "gradient_norm_ratio": candidate_grad.norm().item()
                / control_grad.norm().item(),
                "update_cosine": cosine(control_update, candidate_update),
                "update_norm_ratio": candidate_update.norm().item()
                / control_update.norm().item(),
                "loss_ratio": candidate_loss.item() / control_loss.item(),
            }
            assert alignment["logit_cosine"] >= 0.90
            assert alignment["gradient_cosine"] >= 0.70
            assert 0.50 <= alignment["gradient_norm_ratio"] <= 2.0
            assert alignment["update_cosine"] >= 0.70
            assert 0.50 <= alignment["update_norm_ratio"] <= 2.0
            assert alignment["loss_ratio"] <= 1.50
            trajectory_alignment.append(alignment)
        trajectory.append(
            {
                "step": step + 1,
                "control_loss": control_loss.item(),
                "candidate_loss": candidate_loss.item(),
                "control_concentration": control_concentration,
                "candidate_concentration": candidate_concentration,
            }
        )

    trajectory_iterator = None
    trajectory_workers = train.shutdown_train_loader(trajectory_loader)
    assert len(trajectory_workers) == train.NUM_WORKERS

    heldout_loader = train.make_train_loader(weak_transform())
    heldout_iterator = iter(heldout_loader)
    heldout = [next(heldout_iterator) for _ in range(5)]
    assert all(targets.ndim == 1 for _, targets in heldout)
    heldout_iterator = None
    heldout_workers = train.shutdown_train_loader(heldout_loader)
    assert len(heldout_workers) == train.NUM_WORKERS

    counters, mean_diff, var_diff = bn_differences(control, candidate)
    assert counters and mean_diff <= 0.02 and var_diff <= 0.02
    control.eval()
    candidate.eval()
    eval_metrics = []
    with torch.no_grad():
        for cpu_inputs, cpu_targets in heldout:
            inputs = cpu_inputs.cuda(non_blocking=True)
            targets = cpu_targets.cuda(non_blocking=True)
            control_logits = control(inputs)
            candidate_logits = candidate(inputs)
            control_loss = F.cross_entropy(control_logits, targets)
            candidate_loss = F.cross_entropy(candidate_logits, targets)
            metric = {
                "logit_cosine": cosine(control_logits.float(), candidate_logits.float()),
                "loss_ratio": candidate_loss.item() / control_loss.item(),
            }
            assert metric["logit_cosine"] >= 0.90
            assert metric["loss_ratio"] <= 1.50
            eval_metrics.append(metric)

    assert backend_state() == initial_backends
    result = {
        "status": "pass",
        "backend_state": initial_backends,
        "hard_batches": hard_count,
        "soft_batches": soft_count,
        "observed_dtypes": {k: sorted(v) for k, v in observed_dtypes.items()},
        "one_step": one_step,
        "trajectory_last": trajectory[-1],
        "trajectory_max_candidate_control_loss_ratio": max(
            row["candidate_loss"] / row["control_loss"] for row in trajectory
        ),
        "trajectory_alignment": trajectory_alignment,
        "bn_after_200": {
            "counters_match": counters,
            "running_mean_relative": mean_diff,
            "running_var_relative": var_diff,
        },
        "fp32_eval": eval_metrics,
    }
    NUMERICAL_RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print("NUMERICAL_GATE_PASS")
    print(json.dumps({k: v for k, v in result.items() if k not in {"one_step"}}, indent=2))


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def loader_gate():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    initial_backends = backend_state()
    setup_start = time.perf_counter()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    loader = train.make_train_loader(strong_transform(), collate_fn=train.cutmix_collate)
    model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, 3).cuda()
    optimizer = optim.SGD(
        model.parameters(), lr=train.LR, momentum=train.MOMENTUM, weight_decay=train.WEIGHT_DECAY
    )
    waits = []
    steps = []
    soft = 0
    iterator = iter(loader)
    loader_setup_seconds = time.perf_counter() - setup_start
    for index in range(1000):
        wait_start = time.perf_counter()
        try:
            cpu_inputs, cpu_targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            cpu_inputs, cpu_targets = next(iterator)
        waits.append(time.perf_counter() - wait_start)
        soft += int(cpu_targets.ndim == 2)
        step_start = time.perf_counter()
        inputs = cpu_inputs.cuda(non_blocking=True)
        targets = cpu_targets.cuda(non_blocking=True)
        progress = min(index / 1000, 1.0)
        lr = train.LR if progress <= train.LR_HOLD_FRACTION else train.ANNEAL_START_LR
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        steps.append(time.perf_counter() - step_start)
        assert torch.isfinite(loss)

    iterator = None
    switch_start = time.perf_counter()
    stopped = train.shutdown_train_loader(loader)
    weak_loader = train.make_train_loader(weak_transform())
    weak_iterator = iter(weak_loader)
    _, weak_targets = next(weak_iterator)
    switch_seconds = time.perf_counter() - switch_start
    weak_iterator = None
    weak_stopped = train.shutdown_train_loader(weak_loader)
    wait_median = statistics.median(waits[10:])
    wait_p95 = percentile(waits[10:], 0.95)
    step_median = statistics.median(steps[10:])
    cutmix_rate = soft / 1000
    assert len(stopped) == train.NUM_WORKERS
    assert len(weak_stopped) == train.NUM_WORKERS
    assert weak_targets.ndim == 1
    assert 0.45 <= cutmix_rate <= 0.55
    assert wait_median < 0.10 * step_median
    assert wait_p95 < 0.20 * step_median
    assert backend_state() == initial_backends
    result = {
        "status": "pass",
        "backend_state": initial_backends,
        "batches": 1000,
        "cutmix_rate": cutmix_rate,
        "wait_median_ms": 1000 * wait_median,
        "wait_p95_ms": 1000 * wait_p95,
        "step_median_ms": 1000 * step_median,
        "strong_workers_stopped": len(stopped),
        "weak_workers_stopped": len(weak_stopped),
        "weak_target_ndim": weak_targets.ndim,
        "switch_seconds": switch_seconds,
        "loader_model_setup_seconds": loader_setup_seconds,
    }
    LOADER_RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print("LOADER_GATE_PASS")
    print(json.dumps(result, indent=2))


def eval_child(output):
    start = time.perf_counter()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, 3).cuda()
    evaluator = Eval()
    ready = time.perf_counter()
    loss, accuracy = evaluator.evaluate(model, torch.device("cuda"))
    end = time.perf_counter()
    Path(output).write_text(
        json.dumps(
            {
                "startup_seconds": ready - start,
                "eval_seconds": end - ready,
                "loss": loss,
                "accuracy": accuracy,
                "backend_state": backend_state(),
            },
            indent=2,
        )
        + "\n"
    )


def eval_wall_gate():
    assert LOADER_RESULT.exists()
    assert TIMING_RESULT.exists()
    loader_result = json.loads(LOADER_RESULT.read_text())
    timing_result = json.loads(TIMING_RESULT.read_text())
    conditioning = HERE / "eval-conditioning.json"
    subprocess.run(
        [sys.executable, __file__, "--eval-child", str(conditioning)], check=True
    )
    scored = []
    for index in range(3):
        output = HERE / f"eval-scored-{index}.json"
        subprocess.run([sys.executable, __file__, "--eval-child", str(output)], check=True)
        scored.append(json.loads(output.read_text()))
    states = [row["backend_state"] for row in scored]
    assert all(state == states[0] for state in states)
    startup = max(row["startup_seconds"] for row in scored) + loader_result[
        "loader_model_setup_seconds"
    ]
    eval_seconds = max(row["eval_seconds"] for row in scored)
    projected_steps = max(
        26_898,
        timing_result["ratio_projected_steps"],
        timing_result["absolute_projected_steps"],
    )
    epochs = math.ceil(projected_steps / 390)
    expected_evals = epochs
    projection = (
        300
        + startup
        + loader_result["switch_seconds"]
        + expected_evals * eval_seconds
        + 10
    )
    assert projection < 540
    result = {
        "status": "pass",
        "conditioning_processes": 1,
        "scored_processes": 3,
        "startup_seconds_max": startup,
        "eval_seconds_max": eval_seconds,
        "projected_epochs": epochs,
        "expected_evaluations": expected_evals,
        "switch_seconds": loader_result["switch_seconds"],
        "safety_buffer_seconds": 10,
        "projected_total_seconds": projection,
        "backend_state": states[0],
    }
    EVAL_RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print("EVAL_WALL_GATE_PASS")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader", action="store_true")
    parser.add_argument("--eval-wall", action="store_true")
    parser.add_argument("--eval-child")
    args = parser.parse_args()
    if args.eval_child:
        eval_child(args.eval_child)
    elif args.loader:
        loader_gate()
    elif args.eval_wall:
        eval_wall_gate()
    else:
        numerical_gate()


if __name__ == "__main__":
    main()
