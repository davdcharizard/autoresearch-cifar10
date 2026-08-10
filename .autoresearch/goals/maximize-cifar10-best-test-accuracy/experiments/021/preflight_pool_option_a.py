import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType

# ruff: noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import train


HERE = Path(__file__).resolve().parent
BASELINE_STEPS = 26_898
BASELINE_EPOCHS = 69
BASELINE_EVALUATIONS = 19


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def cv(values):
    return statistics.pstdev(values) / statistics.mean(values)


def tensor_hash_update(digest, tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(memoryview(tensor.numpy()).cast("B"))


def corpus_hash(batches):
    digest = hashlib.sha256()
    for inputs, targets in batches:
        tensor_hash_update(digest, inputs)
        tensor_hash_update(digest, targets)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        tensor_hash_update(digest, tensor)
    return digest.hexdigest()


def rng_hash():
    digest = hashlib.sha256()
    tensor_hash_update(digest, torch.get_rng_state())
    tensor_hash_update(digest, torch.cuda.get_rng_state())
    return digest.hexdigest()


def strong_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
        ]
    )


def accepted_block_forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    shortcut = x
    if self.need_pad:
        shortcut = shortcut[:, :, :: self.stride, :: self.stride]
        shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))
    out += shortcut
    return F.relu(out)


def make_model(arm, device="cpu"):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER)
    if arm == "control":
        for block in all_blocks(model):
            if block.need_pad:
                block.forward = MethodType(accepted_block_forward, block)
    return model.to(device)


def make_optimizer(model):
    return optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )


def all_blocks(model):
    return [
        module for module in model.modules() if isinstance(module, train.BasicBlock)
    ]


def finite_model_and_optimizer(model, optimizer):
    model_finite = all(
        not value.is_floating_point() or torch.isfinite(value).all().item()
        for value in model.state_dict().values()
    )
    optimizer_finite = all(
        not value.is_floating_point() or torch.isfinite(value).all().item()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    return model_finite and optimizer_finite


def backend_state():
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "dtype": str(torch.get_default_dtype()),
        "device": torch.cuda.get_device_name(0),
    }


def semantics(output):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate_cpu = train.ResNet(3, 10, 2)
    candidate_cpu_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state())
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    control_cpu = train.ResNet(3, 10, 2)
    control_cpu_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state())

    state_equal = all(
        torch.equal(left, right)
        for left, right in zip(
            candidate_cpu.state_dict().values(),
            control_cpu.state_dict().values(),
            strict=True,
        )
    )
    rng_equal = all(
        torch.equal(left, right)
        for left, right in zip(candidate_cpu_rng, control_cpu_rng, strict=True)
    )
    candidate_blocks = all_blocks(candidate_cpu)
    transitions = [block for block in candidate_blocks if block.need_pad]
    identities = [block for block in candidate_blocks if not block.need_pad]
    inventory = [
        {
            "stride": block.stride,
            "in_channels": block.conv1.in_channels,
            "out_channels": block.conv1.out_channels,
            "pad_channels": block.pad_channels,
        }
        for block in transitions
    ]

    ramp = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    pooled_ramp = F.avg_pool2d(ramp, 2, 2)
    expected_ramp = torch.tensor([[[[2.5, 4.5], [10.5, 12.5]]]])
    impulses = []
    for row, column in ((0, 0), (0, 1), (1, 0), (1, 1)):
        impulse = torch.zeros(1, 1, 4, 4)
        impulse[0, 0, row, column] = 1
        impulses.append(F.avg_pool2d(impulse, 2, 2))
    impulse_ok = all(
        output_tensor[0, 0, 0, 0].item() == 0.25
        and torch.count_nonzero(output_tensor).item() == 1
        for output_tensor in impulses
    )
    shortcut_input = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    pooled_shortcut = F.avg_pool2d(shortcut_input, 2, 2)
    padded_shortcut = F.pad(pooled_shortcut, (0, 0, 0, 0, 0, 2))
    padded_shortcut_ok = torch.equal(padded_shortcut[:, :2], pooled_shortcut) and (
        torch.count_nonzero(padded_shortcut[:, 2:]).item() == 0
    )

    pool_input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    pool_input.requires_grad_(True)
    F.avg_pool2d(pool_input, 2, 2).sum().backward()
    pool_gradient = pool_input.grad.detach().clone()
    control_input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    control_input.requires_grad_(True)
    control_input[:, :, ::2, ::2].sum().backward()
    control_gradient = control_input.grad.detach().clone()

    candidate = candidate_cpu.cuda().train()
    transition_shapes = []
    hooks = []
    for name, block in (
        ("layer2.0", candidate.layer2[0]),
        ("layer3.0", candidate.layer3[0]),
    ):
        hooks.append(
            block.register_forward_pre_hook(
                lambda _module, args, name=name: transition_shapes.append(
                    {"name": name, "input": list(args[0].shape)}
                )
            )
        )
        hooks.append(
            block.register_forward_hook(
                lambda _module, _args, value, name=name: next(
                    item for item in transition_shapes if item["name"] == name
                ).update({"output": list(value.shape)})
            )
        )

    original_avg_pool2d = F.avg_pool2d
    pool_calls = []

    def recorded_avg_pool2d(inputs, *args, **kwargs):
        result = original_avg_pool2d(inputs, *args, **kwargs)
        pool_calls.append(
            {
                "input": list(inputs.shape),
                "output": list(result.shape),
                "args": list(args),
                "kwargs": kwargs,
            }
        )
        return result

    finite_targets = {}
    try:
        F.avg_pool2d = recorded_avg_pool2d
        fixed_inputs = torch.linspace(
            -0.5, 0.5, 128 * 3 * 32 * 32, device="cuda"
        ).reshape(128, 3, 32, 32)
        for target_name in ("hard", "probability"):
            candidate.zero_grad(set_to_none=True)
            labels = torch.arange(128, device="cuda") % 10
            targets = labels if target_name == "hard" else F.one_hot(labels, 10).float()
            loss = F.cross_entropy(candidate(fixed_inputs), targets)
            loss.backward()
            gradients = [parameter.grad for parameter in candidate.parameters()]
            finite_targets[target_name] = {
                "loss": loss.item(),
                "all_gradients_present": all(value is not None for value in gradients),
                "all_gradients_finite": all(
                    value is not None and torch.isfinite(value).all().item()
                    for value in gradients
                ),
            }
    finally:
        F.avg_pool2d = original_avg_pool2d
        for hook in hooks:
            hook.remove()

    first_pass_calls = pool_calls[:2]
    expected_shapes = [
        {"name": "layer2.0", "input": [128, 32, 32, 32], "output": [128, 64, 16, 16]},
        {"name": "layer3.0", "input": [128, 64, 16, 16], "output": [128, 128, 8, 8]},
    ]
    gates = {
        "parameters": sum(parameter.numel() for parameter in candidate.parameters())
        == 1_073_962,
        "state_equal": state_equal,
        "rng_equal": rng_equal,
        "inventory": len(transitions) == 2
        and len(identities) == 7
        and all(block.stride == 2 for block in transitions),
        "even_inputs": transition_shapes[:2] == expected_shapes,
        "runtime_pool_count": len(pool_calls) == 4,
        "runtime_pool_shapes": [call["input"] for call in first_pass_calls]
        == [[128, 32, 32, 32], [128, 64, 16, 16]],
        "ramp": torch.equal(pooled_ramp, expected_ramp),
        "impulses": impulse_ok,
        "padded_shortcut": padded_shortcut_ok,
        "pool_gradient": torch.equal(
            pool_gradient, torch.full_like(pool_gradient, 0.25)
        ),
        "control_gradient": torch.equal(
            control_gradient,
            torch.tensor(
                [
                    [
                        [
                            [1.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            ),
        ),
        "finite_targets": all(
            item["all_gradients_present"] and item["all_gradients_finite"]
            for item in finite_targets.values()
        ),
    }
    result = {
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "inventory": inventory,
        "transition_shapes": transition_shapes[:2],
        "pool_calls": first_pass_calls,
        "finite_targets": finite_targets,
        "parameter_count": sum(
            parameter.numel() for parameter in candidate.parameters()
        ),
        "state_hash": state_hash(candidate_cpu),
    }
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit("SEMANTICS_GATE_FAIL")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def materialize(batch_count, output, manifest):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    loader = train.make_train_loader(strong_transform(), train.cutmix_collate)
    iterator = iter(loader)
    batches = []
    worker_pids = []
    try:
        while len(batches) < batch_count:
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets = next(iterator)
            batches.append((inputs.clone(), targets.clone()))
    finally:
        iterator = None
        worker_pids = train.shutdown_train_loader(loader)
        del loader
        gc.collect()

    digest = corpus_hash(batches)
    payload = {
        "version": 1,
        "sha256": digest,
        "batch_size": train.BATCH_SIZE,
        "batches": batches,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    hard = sum(targets.ndim == 1 for _, targets in batches)
    mixed = sum(targets.ndim == 2 for _, targets in batches)
    workers_alive = [pid for pid in worker_pids if pid_alive(pid)]
    gates = {
        "batch_count": len(batches) == batch_count,
        "hard_floor": hard >= 80,
        "mixed_floor": mixed >= 80,
        "workers_stopped": len(worker_pids) == train.NUM_WORKERS and not workers_alive,
        "target_shapes": all(
            targets.shape
            in ((train.BATCH_SIZE,), (train.BATCH_SIZE, train.NUM_CLASSES))
            for _, targets in batches
        ),
        "finite": all(
            torch.isfinite(inputs).all().item()
            and (
                not targets.is_floating_point() or torch.isfinite(targets).all().item()
            )
            for inputs, targets in batches
        ),
    }
    result = {
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "sha256": digest,
        "batches": len(batches),
        "hard_batches": hard,
        "mixed_batches": mixed,
        "worker_pids": worker_pids,
        "workers_alive_after_shutdown": workers_alive,
        "bytes": output.stat().st_size,
    }
    write_json(manifest, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit("MATERIALIZE_GATE_FAIL")


def load_corpus(path, pin=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    batches = payload["batches"]
    digest = corpus_hash(batches)
    if digest != payload["sha256"]:
        raise RuntimeError(f"corpus hash mismatch: {digest} != {payload['sha256']}")
    if pin:
        batches = [
            (inputs.pin_memory(), targets.pin_memory()) for inputs, targets in batches
        ]
    return batches, digest


def safety_child(arm, corpus, output):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    batches, digest = load_corpus(corpus)
    model = make_model(arm, "cuda").train()
    starting_state = state_hash(model)
    starting_rng = rng_hash()
    optimizer = make_optimizer(model)
    ema = 0.0
    records = []
    max_gradient_norm = 0.0
    max_update_norm = 0.0
    status = "pass"

    for index, (cpu_inputs, cpu_targets) in enumerate(batches):
        inputs = cpu_inputs.cuda(non_blocking=False)
        targets = cpu_targets.cuda(non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        gradient_square = torch.zeros((), device="cuda")
        gradient_finite = True
        before = []
        for parameter in model.parameters():
            before.append(parameter.detach().clone())
            if (
                parameter.grad is None
                or not torch.isfinite(parameter.grad).all().item()
            ):
                gradient_finite = False
            elif parameter.grad is not None:
                gradient_square += parameter.grad.float().square().sum()
        gradient_norm = gradient_square.sqrt().item()
        optimizer.step()
        update_square = torch.zeros((), device="cuda")
        for parameter, previous in zip(model.parameters(), before, strict=True):
            update_square += (parameter.detach() - previous).float().square().sum()
        update_norm = update_square.sqrt().item()
        histogram = torch.bincount(logits.argmax(1), minlength=10).cpu().tolist()
        concentration = max(histogram) / logits.shape[0]
        ema = 0.95 * ema + 0.05 * loss.item()
        state_finite = finite_model_and_optimizer(model, optimizer)
        step_finite = (
            torch.isfinite(loss).item()
            and gradient_finite
            and math.isfinite(gradient_norm)
            and math.isfinite(update_norm)
            and state_finite
        )
        records.append(
            {
                "step": index + 1,
                "target_ndim": cpu_targets.ndim,
                "loss": loss.item(),
                "loss_ema": ema / (1 - 0.95 ** (index + 1)),
                "histogram": histogram,
                "concentration": concentration,
                "gradient_norm": gradient_norm,
                "update_norm": update_norm,
                "finite": step_finite,
            }
        )
        max_gradient_norm = max(max_gradient_norm, gradient_norm)
        max_update_norm = max(max_update_norm, update_norm)
        if not step_finite:
            status = "fail"
            break

    result = {
        "status": status,
        "arm": arm,
        "corpus_sha256": digest,
        "starting_state_sha256": starting_state,
        "starting_rng_sha256": starting_rng,
        "terminal_rng_sha256": rng_hash(),
        "terminal_loss_ema": records[-1]["loss_ema"],
        "max_gradient_norm": max_gradient_norm,
        "max_update_norm": max_update_norm,
        "hard_batches": sum(item["target_ndim"] == 1 for item in records),
        "mixed_batches": sum(item["target_ndim"] == 2 for item in records),
        "backend": backend_state(),
        "records": records,
    }
    write_json(output, result)


def safety(corpus, output):
    child_results = {}
    for arm in ("control", "candidate"):
        child_output = HERE / f"preflight-safety-{arm}.json"
        subprocess.run(
            [
                sys.executable,
                __file__,
                "safety-child",
                "--arm",
                arm,
                "--corpus",
                str(corpus),
                "--output",
                str(child_output),
            ],
            check=True,
            timeout=240,
        )
        child_results[arm] = json.loads(child_output.read_text())

    control = child_results["control"]
    candidate = child_results["candidate"]
    paired_steps = min(len(control["records"]), len(candidate["records"]))
    concentration_failures = []
    for index in range(paired_steps):
        control_step = control["records"][index]
        candidate_step = candidate["records"][index]
        if (
            candidate_step["concentration"] > 0.95
            and control_step["concentration"] <= 0.95
        ):
            concentration_failures.append(
                {
                    "step": index + 1,
                    "control_histogram": control_step["histogram"],
                    "candidate_histogram": candidate_step["histogram"],
                }
            )
    ema_ratio = candidate["terminal_loss_ema"] / control["terminal_loss_ema"]
    gates = {
        "children": control["status"] == "pass" and candidate["status"] == "pass",
        "corpus_hash": control["corpus_sha256"] == candidate["corpus_sha256"],
        "starting_state": control["starting_state_sha256"]
        == candidate["starting_state_sha256"],
        "starting_rng": control["starting_rng_sha256"]
        == candidate["starting_rng_sha256"],
        "terminal_rng": control["terminal_rng_sha256"]
        == candidate["terminal_rng_sha256"],
        "all_batches": paired_steps == 200,
        "hard_floor": candidate["hard_batches"] >= 80,
        "mixed_floor": candidate["mixed_batches"] >= 80,
        "concentration": not concentration_failures,
        "loss_ema": ema_ratio <= 1.5,
        "backend": control["backend"] == candidate["backend"],
    }
    result = {
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "terminal_loss_ema_ratio": ema_ratio,
        "concentration_failures": concentration_failures,
        "control": {key: value for key, value in control.items() if key != "records"},
        "candidate": {
            key: value for key, value in candidate.items() if key != "records"
        },
        "child_outputs": {
            "control": str(HERE / "preflight-safety-control.json"),
            "candidate": str(HERE / "preflight-safety-candidate.json"),
        },
    }
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit("SAFETY_GATE_FAIL")


def set_lr(optimizer, progress):
    if progress <= train.LR_HOLD_FRACTION:
        lr = train.LR
    else:
        cosine_progress = (progress - train.LR_HOLD_FRACTION) / (
            1 - train.LR_HOLD_FRACTION
        )
        lr = train.MIN_LR + 0.5 * (train.ANNEAL_START_LR - train.MIN_LR) * (
            1 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr


def training_step(model, optimizer, cpu_inputs, cpu_targets, progress):
    started = time.perf_counter()
    inputs = cpu_inputs.cuda(non_blocking=True)
    targets = cpu_targets.cuda(non_blocking=True)
    set_lr(optimizer, progress)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return time.perf_counter() - started, loss


def profiled_step(model, optimizer, cpu_inputs, cpu_targets, progress):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    events[0].record()
    inputs = cpu_inputs.cuda(non_blocking=True)
    targets = cpu_targets.cuda(non_blocking=True)
    events[1].record()
    set_lr(optimizer, progress)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    events[2].record()
    loss = F.cross_entropy(outputs, targets)
    events[3].record()
    loss.backward()
    events[4].record()
    optimizer.step()
    events[5].record()
    torch.cuda.synchronize()
    return {
        "transfer_ms": events[0].elapsed_time(events[1]),
        "forward_ms": events[1].elapsed_time(events[2]),
        "loss_ms": events[2].elapsed_time(events[3]),
        "backward_ms": events[3].elapsed_time(events[4]),
        "optimizer_ms": events[4].elapsed_time(events[5]),
        "loss_value": loss.item(),
    }


def timing_child(arm, corpus, warmup, steps, output):
    batches, digest = load_corpus(corpus, pin=True)
    hard_batches = [batch for batch in batches if batch[1].ndim == 1]
    if len(hard_batches) < 80:
        raise RuntimeError("timing corpus lacks hard batches")
    started = time.perf_counter()
    model = make_model(arm, "cuda").train()
    optimizer = make_optimizer(model)
    startup_seconds = time.perf_counter() - started
    backend = backend_state()
    torch.cuda.reset_peak_memory_stats()

    def selected_batch(index):
        if index % 5 == 4:
            return hard_batches[index % len(hard_batches)], 0.9
        return batches[index % len(batches)], 0.5

    for index in range(warmup):
        (inputs, targets), progress = selected_batch(index)
        _, loss = training_step(model, optimizer, inputs, targets, progress)
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite timing warmup")

    durations = []
    for index in range(steps):
        (inputs, targets), progress = selected_batch(warmup + index)
        duration, loss = training_step(model, optimizer, inputs, targets, progress)
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite measured timing step")
        durations.append(duration)

    stage_records = []
    for index in range(50):
        (inputs, targets), progress = selected_batch(warmup + steps + index)
        stage_records.append(profiled_step(model, optimizer, inputs, targets, progress))

    inference_inputs = batches[0][0][:256]
    if inference_inputs.shape[0] != 128:
        raise RuntimeError("unexpected timing batch")
    inference_inputs = torch.cat((inference_inputs, inference_inputs), dim=0).cuda()
    model.eval()
    with torch.inference_mode():
        for _ in range(100):
            model(inference_inputs)
        torch.cuda.synchronize()
        inference = []
        for _ in range(500):
            inference_started = time.perf_counter()
            model(inference_inputs)
            torch.cuda.synchronize()
            inference.append(time.perf_counter() - inference_started)

    stage_means = {
        key: statistics.mean(record[key] for record in stage_records)
        for key in (
            "transfer_ms",
            "forward_ms",
            "loss_ms",
            "backward_ms",
            "optimizer_ms",
        )
    }
    result = {
        "status": "pass",
        "arm": arm,
        "corpus_sha256": digest,
        "backend": backend,
        "warmup": warmup,
        "steps": steps,
        "mean_ms": 1000 * statistics.mean(durations),
        "median_ms": 1000 * statistics.median(durations),
        "p95_ms": 1000 * percentile(durations, 0.95),
        "stage_mean_ms": stage_means,
        "inference_mean_ms": 1000 * statistics.mean(inference),
        "inference_p95_ms": 1000 * percentile(inference, 0.95),
        "peak_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "startup_seconds": startup_seconds,
    }
    write_json(output, result)


def condition_device():
    model = make_model("control", "cuda").train()
    optimizer = make_optimizer(model)
    inputs = torch.zeros(128, 3, 32, 32).pin_memory()
    targets = torch.arange(128).remainder(10).pin_memory()
    for _ in range(20):
        _, loss = training_step(model, optimizer, inputs, targets, 0.5)
        if not torch.isfinite(loss).item():
            raise RuntimeError("device conditioning became non-finite")
    model.eval()
    with torch.inference_mode():
        evaluation_inputs = torch.zeros(256, 3, 32, 32, device="cuda")
        for _ in range(20):
            model(evaluation_inputs)
        torch.cuda.synchronize()
    print(json.dumps({"status": "pass", "backend": backend_state()}, indent=2))


def timing(corpus, pairs, warmup, steps, output):
    trials = []
    for index in range(pairs):
        order = ("control", "candidate") if index % 2 == 0 else ("candidate", "control")
        trial = {"trial": index + 1, "order": list(order), "arms": {}}
        for arm in order:
            child_output = HERE / f"preflight-timing-{index + 1}-{arm}.json"
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "timing-child",
                    "--arm",
                    arm,
                    "--corpus",
                    str(corpus),
                    "--warmup",
                    str(warmup),
                    "--steps",
                    str(steps),
                    "--output",
                    str(child_output),
                ],
                check=True,
                timeout=240,
            )
            trial["arms"][arm] = json.loads(child_output.read_text())
        trials.append(trial)

    backend_states = [
        trial["arms"][arm]["backend"]
        for trial in trials
        for arm in ("control", "candidate")
    ]
    control_means = [trial["arms"]["control"]["mean_ms"] for trial in trials]
    candidate_means = [trial["arms"]["candidate"]["mean_ms"] for trial in trials]
    pair_ratios = [
        candidate / control
        for candidate, control in zip(candidate_means, control_means, strict=True)
    ]
    control_mean = statistics.mean(control_means)
    candidate_mean = statistics.mean(candidate_means)
    aggregate_ratio = candidate_mean / control_mean
    median_pair_ratio = statistics.median(pair_ratios)
    projected_steps = math.floor(BASELINE_STEPS / aggregate_ratio)
    projected_epochs = math.ceil(projected_steps / 390)
    expected_evaluations = BASELINE_EVALUATIONS + projected_epochs - BASELINE_EPOCHS
    control_p95 = statistics.mean(
        trial["arms"]["control"]["p95_ms"] for trial in trials
    )
    candidate_p95 = statistics.mean(
        trial["arms"]["candidate"]["p95_ms"] for trial in trials
    )
    control_inference = [
        trial["arms"]["control"]["inference_mean_ms"] for trial in trials
    ]
    candidate_inference = [
        trial["arms"]["candidate"]["inference_mean_ms"] for trial in trials
    ]
    inference_ratio = statistics.mean(candidate_inference) / statistics.mean(
        control_inference
    )
    peak_control = max(trial["arms"]["control"]["peak_mb"] for trial in trials)
    peak_candidate = max(trial["arms"]["candidate"]["peak_mb"] for trial in trials)
    projected_total = 330.7 + max(0.0, inference_ratio - 1.0) * 30.7 + 5.0
    gates = {
        "children": all(
            trial["arms"][arm]["status"] == "pass"
            for trial in trials
            for arm in ("control", "candidate")
        ),
        "backend_parity": all(state == backend_states[0] for state in backend_states),
        "corpus_parity": len(
            {
                trial["arms"][arm]["corpus_sha256"]
                for trial in trials
                for arm in ("control", "candidate")
            }
        )
        == 1,
        "median_ratio": median_pair_ratio <= 1.02,
        "steps": projected_steps >= 26_360,
        "control_cv": cv(control_means) < 0.03,
        "candidate_cv": cv(candidate_means) < 0.03,
        "p95": candidate_p95 <= 1.05 * control_p95,
        "memory": peak_candidate < 625 and peak_candidate - peak_control <= 16,
        "inference": inference_ratio <= 1.02,
        "wall": projected_total < 540,
        "evaluations": expected_evaluations == 19,
    }
    result = {
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "backend": backend_states[0],
        "aggregate_training_ratio": aggregate_ratio,
        "median_pair_ratio": median_pair_ratio,
        "pair_ratios": pair_ratios,
        "control_mean_ms": control_mean,
        "candidate_mean_ms": candidate_mean,
        "control_cv": cv(control_means),
        "candidate_cv": cv(candidate_means),
        "p95_ratio": candidate_p95 / control_p95,
        "projected_steps": projected_steps,
        "projected_epochs": projected_epochs,
        "expected_evaluations": expected_evaluations,
        "inference_ratio": inference_ratio,
        "peak_control_mb": peak_control,
        "peak_candidate_mb": peak_candidate,
        "projected_total_seconds": projected_total,
        "trials": trials,
    }
    write_json(output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "trials"},
            indent=2,
            sort_keys=True,
        )
    )
    if result["status"] != "pass":
        raise SystemExit("TIMING_GATE_FAIL")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    semantics_parser = subparsers.add_parser("semantics")
    semantics_parser.add_argument("--output", required=True)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--batches", type=int, required=True)
    materialize_parser.add_argument("--output", required=True)
    materialize_parser.add_argument("--manifest", required=True)

    safety_parser = subparsers.add_parser("safety")
    safety_parser.add_argument("--corpus", required=True)
    safety_parser.add_argument("--output", required=True)

    safety_child_parser = subparsers.add_parser("safety-child")
    safety_child_parser.add_argument(
        "--arm", choices=("control", "candidate"), required=True
    )
    safety_child_parser.add_argument("--corpus", required=True)
    safety_child_parser.add_argument("--output", required=True)

    subparsers.add_parser("condition-device")

    timing_parser = subparsers.add_parser("timing")
    timing_parser.add_argument("--corpus", required=True)
    timing_parser.add_argument("--pairs", type=int, required=True)
    timing_parser.add_argument("--warmup", type=int, required=True)
    timing_parser.add_argument("--steps", type=int, required=True)
    timing_parser.add_argument("--output", required=True)

    timing_child_parser = subparsers.add_parser("timing-child")
    timing_child_parser.add_argument(
        "--arm", choices=("control", "candidate"), required=True
    )
    timing_child_parser.add_argument("--corpus", required=True)
    timing_child_parser.add_argument("--warmup", type=int, required=True)
    timing_child_parser.add_argument("--steps", type=int, required=True)
    timing_child_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "semantics":
        semantics(args.output)
    elif args.command == "materialize":
        materialize(args.batches, args.output, args.manifest)
    elif args.command == "safety":
        safety(args.corpus, args.output)
    elif args.command == "safety-child":
        safety_child(args.arm, args.corpus, args.output)
    elif args.command == "condition-device":
        condition_device()
    elif args.command == "timing":
        timing(args.corpus, args.pairs, args.warmup, args.steps, args.output)
    elif args.command == "timing-child":
        timing_child(args.arm, args.corpus, args.warmup, args.steps, args.output)


if __name__ == "__main__":
    main()
