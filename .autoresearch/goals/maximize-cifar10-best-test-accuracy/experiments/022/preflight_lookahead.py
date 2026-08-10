import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402


def tensor_list_norm(tensors):
    return math.sqrt(sum(tensor.float().square().sum().item() for tensor in tensors))


def parameter_delta_norm(parameters, starts):
    return tensor_list_norm(
        [parameter.detach() - start for parameter, start in zip(parameters, starts, strict=True)]
    )


def optimizer_momentum(optimizer):
    return [optimizer.state[parameter]["momentum_buffer"] for parameter in optimizer.param_groups[0]["params"]]


def all_finite(tensors):
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def class_share(outputs):
    counts = outputs.argmax(1).bincount(minlength=train.NUM_CLASSES)
    return counts.max().item() / outputs.shape[0]


def materialize_batches():
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
    loader = train.make_train_loader(strong_transform, collate_fn=train.cutmix_collate)
    iterator = iter(loader)
    batches = []
    for _ in range(200):
        inputs, targets = next(iterator)
        batches.append((inputs.contiguous().clone(), targets.contiguous().clone()))
    iterator = None
    stopped = train.shutdown_train_loader(loader)
    if len(stopped) != train.NUM_WORKERS:
        raise RuntimeError(f"expected {train.NUM_WORKERS} stopped workers, got {len(stopped)}")

    corpus_path = EXPERIMENT_DIR / "preflight-corpus.pt"
    torch.save(batches, corpus_path)
    hard = next((batch for batch in batches if batch[1].ndim == 1), None)
    soft = next((batch for batch in batches if batch[1].ndim == 2), None)
    if hard is None or soft is None:
        raise RuntimeError("materialized corpus did not contain both hard and CutMix targets")
    torch.save({"hard": hard, "soft": soft}, EXPERIMENT_DIR / "timing-batches.pt")
    return batches, corpus_path


def decay_only_diagnostic(device):
    control = torch.nn.Parameter(torch.ones(4096, device=device))
    candidate = torch.nn.Parameter(control.detach().clone())
    slow = candidate.detach().clone()
    control_opt = torch.optim.SGD([control], lr=train.LR, momentum=train.MOMENTUM, weight_decay=train.WEIGHT_DECAY)
    candidate_opt = torch.optim.SGD([candidate], lr=train.LR, momentum=train.MOMENTUM, weight_decay=train.WEIGHT_DECAY)
    for completed_step in range(1, 51):
        control.grad = torch.zeros_like(control)
        candidate.grad = torch.zeros_like(candidate)
        control_opt.step()
        candidate_opt.step()
        if completed_step % train.LOOKAHEAD_K == 0:
            with torch.no_grad():
                torch._foreach_lerp_([slow], [candidate], train.LOOKAHEAD_ALPHA)
                torch._foreach_copy_([candidate], [slow])
    return {
        "control_norm": control.norm().item(),
        "candidate_norm": candidate.norm().item(),
        "candidate_control_norm_ratio": candidate.norm().item() / control.norm().item(),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    batches, corpus_path = materialize_batches()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    template = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER)
    initial_state = copy.deepcopy(template.state_dict())
    control_model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER).to(device)
    candidate_model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER).to(device)
    control_model.load_state_dict(initial_state)
    candidate_model.load_state_dict(initial_state)
    control_parameters = list(control_model.parameters())
    candidate_parameters = list(candidate_model.parameters())
    slow_parameters = [parameter.detach().clone() for parameter in candidate_parameters]
    if sum(parameter.numel() for parameter in control_parameters) != 1_073_962:
        raise RuntimeError("unexpected parameter count")
    if not all(torch.equal(left, right) for left, right in zip(control_parameters, candidate_parameters, strict=True)):
        raise RuntimeError("aligned models did not start byte-identical")

    control_opt = torch.optim.SGD(
        control_parameters, lr=train.LR, momentum=train.MOMENTUM, weight_decay=train.WEIGHT_DECAY
    )
    candidate_opt = torch.optim.SGD(
        candidate_parameters, lr=train.LR, momentum=train.MOMENTUM, weight_decay=train.WEIGHT_DECAY
    )
    rng_cpu_before = torch.get_rng_state().clone()
    rng_cuda_before = torch.cuda.get_rng_state().clone()
    control_ema = 0.0
    candidate_ema = 0.0
    beta = 0.95
    concentration_failures = []
    cycles = []
    recurrence_max_error = 0.0
    momentum_persisted = True
    step6_momentum_changed = False
    cycle_control_start = [parameter.detach().clone() for parameter in control_parameters]
    cycle_candidate_start = [parameter.detach().clone() for parameter in candidate_parameters]
    momentum_after_step5 = None

    for batch_index, (cpu_inputs, cpu_targets) in enumerate(batches, start=1):
        inputs = cpu_inputs.to(device, non_blocking=True)
        targets = cpu_targets.to(device, non_blocking=True)

        control_model.train()
        control_opt.zero_grad()
        control_outputs = control_model(inputs)
        control_loss = F.cross_entropy(control_outputs, targets)
        control_loss.backward()
        control_opt.step()

        candidate_model.train()
        candidate_opt.zero_grad()
        candidate_outputs = candidate_model(inputs)
        candidate_loss = F.cross_entropy(candidate_outputs, targets)
        candidate_loss.backward()
        candidate_opt.step()

        if batch_index <= 4:
            if not all(
                torch.equal(control, candidate)
                for control, candidate in zip(control_parameters, candidate_parameters, strict=True)
            ):
                raise RuntimeError(f"candidate diverged from control before first sync at step {batch_index}")

        if batch_index % train.LOOKAHEAD_K == 0:
            manual = [
                slow + train.LOOKAHEAD_ALPHA * (fast.detach() - slow)
                for slow, fast in zip(slow_parameters, candidate_parameters, strict=True)
            ]
            momentum_before = [buffer.detach().clone() for buffer in optimizer_momentum(candidate_opt)]
            momentum_ptrs = [buffer.data_ptr() for buffer in optimizer_momentum(candidate_opt)]
            fast_slow_distance = parameter_delta_norm(candidate_parameters, slow_parameters)
            with torch.no_grad():
                torch._foreach_lerp_(slow_parameters, candidate_parameters, train.LOOKAHEAD_ALPHA)
                torch._foreach_copy_(candidate_parameters, slow_parameters)
            recurrence_max_error = max(
                recurrence_max_error,
                max(
                    (actual - expected).abs().max().item()
                    for actual, expected in zip(candidate_parameters, manual, strict=True)
                ),
            )
            momentum_after = optimizer_momentum(candidate_opt)
            momentum_persisted = momentum_persisted and all(
                current.data_ptr() == pointer and torch.equal(current, before)
                for current, pointer, before in zip(momentum_after, momentum_ptrs, momentum_before, strict=True)
            )
            if batch_index == 5:
                momentum_after_step5 = [buffer.detach().clone() for buffer in momentum_after]
            if batch_index <= 50:
                cycles.append(
                    {
                        "step": batch_index,
                        "control_committed_norm": parameter_delta_norm(control_parameters, cycle_control_start),
                        "candidate_committed_norm": parameter_delta_norm(candidate_parameters, cycle_candidate_start),
                        "fast_slow_distance_before_sync": fast_slow_distance,
                        "control_momentum_norm": tensor_list_norm(optimizer_momentum(control_opt)),
                        "candidate_momentum_norm": tensor_list_norm(momentum_after),
                        "control_class_share": class_share(control_outputs),
                        "candidate_class_share": class_share(candidate_outputs),
                    }
                )
            cycle_control_start = [parameter.detach().clone() for parameter in control_parameters]
            cycle_candidate_start = [parameter.detach().clone() for parameter in candidate_parameters]
        elif batch_index == 6 and momentum_after_step5 is not None:
            step6_momentum_changed = any(
                not torch.equal(current, previous)
                for current, previous in zip(optimizer_momentum(candidate_opt), momentum_after_step5, strict=True)
            )

        control_value = control_loss.item()
        candidate_value = candidate_loss.item()
        control_ema = beta * control_ema + (1 - beta) * control_value
        candidate_ema = beta * candidate_ema + (1 - beta) * candidate_value
        control_share = class_share(control_outputs)
        candidate_share = class_share(candidate_outputs)
        if candidate_share > 0.95 and control_share <= 0.95:
            concentration_failures.append(
                {"step": batch_index, "candidate_share": candidate_share, "control_share": control_share}
            )
        tensors_to_check = (
            control_parameters
            + candidate_parameters
            + slow_parameters
            + list(control_model.buffers())
            + list(candidate_model.buffers())
            + optimizer_momentum(control_opt)
            + optimizer_momentum(candidate_opt)
        )
        if not math.isfinite(control_value) or not math.isfinite(candidate_value) or not all_finite(tensors_to_check):
            raise RuntimeError(f"non-finite state at step {batch_index}")

    debias = 1 - beta**len(batches)
    control_terminal_ema = control_ema / debias
    candidate_terminal_ema = candidate_ema / debias
    report = {
        "status": "pass",
        "corpus_path": str(corpus_path),
        "num_batches": len(batches),
        "hard_batches": sum(targets.ndim == 1 for _, targets in batches),
        "soft_batches": sum(targets.ndim == 2 for _, targets in batches),
        "parameter_count": sum(parameter.numel() for parameter in candidate_parameters),
        "steps_1_to_4_bitwise_equal": True,
        "recurrence_max_abs_error": recurrence_max_error,
        "momentum_persisted_at_sync": momentum_persisted,
        "step6_momentum_changed": step6_momentum_changed,
        "candidate_only_concentration_failures": concentration_failures,
        "control_terminal_loss_ema": control_terminal_ema,
        "candidate_terminal_loss_ema": candidate_terminal_ema,
        "candidate_control_loss_ema_ratio": candidate_terminal_ema / control_terminal_ema,
        "cpu_rng_unchanged": torch.equal(rng_cpu_before, torch.get_rng_state()),
        "cuda_rng_unchanged": torch.equal(rng_cuda_before, torch.cuda.get_rng_state()),
        "decay_only": decay_only_diagnostic(device),
        "cycles_through_step_50": cycles,
    }
    report_path = EXPERIMENT_DIR / "preflight-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    if recurrence_max_error != 0.0:
        raise RuntimeError(f"foreach recurrence mismatch: {recurrence_max_error}")
    if not momentum_persisted or not step6_momentum_changed:
        raise RuntimeError("momentum persistence/update gate failed")
    if concentration_failures:
        raise RuntimeError(f"candidate-only concentration: {concentration_failures[0]}")
    if candidate_terminal_ema > 1.5 * control_terminal_ema:
        raise RuntimeError("candidate terminal loss EMA exceeded 1.5x control")
    if not report["cpu_rng_unchanged"] or not report["cuda_rng_unchanged"]:
        raise RuntimeError("paired optimizer path unexpectedly consumed RNG")
    print(json.dumps({key: value for key, value in report.items() if key != "cycles_through_step_50"}))


if __name__ == "__main__":
    main()
