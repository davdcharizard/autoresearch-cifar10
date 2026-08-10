import hashlib
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


def normalized_gradient_norm(parameters):
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return tensor_list_norm(gradients) / max(tensor_list_norm(parameters), 1e-12)


def normalized_update_norm(parameters, starts):
    updates = [parameter.detach() - start for parameter, start in zip(parameters, starts, strict=True)]
    return tensor_list_norm(updates) / max(tensor_list_norm(starts), 1e-12)


def all_finite(model, optimizer):
    tensors = list(model.parameters()) + list(model.buffers())
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def class_share(outputs):
    counts = outputs.argmax(1).bincount(minlength=train.NUM_CLASSES)
    return counts.max().item() / outputs.shape[0]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_batches():
    mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    loader = train.make_train_loader(transform, collate_fn=train.cutmix_collate)
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
        raise RuntimeError("corpus must contain hard and probability targets")
    torch.save({"hard": hard, "soft": soft}, EXPERIMENT_DIR / "timing-batches.pt")
    return batches, corpus_path, stopped


def build_model(num_blocks, width, device):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    return train.ResNet(num_blocks, train.NUM_CLASSES, width).to(device)


def structural_checks(control, candidate, sample):
    control_params = sum(parameter.numel() for parameter in control.parameters())
    candidate_params = sum(parameter.numel() for parameter in candidate.parameters())
    if control_params != 1_073_962 or candidate_params != 1_540_474:
        raise RuntimeError(f"parameter counts {control_params=} {candidate_params=}")
    if len(candidate.layer1) != 2 or len(candidate.layer2) != 2 or len(candidate.layer3) != 2:
        raise RuntimeError("candidate does not have two blocks per stage")
    if sum(isinstance(module, torch.nn.Conv2d) for module in candidate.modules()) != 13:
        raise RuntimeError("candidate convolution count is not 13")
    if not candidate.layer2[0].need_pad or candidate.layer2[0].pad_channels != 48:
        raise RuntimeError("stage-2 Option-A transition mismatch")
    if not candidate.layer3[0].need_pad or candidate.layer3[0].pad_channels != 96:
        raise RuntimeError("stage-3 Option-A transition mismatch")
    shapes = {}
    handles = [
        candidate.layer1.register_forward_hook(lambda _m, _i, o: shapes.__setitem__("layer1", tuple(o.shape))),
        candidate.layer2.register_forward_hook(lambda _m, _i, o: shapes.__setitem__("layer2", tuple(o.shape))),
        candidate.layer3.register_forward_hook(lambda _m, _i, o: shapes.__setitem__("layer3", tuple(o.shape))),
    ]
    candidate.eval()
    buffers_before = [buffer.detach().clone() for buffer in candidate.buffers()]
    with torch.no_grad():
        outputs = candidate(sample)
    for handle in handles:
        handle.remove()
    expected = {
        "layer1": (sample.shape[0], 48, 32, 32),
        "layer2": (sample.shape[0], 96, 16, 16),
        "layer3": (sample.shape[0], 192, 8, 8),
    }
    if shapes != expected or outputs.shape != (sample.shape[0], 10) or not torch.isfinite(outputs).all():
        raise RuntimeError(f"shape/evaluation mismatch: {shapes}")
    if not all(torch.equal(before, after) for before, after in zip(buffers_before, candidate.buffers(), strict=True)):
        raise RuntimeError("evaluation mutated candidate buffers")
    if any(parameter.dtype != torch.float32 for parameter in candidate.parameters()):
        raise RuntimeError("candidate parameters are not FP32")
    return control_params, candidate_params, shapes


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    batches, corpus_path, stopped = materialize_batches()
    control = build_model(3, 2, device)
    candidate = build_model(2, 3, device)
    control_params = list(control.parameters())
    candidate_params = list(candidate.parameters())
    counts = structural_checks(control, candidate, batches[0][0][:2].to(device))
    control_opt = torch.optim.SGD(
        control_params, lr=train.LR, momentum=train.MOMENTUM, weight_decay=train.WEIGHT_DECAY
    )
    candidate_opt = torch.optim.SGD(
        candidate_params, lr=train.LR, momentum=train.MOMENTUM, weight_decay=train.WEIGHT_DECAY
    )
    beta = 0.95
    control_ema = 0.0
    candidate_ema = 0.0
    concentration_events = []
    diagnostics = []

    for step, (cpu_inputs, cpu_targets) in enumerate(batches, start=1):
        inputs = cpu_inputs.to(device, non_blocking=True)
        targets = cpu_targets.to(device, non_blocking=True)

        control.train()
        control_starts = [parameter.detach().clone() for parameter in control_params]
        control_opt.zero_grad()
        control_outputs = control(inputs)
        control_loss = F.cross_entropy(control_outputs, targets)
        control_loss.backward()
        control_grad = normalized_gradient_norm(control_params)
        control_opt.step()
        control_update = normalized_update_norm(control_params, control_starts)

        candidate.train()
        candidate_starts = [parameter.detach().clone() for parameter in candidate_params]
        candidate_opt.zero_grad()
        candidate_outputs = candidate(inputs)
        candidate_loss = F.cross_entropy(candidate_outputs, targets)
        candidate_loss.backward()
        candidate_grad = normalized_gradient_norm(candidate_params)
        candidate_opt.step()
        candidate_update = normalized_update_norm(candidate_params, candidate_starts)

        control_value = control_loss.item()
        candidate_value = candidate_loss.item()
        control_ema = beta * control_ema + (1 - beta) * control_value
        candidate_ema = beta * candidate_ema + (1 - beta) * candidate_value
        control_share = class_share(control_outputs)
        candidate_share = class_share(candidate_outputs)
        if candidate_share > 0.95 and control_share <= 0.90:
            concentration_events.append(
                {"step": step, "candidate_share": candidate_share, "control_share": control_share}
            )
        if step <= 20 or step % 20 == 0:
            diagnostics.append(
                {
                    "step": step,
                    "control_loss": control_value,
                    "candidate_loss": candidate_value,
                    "control_class_share": control_share,
                    "candidate_class_share": candidate_share,
                    "control_normalized_gradient": control_grad,
                    "candidate_normalized_gradient": candidate_grad,
                    "control_normalized_update": control_update,
                    "candidate_normalized_update": candidate_update,
                }
            )
        if not math.isfinite(control_value) or not math.isfinite(candidate_value):
            raise RuntimeError(f"non-finite loss at step {step}")
        if not all_finite(control, control_opt) or not all_finite(candidate, candidate_opt):
            raise RuntimeError(f"non-finite model/optimizer state at step {step}")

    debias = 1 - beta**len(batches)
    control_terminal = control_ema / debias
    candidate_terminal = candidate_ema / debias
    report = {
        "status": "pass",
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256(corpus_path),
        "num_batches": len(batches),
        "hard_batches": sum(targets.ndim == 1 for _, targets in batches),
        "soft_batches": sum(targets.ndim == 2 for _, targets in batches),
        "workers_stopped": len(stopped),
        "control_constructor": "ResNet(3, 10, 2)",
        "candidate_constructor": "ResNet(2, 10, 3)",
        "control_parameter_count": counts[0],
        "candidate_parameter_count": counts[1],
        "candidate_stage_shapes": counts[2],
        "candidate_only_concentration_events": concentration_events,
        "control_terminal_loss_ema": control_terminal,
        "candidate_terminal_loss_ema": candidate_terminal,
        "candidate_control_loss_ema_ratio": candidate_terminal / control_terminal,
        "diagnostics": diagnostics,
    }
    report_path = EXPERIMENT_DIR / "preflight-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if len(concentration_events) >= 2:
        raise RuntimeError(f"repeated candidate-only concentration: {concentration_events[:2]}")
    print(json.dumps({key: value for key, value in report.items() if key != "diagnostics"}))


if __name__ == "__main__":
    main()
