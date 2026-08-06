import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from prepare import DATASET_DIR, NUM_WORKERS, TIME_BUDGET_S, Eval

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

NUM_CLASSES = 10
BATCH_SIZE = 256
PEAK_LR = 0.2
START_LR_RATIO = 0.1
MIN_LR_RATIO = 0.01
WARMUP_FRACTION = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MAX_DROP_PATH = 0.08
DROP_PATH_DECAY_START = 0.75
EVAL_EVERY = 1
CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0
CUTMIX_END = 0.75
CUTMIX_SEED = 42
GC_AUDIT_EVERY = 512
evaluator = Eval()


# ---------------------------------------------------------------------------
# Pre-activation Wide ResNet for CIFAR-10
# ---------------------------------------------------------------------------


class PreActWideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_prob):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
        )
        self.shortcut = (
            nn.Conv2d(
                in_channels,
                out_channels,
                1,
                stride=stride,
                bias=False,
            )
            if stride != 1 or in_channels != out_channels
            else None
        )
        self.drop_prob = drop_prob

    def forward(self, x, drop_scale=0.0):
        preactivated = F.relu(self.bn1(x))
        shortcut = self.shortcut(preactivated) if self.shortcut is not None else x
        out = self.conv1(preactivated)
        out = self.conv2(F.relu(self.bn2(out)))

        drop_prob = self.drop_prob * drop_scale
        if self.training and drop_prob > 0.0:
            keep_prob = 1.0 - drop_prob
            mask = torch.rand(
                (out.shape[0], 1, 1, 1),
                device=out.device,
                dtype=out.dtype,
            )
            mask = (mask < keep_prob).to(out.dtype)
            out = out * mask / keep_prob

        return shortcut + out


class PreActWideResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)

        block_specs = [
            (16, 64, 1),
            (64, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
        ]
        num_blocks = len(block_specs)
        self.blocks = nn.ModuleList(
            [
                PreActWideBlock(
                    in_channels,
                    out_channels,
                    stride,
                    MAX_DROP_PATH * (index + 1) / num_blocks,
                )
                for index, (in_channels, out_channels, stride) in enumerate(block_specs)
            ]
        )
        self.bn = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256, num_classes)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            init.ones_(module.weight)
            init.zeros_(module.bias)

    def forward(self, x, drop_scale=0.0):
        out = self.conv1(x)
        for block in self.blocks:
            out = block(out, drop_scale)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        return self.fc(out.flatten(1))


def learning_rate(progress):
    progress = min(max(progress, 0.0), 1.0)
    if progress < WARMUP_FRACTION:
        warmup_progress = progress / WARMUP_FRACTION
        ratio = START_LR_RATIO + (1.0 - START_LR_RATIO) * warmup_progress
        return PEAK_LR * ratio

    cosine_progress = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
    cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return PEAK_LR * (MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine)


def drop_path_scale(progress):
    if progress <= DROP_PATH_DECAY_START:
        return 1.0
    return max(0.0, (1.0 - progress) / (1.0 - DROP_PATH_DECAY_START))


def gradient_centralization_inventory(model):
    eligible = []
    convolution_count = 0
    linear_count = 0
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            eligible.append((f"{module_name}.weight", module.weight, "conv"))
            convolution_count += 1
        elif isinstance(module, nn.Linear):
            eligible.append((f"{module_name}.weight", module.weight, "classifier"))
            linear_count += 1

    eligible_elements = sum(parameter.numel() for _, parameter, _ in eligible)
    eligible_rows = sum(parameter.shape[0] for _, parameter, _ in eligible)
    assert convolution_count == 16
    assert linear_count == 1
    assert len(eligible) == 17
    assert eligible_elements == 2_745_264
    assert eligible_rows == 2_266
    return eligible, eligible_elements, eligible_rows


def new_gc_audit(device):
    scalar = lambda: torch.zeros((), device=device, dtype=torch.float64)
    return {
        "regularized": scalar(),
        "removed": scalar(),
        "centralized": scalar(),
        "conv_regularized": scalar(),
        "conv_removed": scalar(),
        "classifier_regularized": scalar(),
        "classifier_removed": scalar(),
        "max_residual": scalar(),
        "nonfinite": scalar(),
        "samples": 0,
    }


def squared_l2_fp64(tensor):
    return tensor.to(torch.float64).square().sum()


@torch.no_grad()
def regularize_and_centralize_gradients(
    all_parameters,
    eligible,
    audit,
    one_based_step,
):
    if any(parameter.grad is None for parameter in all_parameters):
        raise RuntimeError("Every model parameter must have a gradient before GC")

    all_gradients = [parameter.grad for parameter in all_parameters]
    torch._foreach_add_(all_gradients, all_parameters, alpha=WEIGHT_DECAY)

    eligible_gradients = [parameter.grad for _, parameter, _ in eligible]
    row_means = [
        gradient.mean(dim=tuple(range(1, gradient.ndim)), keepdim=True)
        for gradient in eligible_gradients
    ]
    should_audit = one_based_step == 1 or one_based_step % GC_AUDIT_EVERY == 0

    if should_audit:
        audit["samples"] += 1
        for gradient, row_mean, (_, _, group) in zip(
            eligible_gradients, row_means, eligible
        ):
            regularized_energy = squared_l2_fp64(gradient)
            removed_energy = squared_l2_fp64(row_mean) * (
                gradient.numel() // gradient.shape[0]
            )
            audit["regularized"].add_(regularized_energy)
            audit["removed"].add_(removed_energy)
            audit[f"{group}_regularized"].add_(regularized_energy)
            audit[f"{group}_removed"].add_(removed_energy)
            audit["nonfinite"].add_(
                torch.count_nonzero(~torch.isfinite(gradient)).to(torch.float64)
            )

    torch._foreach_sub_(eligible_gradients, row_means)

    if should_audit:
        for gradient in eligible_gradients:
            audit["centralized"].add_(squared_l2_fp64(gradient))
            post_mean = gradient.mean(
                dim=tuple(range(1, gradient.ndim)), keepdim=True
            )
            audit["max_residual"].copy_(
                torch.maximum(audit["max_residual"], post_mean.abs().max().double())
            )
            audit["nonfinite"].add_(
                torch.count_nonzero(~torch.isfinite(gradient)).to(torch.float64)
            )


@torch.no_grad()
def optimizer_nonfinite_count(model, optimizer, device):
    count = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in model.parameters():
        count.add_(torch.count_nonzero(~torch.isfinite(parameter)).to(torch.float64))
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                count.add_(torch.count_nonzero(~torch.isfinite(value)).to(torch.float64))
    return count


def cutmix_batch(
    inputs,
    targets,
    cpu_generator,
    cuda_generator,
    lam=None,
    center=None,
    permutation=None,
):
    height, width = inputs.shape[-2:]
    if lam is None:
        lam = torch.rand((), generator=cpu_generator).item()
    lam = min(max(float(lam), 0.0), 1.0)

    cut_ratio = math.sqrt(1.0 - lam)
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    if center is None:
        center_x = int(torch.randint(width, (), generator=cpu_generator).item())
        center_y = int(torch.randint(height, (), generator=cpu_generator).item())
    else:
        center_x, center_y = center

    x1 = max(center_x - cut_width // 2, 0)
    x2 = min(center_x + (cut_width + 1) // 2, width)
    y1 = max(center_y - cut_height // 2, 0)
    y2 = min(center_y + (cut_height + 1) // 2, height)
    area = (x2 - x1) * (y2 - y1)

    if permutation is None:
        permutation = torch.randperm(
            inputs.shape[0], device=inputs.device, generator=cuda_generator
        )
    paired_targets = targets[permutation]
    if area > 0:
        source_patch = inputs[permutation, :, y1:y2, x1:x2].clone()
        inputs[:, :, y1:y2, x1:x2] = source_patch

    adjusted_lam = 1.0 - area / (height * width)
    return inputs, targets, paired_targets, adjusted_lam, area


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def main():
    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mean, std = (
        (0.4914, 0.4822, 0.4465),
        (1, 1, 1),
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = datasets.CIFAR10(
        DATASET_DIR, train=True, download=True, transform=train_tf
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model = PreActWideResNet(NUM_CLASSES).to(device, memory_format=torch.channels_last)
    all_parameters = list(model.parameters())
    num_params = sum(parameter.numel() for parameter in model.parameters())
    eligible, eligible_elements, eligible_rows = gradient_centralization_inventory(model)
    excluded_elements = num_params - eligible_elements
    excluded_tensors = len(all_parameters) - len(eligible)
    assert num_params == 2_748_890
    assert len(all_parameters) == 44
    assert excluded_elements == 3_626
    assert excluded_tensors == 27
    print(f"PreAct WRN-16-4 | params: {num_params:,}")
    print(
        "config: architecture=PreActWideResNet "
        f"params={num_params} peak_lr={PEAK_LR} "
        f"warmup_fraction={WARMUP_FRACTION} "
        f"max_drop_path={MAX_DROP_PATH} eval_every={EVAL_EVERY} "
        f"cutmix_prob={CUTMIX_PROB} cutmix_alpha={CUTMIX_ALPHA} "
        f"cutmix_end={CUTMIX_END} cutmix_seed={CUTMIX_SEED} "
        f"gc=regularized_direction gc_audit_every={GC_AUDIT_EVERY} "
        f"effective_weight_decay={WEIGHT_DECAY} optimizer_weight_decay=0"
    )
    print(
        f"gc_inventory: tensors={len(eligible)} conv=16 classifier=1 "
        f"elements={eligible_elements} rows={eligible_rows} "
        f"excluded_tensors={excluded_tensors} "
        f"excluded_elements={excluded_elements} total_params={num_params}"
    )

    optimizer = optim.SGD(
        all_parameters,
        lr=PEAK_LR * START_LR_RATIO,
        momentum=MOMENTUM,
        weight_decay=0.0,
        nesterov=True,
    )
    print(f"Time budget: {TIME_BUDGET_S}s")
    print(f"Batches per epoch: {len(train_loader)}")
    cutmix_cpu_generator = torch.Generator().manual_seed(CUTMIX_SEED)
    cutmix_cuda_generator = torch.Generator(device=device).manual_seed(CUTMIX_SEED)

    # -----------------------------------------------------------------------
    # Training loop (time-budgeted)
    # -----------------------------------------------------------------------

    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    epoch = 0
    step = 0
    best_acc = 0.0
    test_loss = float("nan")
    test_acc = float("nan")
    cutmix_eligible_batches = 0
    cutmix_applied_batches = 0
    gc_calls = 0
    gc_cutmix_steps = 0
    gc_early_clean_steps = 0
    gc_late_clean_steps = 0
    gc_audit = new_gc_audit(device)
    eval_accuracies = []

    while total_training_time < TIME_BUDGET_S:
        epoch += 1
        model.train()

        for inputs, targets in train_loader:
            t0 = time.time()
            progress = min(total_training_time / TIME_BUDGET_S, 1.0)
            lr = learning_rate(progress)
            current_drop_scale = drop_path_scale(progress)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            inputs = inputs.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            targets = targets.to(device, non_blocking=True)

            targets_a = targets
            targets_b = None
            adjusted_lam = 1.0
            if progress < CUTMIX_END:
                cutmix_eligible_batches += 1
                apply_cutmix = (
                    torch.rand((), generator=cutmix_cpu_generator).item() < CUTMIX_PROB
                )
                if apply_cutmix:
                    inputs, targets_a, targets_b, adjusted_lam, _ = cutmix_batch(
                        inputs,
                        targets,
                        cutmix_cpu_generator,
                        cutmix_cuda_generator,
                    )
                    cutmix_applied_batches += 1

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(inputs, drop_scale=current_drop_scale)
                if targets_b is None:
                    loss = F.cross_entropy(outputs, targets_a)
                else:
                    loss = adjusted_lam * F.cross_entropy(outputs, targets_a)
                    loss += (1.0 - adjusted_lam) * F.cross_entropy(outputs, targets_b)
            loss.backward()
            regularize_and_centralize_gradients(
                all_parameters,
                eligible,
                gc_audit,
                step + 1,
            )
            gc_calls += 1
            if targets_b is not None:
                gc_cutmix_steps += 1
            elif progress < CUTMIX_END:
                gc_early_clean_steps += 1
            else:
                gc_late_clean_steps += 1
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            total_training_time += dt
            step += 1

            train_loss_f = loss.item()
            ema_beta = 0.95
            smooth_train_loss = (
                ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            )
            debiased = smooth_train_loss / (1 - ema_beta**step)

            pct_done = 100 * total_training_time / TIME_BUDGET_S
            img_per_sec = int(BATCH_SIZE / dt)
            remaining = max(0, TIME_BUDGET_S - total_training_time)
            effective_drop = MAX_DROP_PATH * current_drop_scale

            if step % 50 == 0:
                print(
                    f"\rstep {step:05d} ep {epoch} ({pct_done:.1f}%) | "
                    f"loss: {debiased:.4f} | lr: {lr:.4f} | "
                    f"drop: {effective_drop:.3f} | dt: {dt * 1000:.0f}ms | "
                    f"mix: {cutmix_applied_batches}/{cutmix_eligible_batches} | "
                    f"img/s: {img_per_sec:,} | rem: {remaining:.0f}s    ",
                    end="",
                    flush=True,
                )

            if total_training_time >= TIME_BUDGET_S:
                break

        budget_exhausted = total_training_time >= TIME_BUDGET_S
        should_evaluate = epoch % EVAL_EVERY == 0 or budget_exhausted
        if should_evaluate:
            eval_started = time.time()
            test_loss, test_acc = evaluator.evaluate(model, device)
            eval_seconds = time.time() - eval_started

            if test_acc > best_acc:
                best_acc = test_acc
            eval_accuracies.append(test_acc)

            print(
                f"\n  eval ep {epoch:3d} | test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.2f}% | best: {best_acc:.2f}% | "
                f"eval_s: {eval_seconds:.2f}"
            )

        if epoch == 1:
            gc.collect()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    final_nonfinite = optimizer_nonfinite_count(model, optimizer, device)
    audit_values = {
        key: value.item() if torch.is_tensor(value) else value
        for key, value in gc_audit.items()
    }
    final_nonfinite_value = int(final_nonfinite.item())
    t_end = time.time()
    startup_time = t_start_training - t_start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )
    cutmix_ratio = cutmix_applied_batches / max(cutmix_eligible_batches, 1)
    regularized_energy = audit_values["regularized"]
    removed_energy = audit_values["removed"]
    centralized_energy = audit_values["centralized"]
    decomposition_error = abs(
        regularized_energy - centralized_energy - removed_energy
    ) / max(regularized_energy, torch.finfo(torch.float64).tiny)
    removed_norm_ratio = math.sqrt(
        removed_energy / max(regularized_energy, torch.finfo(torch.float64).tiny)
    )
    conv_removed_norm_ratio = math.sqrt(
        audit_values["conv_removed"]
        / max(
            audit_values["conv_regularized"],
            torch.finfo(torch.float64).tiny,
        )
    )
    classifier_removed_norm_ratio = math.sqrt(
        audit_values["classifier_removed"]
        / max(
            audit_values["classifier_regularized"],
            torch.finfo(torch.float64).tiny,
        )
    )
    path_sum = gc_cutmix_steps + gc_early_clean_steps + gc_late_clean_steps
    gc_integrity = (
        gc_calls == step
        and gc_cutmix_steps == cutmix_applied_batches
        and gc_early_clean_steps
        == cutmix_eligible_batches - cutmix_applied_batches
        and gc_late_clean_steps == step - cutmix_eligible_batches
        and path_sum == step
        and audit_values["samples"] == 1 + (step - 1) // GC_AUDIT_EVERY
        and math.isfinite(removed_energy)
        and removed_energy > 0.0
        and decomposition_error <= 1e-5
        and audit_values["max_residual"] <= 1e-6
        and audit_values["nonfinite"] == 0.0
        and final_nonfinite_value == 0
    )
    tail_accuracies = eval_accuracies[-16:]
    tail_mean = sum(tail_accuracies) / len(tail_accuracies)
    tail_min = min(tail_accuracies)
    tail_max = max(tail_accuracies)
    tail_values = ",".join(f"{accuracy:.2f}" for accuracy in tail_accuracies)

    print(
        f"cutmix: applied={cutmix_applied_batches} "
        f"eligible={cutmix_eligible_batches} ratio={cutmix_ratio:.4f}"
    )
    print(
        f"gc_dose: calls={gc_calls} tensor_transforms={len(eligible) * gc_calls} "
        f"element_transforms={eligible_elements * gc_calls} "
        f"row_transforms={eligible_rows * gc_calls} "
        f"cutmix={gc_cutmix_steps} early_clean={gc_early_clean_steps} "
        f"late_clean={gc_late_clean_steps} path_sum={path_sum} "
        f"audit_samples={audit_values['samples']}"
    )
    print(
        f"gc_energy: regularized={regularized_energy:.12e} "
        f"removed={removed_energy:.12e} centralized={centralized_energy:.12e} "
        f"removed_norm_ratio={removed_norm_ratio:.12e} "
        f"decomposition_error={decomposition_error:.12e} "
        f"max_residual={audit_values['max_residual']:.12e}"
    )
    print(
        f"gc_energy_conv: regularized={audit_values['conv_regularized']:.12e} "
        f"removed={audit_values['conv_removed']:.12e} "
        f"removed_norm_ratio={conv_removed_norm_ratio:.12e}"
    )
    print(
        "gc_energy_classifier: "
        f"regularized={audit_values['classifier_regularized']:.12e} "
        f"removed={audit_values['classifier_removed']:.12e} "
        f"removed_norm_ratio={classifier_removed_norm_ratio:.12e}"
    )
    print(
        f"gc_integrity: status={'PASS' if gc_integrity else 'FAIL'} "
        f"audited_nonfinite={int(audit_values['nonfinite'])} "
        f"final_state_nonfinite={final_nonfinite_value} "
        f"eligible_elements={eligible_elements} excluded_elements={excluded_elements}"
    )
    print(
        f"eval_tail16: count={len(tail_accuracies)} mean={tail_mean:.6f} "
        f"min={tail_min:.2f} max={tail_max:.2f} final={tail_accuracies[-1]:.2f} "
        f"best_premium={best_acc - tail_mean:.6f} values={tail_values}"
    )
    print("---")
    print(f"best_test_acc:    {best_acc:.2f}%")
    print(f"final_test_acc:   {test_acc:.2f}%")
    print(f"final_test_loss:  {test_loss:.4f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"startup_seconds:  {startup_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_epochs:       {epoch}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {num_params:,}")


if __name__ == "__main__":
    main()
