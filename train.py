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
LOOKAHEAD_K = 5
LOOKAHEAD_ALPHA = 0.5
LOOKAHEAD_AUDIT_EVERY_SYNCS = 128
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


@torch.no_grad()
def lookahead_distance_stats(fast_parameters, slow_parameters):
    """Return fixed device scalars without retaining parameter-sized temporaries."""
    device = fast_parameters[0].device
    distance_sq = torch.zeros((), device=device, dtype=torch.float64)
    fast_sq = torch.zeros((), device=device, dtype=torch.float64)
    for fast, slow in zip(fast_parameters, slow_parameters, strict=True):
        delta = fast.detach() - slow
        distance_sq.add_(delta.square().sum(dtype=torch.float64))
        fast_sq.add_(fast.detach().square().sum(dtype=torch.float64))
        del delta
    return distance_sq, fast_sq


@torch.no_grad()
def lookahead_sync_(fast_parameters, slow_parameters, alpha):
    """Apply the canonical parameter-only Lookahead interpolation in place."""
    torch._foreach_lerp_(slow_parameters, fast_parameters, alpha)
    torch._foreach_copy_(fast_parameters, slow_parameters)


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
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"PreAct WRN-16-4 | params: {num_params:,}")
    print(
        "config: architecture=PreActWideResNet "
        f"params={num_params} peak_lr={PEAK_LR} "
        f"warmup_fraction={WARMUP_FRACTION} "
        f"max_drop_path={MAX_DROP_PATH} eval_every={EVAL_EVERY} "
        f"cutmix_prob={CUTMIX_PROB} cutmix_alpha={CUTMIX_ALPHA} "
        f"cutmix_end={CUTMIX_END} cutmix_seed={CUTMIX_SEED} "
        f"lookahead_k={LOOKAHEAD_K} lookahead_alpha={LOOKAHEAD_ALPHA} "
        f"lookahead_audit_every_syncs={LOOKAHEAD_AUDIT_EVERY_SYNCS}"
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=PEAK_LR * START_LR_RATIO,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    fast_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    cpu_rng_before_slow = torch.random.get_rng_state()
    cuda_rng_before_slow = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    slow_parameters = [parameter.detach().clone() for parameter in fast_parameters]
    fast_restore = [torch.empty_like(parameter) for parameter in fast_parameters]
    assert torch.equal(cpu_rng_before_slow, torch.random.get_rng_state())
    if cuda_rng_before_slow is not None:
        assert torch.equal(cuda_rng_before_slow, torch.cuda.get_rng_state(device))
    assert len(fast_parameters) == len(slow_parameters) == 44
    assert sum(parameter.numel() for parameter in slow_parameters) == num_params
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameter_ids == {id(parameter) for parameter in fast_parameters}
    assert all(
        slow.shape == fast.shape
        and slow.dtype == fast.dtype == torch.float32
        and slow.device == fast.device
        and not slow.requires_grad
        and id(slow) not in optimizer_parameter_ids
        for fast, slow in zip(fast_parameters, slow_parameters, strict=True)
    )
    print(f"Time budget: {TIME_BUDGET_S}s")
    print(f"Batches per epoch: {len(train_loader)}")
    print(
        f"Lookahead slow state: tensors={len(slow_parameters)} "
        f"elements={sum(parameter.numel() for parameter in slow_parameters)}"
    )
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
    lookahead_syncs = 0
    lookahead_first_sync = 0
    lookahead_last_sync = 0
    lookahead_early_cutmix_syncs = 0
    lookahead_early_clean_syncs = 0
    lookahead_late_clean_syncs = 0
    lookahead_synchronized_evals = 0
    lookahead_swapped_evals = 0
    lookahead_restore_failures = 0
    lookahead_audit_samples = 0
    lookahead_nonfinite_audits = torch.zeros((), device=device, dtype=torch.int64)
    lookahead_pre_norm_sum = torch.zeros((), device=device, dtype=torch.float64)
    lookahead_pre_norm_max = torch.zeros((), device=device, dtype=torch.float64)
    lookahead_displacement_sum = torch.zeros((), device=device, dtype=torch.float64)
    evaluation_accuracies = []
    momentum_buffer_ids = None

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
            optimizer.step()

            completed_step = step + 1
            if completed_step % LOOKAHEAD_K == 0:
                sync_index = lookahead_syncs + 1
                assert completed_step == LOOKAHEAD_K * sync_index
                if sync_index == 1:
                    assert completed_step == LOOKAHEAD_K

                should_audit = (
                    sync_index == 1 or sync_index % LOOKAHEAD_AUDIT_EVERY_SYNCS == 0
                )
                if should_audit:
                    distance_sq, fast_sq = lookahead_distance_stats(
                        fast_parameters,
                        slow_parameters,
                    )
                    normalized_distance = torch.sqrt(
                        distance_sq / fast_sq.clamp_min(torch.finfo(torch.float64).tiny)
                    )
                    lookahead_pre_norm_sum.add_(normalized_distance)
                    lookahead_pre_norm_max.copy_(
                        torch.maximum(lookahead_pre_norm_max, normalized_distance)
                    )
                    lookahead_displacement_sum.add_(
                        LOOKAHEAD_ALPHA * torch.sqrt(distance_sq)
                    )
                    lookahead_nonfinite_audits.add_(
                        (~torch.isfinite(distance_sq)).to(torch.int64)
                        + (~torch.isfinite(fast_sq)).to(torch.int64)
                        + (~torch.isfinite(normalized_distance)).to(torch.int64)
                    )
                    lookahead_audit_samples += 1
                    del distance_sq, fast_sq, normalized_distance

                lookahead_sync_(
                    fast_parameters,
                    slow_parameters,
                    LOOKAHEAD_ALPHA,
                )
                lookahead_syncs = sync_index
                if lookahead_first_sync == 0:
                    lookahead_first_sync = completed_step
                lookahead_last_sync = completed_step
                if progress < CUTMIX_END:
                    if targets_b is None:
                        lookahead_early_clean_syncs += 1
                    else:
                        lookahead_early_cutmix_syncs += 1
                else:
                    lookahead_late_clean_syncs += 1

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
            if step % LOOKAHEAD_K == 0:
                assert lookahead_last_sync == step
                assert all(
                    torch.equal(fast, slow)
                    for fast, slow in zip(
                        fast_parameters,
                        slow_parameters,
                        strict=True,
                    )
                )
                lookahead_synchronized_evals += 1
                test_loss, test_acc = evaluator.evaluate(model, device)
            else:
                module_training_flags = [module.training for module in model.modules()]
                cpu_rng_before_swap = torch.random.get_rng_state()
                cuda_rng_before_swap = (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                )
                with torch.no_grad():
                    torch._foreach_copy_(fast_restore, fast_parameters)
                    torch._foreach_copy_(fast_parameters, slow_parameters)
                assert torch.equal(cpu_rng_before_swap, torch.random.get_rng_state())
                if cuda_rng_before_swap is not None:
                    assert torch.equal(
                        cuda_rng_before_swap,
                        torch.cuda.get_rng_state(device),
                    )
                try:
                    test_loss, test_acc = evaluator.evaluate(model, device)
                finally:
                    cpu_rng_before_restore = torch.random.get_rng_state()
                    cuda_rng_before_restore = (
                        torch.cuda.get_rng_state(device)
                        if device.type == "cuda"
                        else None
                    )
                    with torch.no_grad():
                        torch._foreach_copy_(fast_parameters, fast_restore)
                    for module, training in zip(
                        model.modules(),
                        module_training_flags,
                        strict=True,
                    ):
                        module.training = training
                    assert torch.equal(
                        cpu_rng_before_restore,
                        torch.random.get_rng_state(),
                    )
                    if cuda_rng_before_restore is not None:
                        assert torch.equal(
                            cuda_rng_before_restore,
                            torch.cuda.get_rng_state(device),
                        )
                    restore_failures = sum(
                        not torch.equal(fast, restore)
                        for fast, restore in zip(
                            fast_parameters,
                            fast_restore,
                            strict=True,
                        )
                    )
                    lookahead_restore_failures += restore_failures
                    assert restore_failures == 0
                lookahead_swapped_evals += 1
            eval_seconds = time.time() - eval_started
            evaluation_accuracies.append(test_acc)

            current_momentum_buffer_ids = tuple(
                id(optimizer.state[parameter]["momentum_buffer"])
                for parameter in fast_parameters
            )
            if momentum_buffer_ids is None:
                momentum_buffer_ids = current_momentum_buffer_ids
            else:
                assert momentum_buffer_ids == current_momentum_buffer_ids

            if test_acc > best_acc:
                best_acc = test_acc

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

    t_end = time.time()
    startup_time = t_start_training - t_start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )
    cutmix_ratio = cutmix_applied_batches / max(cutmix_eligible_batches, 1)
    expected_lookahead_syncs = step // LOOKAHEAD_K
    assert lookahead_syncs == expected_lookahead_syncs
    assert lookahead_syncs > 0
    assert lookahead_first_sync == LOOKAHEAD_K
    assert lookahead_last_sync == LOOKAHEAD_K * lookahead_syncs
    assert (
        lookahead_early_cutmix_syncs
        + lookahead_early_clean_syncs
        + lookahead_late_clean_syncs
        == lookahead_syncs
    )
    assert lookahead_synchronized_evals + lookahead_swapped_evals == len(
        evaluation_accuracies
    )
    assert len(evaluation_accuracies) == epoch
    assert lookahead_restore_failures == 0

    final_distance_sq, final_fast_sq = lookahead_distance_stats(
        fast_parameters,
        slow_parameters,
    )
    final_normalized_distance = torch.sqrt(
        final_distance_sq / final_fast_sq.clamp_min(torch.finfo(torch.float64).tiny)
    ).item()
    audit_nonfinite = int(lookahead_nonfinite_audits.item())
    audit_mean_distance = (
        (lookahead_pre_norm_sum / lookahead_audit_samples).item()
        if lookahead_audit_samples
        else float("nan")
    )
    audit_max_distance = lookahead_pre_norm_max.item()
    cumulative_displacement = lookahead_displacement_sum.item()
    final_nonfinite_tensors = sum(
        not bool(torch.isfinite(tensor).all().item())
        for tensor in [*fast_parameters, *slow_parameters]
    )
    final_nonfinite_tensors += sum(
        not bool(torch.isfinite(value).all().item())
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    assert audit_nonfinite == 0
    assert final_nonfinite_tensors == 0
    assert cumulative_displacement > 0.0

    tail_accuracies = evaluation_accuracies[-16:]
    tail_mean = sum(tail_accuracies) / len(tail_accuracies)
    tail_min = min(tail_accuracies)
    tail_max = max(tail_accuracies)
    tail_premium = best_acc - tail_mean

    print(
        f"cutmix: applied={cutmix_applied_batches} "
        f"eligible={cutmix_eligible_batches} ratio={cutmix_ratio:.4f}"
    )
    print(
        f"lookahead: syncs={lookahead_syncs} expected={expected_lookahead_syncs} "
        f"first={lookahead_first_sync} last={lookahead_last_sync} "
        f"steps_since_sync={step - lookahead_last_sync} "
        f"early_cutmix={lookahead_early_cutmix_syncs} "
        f"early_clean={lookahead_early_clean_syncs} "
        f"late_clean={lookahead_late_clean_syncs}"
    )
    print(
        f"lookahead_evals: synchronized={lookahead_synchronized_evals} "
        f"swapped={lookahead_swapped_evals} "
        f"restore_failures={lookahead_restore_failures}"
    )
    print(
        f"lookahead_distance: audit_samples={lookahead_audit_samples} "
        f"mean_pre_norm={audit_mean_distance:.8f} "
        f"max_pre_norm={audit_max_distance:.8f} "
        f"cumulative_displacement_l2={cumulative_displacement:.8f} "
        f"final_norm={final_normalized_distance:.8f} "
        f"audit_nonfinite={audit_nonfinite} "
        f"final_nonfinite_tensors={final_nonfinite_tensors}"
    )
    print(
        "lookahead_tail16: values="
        + ",".join(f"{accuracy:.2f}" for accuracy in tail_accuracies)
        + f" mean={tail_mean:.4f} min={tail_min:.2f} max={tail_max:.2f} "
        f"final={tail_accuracies[-1]:.2f} best_premium={tail_premium:.4f}"
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
