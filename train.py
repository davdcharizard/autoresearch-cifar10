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
ASAM_RHO = 0.5
ASAM_ETA = 0.01
ASAM_START = 0.75
ASAM_PERIOD = 2
ASAM_EPS = 1e-12
ASAM_RADIUS_MIN = 0.499
ASAM_RADIUS_MAX = 0.501
ASAM_EUCLIDEAN_MIN = 1e-3
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


def asam_is_scheduled(progress, next_step):
    return progress >= ASAM_START and next_step % ASAM_PERIOD == 0


@torch.no_grad()
def asam_perturb(named_parameters, snapshots, scales, directions, audit=False):
    parameters = [parameter for _, parameter, _ in named_parameters]
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("ASAM requires a gradient for every trainable parameter")

    torch._foreach_copy_(snapshots, parameters)
    try:
        adapted_indices = [
            index
            for index, (_, _, is_bias) in enumerate(named_parameters)
            if not is_bias
        ]
        adapted_scales = [scales[index] for index in adapted_indices]
        adapted_snapshots = [snapshots[index] for index in adapted_indices]
        torch._foreach_copy_(adapted_scales, adapted_snapshots)
        torch._foreach_abs_(adapted_scales)
        torch._foreach_add_(adapted_scales, ASAM_ETA)

        torch._foreach_copy_(directions, gradients)
        torch._foreach_mul_(directions, scales)
        denominator = torch.linalg.vector_norm(
            torch.stack([direction.float().norm(2) for direction in directions])
        )
        denominator_f = denominator.item()
        if not math.isfinite(denominator_f) or denominator_f <= 0.0:
            raise RuntimeError(f"invalid ASAM scaled-gradient norm: {denominator_f}")

        group_names = tuple(
            "bias"
            if is_bias
            else "fc"
            if name == "fc.weight"
            else "bn"
            if ".bn" in name or name == "bn.weight"
            else "conv"
            for name, _, is_bias in named_parameters
        )
        denominator_group_energy = None
        if audit:
            denominator_group_energy = [
                sum(
                    direction.float().square().sum()
                    for direction, group_name in zip(
                        directions, group_names, strict=True
                    )
                    if group_name == target_group
                )
                for target_group in ("conv", "bn", "fc", "bias")
            ]

        torch._foreach_mul_(directions, scales)
        torch._foreach_mul_(directions, ASAM_RHO / (denominator_f + ASAM_EPS))

        diagnostics = None
        if audit:
            normalized_directions = torch._foreach_div(directions, scales)
            actual_radius = torch.linalg.vector_norm(
                torch.stack(
                    [direction.float().norm(2) for direction in normalized_directions]
                )
            )
            normalized_max = torch.stack(
                [direction.float().abs().max() for direction in normalized_directions]
            ).max()
            euclidean_norm = torch.linalg.vector_norm(
                torch.stack([direction.float().norm(2) for direction in directions])
            )
            max_scale = torch.stack(
                [scale.float().abs().max() for scale in scales]
            ).max()
            epsilon_group_energy = [
                sum(
                    direction.float().square().sum()
                    for direction, group_name in zip(
                        directions, group_names, strict=True
                    )
                    if group_name == target_group
                )
                for target_group in ("conv", "bn", "fc", "bias")
            ]
            denominator_total = sum(denominator_group_energy)
            epsilon_total = sum(epsilon_group_energy)
            diagnostic_values = torch.stack(
                [
                    actual_radius,
                    normalized_max,
                    euclidean_norm,
                    max_scale,
                    *[value / denominator_total for value in denominator_group_energy],
                    *[value / epsilon_total for value in epsilon_group_energy],
                ]
            ).tolist()
            diagnostics = {
                "radius": diagnostic_values[0],
                "normalized_max": diagnostic_values[1],
                "euclidean_norm": diagnostic_values[2],
                "max_scale": diagnostic_values[3],
                "denominator_shares": diagnostic_values[4:8],
                "epsilon_shares": diagnostic_values[8:12],
            }
            if not ASAM_RADIUS_MIN <= diagnostics["radius"] <= ASAM_RADIUS_MAX:
                raise RuntimeError(
                    f"invalid actual ASAM radius: {diagnostics['radius']}"
                )
            if diagnostics["normalized_max"] > ASAM_RADIUS_MAX:
                raise RuntimeError(
                    "invalid maximum normalized ASAM coordinate: "
                    f"{diagnostics['normalized_max']}"
                )
            euclidean_max = ASAM_RADIUS_MAX * diagnostics["max_scale"]
            if not ASAM_EUCLIDEAN_MIN <= diagnostics["euclidean_norm"] <= euclidean_max:
                raise RuntimeError(
                    "invalid ASAM Euclidean norm: "
                    f"{diagnostics['euclidean_norm']} not in "
                    f"[{ASAM_EUCLIDEAN_MIN}, {euclidean_max}]"
                )
            for shares in (
                diagnostics["denominator_shares"],
                diagnostics["epsilon_shares"],
            ):
                if not all(math.isfinite(value) and value >= 0.0 for value in shares):
                    raise RuntimeError(f"invalid ASAM group shares: {shares}")
                if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-5):
                    raise RuntimeError(f"ASAM group shares do not sum to one: {shares}")

        torch._foreach_add_(parameters, directions)
    except Exception:
        torch._foreach_copy_(parameters, snapshots)
        raise
    return denominator_f, diagnostics


@torch.no_grad()
def restore_asam_parameters(parameters, snapshots):
    torch._foreach_copy_(parameters, snapshots)


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
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"PreAct WRN-16-4 | params: {num_params:,}")
    print(
        "config: architecture=PreActWideResNet "
        f"params={num_params} peak_lr={PEAK_LR} "
        f"warmup_fraction={WARMUP_FRACTION} "
        f"max_drop_path={MAX_DROP_PATH} eval_every={EVAL_EVERY} "
        f"cutmix_prob={CUTMIX_PROB} cutmix_alpha={CUTMIX_ALPHA} "
        f"cutmix_end={CUTMIX_END} cutmix_seed={CUTMIX_SEED} "
        f"asam_rho={ASAM_RHO} asam_eta={ASAM_ETA} "
        f"asam_start={ASAM_START} asam_period={ASAM_PERIOD}"
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=PEAK_LR * START_LR_RATIO,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    print(f"Time budget: {TIME_BUDGET_S}s")
    print(f"Batches per epoch: {len(train_loader)}")
    cutmix_cpu_generator = torch.Generator().manual_seed(CUTMIX_SEED)
    cutmix_cuda_generator = torch.Generator(device=device).manual_seed(CUTMIX_SEED)
    asam_named_parameters = [
        (name, parameter, name.endswith(".bias"))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    asam_parameters = [parameter for _, parameter, _ in asam_named_parameters]
    asam_snapshots = [
        torch.empty_like(parameter, memory_format=torch.preserve_format)
        for parameter in asam_parameters
    ]
    asam_scales = [
        torch.empty_like(parameter, memory_format=torch.preserve_format)
        for parameter in asam_parameters
    ]
    asam_directions = [
        torch.empty_like(parameter, memory_format=torch.preserve_format)
        for parameter in asam_parameters
    ]
    for (_, _, is_bias), scale in zip(
        asam_named_parameters, asam_scales, strict=True
    ):
        if is_bias:
            scale.fill_(1.0)
    asam_adapted_tensors = sum(
        not is_bias for _, _, is_bias in asam_named_parameters
    )
    asam_unit_tensors = sum(is_bias for _, _, is_bias in asam_named_parameters)
    asam_adapted_elements = sum(
        parameter.numel()
        for _, parameter, is_bias in asam_named_parameters
        if not is_bias
    )
    asam_unit_elements = sum(
        parameter.numel()
        for _, parameter, is_bias in asam_named_parameters
        if is_bias
    )
    if (
        len(asam_named_parameters) != 44
        or asam_adapted_tensors != 30
        or asam_unit_tensors != 14
        or asam_adapted_elements != 2_747_072
        or asam_unit_elements != 1_818
    ):
        raise RuntimeError("unexpected ASAM parameter inventory")
    batch_norm_modules = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]

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
    cutmix_last_progress = None
    asam_eligible_batches = 0
    asam_applied_batches = 0
    asam_first_step = None
    asam_first_progress = None
    asam_denominator_sum = 0.0
    asam_denominator_min = float("inf")
    asam_denominator_max = 0.0
    asam_first_geometry = None
    asam_nonfinite_failures = 0
    asam_geometry_failures = 0
    asam_restoration_failures = 0
    asam_overlap_failures = 0

    while total_training_time < TIME_BUDGET_S:
        epoch += 1
        model.train()

        for inputs, targets in train_loader:
            t0 = time.time()
            progress = min(total_training_time / TIME_BUDGET_S, 1.0)
            next_step = step + 1
            apply_asam = asam_is_scheduled(progress, next_step)
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
            if progress >= ASAM_START:
                asam_eligible_batches += 1
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
                    cutmix_last_progress = progress

            if apply_asam and targets_b is not None:
                asam_overlap_failures += 1
                raise RuntimeError("ASAM and CutMix must not overlap")

            cuda_rng_state = torch.cuda.get_rng_state(device) if apply_asam else None

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

            if apply_asam:
                audit_geometry = asam_first_geometry is None
                denominator, geometry = asam_perturb(
                    asam_named_parameters,
                    asam_snapshots,
                    asam_scales,
                    asam_directions,
                    audit=audit_geometry,
                )
                asam_denominator_sum += denominator
                asam_denominator_min = min(asam_denominator_min, denominator)
                asam_denominator_max = max(asam_denominator_max, denominator)
                parameters_perturbed = True
                bn_tracking_disabled = False
                batch_norm_tracking = []
                try:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.set_rng_state(cuda_rng_state, device)
                    batch_norm_tracking = [
                        module.track_running_stats for module in batch_norm_modules
                    ]
                    for module in batch_norm_modules:
                        module.track_running_stats = False
                    bn_tracking_disabled = True

                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        outputs = model(inputs, drop_scale=current_drop_scale)
                        loss = F.cross_entropy(outputs, targets)
                    loss.backward()
                finally:
                    if bn_tracking_disabled:
                        for module, tracking in zip(
                            batch_norm_modules,
                            batch_norm_tracking,
                            strict=True,
                        ):
                            module.track_running_stats = tracking
                    if parameters_perturbed:
                        restore_asam_parameters(asam_parameters, asam_snapshots)

                if audit_geometry:
                    if not all(
                        torch.equal(parameter, snapshot)
                        for parameter, snapshot in zip(
                            asam_parameters, asam_snapshots, strict=True
                        )
                    ):
                        asam_restoration_failures += 1
                        raise RuntimeError("ASAM parameters did not restore exactly")
                    asam_first_geometry = geometry
                    denominator_shares = geometry["denominator_shares"]
                    epsilon_shares = geometry["epsilon_shares"]
                    print(
                        "asam_activation: "
                        f"step={next_step} progress={progress:.4f} "
                        f"denominator={denominator:.6f} "
                        f"radius={geometry['radius']:.6f} "
                        f"normalized_max={geometry['normalized_max']:.6f} "
                        f"euclidean_norm={geometry['euclidean_norm']:.6f} "
                        f"max_scale={geometry['max_scale']:.6f} "
                        "denominator_shares="
                        f"{denominator_shares[0]:.6f},"
                        f"{denominator_shares[1]:.6f},"
                        f"{denominator_shares[2]:.6f},"
                        f"{denominator_shares[3]:.6f} "
                        "epsilon_shares="
                        f"{epsilon_shares[0]:.6f},"
                        f"{epsilon_shares[1]:.6f},"
                        f"{epsilon_shares[2]:.6f},"
                        f"{epsilon_shares[3]:.6f}"
                    )

            optimizer.step()

            if apply_asam:
                asam_applied_batches += 1
                if asam_first_step is None:
                    asam_first_step = next_step
                    asam_first_progress = progress

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
                    f"asam: {asam_applied_batches}/{asam_eligible_batches} | "
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
    asam_ratio = asam_applied_batches / max(asam_eligible_batches, 1)
    asam_denominator_mean = asam_denominator_sum / max(asam_applied_batches, 1)

    print(
        f"cutmix: applied={cutmix_applied_batches} "
        f"eligible={cutmix_eligible_batches} ratio={cutmix_ratio:.4f} "
        f"last_progress={cutmix_last_progress if cutmix_last_progress is not None else -1:.4f}"
    )
    print(
        f"asam: applied={asam_applied_batches} eligible={asam_eligible_batches} "
        f"ratio={asam_ratio:.4f} first_step={asam_first_step or -1} "
        f"first_progress={asam_first_progress if asam_first_progress is not None else -1:.4f}"
    )
    geometry = asam_first_geometry or {
        "radius": float("nan"),
        "normalized_max": float("nan"),
        "euclidean_norm": float("nan"),
        "max_scale": float("nan"),
        "denominator_shares": [float("nan")] * 4,
        "epsilon_shares": [float("nan")] * 4,
    }
    denominator_shares = geometry["denominator_shares"]
    epsilon_shares = geometry["epsilon_shares"]
    print(
        f"asam_geometry: rho={ASAM_RHO} eta={ASAM_ETA} "
        f"adapted_tensors={asam_adapted_tensors} unit_tensors={asam_unit_tensors} "
        f"adapted_elements={asam_adapted_elements} unit_elements={asam_unit_elements} "
        f"denominator_min={asam_denominator_min:.6f} "
        f"denominator_mean={asam_denominator_mean:.6f} "
        f"denominator_max={asam_denominator_max:.6f} "
        f"first_radius={geometry['radius']:.6f} "
        f"first_normalized_max={geometry['normalized_max']:.6f} "
        f"first_euclidean_norm={geometry['euclidean_norm']:.6f} "
        f"first_max_scale={geometry['max_scale']:.6f} "
        "denominator_shares="
        f"{denominator_shares[0]:.6f},"
        f"{denominator_shares[1]:.6f},"
        f"{denominator_shares[2]:.6f},"
        f"{denominator_shares[3]:.6f} "
        "epsilon_shares="
        f"{epsilon_shares[0]:.6f},"
        f"{epsilon_shares[1]:.6f},"
        f"{epsilon_shares[2]:.6f},"
        f"{epsilon_shares[3]:.6f} "
        f"nonfinite_failures={asam_nonfinite_failures} "
        f"geometry_failures={asam_geometry_failures} "
        f"restoration_failures={asam_restoration_failures} "
        f"overlap_failures={asam_overlap_failures}"
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
