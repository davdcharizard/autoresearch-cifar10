import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, get_worker_info
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
RANDAUGMENT_NUM_OPS = 1
RANDAUGMENT_MAGNITUDE = 5
RANDAUGMENT_NUM_BINS = 31
RANDAUGMENT_END = 0.75
RANDAUGMENT_SEED = 42
SAM_RHO = 0.05
SAM_START = 0.75
SAM_PERIOD = 2
SAM_EPS = 1e-12
evaluator = Eval()


class PairedRandAugment:
    def __init__(self, mean, std):
        self.spatial_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
        self.clean_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.to_uint8 = transforms.PILToTensor()
        self.randaugment = transforms.RandAugment(
            num_ops=RANDAUGMENT_NUM_OPS,
            magnitude=RANDAUGMENT_MAGNITUDE,
            num_magnitude_bins=RANDAUGMENT_NUM_BINS,
            interpolation=transforms.InterpolationMode.NEAREST,
            fill=0,
        )
        self.private_generator = None
        self.private_seed_key = None

    @staticmethod
    def _seed_key():
        worker_info = get_worker_info()
        if worker_info is None:
            base_seed = RANDAUGMENT_SEED
            worker_id = 0
        else:
            base_seed = int(worker_info.seed)
            worker_id = int(worker_info.id) + 1
        key = base_seed ^ (RANDAUGMENT_SEED * 0x9E3779B97F4A7C15)
        key ^= worker_id * 0xD1B54A32D192ED03
        return key & ((1 << 63) - 1)

    def _get_private_generator(self):
        seed_key = self._seed_key()
        if self.private_generator is None or self.private_seed_key != seed_key:
            self.private_generator = torch.Generator().manual_seed(seed_key)
            self.private_seed_key = seed_key
        return self.private_generator

    def __call__(self, image):
        image = self.spatial_transform(image)
        clean_tensor = self.clean_transform(image)

        private_generator = self._get_private_generator()
        global_state = torch.get_rng_state()
        torch.set_rng_state(private_generator.get_state())
        try:
            augmented_image = self.randaugment(image.copy())
        finally:
            private_state = torch.get_rng_state()
            torch.set_rng_state(global_state)
            private_generator.set_state(private_state)

        return clean_tensor, self.to_uint8(augmented_image)


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


def sam_is_scheduled(progress, next_step):
    return progress >= SAM_START and next_step % SAM_PERIOD == 0


@torch.no_grad()
def sam_perturb(parameters, snapshots):
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("SAM requires a gradient for every trainable parameter")

    grad_norm = torch.linalg.vector_norm(
        torch.stack([gradient.detach().float().norm(2) for gradient in gradients])
    )
    grad_norm_f = grad_norm.item()
    if not math.isfinite(grad_norm_f) or grad_norm_f <= 0.0:
        raise RuntimeError(f"invalid SAM gradient norm: {grad_norm_f}")

    torch._foreach_copy_(snapshots, parameters)
    try:
        torch._foreach_add_(
            parameters,
            gradients,
            alpha=SAM_RHO / (grad_norm_f + SAM_EPS),
        )
    except Exception:
        torch._foreach_copy_(parameters, snapshots)
        raise
    return grad_norm_f


@torch.no_grad()
def restore_sam_parameters(parameters, snapshots):
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
    if not (RANDAUGMENT_END == CUTMIX_END == SAM_START):
        raise RuntimeError("RandAugment, CutMix, and SAM boundaries must match")
    train_tf = PairedRandAugment(mean, std)

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
        f"randaugment_ops={RANDAUGMENT_NUM_OPS} "
        f"randaugment_magnitude={RANDAUGMENT_MAGNITUDE} "
        f"randaugment_bins={RANDAUGMENT_NUM_BINS} "
        f"randaugment_end={RANDAUGMENT_END} "
        f"randaugment_seed={RANDAUGMENT_SEED} "
        f"sam_rho={SAM_RHO} sam_start={SAM_START} sam_period={SAM_PERIOD}"
    )
    operation_space = train_tf.randaugment._augmentation_space(
        RANDAUGMENT_NUM_BINS,
        (32, 32),
    )
    operation_values = []
    for operation_name, (magnitudes, signed) in operation_space.items():
        value = (
            "none"
            if magnitudes.ndim == 0
            else f"{float(magnitudes[RANDAUGMENT_MAGNITUDE]):.6f}"
        )
        operation_values.append(f"{operation_name}:{value}:{int(signed)}")
    print(
        f"randaugment_config: torchvision={torchvision.__version__} "
        "interpolation=nearest fill=0 space=" + ",".join(operation_values)
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
    sam_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    sam_snapshots = [
        torch.empty_like(parameter, memory_format=torch.preserve_format)
        for parameter in sam_parameters
    ]
    batch_norm_modules = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    input_mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    input_std = torch.tensor(std, device=device).view(1, 3, 1, 1)

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
    randaugment_eligible_batches = 0
    randaugment_selected_batches = 0
    randaugment_selected_images = 0
    randaugment_last_progress = None
    randaugment_sam_overlap_failures = 0
    sam_eligible_batches = 0
    sam_applied_batches = 0
    sam_first_step = None
    sam_first_progress = None

    while total_training_time < TIME_BUDGET_S:
        epoch += 1
        model.train()

        for input_views, targets in train_loader:
            t0 = time.time()
            progress = min(total_training_time / TIME_BUDGET_S, 1.0)
            next_step = step + 1
            apply_sam = sam_is_scheduled(progress, next_step)
            use_randaugment = progress < RANDAUGMENT_END
            lr = learning_rate(progress)
            current_drop_scale = drop_path_scale(progress)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            clean_inputs, augmented_inputs = input_views
            if use_randaugment:
                randaugment_eligible_batches += 1
                randaugment_selected_batches += 1
                randaugment_selected_images += augmented_inputs.shape[0]
                randaugment_last_progress = progress
                inputs = augmented_inputs.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
                inputs.div_(255.0).sub_(input_mean).div_(input_std)
            else:
                inputs = clean_inputs.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            targets = targets.to(device, non_blocking=True)

            targets_a = targets
            targets_b = None
            adjusted_lam = 1.0
            if progress >= SAM_START:
                sam_eligible_batches += 1
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

            if apply_sam and targets_b is not None:
                raise RuntimeError("SAM and CutMix must not overlap")
            if apply_sam and use_randaugment:
                randaugment_sam_overlap_failures += 1
                raise RuntimeError("SAM and RandAugment must not overlap")

            cuda_rng_state = torch.cuda.get_rng_state(device) if apply_sam else None

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

            if apply_sam:
                sam_perturb(sam_parameters, sam_snapshots)
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
                        restore_sam_parameters(sam_parameters, sam_snapshots)

            optimizer.step()

            if apply_sam:
                sam_applied_batches += 1
                if sam_first_step is None:
                    sam_first_step = next_step
                    sam_first_progress = progress

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
                    f"sam: {sam_applied_batches}/{sam_eligible_batches} | "
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
    randaugment_ratio = randaugment_selected_batches / max(
        randaugment_eligible_batches,
        1,
    )
    sam_ratio = sam_applied_batches / max(sam_eligible_batches, 1)

    print(
        f"cutmix: applied={cutmix_applied_batches} "
        f"eligible={cutmix_eligible_batches} ratio={cutmix_ratio:.4f}"
    )
    print(
        f"randaugment: selected={randaugment_selected_batches} "
        f"eligible={randaugment_eligible_batches} ratio={randaugment_ratio:.4f} "
        f"images={randaugment_selected_images} "
        f"last_progress={randaugment_last_progress if randaugment_last_progress is not None else -1:.6f} "
        f"cutoff={RANDAUGMENT_END:.6f} "
        f"overlap_failures={randaugment_sam_overlap_failures}"
    )
    print(
        f"sam: applied={sam_applied_batches} eligible={sam_eligible_batches} "
        f"ratio={sam_ratio:.4f} first_step={sam_first_step or -1} "
        f"first_progress={sam_first_progress if sam_first_progress is not None else -1:.4f}"
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
