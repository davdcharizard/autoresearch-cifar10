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
SAM_RHO = 0.05
SAM_START = 0.75
SAM_PERIOD = 2
SAM_EPS = 1e-12
COMPANION_BLOCK_INDEX = 3
COMPANION_CHANNELS = 128
COMPANION_WEIGHT = 0.15
COMPANION_INIT_SEED = 42021
COMPANION_AUDIT_EVERY = 512
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

        cpu_rng_state = torch.get_rng_state()
        try:
            self.companion_fc = nn.Linear(COMPANION_CHANNELS, num_classes)
        finally:
            torch.set_rng_state(cpu_rng_state)
        companion_generator = torch.Generator(device="cpu").manual_seed(
            COMPANION_INIT_SEED
        )
        init.kaiming_normal_(
            self.companion_fc.weight,
            generator=companion_generator,
        )
        init.zeros_(self.companion_fc.bias)
        self.companion_forward_calls = 0

    @staticmethod
    def _weights_init(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            init.ones_(module.weight)
            init.zeros_(module.bias)

    def forward(self, x, drop_scale=0.0, return_companion=False):
        out = self.conv1(x)
        companion_logits = None
        companion_features = None
        for block_index, block in enumerate(self.blocks):
            out = block(out, drop_scale)
            if return_companion and block_index == COMPANION_BLOCK_INDEX:
                companion_features = F.adaptive_avg_pool2d(F.relu(out), 1).flatten(1)
                companion_logits = self.companion_fc(companion_features)
                self.companion_forward_calls += 1
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        main_logits = self.fc(out.flatten(1))
        if return_companion:
            if companion_logits is None or companion_features is None:
                raise RuntimeError("Companion attachment point was not reached")
            return main_logits, companion_logits, companion_features
        return main_logits


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


def classification_loss(logits, targets_a, targets_b=None, lam=1.0):
    if targets_b is None:
        return F.cross_entropy(logits, targets_a)
    return lam * F.cross_entropy(logits, targets_a) + (1.0 - lam) * F.cross_entropy(
        logits, targets_b
    )


def new_companion_audit(device):
    return {
        "feature_norm_sum": torch.zeros(4, device=device, dtype=torch.float64),
        "feature_norm_sq_sum": torch.zeros(4, device=device, dtype=torch.float64),
        "feature_vector_count": torch.zeros(4, device=device, dtype=torch.int64),
        "feature_batch_count": [0, 0, 0, 0],
    }


@torch.no_grad()
def audit_companion_features(audit, features, progress, one_based_step):
    if one_based_step != 1 and one_based_step % COMPANION_AUDIT_EVERY != 0:
        return
    bin_index = min(int(progress * 4.0), 3)
    norms = torch.linalg.vector_norm(features.detach().float(), dim=1).double()
    audit["feature_norm_sum"][bin_index].add_(norms.sum())
    audit["feature_norm_sq_sum"][bin_index].add_(norms.square().sum())
    audit["feature_vector_count"][bin_index].add_(norms.numel())
    audit["feature_batch_count"][bin_index] += 1


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
    companion_num_params = sum(
        parameter.numel() for parameter in model.companion_fc.parameters()
    )
    assert companion_num_params == 1_290
    assert num_params == 2_750_180
    print(f"PreAct WRN-16-4 | params: {num_params:,}")
    print(
        "config: architecture=PreActWideResNet "
        f"params={num_params} peak_lr={PEAK_LR} "
        f"warmup_fraction={WARMUP_FRACTION} "
        f"max_drop_path={MAX_DROP_PATH} eval_every={EVAL_EVERY} "
        f"cutmix_prob={CUTMIX_PROB} cutmix_alpha={CUTMIX_ALPHA} "
        f"cutmix_end={CUTMIX_END} cutmix_seed={CUTMIX_SEED} "
        f"sam_rho={SAM_RHO} sam_start={SAM_START} sam_period={SAM_PERIOD} "
        f"companion_block={COMPANION_BLOCK_INDEX} "
        f"companion_channels={COMPANION_CHANNELS} "
        f"companion_weight={COMPANION_WEIGHT} "
        f"companion_init_seed={COMPANION_INIT_SEED} "
        f"companion_audit_every={COMPANION_AUDIT_EVERY} "
        "companion_schedule=full_run companion_eval=main_only"
    )
    print(
        f"companion_inventory: head_params={companion_num_params} "
        f"total_params={num_params} head=Linear(128,10) "
        "target_policy=shared_area_corrected"
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
    companion_initial_parameters = [
        parameter.detach().clone() for parameter in model.companion_fc.parameters()
    ]
    companion_audit = new_companion_audit(device)

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
    sam_eligible_batches = 0
    sam_applied_batches = 0
    sam_first_step = None
    sam_first_progress = None
    companion_primary_loss_calls = 0
    companion_replay_loss_calls = 0
    companion_eval_events = 0
    eval_accuracies = []

    while total_training_time < TIME_BUDGET_S:
        epoch += 1
        model.train()

        for inputs, targets in train_loader:
            t0 = time.time()
            progress = min(total_training_time / TIME_BUDGET_S, 1.0)
            next_step = step + 1
            apply_sam = sam_is_scheduled(progress, next_step)
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

            cuda_rng_state = torch.cuda.get_rng_state(device) if apply_sam else None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs, companion_outputs, companion_features = model(
                    inputs,
                    drop_scale=current_drop_scale,
                    return_companion=True,
                )
                main_loss = classification_loss(
                    outputs,
                    targets_a,
                    targets_b,
                    adjusted_lam,
                )
                companion_loss = classification_loss(
                    companion_outputs,
                    targets_a,
                    targets_b,
                    adjusted_lam,
                )
                loss = main_loss + COMPANION_WEIGHT * companion_loss
            companion_primary_loss_calls += 1
            audit_companion_features(
                companion_audit,
                companion_features,
                progress,
                next_step,
            )
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
                        outputs, companion_outputs, _ = model(
                            inputs,
                            drop_scale=current_drop_scale,
                            return_companion=True,
                        )
                        main_loss = classification_loss(outputs, targets)
                        companion_loss = classification_loss(
                            companion_outputs,
                            targets,
                        )
                        loss = main_loss + COMPANION_WEIGHT * companion_loss
                    companion_replay_loss_calls += 1
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
            companion_calls_before_eval = model.companion_forward_calls
            test_loss, test_acc = evaluator.evaluate(model, device)
            if model.companion_forward_calls != companion_calls_before_eval:
                raise RuntimeError("Evaluator executed the companion path")
            companion_eval_events += 1
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

    feature_norm_sum = companion_audit["feature_norm_sum"].tolist()
    feature_norm_sq_sum = companion_audit["feature_norm_sq_sum"].tolist()
    feature_vector_count = companion_audit["feature_vector_count"].tolist()
    feature_batch_count = companion_audit["feature_batch_count"]
    feature_sample_count = sum(feature_batch_count)
    expected_feature_samples = 1 + (step - 1) // COMPANION_AUDIT_EVERY
    feature_audit_status = feature_sample_count == expected_feature_samples
    feature_means = [
        feature_norm_sum[index] / max(feature_vector_count[index], 1)
        for index in range(4)
    ]
    feature_rms = [
        math.sqrt(feature_norm_sq_sum[index] / max(feature_vector_count[index], 1))
        for index in range(4)
    ]
    companion_displacement_sq = sum(
        (parameter.detach() - initial).double().square().sum()
        for parameter, initial in zip(
            model.companion_fc.parameters(),
            companion_initial_parameters,
            strict=True,
        )
    )
    companion_displacement = math.sqrt(companion_displacement_sq.item())
    nonfinite_count = sum(
        torch.count_nonzero(~torch.isfinite(parameter)).item()
        for parameter in model.parameters()
    )
    nonfinite_count += sum(
        torch.count_nonzero(~torch.isfinite(value)).item()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    expected_head_calls = step + sam_applied_batches
    companion_integrity = (
        companion_primary_loss_calls == step
        and companion_replay_loss_calls == sam_applied_batches
        and model.companion_forward_calls == expected_head_calls
        and companion_eval_events == epoch
        and num_params == 2_750_180
        and nonfinite_count == 0
    )
    tail_accuracies = eval_accuracies[-16:]
    tail_mean = sum(tail_accuracies) / len(tail_accuracies)
    tail_min = min(tail_accuracies)
    tail_max = max(tail_accuracies)
    tail_values = ",".join(f"{accuracy:.2f}" for accuracy in tail_accuracies)
    t_end = time.time()
    startup_time = t_start_training - t_start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )
    cutmix_ratio = cutmix_applied_batches / max(cutmix_eligible_batches, 1)
    sam_ratio = sam_applied_batches / max(sam_eligible_batches, 1)

    print(
        f"cutmix: applied={cutmix_applied_batches} "
        f"eligible={cutmix_eligible_batches} ratio={cutmix_ratio:.4f}"
    )
    print(
        f"sam: applied={sam_applied_batches} eligible={sam_eligible_batches} "
        f"ratio={sam_ratio:.4f} first_step={sam_first_step or -1} "
        f"first_progress={sam_first_progress if sam_first_progress is not None else -1:.4f}"
    )
    print(
        f"companion_calls: primary_loss={companion_primary_loss_calls} "
        f"replay_loss={companion_replay_loss_calls} "
        f"head_forwards={model.companion_forward_calls} "
        f"expected_head_forwards={expected_head_calls} "
        f"eval_events={companion_eval_events}"
    )
    for index in range(4):
        print(
            f"companion_feature_bin{index}: batches={feature_batch_count[index]} "
            f"vectors={feature_vector_count[index]} "
            f"mean_l2={feature_means[index]:.12e} "
            f"rms_l2={feature_rms[index]:.12e}"
        )
    print(
        f"companion_feature_audit: samples={feature_sample_count} "
        f"expected={expected_feature_samples} "
        f"status={'PASS' if feature_audit_status else 'ANOMALY'}"
    )
    print(
        f"companion_state: displacement_l2={companion_displacement:.12e} "
        f"nonfinite={nonfinite_count}"
    )
    print(
        f"companion_integrity: status={'PASS' if companion_integrity else 'FAIL'} "
        f"head_params={companion_num_params} total_params={num_params}"
    )
    print(
        f"eval_tail16: count={len(tail_accuracies)} mean={tail_mean:.6f} "
        f"min={tail_min:.2f} max={tail_max:.2f} final={tail_accuracies[-1]:.2f} "
        f"best_premium={best_acc - tail_mean:.6f} values={tail_values}"
    )
    if not companion_integrity:
        raise RuntimeError("Companion integrity reconciliation failed")
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
