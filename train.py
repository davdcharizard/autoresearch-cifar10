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
EMA_START = 0.75
EMA_UPDATE_EVERY = 31
EMA_TAIL_HALF_LIVES = 4.0
EMA_HALF_LIFE_S = (
    (1.0 - EMA_START) * TIME_BUDGET_S / EMA_TAIL_HALF_LIVES
)
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
            (128, 320, 2),
            (320, 320, 1),
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
        self.bn = nn.BatchNorm2d(320)
        self.fc = nn.Linear(320, num_classes)
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


class ChargedTimeEMA:
    def __init__(self, model):
        self.parameter_names, self.parameters = zip(*model.named_parameters())
        self.buffer_names, self.buffers = zip(*model.named_buffers())
        state_keys = set(model.state_dict().keys())
        covered_keys = set(self.parameter_names) | set(self.buffer_names)
        if covered_keys != state_keys:
            missing = sorted(state_keys - covered_keys)
            extra = sorted(covered_keys - state_keys)
            raise RuntimeError(f"EMA state coverage mismatch: missing={missing} extra={extra}")

        self.float_buffer_indices = [
            index for index, buffer in enumerate(self.buffers)
            if buffer.is_floating_point()
        ]
        self.int_buffer_indices = [
            index for index, buffer in enumerate(self.buffers)
            if not buffer.is_floating_point()
        ]
        self.float_buffers = [self.buffers[index] for index in self.float_buffer_indices]
        self.int_buffers = [self.buffers[index] for index in self.int_buffer_indices]
        self.float_buffer_names = [
            self.buffer_names[index] for index in self.float_buffer_indices
        ]

        self.ema_parameters = self._empty_like(self.parameters)
        self.ema_float_buffers = self._empty_like(self.float_buffers)
        self.ema_int_buffers = self._empty_like(self.int_buffers)
        self.restore_parameters = self._empty_like(self.parameters)
        self.restore_float_buffers = self._empty_like(self.float_buffers)
        self.restore_int_buffers = self._empty_like(self.int_buffers)
        self.previous_parameters = self._empty_like(self.parameters)

        live_tensors = list(self.parameters) + list(self.buffers)
        shadow_tensors = self.shadow_tensors()
        for shadow, live in zip(
            shadow_tensors,
            self._shadow_sources(),
            strict=True,
        ):
            if (
                shadow.shape != live.shape
                or shadow.dtype != live.dtype
                or shadow.device != live.device
                or shadow.requires_grad
            ):
                raise RuntimeError("invalid EMA shadow tensor")
        if {id(tensor) for tensor in live_tensors} & {
            id(tensor) for tensor in shadow_tensors
        }:
            raise RuntimeError("EMA shadow aliases live model state")

        self.updates = 0
        self.first_step = None
        self.last_step = None
        self.first_progress = None
        self.last_progress = None
        self.first_time = None
        self.last_time = None
        self.last_sample_time = None
        self.sam_samples = 0
        self.ordinary_samples = 0
        self.live_evals = 0
        self.ema_evals = 0
        self.swaps = 0
        self.restore_checks = 0
        self.restore_failures = 0
        self.coverage_failures = 0
        self.nonfinite_failures = 0
        self.rng_failures = 0
        self.intervals = []
        self.decays = []
        self.consecutive_distance_sq = []
        self.eval_parameter_distances = []
        self.eval_parameter_relative_distances = []
        self.final_bn_mean_l2 = 0.0
        self.final_bn_var_l2 = 0.0
        self.final_bn_var_ratios = []

    @staticmethod
    def _empty_like(tensors):
        return [
            torch.empty_like(tensor, memory_format=torch.preserve_format)
            for tensor in tensors
        ]

    @staticmethod
    def _copy(destination, source):
        if destination:
            torch._foreach_copy_(destination, source)

    def _shadow_sources(self):
        return (
            list(self.parameters)
            + self.float_buffers
            + self.int_buffers
            + list(self.parameters)
            + self.float_buffers
            + self.int_buffers
            + list(self.parameters)
        )

    def shadow_tensors(self):
        return (
            self.ema_parameters
            + self.ema_float_buffers
            + self.ema_int_buffers
            + self.restore_parameters
            + self.restore_float_buffers
            + self.restore_int_buffers
            + self.previous_parameters
        )

    @staticmethod
    def _rng_states(device):
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        return cpu_state, cuda_state

    @staticmethod
    def _rng_states_equal(left, right):
        return torch.equal(left[0], right[0]) and (
            left[1] is None or torch.equal(left[1], right[1])
        )

    @torch.no_grad()
    def update(self, sample_time, progress, step, sampled_sam):
        if progress < EMA_START or step % EMA_UPDATE_EVERY != 0:
            return False

        device = self.parameters[0].device
        rng_before = self._rng_states(device)

        if self.updates == 0:
            self._copy(self.ema_parameters, self.parameters)
            self._copy(self.ema_float_buffers, self.float_buffers)
            self._copy(self.ema_int_buffers, self.int_buffers)
            self._copy(self.previous_parameters, self.parameters)
            self.first_step = step
            self.first_progress = progress
            self.first_time = sample_time
        else:
            interval = sample_time - self.last_sample_time
            if interval <= 0.0:
                raise RuntimeError(f"invalid EMA sample interval: {interval}")
            decay = 2.0 ** (-interval / EMA_HALF_LIFE_S)
            if not 0.0 < decay < 1.0:
                raise RuntimeError(f"invalid EMA decay: {decay}")

            distance_sq = torch.stack(
                [
                    (parameter.detach().float() - previous.float()).square().sum()
                    for parameter, previous in zip(
                        self.parameters,
                        self.previous_parameters,
                        strict=True,
                    )
                ]
            ).sum()
            self.consecutive_distance_sq.append(distance_sq)
            self._copy(self.previous_parameters, self.parameters)
            torch._foreach_lerp_(self.ema_parameters, self.parameters, 1.0 - decay)
            torch._foreach_lerp_(
                self.ema_float_buffers,
                self.float_buffers,
                1.0 - decay,
            )
            self._copy(self.ema_int_buffers, self.int_buffers)
            self.intervals.append(interval)
            self.decays.append(decay)

        self.updates += 1
        self.last_step = step
        self.last_progress = progress
        self.last_time = sample_time
        self.last_sample_time = sample_time
        if sampled_sam:
            self.sam_samples += 1
        else:
            self.ordinary_samples += 1
        if not self._rng_states_equal(rng_before, self._rng_states(device)):
            self.rng_failures += 1
            raise RuntimeError("EMA update consumed RNG")
        return True

    @staticmethod
    def _optimizer_identities(optimizer):
        parameter_ids = tuple(
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        momentum_ids = tuple(
            id(optimizer.state[parameter].get("momentum_buffer"))
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        return parameter_ids, momentum_ids

    @staticmethod
    def _vector_distance(left, right):
        return torch.linalg.vector_norm(
            torch.stack(
                [
                    (left_tensor.detach().float() - right_tensor.float()).norm(2)
                    for left_tensor, right_tensor in zip(left, right, strict=True)
                ]
            )
        )

    @torch.no_grad()
    def _record_eval_distances(self):
        parameter_distance = self._vector_distance(
            self.parameters,
            self.ema_parameters,
        )
        parameter_norm = torch.linalg.vector_norm(
            torch.stack([parameter.detach().float().norm(2) for parameter in self.parameters])
        )
        distance_f = parameter_distance.item()
        relative_f = (parameter_distance / parameter_norm.clamp_min(1e-12)).item()
        if not math.isfinite(distance_f) or not math.isfinite(relative_f):
            self.nonfinite_failures += 1
            raise RuntimeError("nonfinite EMA-to-live parameter distance")
        self.eval_parameter_distances.append(distance_f)
        self.eval_parameter_relative_distances.append(relative_f)

        mean_distances = []
        var_distances = []
        var_ratios = []
        for name, online, averaged in zip(
            self.float_buffer_names,
            self.float_buffers,
            self.ema_float_buffers,
            strict=True,
        ):
            distance = (online.detach().float() - averaged.float()).norm(2).item()
            if name.endswith("running_mean"):
                mean_distances.append(distance)
            elif name.endswith("running_var"):
                var_distances.append(distance)
                ratio = averaged.float() / online.detach().float().clamp_min(1e-12)
                var_ratios.extend(ratio.cpu().tolist())
        self.final_bn_mean_l2 = math.sqrt(sum(value * value for value in mean_distances))
        self.final_bn_var_l2 = math.sqrt(sum(value * value for value in var_distances))
        self.final_bn_var_ratios = var_ratios
        diagnostics = [
            self.final_bn_mean_l2,
            self.final_bn_var_l2,
            *self.final_bn_var_ratios,
        ]
        if not all(math.isfinite(value) for value in diagnostics):
            self.nonfinite_failures += 1
            raise RuntimeError("nonfinite EMA BatchNorm diagnostic")

    @torch.no_grad()
    def evaluate(self, evaluator, model, device, optimizer):
        if self.updates == 0:
            self.live_evals += 1
            test_loss, test_acc = evaluator.evaluate(model, device)
            return test_loss, test_acc, "live"

        self._record_eval_distances()
        optimizer_ids = self._optimizer_identities(optimizer)
        module_modes = {name: module.training for name, module in model.named_modules()}
        self._copy(self.restore_parameters, self.parameters)
        self._copy(self.restore_float_buffers, self.float_buffers)
        self._copy(self.restore_int_buffers, self.int_buffers)
        rng_before_swap = self._rng_states(device)
        try:
            self._copy(self.parameters, self.ema_parameters)
            self._copy(self.float_buffers, self.ema_float_buffers)
            self._copy(self.int_buffers, self.ema_int_buffers)
            self.swaps += 1
            if not self._rng_states_equal(rng_before_swap, self._rng_states(device)):
                self.rng_failures += 1
                raise RuntimeError("EMA evaluation swap consumed RNG")
            test_loss, test_acc = evaluator.evaluate(model, device)
        finally:
            rng_before_restore = self._rng_states(device)
            self._copy(self.parameters, self.restore_parameters)
            self._copy(self.float_buffers, self.restore_float_buffers)
            self._copy(self.int_buffers, self.restore_int_buffers)
            for name, module in model.named_modules():
                module.training = module_modes[name]

            fresh_state = model.state_dict(keep_vars=True)
            expected_keys = set(self.parameter_names) | set(self.buffer_names)
            if set(fresh_state.keys()) != expected_keys:
                self.coverage_failures += 1
                raise RuntimeError("post-restore EMA state coverage mismatch")
            restore_by_name = {
                **dict(zip(self.parameter_names, self.restore_parameters, strict=True)),
                **dict(zip(self.float_buffer_names, self.restore_float_buffers, strict=True)),
                **dict(
                    zip(
                        [self.buffer_names[index] for index in self.int_buffer_indices],
                        self.restore_int_buffers,
                        strict=True,
                    )
                ),
            }
            mismatch = any(
                not torch.equal(tensor, restore_by_name[name])
                for name, tensor in fresh_state.items()
            )
            if mismatch or self._optimizer_identities(optimizer) != optimizer_ids:
                self.restore_failures += 1
                raise RuntimeError("EMA evaluation failed exact online restoration")
            if not self._rng_states_equal(rng_before_restore, self._rng_states(device)):
                self.rng_failures += 1
                raise RuntimeError("EMA evaluation restore consumed RNG")
            self.restore_checks += 1

        self.ema_evals += 1
        return test_loss, test_acc, "ema"

    def audit_lines(self):
        if self.consecutive_distance_sq:
            distances = torch.sqrt(torch.stack(self.consecutive_distance_sq)).cpu()
            if not torch.isfinite(distances).all().item():
                self.nonfinite_failures += 1
                raise RuntimeError("nonfinite consecutive EMA sample distance")
            consecutive_min = distances.min().item()
            consecutive_mean = distances.mean().item()
            consecutive_max = distances.max().item()
        else:
            consecutive_min = consecutive_mean = consecutive_max = 0.0

        span = (self.last_time - self.first_time) if self.updates else 0.0
        oldest_weight = 2.0 ** (-span / EMA_HALF_LIFE_S) if self.updates else 1.0
        interval_min = min(self.intervals, default=0.0)
        interval_mean = sum(self.intervals) / max(len(self.intervals), 1)
        interval_max = max(self.intervals, default=0.0)
        decay_min = min(self.decays, default=1.0)
        decay_mean = sum(self.decays) / max(len(self.decays), 1)
        decay_max = max(self.decays, default=1.0)
        parameter_distance = self.eval_parameter_distances[-1] if self.eval_parameter_distances else 0.0
        parameter_relative = (
            self.eval_parameter_relative_distances[-1]
            if self.eval_parameter_relative_distances
            else 0.0
        )
        ratio_min = min(self.final_bn_var_ratios, default=0.0)
        ratio_mean = sum(self.final_bn_var_ratios) / max(len(self.final_bn_var_ratios), 1)
        ratio_max = max(self.final_bn_var_ratios, default=0.0)
        parameter_elements = sum(tensor.numel() for tensor in self.parameters)
        float_buffer_elements = sum(tensor.numel() for tensor in self.float_buffers)
        int_buffer_elements = sum(tensor.numel() for tensor in self.int_buffers)
        float_shadows = [
            tensor
            for tensor in self.ema_parameters + self.ema_float_buffers
            if tensor.is_floating_point()
        ]
        if self.updates and not all(
            torch.isfinite(tensor).all().item() for tensor in float_shadows
        ):
            self.nonfinite_failures += 1
            raise RuntimeError("nonfinite EMA state")
        if self.updates != self.ordinary_samples + self.sam_samples:
            raise RuntimeError("EMA sample classification mismatch")
        if self.updates and len(self.consecutive_distance_sq) != self.updates - 1:
            raise RuntimeError("EMA consecutive-distance count mismatch")

        return [
            (
                f"ema: updates={self.updates} first_step={self.first_step or -1} "
                f"last_step={self.last_step or -1} "
                f"first_progress={self.first_progress if self.first_progress is not None else -1:.4f} "
                f"last_progress={self.last_progress if self.last_progress is not None else -1:.4f} "
                f"first_time={self.first_time if self.first_time is not None else -1:.4f} "
                f"last_time={self.last_time if self.last_time is not None else -1:.4f} "
                f"span={span:.4f} oldest_weight={oldest_weight:.6f} "
                f"ordinary_samples={self.ordinary_samples} sam_samples={self.sam_samples}"
            ),
            (
                f"ema_decay: interval_min={interval_min:.6f} "
                f"interval_mean={interval_mean:.6f} interval_max={interval_max:.6f} "
                f"decay_min={decay_min:.9f} decay_mean={decay_mean:.9f} "
                f"decay_max={decay_max:.9f}"
            ),
            (
                f"ema_eval: live={self.live_evals} ema={self.ema_evals} "
                f"swaps={self.swaps} restore_checks={self.restore_checks} "
                f"restore_failures={self.restore_failures} "
                f"coverage_failures={self.coverage_failures} "
                f"nonfinite_failures={self.nonfinite_failures} "
                f"rng_failures={self.rng_failures}"
            ),
            (
                f"ema_distance: consecutive_count={len(self.consecutive_distance_sq)} "
                f"consecutive_min={consecutive_min:.8f} "
                f"consecutive_mean={consecutive_mean:.8f} "
                f"consecutive_max={consecutive_max:.8f} "
                f"parameter_l2={parameter_distance:.8f} "
                f"parameter_relative={parameter_relative:.10f} "
                f"bn_mean_l2={self.final_bn_mean_l2:.8f} "
                f"bn_var_l2={self.final_bn_var_l2:.8f} "
                f"bn_var_ratio_min={ratio_min:.8f} "
                f"bn_var_ratio_mean={ratio_mean:.8f} "
                f"bn_var_ratio_max={ratio_max:.8f}"
            ),
            (
                f"ema_state: parameters={len(self.parameters)} "
                f"parameter_elements={parameter_elements} "
                f"float_buffers={len(self.float_buffers)} "
                f"float_buffer_elements={float_buffer_elements} "
                f"int_buffers={len(self.int_buffers)} "
                f"int_buffer_elements={int_buffer_elements}"
            ),
        ]


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
    if abs(EMA_HALF_LIFE_S - 18.75) > 1e-12:
        raise RuntimeError(f"unexpected EMA half-life: {EMA_HALF_LIFE_S}")

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
    print(f"PreAct WRN-16-[4,4,5] | params: {num_params:,}")
    print(
        "config: architecture=PreActWideResNet "
        f"stage_widths=64,128,320 params={num_params} peak_lr={PEAK_LR} "
        f"warmup_fraction={WARMUP_FRACTION} "
        f"max_drop_path={MAX_DROP_PATH} eval_every={EVAL_EVERY} "
        f"cutmix_prob={CUTMIX_PROB} cutmix_alpha={CUTMIX_ALPHA} "
        f"cutmix_end={CUTMIX_END} cutmix_seed={CUTMIX_SEED} "
        f"sam_rho={SAM_RHO} sam_start={SAM_START} sam_period={SAM_PERIOD} "
        f"ema_start={EMA_START} ema_update_every={EMA_UPDATE_EVERY} "
        f"ema_tail_half_lives={EMA_TAIL_HALF_LIVES} "
        f"ema_half_life_s={EMA_HALF_LIFE_S}"
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
    model_ema = ChargedTimeEMA(model)
    optimizer_tensor_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    excluded_tensor_ids = optimizer_tensor_ids | {
        id(tensor) for tensor in sam_snapshots
    }
    if excluded_tensor_ids & {id(tensor) for tensor in model_ema.shadow_tensors()}:
        raise RuntimeError("EMA shadow is owned by optimizer or SAM")
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
    sam_eligible_batches = 0
    sam_applied_batches = 0
    sam_first_step = None
    sam_first_progress = None

    while total_training_time < TIME_BUDGET_S:
        epoch += 1
        model.train()

        for inputs, targets in train_loader:
            t0 = time.time()
            step_entry_training_time = total_training_time
            progress = min(step_entry_training_time / TIME_BUDGET_S, 1.0)
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

            model_ema.update(
                sample_time=step_entry_training_time,
                progress=progress,
                step=next_step,
                sampled_sam=apply_sam,
            )

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
            test_loss, test_acc, eval_source = model_ema.evaluate(
                evaluator,
                model,
                device,
                optimizer,
            )
            eval_seconds = time.time() - eval_started

            if test_acc > best_acc:
                best_acc = test_acc

            print(
                f"\n  eval ep {epoch:3d} | test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.2f}% | best: {best_acc:.2f}% | "
                f"source: {eval_source} | eval_s: {eval_seconds:.2f} | "
                f"charged_s: {total_training_time:.3f} | "
                f"progress: {min(total_training_time / TIME_BUDGET_S, 1.0):.6f}"
            )

        if epoch == 1:
            gc.collect()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    final_audit_error = None
    try:
        if model_ema.live_evals + model_ema.ema_evals != epoch:
            raise RuntimeError("EMA evaluation count does not equal epoch count")
        if abs(model_ema.ordinary_samples - model_ema.sam_samples) > 1:
            raise RuntimeError("EMA ordinary/SAM sample parity mismatch")
        ema_audit_lines = model_ema.audit_lines()
    except Exception as error:
        final_audit_error = error
        ema_audit_lines = [
            f"ema_audit_failed: {type(error).__name__}: {error}"
        ]
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
    for audit_line in ema_audit_lines:
        print(audit_line)
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
    print(f"final_train_loss_ema: {debiased:.6f}")
    if final_audit_error is not None:
        raise RuntimeError("EMA final audit failed") from final_audit_error


if __name__ == "__main__":
    main()
