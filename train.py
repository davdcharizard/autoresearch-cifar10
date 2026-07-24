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

NUM_BLOCKS = 2  # WRN-16 = 6*2+4
WIDEN_FACTOR = 2
NUM_CLASSES = 10
BATCH_SIZE = 256
LR = 0.2
MIN_LR = 0.002
WARMUP_FRACTION = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
MAX_STEPS = 64000
EVAL_EVERY = 5
MIXUP_ALPHA = 0.2
MIXUP_END_FRACTION = 0.65
evaluator = Eval()


# ---------------------------------------------------------------------------
# Pre-activation WRN-16-2 for CIFAR-10
# ---------------------------------------------------------------------------


class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x):
        preactivated = F.relu(self.bn1(x))
        shortcut = self.shortcut(preactivated) if self.shortcut is not None else x
        out = self.conv1(preactivated)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, num_blocks, widen_factor, num_classes=10):
        super().__init__()
        widths = [16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(16, widths[0], num_blocks, stride=1)
        self.layer2 = self._make_layer(widths[0], widths[1], num_blocks, stride=2)
        self.layer3 = self._make_layer(widths[1], widths[2], num_blocks, stride=2)
        self.bn = nn.BatchNorm2d(widths[2])
        self.fc = nn.Linear(widths[2], num_classes)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        ch = in_ch
        for s in strides:
            layers.append(PreActBlock(ch, out_ch, s))
            ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def learning_rate(training_time):
    progress = min(max(training_time / TIME_BUDGET_S, 0.0), 1.0)
    if progress < WARMUP_FRACTION:
        warmup_progress = progress / WARMUP_FRACTION
        return MIN_LR + (LR - MIN_LR) * warmup_progress

    cosine_progress = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )


def mixup_batch(inputs, targets, distribution):
    mix = distribution.sample()
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = mix * inputs + (1.0 - mix) * inputs[permutation]
    return mixed_inputs, targets, targets[permutation], mix


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def main():
    # ---------------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------------

    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mean, std = (
        (0.4914, 0.4822, 0.4465),
        (1, 1, 1),
    )  # Yes original paper only mention per-pixel mean and this is per band. See README
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
        persistent_workers=True,
    )

    model = WideResNet(NUM_BLOCKS, WIDEN_FACTOR, NUM_CLASSES).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"WRN-{6 * NUM_BLOCKS + 4}-{WIDEN_FACTOR} | params: {num_params:,}"
    )

    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = optim.SGD(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=MIN_LR,
        momentum=MOMENTUM,
        nesterov=True,
    )
    mixup_distribution = torch.distributions.Beta(
        torch.tensor(MIXUP_ALPHA, device=device),
        torch.tensor(MIXUP_ALPHA, device=device),
    )
    print(f"Time budget: {TIME_BUDGET_S}s")
    print(f"Batches per epoch: {len(train_loader)}")

    # ---------------------------------------------------------------------------
    # Training loop (time-budgeted)
    # ---------------------------------------------------------------------------

    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    epoch = 0
    step = 0
    best_acc = 0.0
    mixup_enabled = True

    while total_training_time < TIME_BUDGET_S and step < MAX_STEPS:
        epoch += 1
        model.train()

        for inputs, targets in train_loader:
            t0 = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            lr = learning_rate(total_training_time)
            for group in optimizer.param_groups:
                group["lr"] = lr

            progress = min(total_training_time / TIME_BUDGET_S, 1.0)
            use_mixup = progress < MIXUP_END_FRACTION
            if mixup_enabled and not use_mixup:
                mixup_enabled = False
                print(
                    f"\nMixup disabled at ep {epoch} step {step} | "
                    f"training_seconds={total_training_time:.1f} "
                    f"({100 * progress:.1f}%) | lr={lr:.4f}"
                )

            optimizer.zero_grad(set_to_none=True)
            if use_mixup:
                mixed_inputs, targets_a, targets_b, mix = mixup_batch(
                    inputs, targets, mixup_distribution
                )
                outputs = model(mixed_inputs)
                loss = mix * F.cross_entropy(outputs, targets_a) + (
                    1.0 - mix
                ) * F.cross_entropy(outputs, targets_b)
            else:
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            optimizer.step()

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

            if step % 50 == 0:
                print(
                    f"\rstep {step:05d} ep {epoch} ({pct_done:.1f}%) | loss: {debiased:.4f} | lr: {lr:.4f} | dt: {dt * 1000:.0f}ms | img/s: {img_per_sec:,} | rem: {remaining:.0f}s    ",
                    end="",
                    flush=True,
                )

            if total_training_time >= TIME_BUDGET_S or step >= MAX_STEPS:
                break

        budget_exhausted = total_training_time >= TIME_BUDGET_S or step >= MAX_STEPS
        if epoch % EVAL_EVERY == 0 or budget_exhausted:
            test_loss, test_acc = evaluator.evaluate(model, device)

            if test_acc > best_acc:
                best_acc = test_acc

            print(
                f"\n  eval ep {epoch:3d} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.2f}% | best: {best_acc:.2f}%"
            )

        if epoch == 1:
            gc.collect()

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------

    t_end = time.time()
    startup_time = t_start_training - t_start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
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
