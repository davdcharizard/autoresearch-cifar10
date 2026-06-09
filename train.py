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

NUM_BLOCKS = 3  # ResNet-20 = 6*3+2
NUM_CLASSES = 10
STAGE_WIDTHS = (28, 56, 112)
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 2e-4
LR_MILESTONES = [21000, 64000]
MAX_STEPS = 64000
USE_CUDNN_BENCHMARK = True
USE_CHANNELS_LAST = True
USE_COMPILE = True
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5
CUTMIX_LABEL_SMOOTHING = 0.05
evaluator = Eval()


def rand_bbox(size, lam, device):
    _, _, height, width = size
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = torch.randint(width, (1,), device=device).item()
    cy = torch.randint(height, (1,), device=device).item()

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, width)
    bby2 = min(cy + cut_h // 2, height)
    return bbx1, bby1, bbx2, bby2


# ---------------------------------------------------------------------------
# ResNet-20 for CIFAR-10 (He et al. 2015, CIFAR variant)
# ---------------------------------------------------------------------------


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.stride = stride
        self.need_pad = stride != 1 or in_channels != out_channels
        self.pad_channels = out_channels - in_channels if self.need_pad else 0

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = x
        if self.need_pad:
            shortcut = shortcut[:, :, :: self.stride, :: self.stride]
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))
        out += shortcut
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        w1, w2, w3 = STAGE_WIDTHS
        self.conv1 = nn.Conv2d(3, w1, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w1)
        self.layer1 = self._make_layer(w1, w1, num_blocks, stride=1)
        self.layer2 = self._make_layer(w1, w2, num_blocks, stride=2)
        self.layer3 = self._make_layer(w2, w3, num_blocks, stride=2)
        self.fc = nn.Linear(w3, num_classes)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        # Kaiming init normal instead of default uniform per "Delving Deep into Rectifiers" (He et al. 2015), cited as
        # [13] in the ResNet paper
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(m.weight)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        ch = in_ch
        for s in strides:
            layers.append(BasicBlock(ch, out_ch, s))
            ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
    print(f"Device: {device}")

    mean, std = (
        (0.4914, 0.4822, 0.4465),
        (1, 1, 1),
    )  # Yes original paper only mention per-pixel mean and this is per band. See README
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
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

    model = ResNet(NUM_BLOCKS, NUM_CLASSES).to(device)
    if device.type == "cuda" and USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    num_params = sum(p.numel() for p in model.parameters())
    if device.type == "cuda" and USE_COMPILE:
        model = torch.compile(model)
    print(f"ResNet-{6 * NUM_BLOCKS + 2} | params: {num_params:,}")
    print(
        f"CutMix alpha: {CUTMIX_ALPHA}, prob: {CUTMIX_PROB}, "
        f"label smoothing: {CUTMIX_LABEL_SMOOTHING}"
    )

    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_MILESTONES, gamma=0.1
    )
    cutmix_dist = torch.distributions.Beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
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

    while total_training_time < TIME_BUDGET_S and step < MAX_STEPS:
        epoch += 1
        model.train()

        for inputs, targets in train_loader:
            t0 = time.time()
            if device.type == "cuda" and USE_CHANNELS_LAST:
                inputs = inputs.to(
                    device, non_blocking=True, memory_format=torch.channels_last
                )
            else:
                inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            apply_cutmix = torch.rand((), device=device).item() < CUTMIX_PROB
            if apply_cutmix:
                lam = float(cutmix_dist.sample().item())
                batch_perm = torch.randperm(inputs.size(0), device=device)
                targets_a = targets
                targets_b = targets[batch_perm]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam, device)
                mixed_inputs = inputs.clone()
                mixed_inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[
                    batch_perm, :, bby1:bby2, bbx1:bbx2
                ]
                patch_area = (bbx2 - bbx1) * (bby2 - bby1)
                lam = 1.0 - patch_area / (inputs.size(-1) * inputs.size(-2))
                inputs_for_model = mixed_inputs
            else:
                inputs_for_model = inputs

            optimizer.zero_grad()
            outputs = model(inputs_for_model)
            if apply_cutmix:
                loss = lam * F.cross_entropy(
                    outputs, targets_a, label_smoothing=CUTMIX_LABEL_SMOOTHING
                ) + (1.0 - lam) * F.cross_entropy(
                    outputs, targets_b, label_smoothing=CUTMIX_LABEL_SMOOTHING
                )
            else:
                loss = F.cross_entropy(outputs, targets, label_smoothing=0.05)
            loss.backward()
            optimizer.step()
            scheduler.step()

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

            lr = optimizer.param_groups[0]["lr"]
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
