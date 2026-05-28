import copy
import gc
import time

import numpy as np
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
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EMA_DECAY = 0.999
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 5
COSINE_T_MAX = 49
WIDTH_MULT = 4
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5
evaluator = Eval()


def cutmix_batch(inputs, targets, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    h, w = inputs.shape[2], inputs.shape[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_rat)
    cut_w = int(w * cut_rat)
    cy = np.random.randint(h)
    cx = np.random.randint(w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    return inputs, targets, targets[index], lam


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

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, width_mult=1):
        super().__init__()
        w = [16 * width_mult, 32 * width_mult, 64 * width_mult]
        self.conv1 = nn.Conv2d(3, w[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w[0])
        self.layer1 = self._make_layer(w[0], w[0], num_blocks, stride=1)
        self.layer2 = self._make_layer(w[0], w[1], num_blocks, stride=2)
        self.layer3 = self._make_layer(w[1], w[2], num_blocks, stride=2)
        self.fc = nn.Linear(w[2], num_classes)
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
    )

    model = ResNet(NUM_BLOCKS, NUM_CLASSES, width_mult=WIDTH_MULT).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-{6 * NUM_BLOCKS + 2} (w={WIDTH_MULT}) | params: {num_params:,}")

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    model = torch.compile(model)
    with torch.amp.autocast("cuda"):
        dummy = torch.randn(2, 3, 32, 32, device=device)
        model(dummy)
    print("torch.compile warmup done")

    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler()
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=COSINE_T_MAX
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
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

    while total_training_time < TIME_BUDGET_S:
        epoch += 1
        model.train()

        for inputs, targets in train_loader:
            t0 = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            use_cutmix = np.random.rand() < CUTMIX_PROB
            if use_cutmix:
                inputs, targets_a, targets_b, lam = cutmix_batch(
                    inputs, targets, CUTMIX_ALPHA
                )

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                if use_cutmix:
                    loss = lam * criterion(outputs, targets_a) + (
                        1.0 - lam
                    ) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)
                for b_ema, b in zip(ema_model.buffers(), model.buffers()):
                    b_ema.data.copy_(b.data)

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

            if total_training_time >= TIME_BUDGET_S:
                break

        scheduler.step()
        test_loss, test_acc = evaluator.evaluate(ema_model, device)

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
