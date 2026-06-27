import copy
import gc
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from prepare import DATASET_DIR, NUM_WORKERS, TIME_BUDGET_S, Eval

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
CUTOUT_LENGTH = 16
evaluator = Eval()


# ---------------------------------------------------------------------------
# Cutout augmentation
# ---------------------------------------------------------------------------
class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        c, h, w = img.shape
        mask = torch.ones(h, w, dtype=img.dtype)
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


# ---------------------------------------------------------------------------
# PreAct ResNet-18
# ---------------------------------------------------------------------------
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if len(self.shortcut) > 0 else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


# ---------------------------------------------------------------------------
# Warmup + Cosine LR scheduler
# ---------------------------------------------------------------------------
class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(seed=42, time_budget_s=TIME_BUDGET_S):
    t_start = time.time()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mean = (0.4914, 0.4822, 0.4465)
    std = (1, 1, 1)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(CUTOUT_LENGTH),
    ])

    train_set = datasets.CIFAR10(DATASET_DIR, train=True, download=True, transform=train_tf)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model = PreActResNet18(NUM_CLASSES).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"PreActResNet-18 | params: {num_params:,}")

    # Try to compile
    try:
        model = torch.compile(model)
        print("Using torch.compile")
    except Exception:
        print("torch.compile not available")

    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True
    )

    # Estimate total steps conservatively
    # From memory: ~390 batches/epoch, ~30-35 epochs in 300s budget
    # Use conservative estimate to ensure single cosine cycle completes
    steps_per_epoch = len(train_loader)
    estimated_epochs = 28  # Conservative: previous runs got 30-34 epochs
    estimated_total_steps = steps_per_epoch * estimated_epochs
    warmup_steps = steps_per_epoch * 1  # 1 epoch warmup

    scheduler = WarmupCosineLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=estimated_total_steps,
        min_lr=1e-6,
    )

    print(f"Time budget: {time_budget_s}s")
    print(f"Batches per epoch: {steps_per_epoch}")
    print(f"Estimated total steps: {estimated_total_steps}")

    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    epoch = 0
    step = 0
    best_acc = 0.0
    best_state = None

    while total_training_time < time_budget_s:
        epoch += 1
        model.train()

        for inputs, targets in train_loader:
            t0 = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize()
            dt = time.time() - t0
            total_training_time += dt
            step += 1

            train_loss_f = loss.item()
            ema_beta = 0.95
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            debiased = smooth_train_loss / (1 - ema_beta ** step)

            lr = optimizer.param_groups[0]["lr"]
            pct_done = 100 * total_training_time / time_budget_s
            remaining = max(0, time_budget_s - total_training_time)

            if step % 100 == 0:
                print(
                    f"\rstep {step:05d} ep {epoch} ({pct_done:.1f}%) | loss: {debiased:.4f} | lr: {lr:.6f} | rem: {remaining:.0f}s    ",
                    end="", flush=True,
                )

            if total_training_time >= time_budget_s:
                break

        test_loss, test_acc = evaluator.evaluate(model, device)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"\n  eval ep {epoch:3d} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.2f}% | best: {best_acc:.2f}% | lr: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if epoch == 1:
            gc.collect()

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    t_end = time.time()
    startup_time = t_start_training - t_start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "model": model,
        "device": device,
        "best_test_acc": best_acc,
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "training_seconds": total_training_time,
        "total_seconds": t_end - t_start,
        "startup_seconds": startup_time,
        "peak_vram_mb": peak_vram_mb,
        "num_epochs": epoch,
        "num_steps": step,
        "num_params": num_params,
    }


def main():
    r = train()
    print("---")
    print(f"best_test_acc:    {r['best_test_acc']:.2f}%")
    print(f"final_test_acc:   {r['final_test_acc']:.2f}%")
    print(f"final_test_loss:  {r['final_test_loss']:.4f}")
    print(f"training_seconds: {r['training_seconds']:.1f}")
    print(f"total_seconds:    {r['total_seconds']:.1f}")
    print(f"startup_seconds:  {r['startup_seconds']:.1f}")
    print(f"peak_vram_mb:     {r['peak_vram_mb']:.1f}")
    print(f"num_epochs:       {r['num_epochs']}")
    print(f"num_steps:        {r['num_steps']}")
    print(f"num_params:       {r['num_params']:,}")


if __name__ == "__main__":
    main()
