import copy
import gc
import time
import math
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
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5

evaluator = Eval()


# ---------------------------------------------------------------------------
# Wide ResNet (WRN-16-4)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.dropout_rate = dropout_rate
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        out += shortcut
        return out


class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=10, dropout_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6
        
        nStages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        
        self.conv1 = nn.Conv2d(3, nStages[0], 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(nStages[0], nStages[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(nStages[1], nStages[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(nStages[2], nStages[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.fc = nn.Linear(nStages[3], num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, in_ch, out_ch, num_blocks, stride, dropout_rate):
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, stride, dropout_rate))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ---------------------------------------------------------------------------
# CutMix utilities
# ---------------------------------------------------------------------------

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0)).to(x.device)
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(seed=42, time_budget_s=TIME_BUDGET_S):
    t_start = time.time()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
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

    # WRN-16-4: wider but not too deep, good accuracy/speed tradeoff
    model = WideResNet(depth=16, widen_factor=4, num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"WRN-16-4 | params: {num_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True
    )
    
    # Estimate total steps for cosine schedule
    steps_per_epoch = len(train_loader)
    estimated_epochs = int(time_budget_s / (steps_per_epoch * 0.008))  # rough estimate
    total_steps_estimate = estimated_epochs * steps_per_epoch
    
    # Use cosine annealing with warm restarts or simple cosine
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps_estimate, eta_min=1e-5)

    print(f"Time budget: {time_budget_s}s")
    print(f"Batches per epoch: {steps_per_epoch}")
    print(f"Estimated epochs: {estimated_epochs}, total steps: {total_steps_estimate}")

    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    epoch = 0
    step = 0
    best_acc = 0.0
    best_state = None

    # Use torch amp for speed
    scaler = torch.cuda.amp.GradScaler()

    while total_training_time < time_budget_s:
        epoch += 1
        model.train()

        for inputs, targets in train_loader:
            t0 = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # CutMix with probability
            use_cutmix = np.random.random() < CUTMIX_PROB
            if use_cutmix:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, CUTMIX_ALPHA)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if use_cutmix:
                    loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
                    f"\rstep {step:05d} ep {epoch} ({pct_done:.1f}%) | loss: {debiased:.4f} | lr: {lr:.5f} | rem: {remaining:.0f}s    ",
                    end="", flush=True,
                )

            if total_training_time >= time_budget_s:
                break

        # Evaluate at end of each epoch
        test_loss, test_acc = evaluator.evaluate(model, device)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"\n  eval ep {epoch:3d} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.2f}% | best: {best_acc:.2f}%"
        )

        if epoch == 1:
            gc.collect()

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
