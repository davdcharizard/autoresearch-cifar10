
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
ACCUM_STEPS = 2  # Gradient accumulation steps (effective batch = 256)
LR = 0.1 * ACCUM_STEPS  # Scale LR linearly with effective batch size
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
CUTOUT_LENGTH = 16
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.1
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
        img = img * mask.unsqueeze(0)
        return img


# ---------------------------------------------------------------------------
# Mixup utility
# ---------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_mult=1):
        super().__init__()
        self.in_planes = 64

        widths = [int(w * width_mult) for w in [64, 128, 256, 512]]

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(widths[3] * block.expansion)
        self.linear = nn.Linear(widths[3] * block.expansion, num_classes)

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
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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
        return self.linear(out)


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


# ---------------------------------------------------------------------------
# TTA wrapper - wraps model for test-time augmentation
# ---------------------------------------------------------------------------
class TTAWrapper(nn.Module):
    """Wraps a model to do test-time augmentation (horizontal flip averaging)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Average predictions over original and horizontally flipped
        out1 = self.model(x)
        out2 = self.model(torch.flip(x, dims=[3]))
        return (out1 + out2) / 2.0


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train(seed=42, time_budget_s=TIME_BUDGET_S):
    t_start = time.time()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
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

    # Try to compile the model
    try:
        model = torch.compile(model)
        print("Model compiled successfully")
    except Exception as e:
        print(f"torch.compile failed: {e}")

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, nesterov=True
    )

    # First, do a few warmup steps to estimate step time
    print("Estimating step time...")
    model.train()
    warmup_iter = iter(train_loader)
    for i in range(10):
        inputs, targets = next(warmup_iter)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # Re-estimate with proper timing
    model.train()
    step_times = []
    for i in range(5):
        inputs, targets = next(warmup_iter)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        t0 = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.time() - t0)

    avg_step_time = sum(step_times) / len(step_times)
    steps_per_epoch = len(train_loader)
    estimated_total_steps = int(time_budget_s / avg_step_time * 0.90)  # 90% safety margin
    # For gradient accumulation, optimizer steps happen every ACCUM_STEPS
    estimated_optimizer_steps = estimated_total_steps // ACCUM_STEPS
    estimated_epochs = estimated_total_steps / steps_per_epoch
    print(f"Avg step time: {avg_step_time*1000:.1f}ms")
    print(f"Estimated total steps: {estimated_total_steps}, optimizer steps: {estimated_optimizer_steps}, epochs: {estimated_epochs:.1f}")

    # Warmup for 5% of training then cosine decay (based on optimizer steps)
    warmup_optimizer_steps = min(250, estimated_optimizer_steps // 20)
    
    def lr_lambda(optim_step):
        if optim_step < warmup_optimizer_steps:
            return (optim_step + 1) / warmup_optimizer_steps
        progress = (optim_step - warmup_optimizer_steps) / max(1, estimated_optimizer_steps - warmup_optimizer_steps)
        progress = min(progress, 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Reset model for actual training
    model_raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_raw._init_weights()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, nesterov=True
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Time budget: {time_budget_s}s")
    print(f"Batches per epoch: {steps_per_epoch}")
    print(f"Gradient accumulation steps: {ACCUM_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * ACCUM_STEPS}")

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    epoch = 0
    step = 0
    optim_step = 0
    best_acc = 0.0
    best_state = None

    while total_training_time < time_budget_s:
        epoch += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            t0 = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Apply Mixup
            inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=MIXUP_ALPHA)

            outputs = model(inputs_mixed)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            # Scale loss by accumulation steps
            loss = loss / ACCUM_STEPS
            loss.backward()

            step += 1

            if step % ACCUM_STEPS == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1

            torch.cuda.synchronize()
            dt = time.time() - t0
            total_training_time += dt

            train_loss_f = loss.item() * ACCUM_STEPS  # Unscale for logging
            ema_beta = 0.95
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            debiased = smooth_train_loss / (1 - ema_beta ** step)

            lr = optimizer.param_groups[0]["lr"]
            pct_done = 100 * total_training_time / time_budget_s
            remaining = max(0, time_budget_s - total_training_time)

            if step % 100 == 0:
                print(
                    f"\rstep {step:05d} optim_step {optim_step} ep {epoch} ({pct_done:.1f}%) | loss: {debiased:.4f} | lr: {lr:.5f} | rem: {remaining:.0f}s    ",
                    end="", flush=True,
                )

            if total_training_time >= time_budget_s:
                break

        # Evaluate at end of each epoch using TTA
        model_raw_eval = model._orig_mod if hasattr(model, '_orig_mod') else model
        tta_model = TTAWrapper(model_raw_eval)
        test_loss, test_acc = evaluator.evaluate(tta_model, device)

        if test_acc > best_acc:
            best_acc = test_acc
            model_raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            best_state = copy.deepcopy(model_raw.state_dict())

        print(
            f"\n  eval ep {epoch:3d} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.2f}% | best: {best_acc:.2f}%"
        )

        if epoch == 1:
            gc.collect()

    # Load the best checkpoint and wrap with TTA
    model_raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    if best_state is not None:
        model_raw.load_state_dict(best_state)

    # Return the TTA-wrapped model for final evaluation
    final_model = TTAWrapper(model_raw)

    t_end = time.time()
    startup_time = t_start_training - t_start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "model": final_model,
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
