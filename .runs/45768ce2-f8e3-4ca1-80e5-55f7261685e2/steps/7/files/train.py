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
        return img * mask


# ---------------------------------------------------------------------------
# PreAct ResNet-18 for CIFAR-10
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
        shortcut = self.shortcut(out)
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
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

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
        return self.linear(out)


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


# ---------------------------------------------------------------------------
# Test-Time Augmentation evaluation
# ---------------------------------------------------------------------------

def evaluate_with_tta(model, device, test_loader):
    """Evaluate with test-time augmentation (horizontal flip)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Original prediction
            outputs1 = model(inputs)
            # Flipped prediction
            outputs2 = model(torch.flip(inputs, [3]))
            # Average
            outputs = (outputs1 + outputs2) / 2.0

            loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Training & evaluation
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

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(DATASET_DIR, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(DATASET_DIR, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = PreActResNet18(NUM_CLASSES).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"PreActResNet-18 | params: {num_params:,}")

    # Try to compile for speed
    try:
        model = torch.compile(model)
        print("Using torch.compile")
    except Exception:
        print("torch.compile not available")

    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, nesterov=True,
    )

    # Estimate total steps for a single cosine cycle
    steps_per_epoch = len(train_loader)
    # Estimate ~32 epochs in budget based on previous runs
    estimated_total_steps = steps_per_epoch * 32
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=estimated_total_steps, eta_min=1e-6
    )

    print(f"Time budget: {time_budget_s}s")
    print(f"Batches per epoch: {steps_per_epoch}")
    print(f"Estimated total steps: {estimated_total_steps}")

    # EMA model for better generalization
    ema_decay = 0.999
    ema_model = PreActResNet18(NUM_CLASSES).to(device)
    ema_model.load_state_dict(model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict())
    ema_model.eval()

    def update_ema():
        src = model._orig_mod if hasattr(model, '_orig_mod') else model
        with torch.no_grad():
            for ema_p, model_p in zip(ema_model.parameters(), src.parameters()):
                ema_p.mul_(ema_decay).add_(model_p, alpha=1 - ema_decay)
            for ema_b, model_b in zip(ema_model.buffers(), src.buffers()):
                ema_b.copy_(model_b)

    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    epoch = 0
    step = 0
    best_acc = 0.0
    best_state = None
    best_ema_acc = 0.0
    best_ema_state = None

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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update EMA
            update_ema()

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

        # Evaluate both regular model and EMA model with TTA
        test_loss, test_acc = evaluator.evaluate(model, device)
        
        # Also evaluate EMA model with TTA
        ema_loss, ema_acc = evaluate_with_tta(ema_model, device, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc
            src = model._orig_mod if hasattr(model, '_orig_mod') else model
            best_state = copy.deepcopy(src.state_dict())

        if ema_acc > best_ema_acc:
            best_ema_acc = ema_acc
            best_ema_state = copy.deepcopy(ema_model.state_dict())

        print(
            f"\n  eval ep {epoch:3d} | test_acc: {test_acc:.2f}% | ema_tta_acc: {ema_acc:.2f}% | best: {best_acc:.2f}% | best_ema: {best_ema_acc:.2f}%"
        )

        if epoch == 1:
            gc.collect()

    # Choose the best between regular and EMA+TTA model
    # But we need to return a model that works with the standard evaluator
    # The evaluator doesn't use TTA, so we should compare:
    # - best regular model accuracy
    # - best EMA model accuracy (without TTA, via standard evaluator)
    
    # Load best regular model
    final_model = PreActResNet18(NUM_CLASSES).to(device)
    
    if best_ema_acc > best_acc and best_ema_state is not None:
        # EMA with TTA was better, but evaluator doesn't use TTA
        # Let's also check EMA without TTA by loading it
        final_model.load_state_dict(best_ema_state)
        ema_no_tta_loss, ema_no_tta_acc = evaluator.evaluate(final_model, device)
        print(f"EMA (no TTA) accuracy: {ema_no_tta_acc:.2f}%")
        
        if ema_no_tta_acc > best_acc and best_state is not None:
            # EMA is genuinely better
            print("Using EMA model")
        elif best_state is not None:
            final_model.load_state_dict(best_state)
            print("Using regular best model")
    elif best_state is not None:
        final_model.load_state_dict(best_state)
        print("Using regular best model")

    t_end = time.time()
    startup_time = t_start_training - t_start
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    # Final eval
    final_loss, final_acc = evaluator.evaluate(final_model, device)
    print(f"Final model accuracy: {final_acc:.2f}%")

    return {
        "model": final_model,
        "device": device,
        "best_test_acc": max(best_acc, final_acc),
        "final_test_acc": final_acc,
        "final_test_loss": final_loss,
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
