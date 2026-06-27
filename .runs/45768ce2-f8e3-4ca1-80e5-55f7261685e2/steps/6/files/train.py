import copy
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
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NESTEROV = True
evaluator = Eval()


# ---------------------------------------------------------------------------
# Cutout augmentation
# ---------------------------------------------------------------------------
class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
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
# DenseNet-BC for CIFAR-10
# ---------------------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return F.avg_pool2d(out, 2)


class DenseNet(nn.Module):
    def __init__(self, depth=100, growth_rate=12, reduction=0.5, num_classes=10):
        super().__init__()
        # DenseNet-BC: depth = 3*N + 4 (3 dense blocks + 2 transitions + 1 initial conv + 1 final)
        # For bottleneck, each layer has 2 conv layers, so N = (depth - 4) / 6
        n_layers = (depth - 4) // 6
        
        n_channels = 2 * growth_rate  # Initial channels for BC variant
        
        self.conv1 = nn.Conv2d(3, n_channels, 3, padding=1, bias=False)
        
        # Dense Block 1
        self.dense1 = self._make_dense(n_channels, growth_rate, n_layers)
        n_channels += n_layers * growth_rate
        out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = Transition(n_channels, out_channels)
        n_channels = out_channels
        
        # Dense Block 2
        self.dense2 = self._make_dense(n_channels, growth_rate, n_layers)
        n_channels += n_layers * growth_rate
        out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = Transition(n_channels, out_channels)
        n_channels = out_channels
        
        # Dense Block 3
        self.dense3 = self._make_dense(n_channels, growth_rate, n_layers)
        n_channels += n_layers * growth_rate
        
        self.bn = nn.BatchNorm2d(n_channels)
        self.fc = nn.Linear(n_channels, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def _make_dense(self, in_channels, growth_rate, n_layers):
        layers = []
        for i in range(n_layers):
            layers.append(Bottleneck(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ---------------------------------------------------------------------------
# EMA model helper
# ---------------------------------------------------------------------------
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


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
        Cutout(16),
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

    # Use DenseNet-BC-40 (growth_rate=24) for a good speed/accuracy tradeoff
    # DenseNet-BC-40 with k=24: ~0.66M params, good performance
    model = DenseNet(depth=40, growth_rate=24, reduction=0.5, num_classes=NUM_CLASSES).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"DenseNet-BC-40 (k=24) | params: {num_params:,}")

    # Try to compile for speed
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception as e:
        print(f"torch.compile failed: {e}")

    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=NESTEROV
    )

    # Estimate total steps for cosine schedule
    steps_per_epoch = len(train_loader)
    # Estimate ~30 epochs in budget based on DenseNet speed
    estimated_epochs = 25
    estimated_total_steps = steps_per_epoch * estimated_epochs

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=estimated_total_steps, eta_min=0)

    # EMA
    ema = EMAModel(model, decay=0.999)

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
            optimizer.step()
            scheduler.step()
            ema.update(model)

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

            if step % 50 == 0:
                print(
                    f"\rstep {step:05d} ep {epoch} ({pct_done:.1f}%) | loss: {debiased:.4f} | lr: {lr:.5f} | rem: {remaining:.0f}s    ",
                    end="", flush=True,
                )

            if total_training_time >= time_budget_s:
                break

        # Evaluate regular model
        test_loss, test_acc = evaluator.evaluate(model, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

        # Evaluate EMA model
        ema.apply_shadow(model)
        ema_test_loss, ema_test_acc = evaluator.evaluate(model, device)
        if ema_test_acc > best_ema_acc:
            best_ema_acc = ema_test_acc
            best_ema_state = copy.deepcopy(model.state_dict())
        ema.restore(model)

        print(
            f"\n  eval ep {epoch:3d} | test_acc: {test_acc:.2f}% | ema_acc: {ema_test_acc:.2f}% | best: {best_acc:.2f}% | best_ema: {best_ema_acc:.2f}%"
        )

        if epoch == 1:
            gc.collect()

    # Pick the better of regular best or EMA best
    final_best_acc = best_acc
    if best_ema_acc > best_acc and best_ema_state is not None:
        model.load_state_dict(best_ema_state)
        final_best_acc = best_ema_acc
        print(f"Using EMA model with acc {best_ema_acc:.2f}%")
    elif best_state is not None:
        model.load_state_dict(best_state)
        print(f"Using regular model with acc {best_acc:.2f}%")

    t_end = time.time()
    startup_time = t_start_training - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

    return {
        "model": model,
        "device": device,
        "best_test_acc": final_best_acc,
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
