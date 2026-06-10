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
WIDTH_MULT = 4  # WRN-style widening: stage widths (16,32,64) -> (64,128,256)
NUM_CLASSES = 10
BATCH_SIZE = 512
PEAK_LR = 0.4  # linear scaling: 0.1 x (512/128)
WARMUP_FRAC = 0.15  # fraction of time budget spent on linear LR warmup
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4  # applied to conv/linear weights only, not BN/bias
LABEL_SMOOTHING = 0.1
MAX_STEPS = 1_000_000  # non-binding; the time budget governs run length
evaluator = Eval()


def lr_at(progress):
    # One-cycle keyed to elapsed-budget-fraction so the anneal always completes:
    # linear warmup to PEAK_LR over the first WARMUP_FRAC, then cosine to ~0.
    if progress < WARMUP_FRAC:
        return PEAK_LR * progress / WARMUP_FRAC
    q = (progress - WARMUP_FRAC) / (1 - WARMUP_FRAC)
    return PEAK_LR * 0.5 * (1 + math.cos(math.pi * q))


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
    def __init__(self, num_blocks, num_classes=10, width_mult=1):
        super().__init__()
        w1, w2, w3 = 16 * width_mult, 32 * width_mult, 64 * width_mult
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
    print(f"Device: {device}")

    torch.set_float32_matmul_precision("high")  # TF32 matmuls
    torch.backends.cudnn.benchmark = True

    mean, std = (
        (0.4914, 0.4822, 0.4465),
        (1, 1, 1),
    )  # Yes original paper only mention per-pixel mean and this is per band. See README
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random"
            ),
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

    model = ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULT).to(
        device, memory_format=torch.channels_last
    )
    base_model = model  # eager reference: eval runs uncompiled, weights shared
    model = torch.compile(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-{6 * NUM_BLOCKS + 2} ({WIDTH_MULT}x wide) | params: {num_params:,}")

    # No weight decay on BN params and biases (ndim <= 1), decay on conv/linear weights
    decay_params = [p for p in model.parameters() if p.ndim > 1]
    no_decay_params = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = optim.SGD(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=0.0,  # set per-step by lr_at()
        momentum=MOMENTUM,
        nesterov=True,
    )
    print(f"Time budget: {TIME_BUDGET_S}s")
    print(f"Batches per epoch: {len(train_loader)}")

    # Compile warmup: one-time inductor compilation must land in startup, not
    # in the per-step timed budget. No optimizer.step() -> weights unchanged.
    warm_x = torch.randn(BATCH_SIZE, 3, 32, 32, device=device).to(
        memory_format=torch.channels_last
    )
    warm_y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)
    model.train()
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            warm_loss = F.cross_entropy(
                model(warm_x), warm_y, label_smoothing=LABEL_SMOOTHING
            )
        warm_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    del warm_x, warm_y, warm_loss

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
            inputs = inputs.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            targets = targets.to(device, non_blocking=True)

            progress = min(total_training_time / TIME_BUDGET_S, 1.0)
            lr_now = lr_at(progress)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = F.cross_entropy(
                    outputs, targets, label_smoothing=LABEL_SMOOTHING
                )
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

        test_loss, test_acc = evaluator.evaluate(base_model, device)

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
