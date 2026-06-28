import gc
import time

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from prepare import DATASET_DIR, NUM_WORKERS, TIME_BUDGET_S, Eval

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

NUM_CLASSES = 10
BATCH_SIZE = 512
PEAK_LR = 0.4  # mean-loss one-cycle peak (DavidNet "lambda" convention)
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.2
PCT_START = 0.15  # fraction of the time budget spent ramping LR 0 -> PEAK_LR
SCALE_OUT = 0.125  # logit scaling (DavidNet: "output scale is important")
MAX_STEPS = 1_000_000  # high guard; the 300s time budget is the real terminator
EMA_DECAY = 0.998  # short-horizon weight EMA (denoised low-LR-tail average)
EMA_WARMUP_FRAC = 0.15  # start EMA once LR ramp completes (matches PCT_START)
TTA_START_FRAC = 0.8  # eval-time flip-TTA only in the final 20% of the budget
evaluator = Eval()

# Normalization MUST match the frozen eval harness in prepare.py (Eval.__init__).
EVAL_MEAN, EVAL_STD = (0.4914, 0.4822, 0.4465), (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Augmentation: Cutout (pure torch, no new deps), operates on normalized CHW tensor
# ---------------------------------------------------------------------------


class Cutout:
    """Zero out a square patch of the (already-normalized) image.

    Filling with 0.0 equals the dataset mean in raw pixel space, since the
    normalization is mean-subtract only (std=1) — the standard cutout-with-mean
    behavior.
    """

    def __init__(self, size=8):
        self.size = size

    def __call__(self, img):  # img: [C, H, W] tensor
        h, w = img.shape[1], img.shape[2]
        cy = int(torch.randint(h, (1,)).item())
        cx = int(torch.randint(w, (1,)).item())
        s = self.size // 2
        y1, y2 = max(0, cy - s), min(h, cy + s)
        x1, x2 = max(0, cx - s), min(w, cx + s)
        img[:, y1:y2, x1:x2] = 0.0
        return img


# ---------------------------------------------------------------------------
# ResNet-9 / "DavidNet" (David Page, cifar10-fast / DAWNBench)
# Wide-shallow residual net that converges in few epochs under a one-cycle LR.
# ---------------------------------------------------------------------------


def conv_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = conv_bn(c, c)
        self.c2 = conv_bn(c, c)

    def forward(self, x):
        return x + self.c2(self.c1(x))


class ResNet9(nn.Module):
    def __init__(self, num_classes=10, scale_out=SCALE_OUT):
        super().__init__()
        self.scale_out = scale_out
        self.prep = conv_bn(3, 64)
        self.layer1 = nn.Sequential(conv_bn(64, 128), nn.MaxPool2d(2), Residual(128))
        self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, num_classes, bias=False)
        self.tta = False  # eval-time horizontal-flip TTA (gated on by the loop)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(m.weight, nonlinearity="relu")

    def _forward_once(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x) * self.scale_out

    def forward(self, x):
        # Training (and eval before the tail) uses a single forward. In eval with
        # TTA enabled, average logits over the image and its horizontal mirror.
        if self.training or not self.tta:
            return self._forward_once(x)
        return 0.5 * (self._forward_once(x) + self._forward_once(x.flip(-1)))


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(EVAL_MEAN, EVAL_STD),
            Cutout(8),
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
        prefetch_factor=4,
    )

    model = ResNet9(NUM_CLASSES).to(device, memory_format=torch.channels_last)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-9 (DavidNet) | params: {num_params:,}")

    optimizer = optim.SGD(
        model.parameters(),
        lr=PEAK_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Weight EMA of the raw model (params + BN buffers averaged with EMA_DECAY).
    # Evaluated each epoch in place of the raw iterate once warmup completes.
    ema_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True
    ).to(device, memory_format=torch.channels_last)
    ema_started = False

    print(f"Time budget: {TIME_BUDGET_S}s")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"EMA decay: {EMA_DECAY} (warmup {EMA_WARMUP_FRAC}) | flip-TTA from {TTA_START_FRAC}")

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

            # Time-based one-cycle LR: triangular ramp 0 -> PEAK over the first
            # PCT_START of the budget, then linear decay PEAK -> ~0 by the budget
            # end. Keyed on elapsed *training* time so the anneal completes
            # regardless of throughput (at most a single-step overshoot).
            progress = min(1.0, total_training_time / TIME_BUDGET_S)
            if progress < PCT_START:
                lr = PEAK_LR * progress / PCT_START
            else:
                lr = PEAK_LR * (1.0 - progress) / (1.0 - PCT_START)
            for g in optimizer.param_groups:
                g["lr"] = lr

            inputs = inputs.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Update the weight EMA once the LR ramp has completed. `progress` is
            # the same time-based value used for the LR schedule above.
            if progress >= EMA_WARMUP_FRAC:
                ema_model.update_parameters(model)
                ema_started = True

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

        # Evaluate the EMA weights once warmup has begun, else the raw model.
        # Enable flip-TTA only in the final TTA_START_FRAC of the budget, where
        # the gain concentrates, to bound the extra eval wall-clock.
        eval_progress = min(1.0, total_training_time / TIME_BUDGET_S)
        use_tta = eval_progress >= TTA_START_FRAC
        if ema_started:
            ema_model.module.tta = use_tta
            eval_target = ema_model
        else:
            model.tta = use_tta  # eval_progress < EMA_WARMUP_FRAC here, so False
            eval_target = model
        test_loss, test_acc = evaluator.evaluate(eval_target, device)

        if test_acc > best_acc:
            best_acc = test_acc

        wall_elapsed = time.time() - t_start
        print(
            f"\n  eval ep {epoch:3d} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.2f}% | best: {best_acc:.2f}% | wall: {wall_elapsed:.0f}s"
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
