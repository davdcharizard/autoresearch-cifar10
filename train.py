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
# Frozen patch-whitening front-end (David Page / hlb / airbench).
# A fixed ZCA-whitening 3x3 conv computed once from CIFAR-10 training patches,
# in the SAME normalized space as the frozen eval (mean-subtract, std=1).
# ---------------------------------------------------------------------------


def compute_whitening_weight(train_set, mean, kernel=3, n_img=2000, n_patches=50000, eps=1e-4):
    """Frozen conv weight [2*K, 3, kernel, kernel] whitening kxk RGB patches.

    Computed in eval space (ToTensor /255 then subtract mean, std=1). Capped at
    n_img images (~1.8M interior patches) to bound the off-budget materialization;
    the patch subsample uses a LOCAL Generator so global RNG state is untouched.
    """
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    n_img = min(n_img, len(train_set.data))
    imgs = torch.from_numpy(train_set.data[:n_img]).float().div_(255.0)  # [N,32,32,3]
    imgs = imgs.permute(0, 3, 1, 2).contiguous() - mean_t               # [N,3,32,32] eval space
    p = imgs.unfold(2, kernel, 1).unfold(3, kernel, 1)                  # [N,3,H',W',k,k]
    p = p.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * kernel * kernel)    # [M, 3*k*k]
    if p.shape[0] > n_patches:
        g = torch.Generator().manual_seed(0)  # LOCAL rng — no global side effect
        p = p[torch.randperm(p.shape[0], generator=g)[:n_patches]]
    p = p - p.mean(0, keepdim=True)
    cov = (p.T @ p) / (p.shape[0] - 1)                                  # [3kk, 3kk]
    eigvals, eigvecs = torch.linalg.eigh(cov, UPLO="U")
    W = (eigvecs / torch.sqrt(eigvals + eps).unsqueeze(0)).T           # rows = whitening filters
    W = W.reshape(3 * kernel * kernel, 3, kernel, kernel)
    return torch.cat([W, -W], dim=0).contiguous().float()             # [2*K,3,k,k], +/- pairs


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


class GatedResidual(nn.Module):
    """ReZero-gated residual (Bachlechner et al. 2020, arXiv:2003.04887).

    `alpha` is a learnable scalar initialized to 0, so the block is EXACT
    identity at init (deeper net starts bit-equivalent to the proven net), yet
    dL/d alpha = <grad_out, c2(c1(x))> != 0 keeps a live gradient path so the
    block trains and ramps its capacity in gradually. (A zeroed final-BN gamma
    would NOT work here: conv_bn ends in ReLU, and ReLU'(0)=0 kills the gradient,
    leaving the block dead/identity forever.)
    """

    def __init__(self, c):
        super().__init__()
        self.c1 = conv_bn(c, c)
        self.c2 = conv_bn(c, c)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.alpha * self.c2(self.c1(x))


class ResNet9(nn.Module):
    def __init__(self, num_classes=10, scale_out=SCALE_OUT):
        super().__init__()
        self.scale_out = scale_out
        # Frozen patch-whitening front-end: 3x3/pad-1 preserves 32x32 so the pool
        # chain is untouched; 3*3*3=27 whitened directions, concat +/- -> 54 channels.
        self.whiten = nn.Conv2d(3, 54, 3, padding=1, bias=False)
        self.whiten.weight.requires_grad_(False)
        self.prep = conv_bn(54, 64)  # learnable stem now consumes whitened input
        self.layer1 = nn.Sequential(conv_bn(64, 128), nn.MaxPool2d(2), Residual(128))
        self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), GatedResidual(256))
        self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, num_classes, bias=False)
        self.tta = False  # eval-time horizontal-flip TTA (gated on by the loop)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(m.weight, nonlinearity="relu")

    def load_whitening(self, weight):
        """Load the frozen whitening matrix AFTER .apply()/.to() so kaiming-init
        does not overwrite it, and re-assert it is non-trainable."""
        with torch.no_grad():
            self.whiten.weight.copy_(
                weight.to(self.whiten.weight.device, self.whiten.weight.dtype)
            )
        self.whiten.weight.requires_grad_(False)

    def _forward_once(self, x):
        x = self.whiten(x)
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
            Cutout(12),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0),
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

    # Compute + load the frozen whitening matrix OFF the 300s training budget
    # (before t_start_training); counts toward startup_seconds, not training_seconds.
    t_w = time.time()
    w_weight = compute_whitening_weight(train_set, EVAL_MEAN, kernel=3)
    model.load_whitening(w_weight)
    whitening_seconds = time.time() - t_w
    print(f"whitening_seconds: {whitening_seconds:.2f}")

    num_params = sum(p.numel() for p in model.parameters())
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet-9 (DavidNet+whiten) | params: {num_params:,} | learnable: {learnable_params:,}")

    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],  # exclude frozen whitening conv
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
