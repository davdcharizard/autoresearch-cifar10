import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets, transforms
from torchvision.transforms import v2

from prepare import DATASET_DIR, NUM_WORKERS, TIME_BUDGET_S, Eval

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

NUM_BLOCKS = 3  # ResNet-20 = 6*3+2
NUM_CLASSES = 10
CUTMIX_ALPHA = 1.0
CUTMIX_PROBABILITY = 0.5
WIDTH_MULTIPLIER = 2
BATCH_SIZE = 128
LR = 0.1
ANNEAL_START_LR = 0.01
MIN_LR = 1e-4
LR_HOLD_FRACTION = 0.8
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MAX_STEPS = 64000
EVAL_CHECKPOINTS = (0.2, 0.4, 0.6, 0.7)
cutmix = v2.CutMix(alpha=CUTMIX_ALPHA, num_classes=NUM_CLASSES)
evaluator = Eval()


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
    def __init__(self, num_blocks, num_classes=10, width_multiplier=1):
        super().__init__()
        c1, c2, c3 = (width_multiplier * channels for channels in (16, 32, 64))
        self.conv1 = nn.Conv2d(3, c1, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.layer1 = self._make_layer(c1, c1, num_blocks, stride=1)
        self.layer2 = self._make_layer(c1, c2, num_blocks, stride=2)
        self.layer3 = self._make_layer(c2, c3, num_blocks, stride=2)
        self.fc = nn.Linear(c3, num_classes)
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


def cutmix_collate(batch):
    inputs, targets = default_collate(batch)
    with torch.random.fork_rng(devices=[]):
        if torch.rand(()).item() < CUTMIX_PROBABILITY:
            return cutmix(inputs, targets)
    return inputs, targets


def make_train_loader(transform, collate_fn=None):
    train_set = datasets.CIFAR10(
        DATASET_DIR, train=True, download=True, transform=transform
    )
    return DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        multiprocessing_context="forkserver",
        collate_fn=collate_fn,
    )


def shutdown_train_loader(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return []

    workers = list(iterator._workers)
    worker_pids = [worker.pid for worker in workers]
    iterator._shutdown_workers()
    loader._iterator = None
    if any(worker.is_alive() for worker in workers):
        raise RuntimeError("training DataLoader workers did not shut down")
    return worker_pids


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
    weak_train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    strong_train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_loader = make_train_loader(strong_train_tf, collate_fn=cutmix_collate)

    model = ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULTIPLIER).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-{6 * NUM_BLOCKS + 2} | params: {num_params:,}")

    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
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
    test_loss = None
    test_acc = None
    eval_checkpoint_index = 0
    randaugment_enabled = True
    strong_batch_count = 0
    cutmix_batch_count = 0

    while total_training_time < TIME_BUDGET_S and step < MAX_STEPS:
        epoch += 1
        model.train()

        train_iterator = iter(train_loader)
        for inputs, targets in train_iterator:
            if randaugment_enabled:
                strong_batch_count += 1
                cutmix_batch_count += int(targets.ndim == 2)
            else:
                assert targets.ndim == 1

            t0 = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            progress = min(total_training_time / TIME_BUDGET_S, 1.0)
            if progress <= LR_HOLD_FRACTION:
                lr = LR
            else:
                cosine_progress = (progress - LR_HOLD_FRACTION) / (
                    1.0 - LR_HOLD_FRACTION
                )
                lr = MIN_LR + 0.5 * (ANNEAL_START_LR - MIN_LR) * (
                    1.0 + math.cos(math.pi * cosine_progress)
                )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
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
            if (
                randaugment_enabled
                and total_training_time >= LR_HOLD_FRACTION * TIME_BUDGET_S
            ):
                break

        train_iterator = None

        progress = min(total_training_time / TIME_BUDGET_S, 1.0)
        training_done = total_training_time >= TIME_BUDGET_S or step >= MAX_STEPS
        checkpoint_due = (
            eval_checkpoint_index < len(EVAL_CHECKPOINTS)
            and progress >= EVAL_CHECKPOINTS[eval_checkpoint_index]
        )
        dense_tail_due = progress >= LR_HOLD_FRACTION
        if checkpoint_due or dense_tail_due or training_done:
            test_loss, test_acc = evaluator.evaluate(model, device)

            if test_acc > best_acc:
                best_acc = test_acc

            print(
                f"\n  eval ep {epoch:3d} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.2f}% | best: {best_acc:.2f}%"
            )

            while (
                eval_checkpoint_index < len(EVAL_CHECKPOINTS)
                and progress >= EVAL_CHECKPOINTS[eval_checkpoint_index]
            ):
                eval_checkpoint_index += 1

        if randaugment_enabled and progress >= LR_HOLD_FRACTION:
            worker_pids = shutdown_train_loader(train_loader)
            del train_loader
            gc.collect()
            train_loader = make_train_loader(weak_train_tf)
            randaugment_enabled = False
            print(
                f"augmentation_switch: randaugment+cutmix->base | epoch: {epoch} | "
                f"progress: {100 * progress:.1f}% | workers_stopped: {len(worker_pids)} | "
                f"cutmix_batches: {cutmix_batch_count}/{strong_batch_count}"
            )

        if epoch == 1:
            gc.collect()

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------

    assert test_loss is not None and test_acc is not None
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
