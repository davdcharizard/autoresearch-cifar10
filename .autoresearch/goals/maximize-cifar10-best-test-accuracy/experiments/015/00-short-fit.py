import gc
import os
import runpy
import subprocess
import sys

import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, os.getcwd())
import train


STEPS = 64
SELECTED = [
    f"layer{stage}.{block}.bn2.weight"
    for stage in (1, 2, 3)
    for block in (1, 2)
]


def load_control():
    namespace = {"__name__": "control"}
    source = subprocess.check_output(["git", "show", "7c1e7d8:train.py"], text=True)
    exec(compile(source, "control_train.py", "exec"), namespace)
    return namespace["ResNet"]


def materialize_batches():
    mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)
    strong = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    torch.manual_seed(20260806)
    loader = train.make_train_loader(strong)
    batches = []
    iterator = iter(loader)
    for step in range(STEPS):
        inputs, targets = next(iterator)
        if step % 2:
            with torch.random.fork_rng(devices=[]):
                inputs, targets = train.cutmix(inputs, targets)
        batches.append((inputs, targets))
    iterator = None
    train.shutdown_train_loader(loader)
    return batches


def fit(cls, batches):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = cls(3, 10, 2).cuda().train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    ema = 0.0
    beta = 0.95
    concentrations = []
    losses = []
    for step, (inputs, targets) in enumerate(batches, 1):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()
        torch.cuda.synchronize()
        loss_value = loss.item()
        losses.append(loss_value)
        ema = beta * ema + (1 - beta) * loss_value
        concentrations.append(
            torch.bincount(logits.argmax(1), minlength=10).max().item() / 128
        )
    terminal_ema = ema / (1 - beta**STEPS)
    named = dict(model.named_parameters())
    gamma_norms = [named[name].norm().item() for name in SELECTED if name in named]
    result = {
        "terminal_loss_ema": terminal_ema,
        "last_loss": losses[-1],
        "max_concentration": max(concentrations),
        "terminal_concentration": concentrations[-1],
        "gamma_norm_min": min(gamma_norms) if gamma_norms else None,
        "gamma_norm_max": max(gamma_norms) if gamma_norms else None,
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    assert torch.cuda.get_device_name() == "NVIDIA H20"
    batches = materialize_batches()
    control = fit(load_control(), batches)
    candidate_cls = runpy.run_path("train.py", run_name="candidate")["ResNet"]
    candidate = fit(candidate_cls, batches)
    print("control", control)
    print("candidate", candidate)
    ratio = candidate["terminal_loss_ema"] / control["terminal_loss_ema"]
    print(f"terminal_loss_ema_ratio={ratio:.6f}")
    assert ratio <= 1.5
    assert candidate["terminal_concentration"] < 1.0
    assert candidate["gamma_norm_min"] is not None
    assert candidate["gamma_norm_min"] > 0
    print("short_fit_gate=PASS")


if __name__ == "__main__":
    main()
