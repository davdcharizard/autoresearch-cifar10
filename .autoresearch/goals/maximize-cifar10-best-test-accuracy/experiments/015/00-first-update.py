import os
import runpy
import subprocess
import sys

import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, os.getcwd())
import train


SELECTED = [
    f"layer{stage}.{block}.bn2.weight"
    for stage in (1, 2, 3)
    for block in (1, 2)
]


def load_control():
    namespace = {"__name__": "control"}
    source = subprocess.check_output(["git", "show", "7c1e7d8:train.py"], text=True)
    exec(compile(source, "control_train.py", "exec"), namespace)
    return namespace


def make_model(cls):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    return cls(3, 10, 2).cuda().train()


def check(control_cls, candidate_cls, inputs, targets, target_kind):
    control = make_model(control_cls)
    candidate = make_model(candidate_cls)
    control_opt = torch.optim.SGD(
        control.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    candidate_opt = torch.optim.SGD(
        candidate.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    control_loss0 = F.cross_entropy(control(inputs), targets)
    candidate_loss0 = F.cross_entropy(candidate(inputs), targets)
    control_loss0.backward()
    candidate_loss0.backward()
    named = dict(candidate.named_parameters())
    gamma_grad_norms = []
    for name in SELECTED:
        gamma_grad = named[name].grad
        assert gamma_grad is not None and torch.isfinite(gamma_grad).all()
        assert torch.count_nonzero(gamma_grad) > 0
        gamma_grad_norms.append(gamma_grad.norm().item())
        prefix = name.removesuffix("bn2.weight")
        for suffix in ("conv1.weight", "bn1.weight", "bn1.bias", "conv2.weight"):
            branch_grad = named[prefix + suffix].grad
            assert branch_grad is not None and torch.count_nonzero(branch_grad) == 0
    control_opt.step()
    candidate_opt.step()
    gamma_max = max(named[name].abs().max().item() for name in SELECTED)
    assert gamma_max <= 0.25
    with torch.no_grad():
        control_logits1 = control(inputs)
        candidate_logits1 = candidate(inputs)
        control_loss1 = F.cross_entropy(control_logits1, targets)
        candidate_loss1 = F.cross_entropy(candidate_logits1, targets)
        control_concentration = (
            torch.bincount(control_logits1.argmax(1), minlength=10).max().item() / 128
        )
        candidate_concentration = (
            torch.bincount(candidate_logits1.argmax(1), minlength=10).max().item()
            / 128
        )
    assert candidate_loss1 <= 2 * candidate_loss0
    assert candidate_loss1 <= 2 * control_loss1
    assert (
        candidate_concentration <= 0.95
        or control_concentration >= candidate_concentration
    )
    candidate_opt.zero_grad()
    F.cross_entropy(candidate(inputs), targets).backward()
    second_conv_norms = []
    for name in SELECTED:
        prefix = name.removesuffix("bn2.weight")
        norms = []
        for suffix in ("conv1.weight", "conv2.weight"):
            grad = named[prefix + suffix].grad
            assert grad is not None and torch.isfinite(grad).all()
            assert torch.count_nonzero(grad) > 0
            norms.append(grad.norm().item())
        second_conv_norms.append(norms)
    print(
        target_kind,
        {
            "candidate_loss0": candidate_loss0.item(),
            "control_loss1": control_loss1.item(),
            "candidate_loss1": candidate_loss1.item(),
            "gamma_grad_min": min(gamma_grad_norms),
            "gamma_grad_max": max(gamma_grad_norms),
            "post_step_gamma_max": gamma_max,
            "candidate_concentration": candidate_concentration,
            "control_concentration": control_concentration,
            "second_conv_norm_min": min(min(values) for values in second_conv_norms),
        },
    )


def main():
    assert torch.cuda.get_device_name() == "NVIDIA H20"
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
    inputs, hard_targets = next(iter(loader))
    train.shutdown_train_loader(loader)
    inputs = inputs.cuda(non_blocking=True)
    hard_targets = hard_targets.cuda(non_blocking=True)
    soft_targets = F.one_hot(hard_targets, 10).float() * 0.9 + 0.01
    control = load_control()["ResNet"]
    candidate = runpy.run_path("train.py", run_name="candidate")["ResNet"]
    check(control, candidate, inputs, hard_targets, "hard")
    check(control, candidate, inputs, soft_targets, "soft")
    print("production_batch_first_update_gate=PASS")


if __name__ == "__main__":
    main()
