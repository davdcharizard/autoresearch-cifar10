import argparse
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision import transforms

import train


HERE = Path(__file__).resolve().parent


class ControlBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
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
        return F.relu(out + shortcut)


class ControlResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._layer(32, 32, 1)
        self.layer2 = self._layer(32, 64, 2)
        self.layer3 = self._layer(64, 128, 2)
        self.fc = nn.Linear(128, 10)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(module.weight)

    @staticmethod
    def _layer(in_channels, out_channels, stride):
        return nn.Sequential(
            ControlBlock(in_channels, out_channels, stride),
            ControlBlock(out_channels, out_channels),
            ControlBlock(out_channels, out_channels),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def strong_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
        ]
    )


def weak_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
        ]
    )


def optimizer(model):
    return optim.SGD(
        model.parameters(),
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )


def shared_state_equal(control, candidate):
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    for key, value in control_state.items():
        if not torch.equal(value, candidate_state[key]):
            return False, key
    return True, None


def structural_gate():
    torch.manual_seed(42)
    control = ControlResNet()
    control_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = train.ResNet(3, 10, 2)
    candidate_rng = torch.random.get_rng_state().clone()
    equal, bad_key = shared_state_equal(control, candidate)
    assert equal, bad_key
    assert torch.equal(control_rng, candidate_rng)
    assert sum(p.numel() for p in candidate.parameters()) == 1_084_586

    transitions = [candidate.layer2[0].shortcut, candidate.layer3[0].shortcut]
    identities = [
        block.shortcut
        for layer in (candidate.layer1, candidate.layer2, candidate.layer3)
        for block in layer
        if block not in (candidate.layer2[0], candidate.layer3[0])
    ]
    assert all(isinstance(module, nn.Identity) for module in identities)
    for shortcut, in_ch, out_ch in zip(transitions, (32, 64), (64, 128)):
        assert len(shortcut) == 3
        pool, projection, bn = shortcut
        assert isinstance(pool, nn.AvgPool2d)
        assert pool.kernel_size == 2 and pool.stride == 2 and pool.padding == 0
        assert not pool.ceil_mode and not pool.count_include_pad
        assert isinstance(projection, train.ShortcutConv)
        assert projection.in_channels == in_ch and projection.out_channels == out_ch
        assert projection.kernel_size == (1, 1) and projection.stride == (1, 1)
        assert projection.bias is None
        assert isinstance(bn, nn.BatchNorm2d) and bn.num_features == out_ch

    generator = torch.Generator(device="cpu").manual_seed(42)
    expected = []
    for shape in ((64, 32, 1, 1), (128, 64, 1, 1)):
        tensor = torch.empty(shape)
        init.kaiming_normal_(tensor, generator=generator)
        expected.append(tensor)
    assert torch.equal(transitions[0][1].weight, expected[0])
    assert torch.equal(transitions[1][1].weight, expected[1])

    ramp = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
    expected_pool = ramp.unfold(2, 2, 2).unfold(3, 2, 2).mean((-1, -2))
    assert torch.equal(transitions[0][0](ramp), expected_pool)
    assert candidate.layer2[0].conv1.stride == (2, 2)
    assert candidate.layer3[0].conv1.stride == (2, 2)

    candidate.cuda().train()
    target_shapes = [(128,), (128, 10)]
    grad_counts = []
    for shape in target_shapes:
        candidate.zero_grad(set_to_none=True)
        inputs = torch.randn(128, 3, 32, 32, device="cuda")
        if len(shape) == 1:
            targets = torch.randint(10, shape, device="cuda")
        else:
            targets = F.one_hot(torch.randint(10, (128,), device="cuda"), 10).float()
        loss = F.cross_entropy(candidate(inputs), targets)
        loss.backward()
        new_params = [parameter for shortcut in transitions for parameter in shortcut.parameters()]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in new_params)
        grad_counts.append(len(new_params))

    result = {
        "status": "pass",
        "parameters": 1_084_586,
        "shared_state_equal": True,
        "rng_equal": True,
        "transition_shortcuts": 2,
        "identity_shortcuts": 7,
        "new_gradient_tensors_per_target": grad_counts,
    }
    (HERE / "preflight-structural.json").write_text(json.dumps(result, indent=2) + "\n")
    print("STRUCTURAL_GATE_PASS")
    print(json.dumps(result, indent=2))


def step(model, opt, inputs, targets):
    opt.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    opt.step()
    updates = [(p.detach() - old).norm().item() for p, old in zip(model.parameters(), before)]
    return logits.detach(), loss.detach(), updates


def concentration(logits):
    return torch.bincount(logits.argmax(1), minlength=10).max().item() / logits.shape[0]


def numerical_gate():
    torch.manual_seed(42)
    control_base = ControlResNet().cuda()
    torch.manual_seed(42)
    candidate_base = train.ResNet(3, 10, 2).cuda()
    control_state = copy.deepcopy(control_base.state_dict())
    candidate_state = copy.deepcopy(candidate_base.state_dict())
    loader = train.make_train_loader(strong_transform(), train.cutmix_collate)
    iterator = iter(loader)
    batches = []
    while len(batches) < 202:
        batches.append(next(iterator))
    iterator = None
    stopped = train.shutdown_train_loader(loader)
    assert len(stopped) == train.NUM_WORKERS
    mixed = sum(targets.ndim == 2 for _, targets in batches)
    assert 0.45 <= mixed / len(batches) <= 0.55

    first_update = []
    hard = next(batch for batch in batches if batch[1].ndim == 1)
    soft = next(batch for batch in batches if batch[1].ndim == 2)
    for name, (cpu_inputs, cpu_targets) in (("hard", hard), ("soft", soft)):
        control = ControlResNet().cuda(); control.load_state_dict(control_state)
        candidate = train.ResNet(3, 10, 2).cuda(); candidate.load_state_dict(candidate_state)
        control_opt, candidate_opt = optimizer(control), optimizer(candidate)
        inputs, targets = cpu_inputs.cuda(), cpu_targets.cuda()
        residuals, shortcuts = [], []
        hooks = []
        for block in (candidate.layer2[0], candidate.layer3[0]):
            hooks.append(block.bn2.register_forward_hook(lambda _m, _a, out: residuals.append(out.detach())))
            hooks.append(block.shortcut.register_forward_hook(lambda _m, _a, out: shortcuts.append(out.detach())))
        with torch.no_grad():
            pre_loss = F.cross_entropy(candidate(inputs), targets).item()
        for hook in hooks: hook.remove()
        ratios = [s.square().mean().sqrt().item() / r.square().mean().sqrt().item() for r, s in zip(residuals, shortcuts)]
        assert all(0.25 <= ratio <= 4.0 for ratio in ratios)
        _, _, _ = step(control, control_opt, inputs, targets)
        projection_before = [candidate.layer2[0].shortcut[1].weight.detach().clone(), candidate.layer3[0].shortcut[1].weight.detach().clone()]
        _, _, _ = step(candidate, candidate_opt, inputs, targets)
        projection_after = [candidate.layer2[0].shortcut[1].weight.detach(), candidate.layer3[0].shortcut[1].weight.detach()]
        update_ratios = [(after - before).norm().item() / before.norm().item() for before, after in zip(projection_before, projection_after)]
        assert all(ratio <= 0.25 for ratio in update_ratios)
        with torch.no_grad():
            control_logits = control(inputs); candidate_logits = candidate(inputs)
            control_replay = F.cross_entropy(control_logits, targets).item()
            candidate_replay = F.cross_entropy(candidate_logits, targets).item()
        assert candidate_replay <= 2 * pre_loss and candidate_replay <= 2 * control_replay
        assert not (concentration(candidate_logits) > 0.95 and concentration(control_logits) <= 0.95)
        first_update.append({"target": name, "rms_ratios": ratios, "update_ratios": update_ratios, "replay_loss": candidate_replay})

    control = ControlResNet().cuda(); control.load_state_dict(control_state)
    candidate = train.ResNet(3, 10, 2).cuda(); candidate.load_state_dict(candidate_state)
    control_opt, candidate_opt = optimizer(control), optimizer(candidate)
    ema_control = ema_candidate = 0.0
    failure = None
    for index, (cpu_inputs, cpu_targets) in enumerate(batches[:200]):
        inputs, targets = cpu_inputs.cuda(non_blocking=True), cpu_targets.cuda(non_blocking=True)
        control_logits, control_loss, _ = step(control, control_opt, inputs, targets)
        candidate_logits, candidate_loss, _ = step(candidate, candidate_opt, inputs, targets)
        ema_control = 0.95 * ema_control + 0.05 * control_loss.item()
        ema_candidate = 0.95 * ema_candidate + 0.05 * candidate_loss.item()
        if not torch.isfinite(candidate_loss) or (concentration(candidate_logits) > 0.95 and concentration(control_logits) <= 0.95):
            failure = {"step": index + 1, "control_loss": control_loss.item(), "candidate_loss": candidate_loss.item(), "control_concentration": concentration(control_logits), "candidate_concentration": concentration(candidate_logits)}
            break
    result = {"status": "fail" if failure else "pass", "cutmix_rate": mixed / len(batches), "first_update": first_update, "failure": failure, "terminal_ema_ratio": ema_candidate / ema_control}
    (HERE / "preflight-numerical.json").write_text(json.dumps(result, indent=2) + "\n")
    assert failure is None
    assert result["terminal_ema_ratio"] <= 1.5
    print("NUMERICAL_GATE_PASS")
    print(json.dumps(result, indent=2))


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def loader_gate():
    torch.manual_seed(42)
    loader = train.make_train_loader(strong_transform(), train.cutmix_collate)
    model = train.ResNet(3, 10, 2).cuda(); opt = optimizer(model)
    waits, durations, mixed = [], [], 0
    iterator = iter(loader)
    for index in range(1000):
        start = time.perf_counter()
        try: cpu_inputs, cpu_targets = next(iterator)
        except StopIteration:
            iterator = iter(loader); cpu_inputs, cpu_targets = next(iterator)
        waits.append(time.perf_counter() - start); mixed += int(cpu_targets.ndim == 2)
        start = time.perf_counter()
        inputs, targets = cpu_inputs.cuda(non_blocking=True), cpu_targets.cuda(non_blocking=True)
        _, loss, _ = step(model, opt, inputs, targets)
        torch.cuda.synchronize(); durations.append(time.perf_counter() - start)
        assert torch.isfinite(loss)
    iterator = None
    stopped = train.shutdown_train_loader(loader)
    weak = train.make_train_loader(weak_transform()); weak_iterator = iter(weak)
    _, weak_targets = next(weak_iterator); weak_iterator = None
    weak_stopped = train.shutdown_train_loader(weak)
    wait_median = statistics.median(waits[10:]); wait_p95 = percentile(waits[10:], 0.95); step_median = statistics.median(durations[10:])
    result = {"status": "pass", "cutmix_rate": mixed / 1000, "wait_median_ms": 1000 * wait_median, "wait_p95_ms": 1000 * wait_p95, "step_median_ms": 1000 * step_median, "strong_workers_stopped": len(stopped), "weak_workers_stopped": len(weak_stopped), "weak_target_ndim": weak_targets.ndim}
    assert 0.45 <= result["cutmix_rate"] <= 0.55 and wait_median < 0.1 * step_median and wait_p95 < 0.2 * step_median
    assert len(stopped) == train.NUM_WORKERS and len(weak_stopped) == train.NUM_WORKERS and weak_targets.ndim == 1
    (HERE / "preflight-loader.json").write_text(json.dumps(result, indent=2) + "\n")
    print("LOADER_GATE_PASS"); print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural", action="store_true")
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument("--loader", action="store_true")
    args = parser.parse_args()
    if args.structural: structural_gate()
    elif args.numerical: numerical_gate()
    elif args.loader: loader_gate()
    else: parser.error("choose one gate")


if __name__ == "__main__":
    main()
