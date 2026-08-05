import importlib
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
import prepare


class FailClosedEval:
    def __init__(self):
        pass

    def evaluate(self, *args):
        raise AssertionError("preflight must not evaluate test data")


prepare.Eval = FailClosedEval
sys.modules.pop("train", None)
train = importlib.import_module("train")
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "NVIDIA H20"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

for invalid in (0, 1, None, "true"):
    try:
        train.WideResNet(2, 2, 10, zero_init_residual=invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted invalid Boolean {invalid!r}")

torch.manual_seed(1234)
accepted = train.WideResNet(2, 2, 10, zero_init_residual=False)
accepted_rng = torch.get_rng_state().clone()
torch.manual_seed(1234)
candidate = train.WideResNet(2, 2, 10, zero_init_residual=True)
candidate_rng = torch.get_rng_state().clone()
assert torch.equal(accepted_rng, candidate_rng)
assert accepted.state_dict().keys() == candidate.state_dict().keys()
assert sum(p.numel() for p in accepted.parameters()) == 691_674
assert sum(p.numel() for p in candidate.parameters()) == 691_674
accepted_blocks = [m for m in accepted.modules() if isinstance(m, train.PreActBlock)]
candidate_blocks = [m for m in candidate.modules() if isinstance(m, train.PreActBlock)]
assert len(accepted_blocks) == len(candidate_blocks) == 6
assert accepted.zeroed_residual_blocks == 0 and candidate.zeroed_residual_blocks == 6
endpoint_keys = {f"layer{stage}.{index}.conv2.weight" for stage in range(1, 4) for index in range(2)}
for key, value in accepted.state_dict().items():
    other = candidate.state_dict()[key]
    if key in endpoint_keys:
        assert torch.count_nonzero(value) > 0 and torch.count_nonzero(other) == 0
    else:
        assert torch.equal(value, other), key
for block in candidate_blocks:
    assert torch.equal(block.bn1.weight, torch.ones_like(block.bn1.weight))
    assert torch.equal(block.bn2.weight, torch.ones_like(block.bn2.weight))
    assert torch.equal(block.bn1.bias, torch.zeros_like(block.bn1.bias))
    assert torch.equal(block.bn2.bias, torch.zeros_like(block.bn2.bias))

shapes = [(16, 32, 1), (32, 32, 1), (32, 64, 2), (64, 64, 1), (64, 128, 2), (128, 128, 1)]
for mode in (True, False):
    for block, (in_ch, out_ch, stride) in zip(candidate_blocks, shapes):
        block.train(mode)
        x = torch.randn(4, in_ch, 16, 16)
        with torch.no_grad():
            pre = F.relu(block.bn1(x))
            shortcut = block.shortcut(pre) if block.shortcut is not None else x
            actual = block(x)
        assert torch.equal(actual, shortcut)
        assert actual.shape == (4, out_ch, 16 // stride, 16 // stride)

candidate = candidate.to(device).train()
optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01, momentum=0.9, nesterov=True)
x = torch.randn(32, 3, 32, 32, device=device)
y = torch.randint(0, 10, (32,), device=device)
loss = F.cross_entropy(candidate(x), y)
loss.backward()
blocks = [m for m in candidate.modules() if isinstance(m, train.PreActBlock)]
for block in blocks:
    assert torch.isfinite(block.conv2.weight.grad).all() and block.conv2.weight.grad.norm() > 0
    assert block.conv1.weight.grad is not None and block.conv1.weight.grad.norm() == 0
    assert block.bn2.weight.grad is not None and block.bn2.weight.grad.norm() == 0
for block in (candidate.layer1[0], candidate.layer2[0], candidate.layer3[0]):
    assert torch.isfinite(block.shortcut.weight.grad).all() and block.shortcut.weight.grad.norm() > 0
assert candidate.fc.weight.grad.norm() > 0
optimizer.step()
for block in blocks:
    assert torch.isfinite(block.conv2.weight).all() and block.conv2.weight.norm() > 0
optimizer.zero_grad(set_to_none=True)
F.cross_entropy(candidate(x), y).backward()
for block in blocks:
    assert torch.isfinite(block.conv1.weight.grad).all() and block.conv1.weight.grad.norm() > 0
    assert torch.isfinite(block.bn2.weight.grad).all() and block.bn2.weight.grad.norm() > 0

dead = train.PreActBlock(32, 32)
torch.nn.init.zeros_(dead.bn2.weight)
torch.nn.init.zeros_(dead.bn2.bias)
dead(torch.randn(4, 32, 8, 8)).sum().backward()
assert dead.bn2.weight.grad is not None and dead.bn2.weight.grad.norm() == 0

torch.manual_seed(4321)
accepted = train.WideResNet(2, 2, 10, zero_init_residual=False).to(device)
torch.manual_seed(4321)
candidate = train.WideResNet(2, 2, 10, zero_init_residual=True).to(device)
models = {"accepted": accepted, "candidate": candidate}


def make_optimizer(model):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": train.WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
        lr=train.MIN_LR,
        momentum=train.MOMENTUM,
        nesterov=True,
    )


optimizers = {k: make_optimizer(v) for k, v in models.items()}
distributions = {
    k: torch.distributions.Beta(torch.tensor(0.2, device=device), torch.tensor(0.2, device=device))
    for k in models
}
generator = torch.Generator().manual_seed(9876)
host_x = torch.randn(256, 3, 32, 32, generator=generator, pin_memory=True)
host_y = torch.randint(0, 10, (256,), generator=generator, pin_memory=True)
states = {}
for key in models:
    torch.manual_seed(10)
    torch.cuda.manual_seed(20)
    states[key] = (torch.get_rng_state(), torch.cuda.get_rng_state())


def step(key, progress):
    start = time.perf_counter()
    inputs = host_x.to(device, non_blocking=True)
    targets = host_y.to(device, non_blocking=True)
    optimizer = optimizers[key]
    for group in optimizer.param_groups:
        group["lr"] = train.learning_rate(progress * train.TIME_BUDGET_S)
    optimizer.zero_grad(set_to_none=True)
    if progress < 0.65:
        mixed, a, b, mix = train.mixup_batch(inputs, targets, distributions[key])
        logits = models[key](mixed)
        loss = mix * F.cross_entropy(logits, a) + (1 - mix) * F.cross_entropy(logits, b)
    else:
        logits = models[key](inputs)
        loss = F.cross_entropy(logits, targets)
    assert torch.isfinite(loss) and logits.shape == (256, 10)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start)


def window(key, progress, count):
    torch.set_rng_state(states[key][0])
    torch.cuda.set_rng_state(states[key][1])
    values = [step(key, progress) for _ in range(count)]
    states[key] = (torch.get_rng_state(), torch.cuda.get_rng_state())
    return statistics.mean(values)


for key in models:
    window(key, 0.5, 25)
order = (("accepted", "A"), ("candidate", "A"), ("candidate", "B"), ("accepted", "B"), ("accepted", "C"), ("candidate", "C"))
results = {regime: {key: [] for key in models} for regime in ("mixup", "hard")}
torch.cuda.reset_peak_memory_stats()
for regime, progress in (("mixup", 0.5), ("hard", 0.8)):
    for key, label in order:
        value = window(key, progress, 50)
        results[regime][key].append(value)
        print(f"{regime} {key}-{label}: {value:.6f} ms")
medians = {regime: {key: statistics.median(v) for key, v in paths.items()} for regime, paths in results.items()}
cvs = {regime: {key: statistics.pstdev(v) / statistics.mean(v) for key, v in paths.items()} for regime, paths in results.items()}
aggregate = {key: 0.65 * medians["mixup"][key] + 0.35 * medians["hard"][key] for key in models}
retention = aggregate["accepted"] / aggregate["candidate"]
projection = 141.9 * retention
print(f"cvs={cvs}")
print(f"aggregates={aggregate} retention={retention:.6f} projection={projection:.6f}")
print(f"peak_memory_mb={torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")
assert all(value <= 0.05 for paths in cvs.values() for value in paths.values())
assert retention >= 0.97 and projection >= 135.0
print("PREFLIGHT PASS")
