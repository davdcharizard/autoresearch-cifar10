import argparse
import copy
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import prepare


class GuardEval:
    constructions = 0

    def __init__(self):
        type(self).constructions += 1

    def evaluate(self, model, device):
        raise AssertionError("preflight may not evaluate")


prepare.Eval = GuardEval
import train


DEVICE = torch.device("cuda")
BATCH = train.BATCH_SIZE


def optimizer_for(model):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return optim.SGD(
        [{"params": decay, "weight_decay": train.WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=train.LR,
        momentum=train.MOMENTUM,
        nesterov=True,
    )


def distribution():
    return torch.distributions.Beta(
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
    )


def model_pair():
    torch.empty(1, device=DEVICE)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone()
    accepted = train.WideResNet(train.NUM_BLOCKS, train.WIDEN_FACTOR)
    accepted_cpu_after = torch.random.get_rng_state().clone()
    accepted_cuda_after = torch.cuda.get_rng_state().clone()
    torch.random.set_rng_state(cpu_before)
    torch.cuda.set_rng_state(cuda_before)
    candidate = train.WideResNet(
        train.NUM_BLOCKS,
        train.WIDEN_FACTOR,
        stage_blocks=train.STAGE_BLOCKS,
    )
    assert torch.equal(torch.random.get_rng_state(), accepted_cpu_after)
    assert torch.equal(torch.cuda.get_rng_state(), accepted_cuda_after)
    return accepted, candidate


def macs(model):
    total = 0
    hooks = []

    def hook(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += (
                output.shape[2]
                * output.shape[3]
                * module.out_channels
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
        elif isinstance(module, nn.Linear):
            total += module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 3, 32, 32))
    for item in hooks:
        item.remove()
    return total


def semantics():
    assert GuardEval.constructions == 1
    assert train.NEW_BLOCK_INIT_SEED == 16016
    assert train.STAGE_BLOCKS == (1, 2, 3)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    accepted, candidate = model_pair()
    assert [len(candidate.layer1), len(candidate.layer2), len(candidate.layer3)] == [1, 2, 3]
    assert sum(isinstance(m, train.PreActBlock) for m in candidate.modules()) == 6
    shortcuts = [m.shortcut for m in candidate.modules() if isinstance(m, train.PreActBlock) and m.shortcut is not None]
    assert len(shortcuts) == 3
    assert sum(p.numel() for p in accepted.parameters()) == 691_674
    assert sum(p.numel() for p in candidate.parameters()) == 968_538
    assert macs(accepted) == macs(candidate) == 101_106_944

    accepted_state = accepted.state_dict()
    candidate_state = candidate.state_dict()
    for name, tensor in accepted_state.items():
        if name.startswith("layer1.1."):
            continue
        assert name in candidate_state and torch.equal(tensor, candidate_state[name]), name

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(16016)
        oracle = train.PreActBlock(128, 128)
        oracle.apply(train.WideResNet._weights_init)
    oracle_state = oracle.state_dict()
    for name, tensor in oracle_state.items():
        assert torch.equal(tensor, candidate_state[f"layer3.2.{name}"]), name

    shapes = {}
    hooks = [
        candidate.layer1.register_forward_hook(lambda m, i, o: shapes.__setitem__("l1", tuple(o.shape))),
        candidate.layer2.register_forward_hook(lambda m, i, o: shapes.__setitem__("l2", tuple(o.shape))),
        candidate.layer3.register_forward_hook(lambda m, i, o: shapes.__setitem__("l3", tuple(o.shape))),
    ]
    candidate = candidate.to(DEVICE).train()
    inputs = torch.randn(BATCH, 3, 32, 32, device=DEVICE)
    targets = torch.arange(BATCH, device=DEVICE) % train.NUM_CLASSES
    output = candidate(inputs)
    assert output.shape == (BATCH, train.NUM_CLASSES) and torch.isfinite(output).all()
    assert shapes == {"l1": (BATCH, 32, 32, 32), "l2": (BATCH, 64, 16, 16), "l3": (BATCH, 128, 8, 8)}
    for item in hooks:
        item.remove()
    loss = F.cross_entropy(output, targets)
    loss.backward()
    new_params = list(candidate.layer3[2].parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
    assert all(p.grad.norm().item() > 0 for p in new_params if p.ndim >= 2)
    before = [p.detach().clone() for p in new_params]
    optimizer_for(candidate).step()
    assert any(not torch.equal(old, new) for old, new in zip(before, new_params))
    print("depths=[1,2,3] params=968538 macs=101106944 seed=16016")
    print("SEMANTICS PASS")


def timed_step(model, optimizer, host_x, host_y, dist, use_mixup):
    start = time.perf_counter()
    x = host_x.to(DEVICE, non_blocking=True)
    y = host_y.to(DEVICE, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    if use_mixup:
        mixed, a, b, mix = train.mixup_batch(x, y, dist)
        out = model(mixed)
        loss = mix * F.cross_entropy(out, a) + (1 - mix) * F.cross_entropy(out, b)
    else:
        out = model(x)
        loss = F.cross_entropy(out, y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start)


def throughput():
    assert GuardEval.constructions == 1
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    accepted, candidate = model_pair()
    accepted, candidate = accepted.to(DEVICE).train(), candidate.to(DEVICE).train()
    opts = [optimizer_for(accepted), optimizer_for(candidate)]
    dists = [distribution(), distribution()]
    x = torch.randn(BATCH, 3, 32, 32, pin_memory=True)
    y = (torch.arange(BATCH) % train.NUM_CLASSES).pin_memory()
    regime_results = []
    for use_mixup in (True, False):
        for idx, model in enumerate((accepted, candidate)):
            for _ in range(25):
                timed_step(model, opts[idx], x, y, dists[idx], use_mixup)
        windows = [[], []]
        for window in range(3):
            order = (0, 1) if window % 2 == 0 else (1, 0)
            for idx in order:
                values = [timed_step((accepted, candidate)[idx], opts[idx], x, y, dists[idx], use_mixup) for _ in range(50)]
                windows[idx].append(statistics.mean(values))
        medians = [statistics.median(v) for v in windows]
        cvs = [statistics.pstdev(v) / statistics.mean(v) for v in windows]
        assert all(cv <= 0.05 for cv in cvs)
        regime_results.append((medians, cvs, windows))
    accepted_ms = 0.65 * regime_results[0][0][0] + 0.35 * regime_results[1][0][0]
    candidate_ms = 0.65 * regime_results[0][0][1] + 0.35 * regime_results[1][0][1]
    retention = accepted_ms / candidate_ms
    projected = 141.9 * retention
    print(f"mixup={regime_results[0]} hard={regime_results[1]}")
    print(f"accepted_ms={accepted_ms:.6f} candidate_ms={candidate_ms:.6f} retention={retention:.6f} projected_passes={projected:.6f}")
    assert retention >= 0.97 and projected >= 137.6
    print("THROUGHPUT PASS")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--semantics", action="store_true")
    group.add_argument("--throughput", action="store_true")
    args = parser.parse_args()
    semantics() if args.semantics else throughput()


if __name__ == "__main__":
    main()
