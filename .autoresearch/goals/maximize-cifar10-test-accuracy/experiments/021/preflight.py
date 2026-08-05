import argparse
import importlib.util
import math
import statistics
import sys
import time
import types

import torch
import torch.nn.functional as F
import torch.optim as optim


class DummyEval:
    pass


prepare = types.ModuleType("prepare")
prepare.DATASET_DIR = "./data"
prepare.NUM_WORKERS = 0
prepare.TIME_BUDGET_S = 300
prepare.Eval = DummyEval
sys.modules["prepare"] = prepare
spec = importlib.util.spec_from_file_location("candidate_train", "train.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def optimizer_for(model):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return optim.SGD(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=0.01,
        momentum=0.9,
        nesterov=True,
    )


def semantics():
    assert candidate.SAM_RHO == 0.05
    assert candidate.SAM_START_FRACTION == 0.90
    assert [p >= candidate.SAM_START_FRACTION for p in [0.899999, 0.9, 0.900001]] == [False, True, True]
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = candidate.WideResNet(2, 2, 10).cuda().train()
    assert sum(p.numel() for p in model.parameters()) == 691674
    optimizer = optimizer_for(model)
    x = torch.linspace(-1, 1, 8 * 3 * 32 * 32, device="cuda").view(8, 3, 32, 32)
    y = torch.arange(8, device="cuda") % 10
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    originals = [p.detach().clone() for p in model.parameters()]
    bn_after_first = [
        (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
        for m in model.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    ]
    second = candidate.sam_second_backward(
        model, lambda: F.cross_entropy(model(x), y), candidate.SAM_RHO
    )
    assert torch.isfinite(second)
    assert all(torch.equal(p, old) for p, old in zip(model.parameters(), originals))
    bn_after_sam = [
        (m.running_mean, m.running_var, m.num_batches_tracked)
        for m in model.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    ]
    assert all(
        torch.equal(a, b)
        for expected, actual in zip(bn_after_first, bn_after_sam)
        for a, b in zip(expected, actual)
    )
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    print("SEMANTICS PASS")
    print("boundary=below:false equal:true above:true")
    print("params=691674 restoration=exact bn_updates=one gradients=finite")


def timed(model, optimizer, x, y, sam, steps):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        if sam:
            loss = candidate.sam_second_backward(
                model, lambda: F.cross_entropy(model(x), y), candidate.SAM_RHO
            )
        optimizer.step()
    torch.cuda.synchronize()
    assert math.isfinite(loss.item())
    return (time.perf_counter() - start) / steps


def throughput():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = candidate.WideResNet(2, 2, 10).cuda().train()
    optimizer = optimizer_for(model)
    x = torch.linspace(-1, 1, 256 * 3 * 32 * 32, device="cuda").view(256, 3, 32, 32)
    y = torch.arange(256, device="cuda") % 10
    timed(model, optimizer, x, y, False, 25)
    timed(model, optimizer, x, y, True, 25)
    normal = [timed(model, optimizer, x, y, False, 50) for _ in range(3)]
    sam = [timed(model, optimizer, x, y, True, 50) for _ in range(3)]
    normal_mean = statistics.mean(normal)
    sam_mean = statistics.mean(sam)
    normal_cv = statistics.pstdev(normal) / normal_mean
    sam_cv = statistics.pstdev(sam) / sam_mean
    retention = normal_mean / (0.9 * normal_mean + 0.1 * sam_mean)
    projected = 141.9 * retention
    print(f"normal_ms={normal_mean * 1000:.6f} cv={normal_cv:.6f}")
    print(f"sam_ms={sam_mean * 1000:.6f} cv={sam_cv:.6f}")
    print(f"whole_run_retention={retention:.6f}")
    print(f"projected_passes={projected:.6f}")
    assert normal_cv <= 0.05 and sam_cv <= 0.05
    assert retention >= 0.90 and projected >= 127
    print("THROUGHPUT PASS")


parser = argparse.ArgumentParser()
parser.add_argument("--semantics", action="store_true")
parser.add_argument("--throughput", action="store_true")
args = parser.parse_args()
assert args.semantics != args.throughput
semantics() if args.semantics else throughput()

