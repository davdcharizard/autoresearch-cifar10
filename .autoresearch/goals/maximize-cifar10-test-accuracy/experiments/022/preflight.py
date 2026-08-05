import argparse
import copy
import importlib.util
import inspect
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


def bn_state(model):
    return [
        (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
        for m in model.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    ]


def semantics():
    assert candidate.SAM_RHO == 0.05
    assert candidate.SAM_START_FRACTION == 0.90
    cases = [(0.899999, 200), (0.9, 199), (0.9, 200), (0.900001, 202)]
    assert [candidate.should_use_sam(*case) for case in cases] == [
        False,
        False,
        True,
        True,
    ]
    source = inspect.getsource(candidate.main)
    assert "if should_use_sam(progress, step):" in source
    assert "if not sam_activated:" in source

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = candidate.WideResNet(2, 2, 10).cuda().train()
    assert sum(p.numel() for p in model.parameters()) == 691674
    optimizer = optimizer_for(model)
    groups_before = [
        (group["weight_decay"], group["momentum"], group["nesterov"])
        for group in optimizer.param_groups
    ]
    x = torch.linspace(-1, 1, 8 * 3 * 32 * 32, device="cuda").view(8, 3, 32, 32)
    y = torch.arange(8, device="cuda") % 10
    optimizer.zero_grad(set_to_none=True)
    first_loss = F.cross_entropy(model(x), y)
    first_loss.backward()
    originals = [p.detach().clone() for p in model.parameters()]
    first_grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    bn_after_first = bn_state(model)

    oracle = copy.deepcopy(model)
    oracle_params = [p for p in oracle.parameters() if p.requires_grad]
    oracle_grad_norm = torch.linalg.vector_norm(
        torch.stack([grad.norm(2) for grad in first_grads if grad is not None]), 2
    )
    with torch.no_grad():
        for p, grad in zip(oracle_params, first_grads):
            if grad is not None:
                p.add_(grad * (candidate.SAM_RHO / oracle_grad_norm))
    oracle.zero_grad(set_to_none=True)
    oracle_loss = F.cross_entropy(oracle(x), y)
    oracle_loss.backward()
    oracle_grads = [None if p.grad is None else p.grad.clone() for p in oracle.parameters()]

    second_loss = candidate.sam_second_backward(
        model, lambda: F.cross_entropy(model(x), y), candidate.SAM_RHO
    )
    assert torch.isfinite(second_loss)
    assert all(torch.equal(p, old) for p, old in zip(model.parameters(), originals))
    assert all(
        torch.equal(expected, actual)
        for expected_group, actual_group in zip(bn_after_first, bn_state(model))
        for expected, actual in zip(expected_group, actual_group)
    )
    max_grad_diff = 0.0
    for parameter, expected in zip(model.parameters(), oracle_grads):
        if expected is None:
            assert parameter.grad is None
        else:
            max_grad_diff = max(
                max_grad_diff, (parameter.grad - expected).abs().max().item()
            )
    print(f"oracle_max_grad_diff={max_grad_diff:.9e}")
    assert max_grad_diff <= 5e-5
    groups_after = [
        (group["weight_decay"], group["momentum"], group["nesterov"])
        for group in optimizer.param_groups
    ]
    assert groups_after == groups_before
    optimizer.step()
    print("SEMANTICS PASS")
    print("predicate=below:false odd:false even:true transition=first_true_branch")
    print(
        "params=691674 restoration=exact bn_updates=one "
        f"pure_second_grad_max_diff={max_grad_diff:.3e} groups=unchanged"
    )


def timed_pattern(model, optimizer, x, y, alternating, steps):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        if alternating and candidate.should_use_sam(0.90, step):
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
    timed_pattern(model, optimizer, x, y, False, 20)
    timed_pattern(model, optimizer, x, y, True, 20)
    normal = [timed_pattern(model, optimizer, x, y, False, 40) for _ in range(3)]
    alternating = [
        timed_pattern(model, optimizer, x, y, True, 40) for _ in range(3)
    ]
    normal_mean = statistics.mean(normal)
    alternating_mean = statistics.mean(alternating)
    normal_cv = statistics.pstdev(normal) / normal_mean
    alternating_cv = statistics.pstdev(alternating) / alternating_mean
    retention = normal_mean / (0.9 * normal_mean + 0.1 * alternating_mean)
    projected = 141.9 * retention
    print(f"normal_ms={normal_mean * 1000:.6f} cv={normal_cv:.6f}")
    print(
        f"alternating_ms={alternating_mean * 1000:.6f} "
        f"cv={alternating_cv:.6f}"
    )
    print(f"whole_run_retention={retention:.6f}")
    print(f"projected_passes={projected:.6f}")
    assert normal_cv <= 0.05 and alternating_cv <= 0.05
    assert retention >= 0.90 and projected >= 127.71
    print("THROUGHPUT PASS")


parser = argparse.ArgumentParser()
parser.add_argument("--semantics", action="store_true")
parser.add_argument("--throughput", action="store_true")
args = parser.parse_args()
assert args.semantics != args.throughput
semantics() if args.semantics else throughput()
