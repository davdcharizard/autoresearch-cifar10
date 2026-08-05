import argparse
import copy
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import prepare


class GuardEval:
    constructions = 0

    def __init__(self):
        type(self).constructions += 1

    def evaluate(self, model, device):
        raise AssertionError("preflight must never evaluate or load test data")


prepare.Eval = GuardEval
import train


DEVICE = torch.device("cuda")
BATCH_SIZE = train.BATCH_SIZE


def make_distribution():
    return torch.distributions.Beta(
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
    )


def make_optimizer(model):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return optim.SGD(
        [
            {"params": decay, "weight_decay": train.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train.LR,
        momentum=train.MOMENTUM,
        nesterov=True,
    )


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def semantics():
    assert torch.cuda.is_available()
    assert GuardEval.constructions == 1
    torch.empty(1, device=DEVICE)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_a = train.WideResNet(train.NUM_BLOCKS, train.WIDEN_FACTOR)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_b = train.WideResNet(train.NUM_BLOCKS, train.WIDEN_FACTOR)
    assert sum(p.numel() for p in model_a.parameters()) == 691_674
    assert_state_equal(model_a.state_dict(), model_b.state_dict())
    model_a = model_a.to(DEVICE)
    model_b = model_b.to(DEVICE)

    distribution = make_distribution()
    torch.cuda.manual_seed(15015)
    samples = torch.cat([distribution.sample((BATCH_SIZE,)) for _ in range(16)])
    assert samples.shape == (4096,)
    assert samples.dtype == torch.float32
    assert torch.isfinite(samples).all()
    assert ((samples >= 0.0) & (samples <= 1.0)).all()
    sample_mean = samples.mean().item()
    sample_variance = samples.var(unbiased=False).item()
    assert 0.47 <= sample_mean <= 0.53, sample_mean
    assert 0.15 <= sample_variance <= 0.21, sample_variance
    assert all(chunk.unique().numel() > 1 for chunk in samples.split(BATCH_SIZE))

    torch.manual_seed(1515)
    torch.cuda.manual_seed(1515)
    inputs = torch.randn(BATCH_SIZE, 3, 32, 32, device=DEVICE)
    identity_targets = torch.arange(BATCH_SIZE, device=DEVICE)
    mixed, targets_a, targets_b, mix = train.mixup_batch(
        inputs, identity_targets, distribution
    )
    assert mix.shape == (BATCH_SIZE,)
    assert torch.equal(targets_a, identity_targets)
    permutation = targets_b
    expected = mix[:, None, None, None] * inputs + (
        1.0 - mix[:, None, None, None]
    ) * inputs[permutation]
    torch.testing.assert_close(mixed, expected, rtol=0.0, atol=0.0)
    assert mix.unique().numel() > 1

    targets_a = torch.arange(BATCH_SIZE, device=DEVICE) % train.NUM_CLASSES
    targets_b = (targets_a + 3) % train.NUM_CLASSES
    outputs = torch.randn(
        BATCH_SIZE, train.NUM_CLASSES, device=DEVICE, requires_grad=True
    )
    production_loss = train.mixup_loss(outputs, targets_a, targets_b, mix)
    reference_loss = (
        mix * F.cross_entropy(outputs, targets_a, reduction="none")
        + (1.0 - mix) * F.cross_entropy(outputs, targets_b, reduction="none")
    ).mean()
    torch.testing.assert_close(production_loss, reference_loss, rtol=0.0, atol=0.0)

    constant = 0.37
    constant_mix = torch.full((BATCH_SIZE,), constant, device=DEVICE)
    candidate_outputs = torch.randn(
        BATCH_SIZE, train.NUM_CLASSES, device=DEVICE, requires_grad=True
    )
    accepted_outputs = candidate_outputs.detach().clone().requires_grad_(True)
    candidate_loss = train.mixup_loss(
        candidate_outputs, targets_a, targets_b, constant_mix
    )
    accepted_loss = constant * F.cross_entropy(accepted_outputs, targets_a) + (
        1.0 - constant
    ) * F.cross_entropy(accepted_outputs, targets_b)
    candidate_gradient = torch.autograd.grad(candidate_loss, candidate_outputs)[0]
    accepted_gradient = torch.autograd.grad(accepted_loss, accepted_outputs)[0]
    torch.testing.assert_close(candidate_loss, accepted_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        candidate_gradient, accepted_gradient, rtol=1e-5, atol=1e-6
    )

    hard_inputs = torch.randn(BATCH_SIZE, 3, 32, 32, device=DEVICE)
    hard_targets = torch.arange(BATCH_SIZE, device=DEVICE) % train.NUM_CLASSES
    opt_a = make_optimizer(model_a)
    opt_b = make_optimizer(model_b)
    for model, optimizer in ((model_a, opt_a), (model_b, opt_b)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(hard_inputs), hard_targets)
        loss.backward()
        optimizer.step()
    assert_state_equal(model_a.state_dict(), model_b.state_dict())
    assert opt_a.state_dict().keys() == opt_b.state_dict().keys()
    for state_a, state_b in zip(
        opt_a.state_dict()["state"].values(), opt_b.state_dict()["state"].values()
    ):
        assert state_a.keys() == state_b.keys()
        for key in state_a:
            assert torch.equal(state_a[key], state_b[key])

    print(
        f"beta_mean={sample_mean:.6f} beta_variance={sample_variance:.6f} "
        f"unique_first_batch={samples[:BATCH_SIZE].unique().numel()}"
    )
    print("SEMANTICS PASS")


def scalar_mixup(inputs, targets, distribution):
    mix = distribution.sample()
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = mix * inputs + (1.0 - mix) * inputs[permutation]
    return mixed, targets, targets[permutation], mix


def timed_step(model, optimizer, host_inputs, host_targets, distribution, candidate):
    start = time.perf_counter()
    inputs = host_inputs.to(DEVICE, non_blocking=True)
    targets = host_targets.to(DEVICE, non_blocking=True)
    for group in optimizer.param_groups:
        group["lr"] = train.LR
    optimizer.zero_grad(set_to_none=True)
    if candidate:
        mixed, targets_a, targets_b, mix = train.mixup_batch(
            inputs, targets, distribution
        )
        outputs = model(mixed)
        loss = train.mixup_loss(outputs, targets_a, targets_b, mix)
    else:
        mixed, targets_a, targets_b, mix = scalar_mixup(
            inputs, targets, distribution
        )
        outputs = model(mixed)
        loss = mix * F.cross_entropy(outputs, targets_a) + (
            1.0 - mix
        ) * F.cross_entropy(outputs, targets_b)
    assert outputs.shape == (BATCH_SIZE, train.NUM_CLASSES)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start)


def throughput():
    assert torch.cuda.is_available()
    assert GuardEval.constructions == 1
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    accepted_model = train.WideResNet(train.NUM_BLOCKS, train.WIDEN_FACTOR).to(DEVICE)
    candidate_model = copy.deepcopy(accepted_model)
    accepted_optimizer = make_optimizer(accepted_model)
    candidate_optimizer = make_optimizer(candidate_model)
    accepted_distribution = make_distribution()
    candidate_distribution = make_distribution()
    host_inputs = torch.randn(BATCH_SIZE, 3, 32, 32, pin_memory=True)
    host_targets = torch.arange(BATCH_SIZE) % train.NUM_CLASSES
    host_targets = host_targets.pin_memory()

    for _ in range(25):
        timed_step(
            accepted_model,
            accepted_optimizer,
            host_inputs,
            host_targets,
            accepted_distribution,
            False,
        )
        timed_step(
            candidate_model,
            candidate_optimizer,
            host_inputs,
            host_targets,
            candidate_distribution,
            True,
        )

    accepted_windows = []
    candidate_windows = []
    for window in range(3):
        order = (False, True) if window % 2 == 0 else (True, False)
        for candidate in order:
            model = candidate_model if candidate else accepted_model
            optimizer = candidate_optimizer if candidate else accepted_optimizer
            distribution = (
                candidate_distribution if candidate else accepted_distribution
            )
            measurements = [
                timed_step(
                    model,
                    optimizer,
                    host_inputs,
                    host_targets,
                    distribution,
                    candidate,
                )
                for _ in range(50)
            ]
            target = candidate_windows if candidate else accepted_windows
            target.append(statistics.mean(measurements))

    accepted_median = statistics.median(accepted_windows)
    candidate_median = statistics.median(candidate_windows)
    accepted_cv = statistics.pstdev(accepted_windows) / statistics.mean(
        accepted_windows
    )
    candidate_cv = statistics.pstdev(candidate_windows) / statistics.mean(
        candidate_windows
    )
    retention = accepted_median / candidate_median
    projected_passes = 141.9 * retention
    print(f"accepted_windows_ms={accepted_windows}")
    print(f"candidate_windows_ms={candidate_windows}")
    print(
        f"accepted_median_ms={accepted_median:.6f} "
        f"candidate_median_ms={candidate_median:.6f} "
        f"accepted_cv={accepted_cv:.6f} candidate_cv={candidate_cv:.6f} "
        f"retention={retention:.6f} projected_passes={projected_passes:.6f}"
    )
    assert accepted_cv <= 0.05, accepted_cv
    assert candidate_cv <= 0.05, candidate_cv
    assert retention >= 0.95, retention
    assert projected_passes >= 134.8, projected_passes
    print("THROUGHPUT PASS")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--semantics", action="store_true")
    mode.add_argument("--throughput", action="store_true")
    args = parser.parse_args()
    if args.semantics:
        semantics()
    else:
        throughput()


if __name__ == "__main__":
    main()
