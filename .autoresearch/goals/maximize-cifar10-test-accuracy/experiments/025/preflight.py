import argparse
import inspect
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
ACCEPTED_PASSES = 141.9
MIN_PROJECTED_PASSES = 137.0


def optimizer_for(model):
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
    cpu_after = torch.random.get_rng_state().clone()
    cuda_after = torch.cuda.get_rng_state().clone()
    torch.random.set_rng_state(cpu_before)
    torch.cuda.set_rng_state(cuda_before)
    candidate = train.WideResNet(
        train.NUM_BLOCKS, train.WIDEN_FACTOR, stage3_attention=True
    )
    assert torch.equal(torch.random.get_rng_state(), cpu_after)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_after)
    return accepted, candidate


def semantic_checks():
    assert GuardEval.constructions == 1
    assert train.ATTENTION_INIT_SEED == 17017
    assert "diag" not in inspect.getsource(train.Stage3SE).lower()
    accepted, candidate = model_pair()
    assert sum(p.numel() for p in accepted.parameters()) == 691_674
    assert sum(p.numel() for p in candidate.parameters()) == 696_042
    gates = [block.attention for block in candidate.layer3]
    assert len(gates) == 2 and all(isinstance(gate, train.Stage3SE) for gate in gates)
    assert all(block.attention is None for block in accepted.layer3)
    for gate in gates:
        assert gate.fc1.weight.shape == (8, 128)
        assert gate.fc2.weight.shape == (128, 8)
        assert not list(gate.named_buffers())
        assert torch.count_nonzero(gate.fc1.weight) > 0
        assert torch.count_nonzero(gate.fc1.bias) == 0
        assert torch.count_nonzero(gate.fc2.weight) == 0
        assert torch.count_nonzero(gate.fc2.bias) == 0

    accepted_state = accepted.state_dict()
    candidate_state = candidate.state_dict()
    for name, tensor in accepted_state.items():
        assert name in candidate_state and torch.equal(tensor, candidate_state[name]), name

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17017)
        oracle = []
        for _ in range(2):
            gate = train.Stage3SE(128, reduction=16)
            torch.nn.init.kaiming_normal_(
                gate.fc1.weight, mode="fan_in", nonlinearity="relu"
            )
            torch.nn.init.zeros_(gate.fc1.bias)
            torch.nn.init.zeros_(gate.fc2.weight)
            torch.nn.init.zeros_(gate.fc2.bias)
            oracle.append(gate)
    for expected, actual in zip(oracle, gates):
        for name, tensor in expected.state_dict().items():
            assert torch.equal(tensor, actual.state_dict()[name]), name

    accepted = accepted.to(DEVICE).eval()
    candidate = candidate.to(DEVICE).eval()
    assert all(
        p.device.type == "cuda" and p.dtype == torch.float32
        for gate in gates
        for p in gate.parameters()
    )
    inputs = torch.randn(16, 3, 32, 32, device=DEVICE)
    with torch.inference_mode():
        accepted_logits = accepted(inputs)
        candidate_logits = candidate(inputs)
        scales = []
        for gate in gates:
            residual = torch.randn(4, 128, 8, 8, device=DEVICE)
            pooled = F.adaptive_avg_pool2d(residual, 1).flatten(1)
            scales.append(2.0 * torch.sigmoid(gate.fc2(F.relu(gate.fc1(pooled)))))
        assert all(torch.equal(scale, torch.ones_like(scale)) for scale in scales)
    assert torch.equal(accepted_logits, candidate_logits)

    candidate.train()
    optimizer = optimizer_for(candidate)
    optimizer.zero_grad(set_to_none=True)
    targets = torch.arange(16, device=DEVICE) % train.NUM_CLASSES
    F.cross_entropy(candidate(inputs), targets).backward()
    for gate in gates:
        assert gate.fc1.weight.grad is not None
        assert torch.count_nonzero(gate.fc1.weight.grad) == 0
        assert gate.fc2.weight.grad is not None
        assert gate.fc2.weight.grad.norm().item() > 0
        assert torch.isfinite(gate.fc2.weight.grad).all()
        assert torch.isfinite(gate.fc2.bias.grad).all()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(candidate(inputs), targets).backward()
    assert all(
        gate.fc1.weight.grad.norm().item() > 0
        and torch.isfinite(gate.fc1.weight.grad).all()
        for gate in gates
    )

    decay_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    for gate in gates:
        assert id(gate.fc1.weight) in decay_ids and id(gate.fc2.weight) in decay_ids
        assert id(gate.fc1.bias) in no_decay_ids and id(gate.fc2.bias) in no_decay_ids

    class ZeroGate(nn.Module):
        def forward(self, residual):
            return torch.zeros_like(residual)

    block = candidate.layer3[1]
    block.attention = ZeroGate()
    probe = torch.randn(4, 128, 8, 8, device=DEVICE)
    block.eval()
    with torch.inference_mode():
        assert torch.equal(block(probe), probe)

    print("gates=2 params=696042 seed=17017 identity_no_diagnostics=pass")
    print("SEMANTICS PASS")


def timed_step(model, optimizer, host_x, host_y, dist, use_mixup, rng_state):
    torch.cuda.set_rng_state(rng_state)
    start = time.perf_counter()
    inputs = host_x.to(DEVICE, non_blocking=True)
    targets = host_y.to(DEVICE, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    if use_mixup:
        mixed, targets_a, targets_b, mix = train.mixup_batch(inputs, targets, dist)
        outputs = model(mixed)
        loss = mix * F.cross_entropy(outputs, targets_a) + (1.0 - mix) * F.cross_entropy(
            outputs, targets_b
        )
    else:
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed_ms = 1000 * (time.perf_counter() - start)
    return elapsed_ms, torch.cuda.get_rng_state().clone()


def throughput_checks():
    accepted, candidate = model_pair()
    models = [accepted.to(DEVICE).train(), candidate.to(DEVICE).train()]
    optimizers = [optimizer_for(model) for model in models]
    distributions = [distribution(), distribution()]
    host_x = torch.randn(BATCH, 3, 32, 32, pin_memory=True)
    host_y = (torch.arange(BATCH) % train.NUM_CLASSES).pin_memory()
    torch.cuda.manual_seed(25025)
    rng_states = [torch.cuda.get_rng_state().clone() for _ in models]
    results = []
    for use_mixup in (True, False):
        for index in range(2):
            for _ in range(25):
                _, rng_states[index] = timed_step(
                    models[index], optimizers[index], host_x, host_y,
                    distributions[index], use_mixup, rng_states[index]
                )
        windows = [[], []]
        for window in range(3):
            order = (0, 1) if window % 2 == 0 else (1, 0)
            for index in order:
                values = []
                for _ in range(50):
                    elapsed, rng_states[index] = timed_step(
                        models[index], optimizers[index], host_x, host_y,
                        distributions[index], use_mixup, rng_states[index]
                    )
                    values.append(elapsed)
                windows[index].append(statistics.mean(values))
        medians = [statistics.median(values) for values in windows]
        cvs = [statistics.pstdev(values) / statistics.mean(values) for values in windows]
        assert all(cv <= 0.05 for cv in cvs), (use_mixup, cvs, windows)
        results.append((medians, cvs, windows))
    accepted_ms = 0.65 * results[0][0][0] + 0.35 * results[1][0][0]
    candidate_ms = 0.65 * results[0][0][1] + 0.35 * results[1][0][1]
    retention = accepted_ms / candidate_ms
    projected = ACCEPTED_PASSES * retention
    print(f"mixup={results[0]} hard={results[1]}")
    print(
        f"accepted_ms={accepted_ms:.6f} candidate_ms={candidate_ms:.6f} "
        f"retention={retention:.6f} projected_passes={projected:.6f}"
    )
    assert projected >= MIN_PROJECTED_PASSES, projected
    print("THROUGHPUT PASS")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--semantics", action="store_true")
    group.add_argument("--throughput", action="store_true")
    args = parser.parse_args()
    semantic_checks() if args.semantics else throughput_checks()


if __name__ == "__main__":
    main()
