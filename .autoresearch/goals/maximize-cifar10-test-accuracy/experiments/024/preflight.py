import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
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
    accepted = train.WideResNet(
        train.NUM_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES, stage3_gates=False
    )
    cpu_after = torch.random.get_rng_state().clone()
    cuda_after = torch.cuda.get_rng_state().clone()
    torch.random.set_rng_state(cpu_before)
    torch.cuda.set_rng_state(cuda_before)
    candidate = train.WideResNet(
        train.NUM_BLOCKS, train.WIDEN_FACTOR, train.NUM_CLASSES, stage3_gates=True
    )
    assert torch.equal(torch.random.get_rng_state(), cpu_after)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_after)
    return accepted, candidate


def semantic_checks():
    assert GuardEval.constructions == 1
    accepted, candidate = model_pair()
    assert sum(p.numel() for p in accepted.parameters()) == 691_674
    assert sum(p.numel() for p in candidate.parameters()) == 692_186
    gates = [block.gate for block in candidate.layer3]
    assert len(gates) == 2
    assert all(isinstance(gate, train.DiagonalStage3Gate) for gate in gates)
    assert all(block.gate is None for block in accepted.layer3)
    for gate in gates:
        assert gate.weight.shape == (128,) and gate.bias.shape == (128,)
        assert torch.count_nonzero(gate.weight) == 0
        assert torch.count_nonzero(gate.bias) == 0
        assert len(list(gate.buffers())) == 0

    accepted_state = accepted.state_dict()
    candidate_state = candidate.state_dict()
    for name, tensor in accepted_state.items():
        assert name in candidate_state and torch.equal(tensor, candidate_state[name]), name

    accepted = accepted.to(DEVICE).eval()
    candidate = candidate.to(DEVICE).eval()
    inputs = torch.randn(16, 3, 32, 32, device=DEVICE)
    residual = torch.randn(16, 128, 8, 8, device=DEVICE)
    with torch.inference_mode():
        for gate in gates:
            assert torch.equal(gate(residual), residual)
        assert torch.equal(accepted(inputs), candidate(inputs))
    assert all(
        p.device.type == "cuda" and p.dtype == torch.float32
        for gate in gates
        for p in gate.parameters()
    )
    assert candidate.layer3[0].shortcut is not None
    assert candidate.layer3[1].shortcut is None

    candidate.train()
    optimizer = optimizer_for(candidate)
    optimizer.zero_grad(set_to_none=True)
    targets = torch.arange(16, device=DEVICE) % train.NUM_CLASSES
    F.cross_entropy(candidate(inputs), targets).backward()
    for gate in gates:
        assert gate.weight.grad is not None and torch.isfinite(gate.weight.grad).all()
        assert gate.bias.grad is not None and torch.isfinite(gate.bias.grad).all()
        assert gate.weight.grad.norm().item() > 0
        assert gate.bias.grad.norm().item() > 0
    decay_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    for gate in gates:
        assert id(gate.weight) not in decay_ids and id(gate.bias) not in decay_ids
        assert id(gate.weight) in no_decay_ids and id(gate.bias) in no_decay_ids
    print("accepted=691674 candidate=692186 gates=2 shape=128")
    print("state_rng_logits=exact scales=unit gradients=open groups=no_decay")
    print("SEMANTICS PASS")


def timed_step(model, optimizer, host_x, host_y, dist, use_mixup):
    start = time.perf_counter()
    x = host_x.to(DEVICE, non_blocking=True)
    y = host_y.to(DEVICE, non_blocking=True)
    lr = train.learning_rate(150.0 if use_mixup else 240.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    if use_mixup:
        mixed, a, b, mix = train.mixup_batch(x, y, dist)
        outputs = model(mixed)
        loss = mix * F.cross_entropy(outputs, a) + (1.0 - mix) * F.cross_entropy(
            outputs, b
        )
    else:
        outputs = model(x)
        loss = F.cross_entropy(outputs, y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start)


def throughput_checks():
    accepted, candidate = model_pair()
    models = [accepted.to(DEVICE).train(), candidate.to(DEVICE).train()]
    optimizers = [optimizer_for(model) for model in models]
    distributions = [distribution(), distribution()]
    host_x = torch.randn(BATCH, 3, 32, 32, pin_memory=True)
    host_y = (torch.arange(BATCH) % train.NUM_CLASSES).pin_memory()
    rng_states = []
    for index in range(2):
        torch.cuda.manual_seed(4240 + index)
        rng_states.append(torch.cuda.get_rng_state().clone())

    results = []
    for use_mixup in (True, False):
        for index in range(2):
            torch.cuda.set_rng_state(rng_states[index])
            for _ in range(20):
                timed_step(
                    models[index], optimizers[index], host_x, host_y,
                    distributions[index], use_mixup
                )
            rng_states[index] = torch.cuda.get_rng_state().clone()
        windows = [[], []]
        for window in range(3):
            order = (0, 1) if window % 2 == 0 else (1, 0)
            for index in order:
                torch.cuda.set_rng_state(rng_states[index])
                values = [
                    timed_step(
                        models[index], optimizers[index], host_x, host_y,
                        distributions[index], use_mixup
                    )
                    for _ in range(40)
                ]
                rng_states[index] = torch.cuda.get_rng_state().clone()
                windows[index].append(statistics.mean(values))
        medians = [statistics.median(values) for values in windows]
        cvs = [
            statistics.pstdev(values) / statistics.mean(values) for values in windows
        ]
        assert all(cv <= 0.05 for cv in cvs)
        results.append((medians, cvs, windows))

    accepted_ms = 0.65 * results[0][0][0] + 0.35 * results[1][0][0]
    candidate_ms = 0.65 * results[0][0][1] + 0.35 * results[1][0][1]
    retention = accepted_ms / candidate_ms
    projected = 141.9 * retention
    print(f"mixup={results[0]}")
    print(f"hard={results[1]}")
    print(
        f"accepted_ms={accepted_ms:.6f} candidate_ms={candidate_ms:.6f} "
        f"retention={retention:.6f} projected_passes={projected:.6f}"
    )
    assert projected >= 138.0
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
