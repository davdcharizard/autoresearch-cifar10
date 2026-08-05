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


def construct(widths, attention):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = train.WideResNet(
        train.NUM_BLOCKS, widths, train.NUM_CLASSES, use_attention=attention
    )
    return model, torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def semantic_checks():
    assert GuardEval.constructions == 1
    assert train.STAGE_WIDTHS == (32, 64, 160)
    assert train.ATTENTION_INIT_SEED == 23017
    torch.empty(1, device=DEVICE)
    accepted, _, _ = construct((32, 64, 128), False)
    width_only, width_cpu, width_cuda = construct((32, 64, 160), False)
    candidate, candidate_cpu, candidate_cuda = construct((32, 64, 160), True)
    assert torch.equal(width_cpu, candidate_cpu)
    assert torch.equal(width_cuda, candidate_cuda)
    assert sum(p.numel() for p in accepted.parameters()) == 691_674
    assert sum(p.numel() for p in width_only.parameters()) == 961_562
    assert sum(p.numel() for p in candidate.parameters()) == 968_302

    for model, widths in (
        (accepted, (32, 64, 128)),
        (width_only, (32, 64, 160)),
        (candidate, (32, 64, 160)),
    ):
        assert model.layer1[0].conv1.out_channels == widths[0]
        assert model.layer2[0].conv1.out_channels == widths[1]
        assert model.layer3[0].conv1.out_channels == widths[2]
        assert model.layer3[1].conv2.out_channels == widths[2]
        assert model.bn.num_features == widths[2]
        assert model.fc.in_features == widths[2]
        assert model.layer3[0].shortcut is not None
        assert model.layer3[1].shortcut is None

    width_state = width_only.state_dict()
    candidate_state = candidate.state_dict()
    for name, tensor in width_state.items():
        assert name in candidate_state and torch.equal(tensor, candidate_state[name]), name

    assert all(block.se is None for block in width_only.layer3)
    gates = [block.se for block in candidate.layer3]
    assert len(gates) == 2 and all(isinstance(gate, train.Stage3SE) for gate in gates)
    for gate in gates:
        assert gate.fc1.weight.shape == (10, 160)
        assert gate.fc2.weight.shape == (160, 10)
        assert torch.count_nonzero(gate.fc1.weight) > 0
        assert torch.count_nonzero(gate.fc1.bias) == 0
        assert torch.count_nonzero(gate.fc2.weight) == 0
        assert torch.count_nonzero(gate.fc2.bias) == 0
        assert len(list(gate.buffers())) == 0

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(23017)
        oracle = [train.Stage3SE(160, 16) for _ in range(2)]
    for expected, actual in zip(oracle, gates):
        for name, tensor in expected.state_dict().items():
            assert torch.equal(tensor, actual.state_dict()[name]), name

    width_only = width_only.to(DEVICE).eval()
    candidate = candidate.to(DEVICE).eval()
    assert all(
        p.device.type == "cuda" and p.dtype == torch.float32
        for gate in gates
        for p in gate.parameters()
    )
    inputs = torch.randn(16, 3, 32, 32, device=DEVICE)
    with torch.inference_mode():
        width_logits = width_only(inputs)
        candidate_logits = candidate(inputs)
    assert torch.equal(width_logits, candidate_logits)

    candidate.train()
    optimizer = optimizer_for(candidate)
    optimizer.zero_grad(set_to_none=True)
    targets = torch.arange(16, device=DEVICE) % train.NUM_CLASSES
    F.cross_entropy(candidate(inputs), targets).backward()
    for gate in gates:
        assert gate.fc1.weight.grad is not None
        assert torch.count_nonzero(gate.fc1.weight.grad) == 0
        assert gate.fc2.weight.grad is not None and gate.fc2.weight.grad.norm() > 0
        assert gate.fc2.bias.grad is not None and torch.isfinite(gate.fc2.bias.grad).all()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(candidate(inputs), targets).backward()
    assert all(gate.fc1.weight.grad.norm() > 0 for gate in gates)

    decay_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(p) for p in optimizer.param_groups[1]["params"]}
    for gate in gates:
        assert id(gate.fc1.weight) in decay_ids and id(gate.fc2.weight) in decay_ids
        assert id(gate.fc1.bias) in no_decay_ids and id(gate.fc2.bias) in no_decay_ids
    print("accepted=691674 width_only=961562 candidate=968302")
    print("gates=2 shapes=160x10x160 seed=23017 identity=exact rng=preserved")
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
    accepted, _, _ = construct((32, 64, 128), False)
    candidate, _, _ = construct((32, 64, 160), True)
    models = [accepted.to(DEVICE).train(), candidate.to(DEVICE).train()]
    optimizers = [optimizer_for(model) for model in models]
    distributions = [distribution(), distribution()]
    host_x = torch.randn(BATCH, 3, 32, 32, pin_memory=True)
    host_y = (torch.arange(BATCH) % train.NUM_CLASSES).pin_memory()
    rng_states = []
    for index in range(2):
        torch.cuda.manual_seed(4200 + index)
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
    assert projected >= 127.0
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
