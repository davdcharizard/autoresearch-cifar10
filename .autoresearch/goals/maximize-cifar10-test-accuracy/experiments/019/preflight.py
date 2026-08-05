import argparse
import importlib.util
import math
import statistics
import subprocess
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DummyEval:
    def evaluate(self, model, device):
        raise AssertionError("evaluator must not be used by preflight")


def load_modules():
    prepare = types.ModuleType("prepare")
    prepare.DATASET_DIR = "./data"
    prepare.NUM_WORKERS = 0
    prepare.TIME_BUDGET_S = 300
    prepare.Eval = DummyEval
    sys.modules["prepare"] = prepare

    accepted = types.ModuleType("accepted_train")
    accepted.__file__ = "eb08811:train.py"
    source = subprocess.check_output(
        ["git", "show", "eb08811:train.py"], text=True
    )
    exec(compile(source, accepted.__file__, "exec"), accepted.__dict__)

    spec = importlib.util.spec_from_file_location("candidate_train", "train.py")
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    return accepted, candidate


def seed_all():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def semantics():
    accepted, candidate = load_modules()
    assert torch.cuda.is_available()

    seed_all()
    accepted_model = accepted.WideResNet(2, 2, 10)
    accepted_cpu_rng = torch.random.get_rng_state().clone()
    accepted_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]

    seed_all()
    candidate_model = candidate.WideResNet(2, 2, 10)
    candidate_cpu_rng = torch.random.get_rng_state().clone()
    candidate_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]

    assert torch.equal(accepted_cpu_rng, candidate_cpu_rng)
    assert all(
        torch.equal(left, right)
        for left, right in zip(accepted_cuda_rng, candidate_cuda_rng)
    )
    accepted_state = accepted_model.state_dict()
    candidate_state = candidate_model.state_dict()
    extras = set(candidate_state) - set(accepted_state)
    assert extras == {
        "layer3.0.residual_transform.scale",
        "layer3.1.residual_transform.fc1.weight",
        "layer3.1.residual_transform.fc1.bias",
        "layer3.1.residual_transform.fc2.weight",
        "layer3.1.residual_transform.fc2.bias",
    }
    for name, value in accepted_state.items():
        assert torch.equal(value, candidate_state[name]), name

    transforms = [
        (stage_index, block_index, block.residual_transform)
        for stage_index, layer in enumerate(
            [candidate_model.layer1, candidate_model.layer2, candidate_model.layer3],
            start=1,
        )
        for block_index, block in enumerate(layer)
        if block.residual_transform is not None
    ]
    assert len(transforms) == 2
    assert isinstance(transforms[0][2], candidate.StaticChannelScale)
    assert isinstance(transforms[1][2], candidate.Stage3SE)
    assert [(item[0], item[1]) for item in transforms] == [(3, 0), (3, 1)]
    static = candidate_model.layer3[0].residual_transform
    gate = candidate_model.layer3[1].residual_transform
    assert static.scale.shape == (128,)
    assert torch.equal(static.scale, torch.ones_like(static.scale))

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        oracle = candidate.Stage3SE(128, reduction=16)
        nn.init.kaiming_normal_(
            oracle.fc1.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(oracle.fc1.bias)
        nn.init.zeros_(oracle.fc2.weight)
        nn.init.zeros_(oracle.fc2.bias)
    for actual, expected in zip(gate.parameters(), oracle.parameters()):
        assert torch.equal(actual, expected)

    x = torch.linspace(-1.0, 1.0, 3 * 32 * 32).view(1, 3, 32, 32)
    accepted_model.eval()
    candidate_model.eval()
    with torch.inference_mode():
        accepted_logits = accepted_model(x)
        candidate_logits = candidate_model(x)
        gate_scale = 2.0 * torch.sigmoid(
            gate.fc2(F.relu(gate.fc1(torch.ones(4, 128))))
        )
    assert torch.equal(accepted_logits, candidate_logits)
    assert torch.equal(gate_scale, torch.ones_like(gate_scale))

    class ZeroResidual(nn.Module):
        def forward(self, value):
            return torch.zeros_like(value)

    for block in (candidate_model.layer3[0], candidate_model.layer3[1]):
        transform = block.residual_transform
        block.residual_transform = ZeroResidual()
        block.eval()
        block_input = torch.linspace(
            -1.0, 1.0, 2 * block.bn1.num_features * 8 * 8
        ).view(2, block.bn1.num_features, 8, 8)
        with torch.inference_mode():
            preactivated = F.relu(block.bn1(block_input))
            expected = (
                block.shortcut(preactivated)
                if block.shortcut is not None
                else block_input
            )
            actual = block(block_input)
        block.residual_transform = transform
        assert torch.equal(actual, expected)

    candidate_model = candidate_model.cuda()
    static = candidate_model.layer3[0].residual_transform
    gate = candidate_model.layer3[1].residual_transform
    assert sum(p.numel() for p in candidate_model.parameters()) == 693986
    assert all(
        p.device.type == "cuda" and p.dtype == torch.float32
        for p in list(static.parameters()) + list(gate.parameters())
    )
    decay = {
        id(p)
        for p in candidate_model.parameters()
        if p.requires_grad and p.ndim >= 2
    }
    no_decay = {
        id(p)
        for p in candidate_model.parameters()
        if p.requires_grad and p.ndim < 2
    }
    assert id(static.scale) in no_decay
    assert id(gate.fc1.weight) in decay and id(gate.fc2.weight) in decay
    assert id(gate.fc1.bias) in no_decay and id(gate.fc2.bias) in no_decay
    assert not list(static.buffers()) and not list(gate.buffers())

    optimizer = optim.SGD(
        [
            {"params": [gate.fc1.weight, gate.fc2.weight], "weight_decay": 5e-4},
            {
                "params": [static.scale, gate.fc1.bias, gate.fc2.bias],
                "weight_decay": 0.0,
            },
        ],
        lr=0.1,
        momentum=0.9,
        nesterov=True,
    )
    residual0 = torch.linspace(
        -1.0, 1.0, 4 * 128 * 8 * 8, device="cuda"
    ).view(4, 128, 8, 8)
    residual1 = residual0.flip(0)
    optimizer.zero_grad(set_to_none=True)
    loss = static(residual0).square().mean() + gate(residual1).square().mean()
    loss.backward()
    assert torch.count_nonzero(static.scale.grad).item() > 0
    assert torch.count_nonzero(gate.fc2.weight.grad).item() > 0
    assert torch.count_nonzero(gate.fc1.weight.grad).item() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    gate(residual1).square().mean().backward()
    assert torch.isfinite(gate.fc1.weight.grad).all()
    assert torch.count_nonzero(gate.fc1.weight.grad).item() > 0

    print("SEMANTICS PASS")
    print("transforms=2 placement=layer3.0.static,layer3.1.se")
    print("params=693986")
    print("seed=42 common_state=exact cpu_rng=exact cuda_rng=exact")
    print("initial_logits=exact initial_scales=1 shortcuts=ungated")
    print("gradients=pass optimizer_groups=pass diagnostics=none")


def make_optimizer(model):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return optim.SGD(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=0.05,
        momentum=0.9,
        nesterov=True,
    )


def train_steps(model, optimizer, inputs, targets, mixed, permuted, mixup, steps):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if mixup:
            outputs = model(mixed)
            loss = 0.4 * F.cross_entropy(outputs, targets) + 0.6 * F.cross_entropy(
                outputs, permuted
            )
        else:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert math.isfinite(loss.item())
    return elapsed / steps


def cv(values):
    return statistics.pstdev(values) / statistics.mean(values)


def throughput():
    accepted, candidate = load_modules()
    assert torch.cuda.is_available()
    seed_all()
    accepted_model = accepted.WideResNet(2, 2, 10).cuda().train()
    seed_all()
    candidate_model = candidate.WideResNet(2, 2, 10).cuda().train()
    accepted_optimizer = make_optimizer(accepted_model)
    candidate_optimizer = make_optimizer(candidate_model)

    inputs = torch.linspace(
        -1.0, 1.0, 256 * 3 * 32 * 32, device="cuda"
    ).view(256, 3, 32, 32)
    targets = torch.arange(256, device="cuda") % 10
    permuted = targets.flip(0)
    mixed = 0.4 * inputs + 0.6 * inputs.flip(0)

    for mixup in (True, False):
        train_steps(
            accepted_model,
            accepted_optimizer,
            inputs,
            targets,
            mixed,
            permuted,
            mixup,
            25,
        )
        train_steps(
            candidate_model,
            candidate_optimizer,
            inputs,
            targets,
            mixed,
            permuted,
            mixup,
            25,
        )

    values = {
        "accepted_mixup": [],
        "candidate_mixup": [],
        "accepted_hard": [],
        "candidate_hard": [],
    }
    for window in range(3):
        order = (
            [
                ("accepted", accepted_model, accepted_optimizer),
                ("candidate", candidate_model, candidate_optimizer),
            ]
            if window % 2 == 0
            else [
                ("candidate", candidate_model, candidate_optimizer),
                ("accepted", accepted_model, accepted_optimizer),
            ]
        )
        for regime, mixup in (("mixup", True), ("hard", False)):
            for name, model, optimizer in order:
                values[f"{name}_{regime}"].append(
                    train_steps(
                        model,
                        optimizer,
                        inputs,
                        targets,
                        mixed,
                        permuted,
                        mixup,
                        50,
                    )
                )

    means = {key: statistics.mean(items) for key, items in values.items()}
    cvs = {key: cv(items) for key, items in values.items()}
    accepted_weighted = 0.65 * means["accepted_mixup"] + 0.35 * means["accepted_hard"]
    candidate_weighted = 0.65 * means["candidate_mixup"] + 0.35 * means["candidate_hard"]
    retention = accepted_weighted / candidate_weighted
    projected_passes = 300.0 / candidate_weighted * 256 / 50000

    for key, items in values.items():
        windows = ",".join(f"{item * 1000:.6f}" for item in items)
        print(
            f"{key}: windows_ms=[{windows}] "
            f"mean_ms={means[key] * 1000:.6f} cv={cvs[key]:.6f}"
        )
    print(f"accepted_weighted_ms={accepted_weighted * 1000:.6f}")
    print(f"candidate_weighted_ms={candidate_weighted * 1000:.6f}")
    print(f"retention={retention:.6f}")
    print(f"projected_passes={projected_passes:.6f}")
    assert all(value <= 0.05 for value in cvs.values())
    assert retention >= 0.97
    assert math.isfinite(projected_passes)
    print("THROUGHPUT PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantics", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    args = parser.parse_args()
    assert args.semantics != args.throughput
    semantics() if args.semantics else throughput()


if __name__ == "__main__":
    main()

