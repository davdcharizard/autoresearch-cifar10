import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

import train as candidate


BASELINE_COMMIT = "7c1e7d8"


def load_baseline():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:train.py"], text=True
    )
    namespace = {"__name__": "baseline_train", "__file__": "baseline_train.py"}
    exec(compile(source, "baseline_train.py", "exec"), namespace)
    return namespace


def construct(model_class):
    torch.manual_seed(42)
    model = model_class(3, 10, 2)
    return model, torch.random.get_rng_state().clone()


def verify_shared_initialization(baseline):
    reference, reference_rng = construct(baseline["ResNet"])
    trial, trial_rng = construct(candidate.ResNet)
    reference_modules = dict(reference.named_modules())
    trial_modules = dict(trial.named_modules())
    randomized_names = [
        name
        for name, module in reference_modules.items()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    assert randomized_names == [
        name
        for name, module in trial_modules.items()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    for name in randomized_names:
        reference_module = reference_modules[name]
        trial_module = trial_modules[name]
        assert torch.equal(reference_module.weight, trial_module.weight), name
        if reference_module.bias is not None:
            assert torch.equal(reference_module.bias, trial_module.bias), name
    assert torch.equal(reference_rng, trial_rng)
    print("shared Conv/Linear tensors and post-construction CPU RNG: bitwise equal")


def zero_residual(block):
    nn.init.zeros_(block.conv1.weight)
    nn.init.zeros_(block.conv2.weight)
    block.eval()


def verify_shortcuts():
    generator = torch.Generator().manual_seed(12012)

    first = candidate.BasicBlock(32, 32, preactivate_shortcut=True)
    zero_residual(first)
    first_input = torch.randn(4, 32, 32, 32, generator=generator)
    first_expected = F.relu(first.bn1(first_input))
    assert torch.equal(first(first_input), first_expected)

    ordinary = candidate.BasicBlock(32, 32)
    zero_residual(ordinary)
    ordinary_input = torch.randn(4, 32, 32, 32, generator=generator)
    ordinary_input[0, 0, 0, 0] = -3.0
    assert torch.equal(ordinary(ordinary_input), ordinary_input)
    assert ordinary(ordinary_input)[0, 0, 0, 0] < 0

    for in_channels, out_channels, spatial in ((32, 64, 32), (64, 128, 16)):
        transition = candidate.BasicBlock(in_channels, out_channels, stride=2)
        zero_residual(transition)
        transition_input = torch.randn(
            4, in_channels, spatial, spatial, generator=generator
        )
        preactivated = F.relu(transition.bn1(transition_input))
        expected = preactivated[:, :, ::2, ::2]
        expected = F.pad(
            expected, (0, 0, 0, 0, 0, out_channels - in_channels)
        )
        assert torch.equal(transition(transition_input), expected)
    print("first, ordinary, and transition shortcut semantics: exact")


def verify_structure_and_gradients():
    torch.manual_seed(42)
    model = candidate.ResNet(3, 10, 2)
    blocks = [module for module in model.modules() if isinstance(module, candidate.BasicBlock)]
    assert len(blocks) == 9
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_073_962
    assert sum(isinstance(module, nn.Conv2d) for module in model.modules()) == 19
    assert sum(isinstance(module, nn.BatchNorm2d) for module in model.modules()) == 19
    assert [block.bn1.num_features for block in blocks] == [
        32,
        32,
        32,
        32,
        64,
        64,
        64,
        128,
        128,
    ]
    assert [block.bn2.num_features for block in blocks] == [
        32,
        32,
        32,
        64,
        64,
        64,
        128,
        128,
        128,
    ]
    assert [block.preactivate_shortcut for block in blocks] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len(optimizer_parameters) == len(list(model.parameters()))
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter) for parameter in model.parameters()
    }
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4

    inputs = torch.randn(128, 3, 32, 32)
    hard_targets = torch.randint(0, 10, (128,))
    soft_targets = F.one_hot(torch.randint(0, 10, (128,)), 10).float()
    for targets in (hard_targets, soft_targets):
        optimizer.zero_grad()
        logits = model(inputs)
        assert logits.shape == (128, 10)
        assert torch.isfinite(logits).all()
        loss = F.cross_entropy(logits, targets)
        assert torch.isfinite(loss)
        loss.backward()
        for index, block in enumerate(blocks):
            for name, parameter in (
                ("conv1", block.conv1.weight),
                ("conv2", block.conv2.weight),
                ("bn1.weight", block.bn1.weight),
                ("bn1.bias", block.bn1.bias),
                ("bn2.weight", block.bn2.weight),
                ("bn2.bias", block.bn2.bias),
            ):
                assert parameter.grad is not None, (index, name)
                assert torch.isfinite(parameter.grad).all(), (index, name)
                assert torch.count_nonzero(parameter.grad), (index, name)
    print("structure, initialization, optimizer membership, targets, gradients: pass")


def main():
    baseline = load_baseline()
    verify_shared_initialization(baseline)
    verify_shortcuts()
    verify_structure_and_gradients()
    print("PREACTIVATION STRUCTURAL GATE: PASS")


if __name__ == "__main__":
    main()
