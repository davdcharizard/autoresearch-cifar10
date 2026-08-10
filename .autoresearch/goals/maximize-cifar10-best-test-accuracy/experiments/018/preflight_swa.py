import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent


def clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested(item) for item in value)
    return copy.deepcopy(value)


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def parameter_vector(parameters):
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def weak_transform():
    mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def arithmetic_gate():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device("cuda")

    small_parameters = (
        torch.nn.Parameter(torch.zeros(3, 4, device=device)),
        torch.nn.Parameter(torch.zeros(5, device=device)),
    )
    accumulator = train.SWAAccumulator(small_parameters)
    references = []
    durations = []
    for index in range(7):
        with torch.no_grad():
            small_parameters[0].copy_(
                torch.arange(12, device=device).reshape(3, 4) * (index + 1) / 17
            )
            small_parameters[1].copy_(torch.arange(5, device=device) * (index + 2) / 13)
        references.append(parameter_vector(small_parameters).double())
        durations.append(
            train.timed_swa_snapshot(accumulator, small_parameters, 0.86 + 0.01 * index)
        )

    reference_mean = torch.stack(references).mean(0)
    torch.testing.assert_close(
        accumulator.average.double(), reference_mean, rtol=2e-6, atol=1e-6
    )
    assert accumulator.count == 7
    assert accumulator.median_consecutive_rms >= train.SWA_MIN_CONSECUTIVE_RMS
    assert accumulator.first_last_rms >= train.SWA_MIN_FIRST_LAST_RMS
    assert all(duration >= 0 for duration in durations)

    accumulator.install(small_parameters)
    torch.testing.assert_close(
        parameter_vector(small_parameters).double(),
        reference_mean,
        rtol=2e-6,
        atol=1e-6,
    )
    try:
        accumulator.install(tuple(reversed(small_parameters)))
    except RuntimeError:
        reorder_rejected = True
    else:
        reorder_rejected = False
    assert reorder_rejected

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = train.ResNet(
        train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER
    ).to(device)
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    optimizer.state[parameters[0]]["momentum_buffer"] = torch.ones_like(parameters[0])
    parameters[0].grad = torch.full_like(parameters[0], 0.125)

    state_before = clone_nested(model.state_dict())
    optimizer_before = clone_nested(optimizer.state_dict())
    grad_before = parameters[0].grad.clone()
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state().clone()
    real_accumulator = train.SWAAccumulator(parameters)
    real_duration = train.timed_swa_snapshot(real_accumulator, parameters, 0.86)

    assert_nested_equal(model.state_dict(), state_before)
    assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert torch.equal(parameters[0].grad, grad_before)
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)
    assert real_accumulator.average.dtype == torch.float32
    assert not real_accumulator.average.requires_grad

    result = {
        "status": "pass",
        "reference_snapshots": accumulator.count,
        "max_reference_error": float(
            (accumulator.average.double() - reference_mean).abs().max().item()
        ),
        "median_consecutive_rms": accumulator.median_consecutive_rms,
        "first_last_rms": accumulator.first_last_rms,
        "snapshot_seconds": durations,
        "real_snapshot_seconds": real_duration,
        "online_state_equal": True,
        "optimizer_state_equal": True,
        "rng_equal": True,
        "reorder_rejected": reorder_rejected,
    }
    (OUTPUT_DIR / "preflight-arithmetic.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("ARITHMETIC_GATE_PASS")
    print(json.dumps(result, indent=2))


def refresh_gate():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device("cuda")
    model = train.ResNet(
        train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER
    ).to(device)
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=train.LR,
        momentum=train.MOMENTUM,
        weight_decay=train.WEIGHT_DECAY,
    )
    optimizer.state[parameters[0]]["momentum_buffer"] = torch.ones_like(parameters[0])
    parameters[0].grad = torch.full_like(parameters[0], 0.125)

    accumulator = train.SWAAccumulator(parameters)
    for index in range(7):
        with torch.no_grad():
            scale = (index + 1) * 2e-4
            for parameter_index, parameter in enumerate(parameters):
                parameter.add_(scale * ((parameter_index % 3) - 1))
        train.timed_swa_snapshot(accumulator, parameters, 0.86 + index * 0.015)

    expected_average = accumulator.average.clone()
    optimizer_before = clone_nested(optimizer.state_dict())
    grad_before = parameters[0].grad.clone()
    install_seconds, momenta = train.timed_swa_install(accumulator, model, parameters)
    torch.testing.assert_close(
        parameter_vector(parameters), expected_average, rtol=0, atol=0
    )
    assert all(module.momentum is None for module, _ in momenta)
    assert all(original == 0.1 for _, original in momenta)
    for module, _ in momenta:
        assert torch.count_nonzero(module.running_mean).item() == 0
        assert torch.all(module.running_var == 1).item()
        assert module.num_batches_tracked.item() == 0

    parameters_before_refresh = parameter_vector(parameters).clone()
    loader = train.make_train_loader(weak_transform())
    refresh_seconds, refresh_batches = train.refresh_batch_norm(
        model, loader, device, max_batches=780
    )
    finish_seconds, bn_batches = train.finish_batch_norm_refresh(momenta, parameters)

    assert refresh_batches == 780
    assert bn_batches == refresh_batches
    assert refresh_batches > len(loader)
    assert all(module.momentum == 0.1 for module, _ in momenta)
    torch.testing.assert_close(
        parameter_vector(parameters), parameters_before_refresh, rtol=0, atol=0
    )
    assert torch.equal(parameters[0].grad, grad_before)
    assert_nested_equal(optimizer.state_dict(), optimizer_before)
    assert any(torch.count_nonzero(module.running_mean).item() for module, _ in momenta)
    assert all(torch.all(module.running_var > 0).item() for module, _ in momenta)

    stopped_workers = len(train.shutdown_train_loader(loader))
    assert stopped_workers == train.NUM_WORKERS
    result = {
        "status": "pass",
        "refresh_batches": refresh_batches,
        "bn_batches": bn_batches,
        "loader_batches": len(loader),
        "iterator_recreated": refresh_batches > len(loader),
        "install_seconds": install_seconds + finish_seconds,
        "refresh_seconds": refresh_seconds,
        "parameters_equal": True,
        "optimizer_state_equal": True,
        "gradients_equal": True,
        "momenta_restored": True,
        "stopped_workers": stopped_workers,
    }
    (OUTPUT_DIR / "preflight-refresh.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("REFRESH_GATE_PASS")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arithmetic", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.arithmetic == args.refresh:
        parser.error("select exactly one gate")
    if args.arithmetic:
        arithmetic_gate()
    else:
        refresh_gate()


if __name__ == "__main__":
    main()
