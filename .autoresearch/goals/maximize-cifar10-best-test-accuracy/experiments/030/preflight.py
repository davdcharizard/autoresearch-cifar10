import ast
import hashlib
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[5]
SOURCE = (ROOT / "train.py").read_text()
TREE = ast.parse(SOURCE)


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def schedule(progress, anneal_start_lr):
    lr = 0.1
    hold = 0.8
    minimum = 1e-4
    if progress <= hold:
        return lr
    cosine_progress = (progress - hold) / (1.0 - hold)
    return minimum + 0.5 * (anneal_start_lr - minimum) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )


def main():
    assignments = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ANNEAL_START_LR"
            for target in node.targets
        )
    ]
    loads = [
        node.lineno
        for node in ast.walk(TREE)
        if isinstance(node, ast.Name)
        and node.id == "ANNEAL_START_LR"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(assignments) == 1
    assert ast.literal_eval(assignments[0].value) == 0.02
    assert len(loads) == 1
    load_line = loads[0]
    assert "ANNEAL_START_LR - MIN_LR" in SOURCE.splitlines()[load_line - 1]
    assert "if progress <= LR_HOLD_FRACTION:" in SOURCE
    assert SOURCE.count("evaluator.evaluate(") == 1

    points = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.7,
        0.8,
        math.nextafter(0.8, 1.0),
        0.85,
        0.9,
        0.95,
        1.0,
    ]
    accepted = [schedule(point, 0.01) for point in points]
    candidate = [schedule(point, 0.02) for point in points]
    assert accepted[:6] == candidate[:6] == [0.1] * 6
    assert math.isclose(candidate[6], 0.02, rel_tol=0.0, abs_tol=1e-15)
    assert all(a >= b for a, b in zip(candidate[6:], candidate[7:]))
    assert accepted[-1] == candidate[-1] == 1e-4

    class_nodes = [
        node
        for node in TREE.body
        if isinstance(node, ast.ClassDef) and node.name in {"BasicBlock", "ResNet"}
    ]
    module = ast.Module(body=class_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "nn": nn, "F": F, "init": init}
    exec(compile(module, str(ROOT / "train.py"), "exec"), namespace)
    resnet = namespace["ResNet"]

    torch.manual_seed(42)
    control_model = resnet(3, 10, 2)
    control_rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    torch.manual_seed(42)
    candidate_model = resnet(3, 10, 2)
    candidate_rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    control_hash = tensor_hash(control_model.state_dict().items())
    candidate_hash = tensor_hash(candidate_model.state_dict().items())
    assert control_hash == candidate_hash
    assert control_rng_hash == candidate_rng_hash
    num_params = sum(parameter.numel() for parameter in control_model.parameters())
    assert num_params == 1_073_962

    control_optimizer = optim.SGD(
        control_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    candidate_optimizer = optim.SGD(
        candidate_model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4
    )
    torch.manual_seed(314159)
    for control_parameter, candidate_parameter in zip(
        control_model.parameters(), candidate_model.parameters(), strict=True
    ):
        buffer = 0.01 * torch.randn_like(control_parameter)
        control_optimizer.state[control_parameter]["momentum_buffer"] = buffer.clone()
        candidate_optimizer.state[candidate_parameter]["momentum_buffer"] = buffer.clone()

    torch.manual_seed(271828)
    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))
    before = [parameter.detach().clone() for parameter in control_model.parameters()]
    for model, optimizer in (
        (control_model, control_optimizer),
        (candidate_model, candidate_optimizer),
    ):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(inputs), targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()

    buffer_equal = all(
        torch.equal(
            control_optimizer.state[control_parameter]["momentum_buffer"],
            candidate_optimizer.state[candidate_parameter]["momentum_buffer"],
        )
        for control_parameter, candidate_parameter in zip(
            control_model.parameters(), candidate_model.parameters(), strict=True
        )
    )
    assert buffer_equal
    control_deltas = [
        prior - parameter.detach()
        for prior, parameter in zip(before, control_model.parameters(), strict=True)
    ]
    candidate_deltas = [
        prior - parameter.detach()
        for prior, parameter in zip(before, candidate_model.parameters(), strict=True)
    ]
    control_norm = torch.linalg.vector_norm(
        torch.cat([delta.flatten() for delta in control_deltas])
    ).item()
    candidate_norm = torch.linalg.vector_norm(
        torch.cat([delta.flatten() for delta in candidate_deltas])
    ).item()
    update_ratio = candidate_norm / control_norm
    assert math.isclose(update_ratio, 2.0, rel_tol=2e-3, abs_tol=2e-3)
    tensor_ratios = [
        candidate_delta.norm().item() / control_delta.norm().item()
        for control_delta, candidate_delta in zip(
            control_deltas, candidate_deltas, strict=True
        )
        if control_delta.norm().item() > 1e-6
    ]
    assert min(tensor_ratios) > 1.98 and max(tensor_ratios) < 2.02

    print(f"source_assignment_line={assignments[0].lineno}")
    print(f"source_load_line={load_line}")
    print(f"schedule_points={points}")
    print(f"accepted_lrs={accepted}")
    print(f"candidate_lrs={candidate}")
    print(f"state_hash={control_hash}")
    print(f"rng_hash={control_rng_hash}")
    print(f"num_params={num_params}")
    print(f"momentum_buffers_equal={buffer_equal}")
    print(f"aggregate_update_ratio={update_ratio:.9f}")
    print(f"tensor_update_ratio_min={min(tensor_ratios):.9f}")
    print(f"tensor_update_ratio_max={max(tensor_ratios):.9f}")
    print("preflight=pass")


if __name__ == "__main__":
    main()
