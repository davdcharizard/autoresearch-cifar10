import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import train


EXP = Path(__file__).resolve().parent
STRONG_PATH = EXP.parent / "022" / "preflight-corpus.pt"
WEAK_PATH = EXP.parent / "028" / "weak-corpus.pt"
HASHES = {
    STRONG_PATH: "e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946",
    WEAK_PATH: "ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032",
}


def file_hash(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_buffers(optimizer):
    return [
        optimizer.state[parameter]["momentum_buffer"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]


def total_norm(tensors):
    return math.sqrt(sum(t.detach().double().square().sum().item() for t in tensors))


def finite(model, optimizer):
    return all(torch.isfinite(p).all() for p in model.parameters()) and all(
        torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def percentile(values, q):
    return torch.tensor(values, dtype=torch.float64).quantile(q).item()


def main():
    observed_hashes = {}
    for path, expected in HASHES.items():
        assert path.is_file()
        observed_hashes[path.name] = file_hash(path)
        assert observed_hashes[path.name] == expected
    strong = torch.load(STRONG_PATH, map_location="cpu", weights_only=False)
    weak = torch.load(WEAK_PATH, map_location="cpu", weights_only=False)
    assert len(strong) == 200 and len(weak) == 64
    assert all(target.ndim == 1 for _, target in weak)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    source = train.ResNet(3, 10, 2).cuda().train()
    source_opt = torch.optim.SGD(
        source.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    for cpu_inputs, cpu_targets in strong:
        inputs = cpu_inputs.cuda()
        targets = cpu_targets.cuda()
        source_opt.zero_grad()
        loss = F.cross_entropy(source(inputs), targets)
        loss.backward()
        source_opt.step()
    torch.cuda.synchronize()
    assert len(optimizer_buffers(source_opt)) == 59

    boundary_model = copy.deepcopy(source.state_dict())
    boundary_optimizer = copy.deepcopy(source_opt.state_dict())
    control = train.ResNet(3, 10, 2).cuda().train()
    candidate = train.ResNet(3, 10, 2).cuda().train()
    control.load_state_dict(boundary_model)
    candidate.load_state_dict(boundary_model)
    control_opt = torch.optim.SGD(
        control.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    candidate_opt = torch.optim.SGD(
        candidate.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    control_opt.load_state_dict(copy.deepcopy(boundary_optimizer))
    candidate_opt.load_state_dict(copy.deepcopy(boundary_optimizer))
    for optimizer in (control_opt, candidate_opt):
        for group in optimizer.param_groups:
            group["lr"] = 0.01

    first_inputs = weak[0][0].cuda()
    with torch.no_grad():
        logits_before_control = control(first_inputs)
        logits_before_candidate = candidate(first_inputs)
    assert torch.equal(logits_before_control, logits_before_candidate)
    param_hash_before = tensor_hash(candidate.named_parameters())
    buffer_hash_before = tensor_hash(candidate.named_buffers())
    gradient_hash_before = tensor_hash(
        (name, parameter.grad)
        for name, parameter in candidate.named_parameters()
        if parameter.grad is not None
    )
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state().clone()
    group_config_before = [
        {key: value for key, value in group.items() if key != "params"}
        for group in candidate_opt.param_groups
    ]
    parameter_ids_before = [
        id(parameter)
        for group in candidate_opt.param_groups
        for parameter in group["params"]
    ]
    reset_count = train.reset_sgd_momentum(candidate_opt)
    torch.cuda.synchronize()
    assert reset_count == 59
    assert all(torch.count_nonzero(buffer).item() == 0 for buffer in optimizer_buffers(candidate_opt))
    assert tensor_hash(candidate.named_parameters()) == param_hash_before
    assert tensor_hash(candidate.named_buffers()) == buffer_hash_before
    assert tensor_hash(
        (name, parameter.grad)
        for name, parameter in candidate.named_parameters()
        if parameter.grad is not None
    ) == gradient_hash_before
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)
    assert [
        {key: value for key, value in group.items() if key != "params"}
        for group in candidate_opt.param_groups
    ] == group_config_before
    assert [
        id(parameter)
        for group in candidate_opt.param_groups
        for parameter in group["params"]
    ] == parameter_ids_before
    with torch.no_grad():
        assert torch.equal(logits_before_candidate, candidate(first_inputs))

    report = {
        "corpus_hashes": observed_hashes,
        "boundary_model_hash": tensor_hash(source.state_dict().items()),
        "reset_count": reset_count,
        "buffer_norm_before": total_norm(optimizer_buffers(control_opt)),
        "buffer_norm_after": total_norm(optimizer_buffers(candidate_opt)),
        "candidate_only_concentration_steps": [],
        "control_losses": [],
        "candidate_losses": [],
        "update_ratios": [],
        "relative_parameter_updates": [],
        "own_median_ratios": [],
        "momentum_difference_norms": [],
    }
    for step, (cpu_inputs, cpu_targets) in enumerate(weak, start=1):
        inputs = cpu_inputs.cuda()
        targets = cpu_targets.cuda()
        arm_data = []
        for model, optimizer in ((control, control_opt), (candidate, candidate_opt)):
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            before = [p.detach().clone() for p in model.parameters()]
            parameter_norm = total_norm(before)
            if model is candidate and step == 1:
                expected = [
                    old - 0.01 * (p.grad.detach() + 1e-4 * old)
                    for old, p in zip(before, model.parameters(), strict=True)
                ]
            optimizer.step()
            if model is candidate and step == 1:
                assert all(
                    torch.allclose(p, e, atol=1e-7, rtol=1e-6)
                    for p, e in zip(model.parameters(), expected, strict=True)
                )
            updates = [
                old - p.detach()
                for old, p in zip(before, model.parameters(), strict=True)
            ]
            counts = torch.bincount(logits.argmax(1), minlength=10)
            arm_data.append((loss.item(), total_norm(updates), parameter_norm, counts))
        c_loss, c_update, _, c_counts = arm_data[0]
        k_loss, k_update, k_parameter, k_counts = arm_data[1]
        report["control_losses"].append(c_loss)
        report["candidate_losses"].append(k_loss)
        report["update_ratios"].append(k_update / c_update)
        report["relative_parameter_updates"].append(k_update / k_parameter)
        if len(report["relative_parameter_updates"]) > 16:
            prior = report["relative_parameter_updates"][-17:-1]
            median = torch.tensor(prior).median().item()
            report["own_median_ratios"].append((k_update / k_parameter) / median)
        report["momentum_difference_norms"].append(
            total_norm(
                control_buffer - candidate_buffer
                for control_buffer, candidate_buffer in zip(
                    optimizer_buffers(control_opt),
                    optimizer_buffers(candidate_opt),
                    strict=True,
                )
            )
        )
        if k_counts.max().item() > 0.95 * 128 and c_counts.max().item() <= 0.95 * 128:
            report["candidate_only_concentration_steps"].append(step)
        assert finite(control, control_opt) and finite(candidate, candidate_opt)

    def ema(values, beta=0.95):
        value = 0.0
        for item in values:
            value = beta * value + (1 - beta) * item
        return value / (1 - beta ** len(values))

    report["loss_ema_ratio"] = ema(report["candidate_losses"]) / ema(report["control_losses"])
    report["update_ratio_max"] = max(report["update_ratios"])
    report["relative_parameter_update_max"] = max(report["relative_parameter_updates"])
    report["own_median_ratio_max"] = max(report["own_median_ratios"])
    report["first_update_ratio"] = report["update_ratios"][0]
    report["momentum_decay_ratio_step_44"] = report["momentum_difference_norms"][43] / report["momentum_difference_norms"][0]
    report_path = EXP / "preflight-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    with report_path.open() as handle:
        os.fsync(handle.fileno())

    assert not report["candidate_only_concentration_steps"]
    assert report["loss_ema_ratio"] <= 1.5
    assert report["update_ratio_max"] <= 5.0
    assert report["relative_parameter_update_max"] <= 0.25
    assert report["own_median_ratio_max"] <= 5.0
    print(json.dumps({key: value for key, value in report.items() if key not in {"control_losses", "candidate_losses", "update_ratios", "relative_parameter_updates", "own_median_ratios", "momentum_difference_norms"}}, indent=2))
    print("preflight=pass")


if __name__ == "__main__":
    main()
