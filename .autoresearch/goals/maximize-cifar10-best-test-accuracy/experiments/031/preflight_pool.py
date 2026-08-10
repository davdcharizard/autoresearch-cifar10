import hashlib
import json
import math
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import train


EXP = Path(__file__).resolve().parent
STRONG_PATH = EXP.parent / "022" / "preflight-corpus.pt"
WEAK_PATH = EXP.parent / "028" / "weak-corpus.pt"
EXPECTED = {
    STRONG_PATH: "e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946",
    WEAK_PATH: "ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032",
}
C = 0.10 * train.POOL_RESIDUAL_SCALE


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def control_forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    avg = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
    self._last_pool_avg = avg.detach()
    self._last_pool_residual = torch.zeros_like(avg.detach())
    return self.fc(avg)


def norm_tensors(tensors):
    return math.sqrt(sum(t.detach().double().square().sum().item() for t in tensors))


def finite_model(model, optimizer):
    return all(torch.isfinite(p).all() for p in model.parameters()) and all(
        torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def percentile(values, q):
    return torch.tensor(values, dtype=torch.float64).quantile(q).item()


def main():
    assert torch.cuda.is_available()
    hashes = {}
    for path, expected in EXPECTED.items():
        assert path.is_file()
        hashes[path.name] = file_hash(path)
        assert hashes[path.name] == expected
    strong = torch.load(STRONG_PATH, map_location="cpu", weights_only=False)
    weak = torch.load(WEAK_PATH, map_location="cpu", weights_only=False)
    assert len(strong) == 200 and len(weak) == 64
    assert sum(target.ndim == 2 for _, target in strong) == 106
    assert all(inputs.shape == (128, 3, 32, 32) for inputs, _ in strong + weak)
    assert all(target.ndim == 1 for _, target in weak)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    control = train.ResNet(3, 10, 2).cuda()
    control.forward = types.MethodType(control_forward, control)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    candidate = train.ResNet(3, 10, 2).cuda()
    assert state_hash(control) == state_hash(candidate)
    assert sum(p.numel() for p in candidate.parameters()) == 1_073_962
    control_opt = torch.optim.SGD(
        control.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    candidate_opt = torch.optim.SGD(
        candidate.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )

    constant = torch.full((4, 128), 2.0, device="cuda")
    assert torch.equal(constant + C * (constant - constant), constant)
    random_map = torch.randn(4, 128, 8, 8, device="cuda")
    avg = random_map.mean((2, 3))
    residual = random_map.amax((2, 3)) - avg
    reference = (1.0 - C) * avg.double() + C * random_map.double().amax((2, 3))
    actual = avg.double() + C * residual.double()
    assert torch.allclose(actual, reference, rtol=1e-6, atol=1e-7)

    report = {
        "corpus_hashes": hashes,
        "state_hash": state_hash(candidate),
        "scale": train.POOL_RESIDUAL_SCALE,
        "coefficient": C,
        "strong_steps": len(strong),
        "weak_steps": len(weak),
        "ratios": [],
        "update_ratios": [],
        "classifier_grad_ratios": [],
        "candidate_only_concentration_steps": [],
        "losses": {"control_strong": [], "candidate_strong": [], "control_weak": [], "candidate_weak": []},
    }
    first = None
    global_step = 0
    for phase, corpus in (("strong", strong), ("weak", weak)):
        for phase_step, (cpu_inputs, cpu_targets) in enumerate(corpus, start=1):
            global_step += 1
            inputs = cpu_inputs.cuda()
            targets = cpu_targets.cuda()
            snapshots = []
            step_data = []
            for model, optimizer, arm in (
                (control, control_opt, "control"),
                (candidate, candidate_opt, "candidate"),
            ):
                model.train()
                for group in optimizer.param_groups:
                    group["lr"] = 0.1 if phase == "strong" else 0.01
                optimizer.zero_grad()
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                grad_norm = norm_tensors(p.grad for p in model.parameters())
                fc_grad = model.fc.weight.grad.norm().item()
                before = [p.detach().clone() for p in model.parameters()]
                optimizer.step()
                update_norm = norm_tensors(
                    old - parameter.detach()
                    for old, parameter in zip(before, model.parameters(), strict=True)
                )
                counts = torch.bincount(logits.argmax(1), minlength=10)
                step_data.append((loss.item(), grad_norm, fc_grad, update_norm, counts))
                report["losses"][f"{arm}_{phase}"].append(loss.item())
                snapshots.append(before)
            c_loss, _, c_fc_grad, c_update, c_counts = step_data[0]
            k_loss, _, k_fc_grad, k_update, k_counts = step_data[1]
            report["update_ratios"].append(k_update / c_update)
            report["classifier_grad_ratios"].append(k_fc_grad / c_fc_grad)
            if k_counts.max().item() > 0.95 * 128 and c_counts.max().item() <= 0.95 * 128:
                report["candidate_only_concentration_steps"].append(global_step)
            ratio = (
                C
                * candidate._last_pool_residual.double().square().mean().sqrt()
                / candidate._last_pool_avg.double().square().mean().sqrt()
            ).item()
            per_example = (
                C
                * candidate._last_pool_residual.norm(dim=1)
                / candidate._last_pool_avg.norm(dim=1).clamp_min(1e-12)
            )
            if first is None:
                with torch.no_grad():
                    control.eval()
                    candidate.eval()
                    control_logits = control(inputs)
                    candidate.reset_pool_diagnostics()
                    candidate_logits = candidate(inputs)
                    candidate_logits_repeat = candidate(inputs)
                    diag_ratio = candidate.pool_diagnostic_ratio()
                first = {
                    "loss_ratio": k_loss / c_loss,
                    "logit_cosine": F.cosine_similarity(
                        control_logits.flatten(), candidate_logits.flatten(), dim=0
                    ).item(),
                    "classifier_grad_ratio": k_fc_grad / c_fc_grad,
                    "update_ratio": k_update / c_update,
                    "eval_repeat_equal": torch.equal(candidate_logits, candidate_logits_repeat),
                    "eval_diag_ratio": diag_ratio,
                }
            if phase_step % (20 if phase == "strong" else 16) == 0 or phase_step == len(corpus):
                report["ratios"].append(
                    {"phase": phase, "step": phase_step, "aggregate": ratio, "max_per_example": per_example.max().item()}
                )
            assert finite_model(control, control_opt) and finite_model(candidate, candidate_opt)

    def ema(values, beta=0.95):
        value = 0.0
        for item in values:
            value = beta * value + (1 - beta) * item
        return value / (1 - beta ** len(values))

    report["first"] = first
    report["strong_ema_ratio"] = ema(report["losses"]["candidate_strong"]) / ema(report["losses"]["control_strong"])
    report["weak_ema_ratio"] = ema(report["losses"]["candidate_weak"]) / ema(report["losses"]["control_weak"])
    report["update_ratio_p95"] = percentile(report["update_ratios"], 0.95)
    report["update_ratio_max"] = max(report["update_ratios"])
    report["classifier_grad_ratio_p95"] = percentile(report["classifier_grad_ratios"], 0.95)
    report["max_aggregate_ratio"] = max(item["aggregate"] for item in report["ratios"])
    report["max_per_example_ratio"] = max(item["max_per_example"] for item in report["ratios"])
    report_path = EXP / "preflight-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    with report_path.open() as handle:
        os.fsync(handle.fileno())

    assert first["logit_cosine"] >= 0.995
    assert 0.8 <= first["classifier_grad_ratio"] <= 1.25
    assert 0.8 <= first["update_ratio"] <= 1.25
    assert first["eval_repeat_equal"]
    assert not report["candidate_only_concentration_steps"]
    assert report["strong_ema_ratio"] <= 1.10
    assert report["weak_ema_ratio"] <= 1.10
    assert report["update_ratio_p95"] <= 1.25
    assert report["update_ratio_max"] <= 1.50
    assert report["classifier_grad_ratio_p95"] <= 1.30
    assert report["max_aggregate_ratio"] <= 0.25
    assert report["max_per_example_ratio"] <= 0.75
    print(json.dumps({key: value for key, value in report.items() if key not in {"losses", "update_ratios", "classifier_grad_ratios"}}, indent=2))
    print("preflight=pass")


if __name__ == "__main__":
    main()
