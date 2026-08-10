import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parent
STRONG = ROOT / ".autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight-corpus.pt"
REPORT = EXP / "mechanism-report.json"
sys.path.insert(0, str(ROOT))
import train  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_module():
    source = subprocess.check_output(["git", "show", "7c1e7d8:train.py"], cwd=ROOT, text=True)
    module = types.ModuleType("control_train_037")
    module.__file__ = str(ROOT / "train.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def seeded(arm):
    module = train if arm == "candidate" else control_module()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = module.ResNet(module.NUM_BLOCKS, module.NUM_CLASSES, module.WIDTH_MULTIPLIER)
    cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    return model.cuda(), module, cpu, cuda


def construction():
    c, _cm, ccpu, ccuda = seeded("control")
    x, _xm, xcpu, xcuda = seeded("candidate")
    cs, xs = c.state_dict(), x.state_dict()
    if list(cs) != list(xs) or any(not torch.equal(cs[k], xs[k]) for k in cs):
        raise RuntimeError("initial state mismatch")
    if not torch.equal(ccpu, xcpu) or not torch.equal(ccuda, xcuda):
        raise RuntimeError("RNG mismatch")
    modules = dict(x.named_modules())
    projected = [n for n, m in modules.items() if isinstance(m, train.MeanCenteredConv2d)]
    inventory = {
        "conv": sum(isinstance(m, torch.nn.Conv2d) for m in modules.values()),
        "bn": sum(isinstance(m, torch.nn.BatchNorm2d) for m in modules.values()),
        "linear": sum(isinstance(m, torch.nn.Linear) for m in modules.values()),
        "params": sum(p.numel() for p in x.parameters()),
        "projected": projected,
    }
    if inventory != {"conv": 19, "bn": 19, "linear": 1, "params": 1_073_962, "projected": ["conv1"]}:
        raise RuntimeError(f"inventory {inventory}")
    return inventory


def oracle():
    torch.manual_seed(370)
    module = train.MeanCenteredConv2d(3, 5, 3, padding=1, bias=False).double()
    inputs = torch.randn(4, 3, 8, 8, dtype=torch.float64, requires_grad=True)
    expected_inputs = inputs.detach().clone().requires_grad_(True)
    actual = module(inputs)
    centered = module.weight - module.weight.mean((1, 2, 3), keepdim=True)
    expected = F.conv2d(expected_inputs, centered, padding=1)
    probe = torch.randn_like(actual)
    actual.backward(probe, retain_graph=True)
    actual_input_grad = inputs.grad.clone()
    actual_weight_grad = module.weight.grad.clone()
    module.zero_grad(set_to_none=True)
    expected.backward(probe)
    result = {
        "output_error": float((actual - expected).abs().max()),
        "input_grad_error": float((actual_input_grad - expected_inputs.grad).abs().max()),
        "effective_mean_max": float(centered.mean((1, 2, 3)).abs().max()),
        "raw_norm": float(module.weight.norm()),
        "effective_norm": float(centered.norm()),
        "gradient_mean_max": float(actual_weight_grad.mean((1, 2, 3)).abs().max()),
    }
    if result["output_error"] > 1e-10 or result["input_grad_error"] > 1e-10:
        raise RuntimeError("oracle mismatch")
    if result["effective_mean_max"] > 1e-10 or result["effective_norm"] > result["raw_norm"] + 1e-12:
        raise RuntimeError("projection invariant")
    return result


def rms(t):
    return float(t.float().square().mean().sqrt())


def capture(model, inputs, targets):
    values = {}
    handles = [
        model.conv1.register_forward_hook(lambda _m, _a, o: values.__setitem__("pre", o.detach().clone())),
        model.bn1.register_forward_hook(lambda _m, _a, o: values.__setitem__("post", o.detach().clone())),
        model.fc.register_forward_pre_hook(lambda _m, a: values.__setitem__("pooled", a[0].detach().clone())),
    ]
    model.train()
    out = model(inputs.cuda(non_blocking=True))
    loss = F.cross_entropy(out, targets.cuda(non_blocking=True))
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    return {"pre_rms": rms(values["pre"]), "post_rms": rms(values["post"]), "pooled": values["pooled"].cpu(), "logits": out.detach().cpu(), "loss": float(loss), "class_share": float(out.argmax(1).bincount(minlength=10).max()) / out.shape[0]}


def relative_l2(a, b):
    return float((a - b).norm() / max(float(a.norm()), 1e-30))


def compare(c, x):
    return {"post_rms_relative": abs(x["post_rms"] / c["post_rms"] - 1), "pooled_relative_l2": relative_l2(c["pooled"], x["pooled"]), "logit_relative_l2": relative_l2(c["logits"], x["logits"]), "loss_ratio": x["loss"] / c["loss"], "logit_rms_ratio": rms(x["logits"]) / rms(c["logits"]), "control_share": c["class_share"], "candidate_share": x["class_share"]}


def train64(arm, batches):
    model, module, _cpu, _cuda = seeded(arm)
    opt = torch.optim.SGD(model.parameters(), lr=module.LR, momentum=module.MOMENTUM, weight_decay=module.WEIGHT_DECAY)
    for inputs, targets in batches[:64]:
        opt.zero_grad()
        out = model(inputs.cuda(non_blocking=True))
        loss = F.cross_entropy(out, targets.cuda(non_blocking=True))
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return model


def main():
    batches = torch.load(STRONG, map_location="cpu", weights_only=False)
    hard = next(b for b in batches if b[1].ndim == 1)
    soft = next(b for b in batches if b[1].ndim == 2)
    inventory = construction()
    oracle_report = oracle()
    cases = {}
    failures = []
    for name, batch in (("hard", hard), ("cutmix", soft)):
        c1, *_ = seeded("control")
        c2, *_ = seeded("control")
        x, *_ = seeded("candidate")
        c1_initial = capture(c1, *batch)
        c2_initial = capture(c2, *batch)
        x_initial = capture(x, *batch)
        initial_noise = compare(c1_initial, c2_initial)
        initial = compare(c1_initial, x_initial)
        c1_64 = train64("control", batches)
        c2_64 = train64("control", batches)
        x64 = train64("candidate", batches)
        c1_late = capture(c1_64, *batch)
        c2_late = capture(c2_64, *batch)
        x_late = capture(x64, *batch)
        late_noise = compare(c1_late, c2_late)
        late = compare(c1_late, x_late)
        for stage, noise, candidate in (("initial", initial_noise, initial), ("step64", late_noise, late)):
            survival = max(candidate["post_rms_relative"], candidate["pooled_relative_l2"], candidate["logit_relative_l2"])
            control_floor = max(noise["post_rms_relative"], noise["pooled_relative_l2"], noise["logit_relative_l2"])
            if survival < max(1e-4, 5 * control_floor):
                failures.append(f"{name} {stage} mechanism null {survival}/{control_floor}")
            if not 0.8 <= candidate["loss_ratio"] <= 1.2 or candidate["logit_rms_ratio"] >= 2:
                failures.append(f"{name} {stage} ratio failure")
            if candidate["candidate_share"] > 0.95 and candidate["control_share"] <= 0.95:
                failures.append(f"{name} {stage} candidate concentration")
        cases[name] = {"initial_control_noise": initial_noise, "initial_candidate": initial, "step64_control_noise": late_noise, "step64_candidate": late}
    report = {"status": "failed" if failures else "pass", "controller_sha256": sha(Path(__file__)), "train_sha256": sha(ROOT / "train.py"), "corpus_sha256": sha(STRONG), "inventory": inventory, "oracle": oracle_report, "cases": cases, "failures": failures}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    with REPORT.open("rb") as f:
        os.fsync(f.fileno())
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps({"status": "pass", "report": str(REPORT)}))


if __name__ == "__main__":
    main()
