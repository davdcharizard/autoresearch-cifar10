import json
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import train  # noqa: E402


def control_module():
    source = subprocess.check_output(
        ["git", "show", "7c1e7d8:train.py"], cwd=ROOT, text=True
    )
    module = types.ModuleType("control_train_039")
    module.__file__ = str(ROOT / "train.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def pool(x):
    avg = x.mean((2, 3))
    rms = torch.linalg.vector_norm(x, dim=(2, 3)) / 8
    return torch.lerp(avg, rms, 1 / 64)


def main():
    failures = []
    control = control_module()
    torch.manual_seed(42)
    accepted = control.ResNet(
        control.NUM_BLOCKS, control.NUM_CLASSES, control.WIDTH_MULTIPLIER
    )
    accepted_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = train.ResNet(
        train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER
    )
    candidate_rng = torch.get_rng_state().clone()
    state_equal = all(
        torch.equal(accepted.state_dict()[k], candidate.state_dict()[k])
        for k in accepted.state_dict()
    )
    inventory = (
        sum(isinstance(m, torch.nn.Conv2d) for m in candidate.modules()),
        sum(p.numel() for p in candidate.parameters()),
    )
    if (
        not state_equal
        or not torch.equal(accepted_rng, candidate_rng)
        or inventory != (19, 1_073_962)
    ):
        failures.append("construction")

    cases = {
        "zero": torch.zeros(3, 5, 8, 8, dtype=torch.float64),
        "constant": torch.ones(3, 5, 8, 8, dtype=torch.float64),
        "one_hot": F.pad(torch.ones(3, 5, 1, 1, dtype=torch.float64), (0, 7, 0, 7)),
        "random": torch.rand(3, 5, 8, 8, dtype=torch.float64),
    }
    oracle = {}
    for name, raw in cases.items():
        x = raw.requires_grad_(True)
        out = pool(x)
        out.sum().backward()
        avg = raw.mean((2, 3))
        ratio = out.detach() / avg.clamp_min(1e-30)
        oracle[name] = {
            "finite": bool(torch.isfinite(x.grad).all()),
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "grad_min": float(x.grad.min()),
            "grad_max": float(x.grad.max()),
        }
        if not oracle[name]["finite"] or (
            name != "zero"
            and (ratio.min() < 1 - 1e-12 or ratio.max() > 71 / 64 + 1e-12)
        ):
            failures.append(f"oracle {name}")
        if x.grad.min() < 63 / 4096 - 1e-12 or x.grad.max() > 71 / 4096 + 1e-12:
            failures.append(f"jacobian {name}")

    report = {
        "status": "failed" if failures else "pass",
        "failures": failures,
        "state_equal": state_equal,
        "inventory": inventory,
        "oracle": oracle,
    }
    path = Path(__file__).with_name("math-report.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps({"status": "pass", "report": str(path)}))


if __name__ == "__main__":
    main()
