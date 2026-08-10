import ast
import hashlib
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from prepare import DATASET_DIR


SOURCE = (ROOT / "train.py").read_text()


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, tensor in items:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model_class():
    tree = ast.parse(SOURCE)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"BasicBlock", "ResNet"}
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "nn": nn, "F": F, "init": init}
    exec(compile(module, str(ROOT / "train.py"), "exec"), namespace)
    return namespace["ResNet"]


def run_once(resnet):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = resnet(3, 10, 2)
    state_hash = tensor_hash(model.state_dict().items())
    cpu_rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    model = model.to("cuda")
    cuda_rng_hash = hashlib.sha256(
        torch.cuda.get_rng_state().cpu().numpy().tobytes()
    ).hexdigest()
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (1, 1, 1)),
        ]
    )
    dataset = datasets.CIFAR10(
        DATASET_DIR, train=True, download=False, transform=transform
    )
    subset = Subset(dataset, list(range(1024)))
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)
    input_digest = hashlib.sha256()
    avg_sq = torch.zeros((), dtype=torch.float64, device="cuda")
    residual_sq = torch.zeros((), dtype=torch.float64, device="cuda")
    count = 0
    with torch.no_grad():
        for inputs, _ in loader:
            input_digest.update(inputs.contiguous().numpy().tobytes())
            out = inputs.to("cuda")
            out = F.relu(model.bn1(model.conv1(out)))
            out = model.layer1(out)
            out = model.layer2(out)
            out = model.layer3(out)
            avg = out.mean(dim=(2, 3))
            residual = out.amax(dim=(2, 3)) - avg
            avg_sq += avg.double().square().sum()
            residual_sq += residual.double().square().sum()
            count += avg.numel()
    rms_avg = math.sqrt((avg_sq / count).item())
    rms_residual = math.sqrt((residual_sq / count).item())
    scale = min(1.0, rms_avg / max(rms_residual, 1e-12))
    return {
        "state_hash": state_hash,
        "cpu_rng_hash": cpu_rng_hash,
        "cuda_rng_hash": cuda_rng_hash,
        "input_hash": input_digest.hexdigest(),
        "count": count,
        "rms_avg": rms_avg,
        "rms_residual": rms_residual,
        "raw_ratio": rms_avg / rms_residual,
        "scale": scale,
        "added_ratio": 0.10 * scale * rms_residual / rms_avg,
    }


def main():
    assert torch.cuda.is_available()
    resnet = load_model_class()
    first = run_once(resnet)
    second = run_once(resnet)
    assert first == second
    assert first["count"] == 1024 * 128
    assert 0.0 < first["scale"] <= 1.0
    assert first["rms_residual"] > 0.0
    assert first["added_ratio"] <= 0.100001
    source_hash = hashlib.sha256(SOURCE.encode()).hexdigest()
    print(f"source_hash={source_hash}")
    for key, value in first.items():
        print(f"{key}={value}")
    print(f"frozen_scale_8sig={first['scale']:.8g}")
    print("calibration=pass")


if __name__ == "__main__":
    main()
