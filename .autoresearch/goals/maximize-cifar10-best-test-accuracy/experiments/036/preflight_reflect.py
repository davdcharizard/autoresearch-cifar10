import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import v2


ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parent
CORPUS = EXP / "paired-corpus.pt"
REPORT = EXP / "preflight-report.json"
sys.path.insert(0, str(ROOT))
import train  # noqa: E402
from prepare import DATASET_DIR  # noqa: E402

MEAN = (0.4914, 0.4822, 0.4465)
STD = (1, 1, 1)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_batches(batches):
    h = hashlib.sha256()
    for x, y in batches:
        h.update(x.contiguous().numpy().tobytes())
        h.update(y.contiguous().numpy().tobytes())
    return h.hexdigest()


def tfm(mode, strong):
    ops = [
        transforms.RandomCrop(32, padding=4, padding_mode=mode),
        transforms.RandomHorizontalFlip(),
    ]
    if strong:
        ops.append(transforms.RandAugment(num_ops=1, magnitude=7))
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def offset_gate(data):
    image, label = data[0]
    reports = []
    for top in range(9):
        for left in range(9):
            control = TF.crop(TF.pad(image, 4, fill=0, padding_mode="constant"), top, left, 32, 32)
            candidate = TF.crop(TF.pad(image, 4, padding_mode="reflect"), top, left, 32, 32)
            c = TF.to_tensor(control)
            x = TF.to_tensor(candidate)
            expected = torch.zeros(32, 32, dtype=torch.bool)
            for row in range(32):
                for col in range(32):
                    sy, sx = top + row - 4, left + col - 4
                    expected[row, col] = not (0 <= sy < 32 and 0 <= sx < 32)
            actual = (c != x).any(0)
            if (actual & ~expected).any():
                raise RuntimeError(f"interior changed at offset {top},{left}")
            if (top, left) == (4, 4) and not torch.equal(c, x):
                raise RuntimeError("center crop differs")
            reports.append({"top": top, "left": left, "expected_area": float(expected.float().mean()), "changed_area": float(actual.float().mean())})
    return {"label": int(label), "offsets": reports, "mean_expected_area": statistics.mean(r["expected_area"] for r in reports), "mean_changed_area": statistics.mean(r["changed_area"] for r in reports)}


def make_corpus():
    data = datasets.CIFAR10(DATASET_DIR, train=True, download=False)
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(360)).tolist()[: 48 * 128]
    cutmix = v2.CutMix(alpha=1.0, num_classes=10)
    arms = {"control": [], "candidate": []}
    rng_equal = True
    changed = []
    for batch_index in range(48):
        strong = batch_index < 32
        control_tf = tfm("constant", strong)
        candidate_tf = tfm("reflect", strong)
        c_samples, x_samples = [], []
        for index in indices[batch_index * 128 : (batch_index + 1) * 128]:
            image, label = data[index]
            state = torch.Generator().manual_seed(360000 + index + batch_index * 50000).get_state().clone()
            torch.set_rng_state(state.clone())
            c = control_tf(image)
            c_after = torch.get_rng_state().clone()
            torch.set_rng_state(state.clone())
            x = candidate_tf(image)
            x_after = torch.get_rng_state().clone()
            rng_equal &= torch.equal(c_after, x_after)
            c_samples.append((c, label))
            x_samples.append((x, label))
            changed.append(float((c != x).float().mean()))
        c_batch = default_collate(c_samples)
        x_batch = default_collate(x_samples)
        if strong and batch_index % 2 == 1:
            mix_state = torch.Generator().manual_seed(880000 + batch_index).get_state().clone()
            torch.set_rng_state(mix_state.clone())
            c_batch = cutmix(*c_batch)
            c_after = torch.get_rng_state().clone()
            torch.set_rng_state(mix_state.clone())
            x_batch = cutmix(*x_batch)
            x_after = torch.get_rng_state().clone()
            rng_equal &= torch.equal(c_after, x_after)
        if not torch.equal(c_batch[1], x_batch[1]):
            raise RuntimeError(f"target mismatch batch {batch_index}")
        arms["control"].append(tuple(t.contiguous() for t in c_batch))
        arms["candidate"].append(tuple(t.contiguous() for t in x_batch))
    if not rng_equal:
        raise RuntimeError("paired transform RNG mismatch")
    metadata = {"indices_sha256": hashlib.sha256(bytes(str(indices), "utf-8")).hexdigest(), "rng_equal": True, "strong_batches": 32, "strong_hard": 16, "strong_cutmix": 16, "weak_batches": 16, "mean_changed_fraction": statistics.mean(changed)}
    torch.save({"arms": arms, "metadata": metadata}, CORPUS)
    with CORPUS.open("rb") as f:
        os.fsync(f.fileno())
    return arms, {**metadata, "file_sha256": sha(CORPUS), "control_tensor_sha256": digest_batches(arms["control"]), "candidate_tensor_sha256": digest_batches(arms["candidate"])}


def norm(tensors):
    return math.sqrt(sum(float(t.float().square().sum()) for t in tensors))


def share(outputs):
    return float(outputs.argmax(1).bincount(minlength=10).max()) / outputs.shape[0]


def weak_lr(index):
    progress = 0.8 + 0.2 * index / 15
    cosine = (progress - 0.8) / 0.2
    return 1e-4 + 0.5 * (0.01 - 1e-4) * (1 + math.cos(math.pi * cosine))


def trajectory(arm):
    corpus = torch.load(CORPUS, map_location="cpu", weights_only=False)["arms"]
    source = corpus["candidate" if arm == "candidate" else "control"]
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = train.ResNet(train.NUM_BLOCKS, train.NUM_CLASSES, train.WIDTH_MULTIPLIER).cuda().train()
    params = list(model.parameters())
    opt = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-4)
    records, history = [], []
    emas, counts = {"strong": 0.0, "weak": 0.0}, {"strong": 0, "weak": 0}
    for step, (cpu_x, cpu_y) in enumerate(source, 1):
        phase = "strong" if step <= 32 else "weak"
        opt.param_groups[0]["lr"] = 0.1 if phase == "strong" else weak_lr(step - 33)
        x, y = cpu_x.cuda(non_blocking=True), cpu_y.cuda(non_blocking=True)
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        starts = [p.detach().clone() for p in params]
        pnorm = norm(starts)
        gnorm = norm([p.grad for p in params if p.grad is not None])
        opt.step()
        torch.cuda.synchronize()
        unorm = norm([p.detach() - s for p, s in zip(params, starts, strict=True)])
        preceding = unorm / statistics.median(history[-16:]) if len(history) >= 16 else None
        history.append(unorm)
        value = float(loss)
        emas[phase] = 0.95 * emas[phase] + 0.05 * value
        counts[phase] += 1
        records.append({"step": step, "phase": phase, "loss": value, "class_share": share(out), "logit_rms": float(out.float().square().mean().sqrt()), "gradient_norm": gnorm, "update_norm": unorm, "update_parameter_ratio": unorm / pnorm, "preceding_ratio": preceding})
        tensors = list(model.parameters()) + list(model.buffers()) + [p.grad for p in params if p.grad is not None]
        if not all(torch.isfinite(t).all() for t in tensors):
            raise RuntimeError(f"nonfinite {arm} step {step}")
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    print(json.dumps({"arm": arm, "records": records, "ema": {p: emas[p] / (1 - 0.95 ** counts[p]) for p in emas}, "bn_counts": sorted({int(m.num_batches_tracked) for m in bns}), "min_var": min(float(m.running_var.min()) for m in bns), "momentum_buffers": sum("momentum_buffer" in s for s in opt.state.values()), "param_tensors": len(params)}))


def child(arm):
    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--trajectory", arm], cwd=ROOT, capture_output=True, text=True, check=True, timeout=240)
    return json.loads([line for line in p.stdout.splitlines() if line.strip()][-1])


def compare(a, b):
    maxima = {"logit": 0.0, "gradient": 0.0, "update": 0.0, "update_parameter": 0.0, "preceding": 0.0}
    concentrations = []
    failures = []
    for ra, rb in zip(a["records"], b["records"], strict=True):
        for key, field in (("logit", "logit_rms"), ("gradient", "gradient_norm"), ("update", "update_norm")):
            ratio = rb[field] / max(ra[field], 1e-30)
            maxima[key] = max(maxima[key], ratio)
            if ratio > 5:
                failures.append(f"{key} ratio step {ra['step']}")
        maxima["update_parameter"] = max(maxima["update_parameter"], rb["update_parameter_ratio"])
        if rb["update_parameter_ratio"] > 0.25:
            failures.append(f"update/parameter step {ra['step']}")
        if rb["preceding_ratio"] is not None:
            maxima["preceding"] = max(maxima["preceding"], rb["preceding_ratio"])
            if rb["preceding_ratio"] > 5:
                failures.append(f"update/median step {ra['step']}")
        if rb["class_share"] > 0.95 and ra["class_share"] <= 0.95:
            concentrations.append(ra["step"])
    persistent = any(b == a + 1 for a, b in zip(concentrations, concentrations[1:])) or len(concentrations) >= 3
    if persistent:
        failures.append(f"persistent concentration {concentrations}")
    ema_ratios = {p: b["ema"][p] / a["ema"][p] for p in ("strong", "weak")}
    if any(v > 1.5 for v in ema_ratios.values()):
        failures.append(f"EMA {ema_ratios}")
    for arm in (a, b):
        if arm["bn_counts"] != [48] or arm["min_var"] <= 0 or arm["momentum_buffers"] != arm["param_tensors"]:
            failures.append(f"state incomplete {arm['arm']}")
    return {"status": "failed" if failures else "pass", "maxima": maxima, "concentration_steps": concentrations, "persistent_concentration": persistent, "ema_ratios": ema_ratios, "failures": failures}


def parent():
    data = datasets.CIFAR10(DATASET_DIR, train=True, download=False)
    offsets = offset_gate(data)
    arms, corpus = make_corpus()
    del arms
    controls = [child(f"control-{i}") for i in range(4)]
    calibrations = [compare(controls[0], controls[1]), compare(controls[2], controls[3])]
    pre_candidate_failures = [failure for calibration in calibrations for failure in calibration["failures"]]
    if pre_candidate_failures:
        report = {"status": "control-failed", "controller_sha256": sha(Path(__file__)), "train_sha256": sha(ROOT / "train.py"), "offsets": offsets, "corpus": corpus, "control_calibrations": calibrations, "failures": pre_candidate_failures}
    else:
        candidate = child("candidate")
        result = compare(controls[0], candidate)
        corpus_after = torch.load(CORPUS, map_location="cpu", weights_only=False)["arms"]
        after = {"file_sha256": sha(CORPUS), "control_tensor_sha256": digest_batches(corpus_after["control"]), "candidate_tensor_sha256": digest_batches(corpus_after["candidate"])}
        failures = result["failures"] + ([] if all(after[k] == corpus[k] for k in after) else ["corpus changed"])
        report = {"status": "failed" if failures else "pass", "controller_sha256": sha(Path(__file__)), "train_sha256": sha(ROOT / "train.py"), "offsets": offsets, "corpus": corpus, "control_calibrations": calibrations, "trajectory": result, "corpus_after": after, "failures": failures}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    with REPORT.open("rb") as f:
        os.fsync(f.fileno())
    if report["status"] != "pass":
        raise RuntimeError("; ".join(report["failures"]))
    print(json.dumps({"status": "pass", "report": str(REPORT)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory")
    args = parser.parse_args()
    trajectory(args.trajectory) if args.trajectory else parent()
