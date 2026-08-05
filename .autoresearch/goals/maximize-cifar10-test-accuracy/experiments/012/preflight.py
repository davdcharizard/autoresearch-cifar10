import importlib
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
import prepare


class FailClosedEval:
    instances = 0

    def __init__(self):
        type(self).instances += 1

    def evaluate(self, *args, **kwargs):
        raise AssertionError("preflight must not evaluate test data")


prepare.Eval = FailClosedEval
sys.modules.pop("train", None)
train = importlib.import_module("train")
assert FailClosedEval.instances == 1
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "NVIDIA H20"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

for bad in (0, -1, 64.0, True, "64"):
    try:
        train.WideResNet(2, 2, 10, refinement_width=bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted invalid refinement_width={bad!r}")

torch.manual_seed(1234)
accepted = train.WideResNet(2, 2, 10, refinement_width=None)
accepted_rng = torch.get_rng_state().clone()
torch.manual_seed(1234)
candidate = train.WideResNet(2, 2, 10, refinement_width=64)
candidate_rng = torch.get_rng_state().clone()
assert torch.equal(accepted_rng, candidate_rng)

accepted_state = accepted.state_dict()
candidate_state = candidate.state_dict()
for key, value in accepted_state.items():
    assert key in candidate_state and torch.equal(value, candidate_state[key]), key
extra_keys = {key for key in candidate_state if key not in accepted_state}
assert extra_keys and all(key.startswith("refinement.") for key in extra_keys)
assert sum(p.numel() for p in accepted.parameters()) == 691_674
assert sum(p.numel() for p in candidate.parameters()) == 745_434

refinement = candidate.refinement
assert isinstance(refinement, train.PreActBottleneck)
for bn, width in ((refinement.bn1, 128), (refinement.bn2, 64), (refinement.bn3, 64)):
    assert bn.num_features == width
    assert torch.equal(bn.weight, torch.ones_like(bn.weight))
    assert torch.equal(bn.bias, torch.zeros_like(bn.bias))
for conv, shape, kernel, padding in (
    (refinement.conv1, (128, 64), (1, 1), (0, 0)),
    (refinement.conv2, (64, 64), (3, 3), (1, 1)),
    (refinement.conv3, (64, 128), (1, 1), (0, 0)),
):
    assert (conv.in_channels, conv.out_channels) == shape
    assert conv.kernel_size == kernel and conv.stride == (1, 1)
    assert conv.padding == padding and conv.bias is None
    assert torch.count_nonzero(conv.weight) > 0
assert candidate.bn.num_features == 128
assert (candidate.fc.in_features, candidate.fc.out_features) == (128, 10)
assert [len(candidate.layer1), len(candidate.layer2), len(candidate.layer3)] == [2, 2, 2]

identity_copy = train.PreActBottleneck(128, 64).eval()
for conv in (identity_copy.conv1, identity_copy.conv2, identity_copy.conv3):
    torch.nn.init.zeros_(conv.weight)
identity_input = torch.randn(4, 128, 8, 8)
assert torch.equal(identity_copy(identity_input), identity_input)

hook_data = {"layer3": None, "calls": 0}
candidate.layer3.register_forward_hook(
    lambda module, inputs, output: hook_data.__setitem__("layer3", output.detach().clone())
)

def refinement_pre_hook(module, inputs):
    hook_data["calls"] += 1
    assert torch.equal(inputs[0], hook_data["layer3"])

candidate.refinement.register_forward_pre_hook(refinement_pre_hook)
with torch.no_grad():
    output = candidate(torch.randn(256, 3, 32, 32))
assert output.shape == (256, 10) and torch.isfinite(output).all()
assert hook_data["calls"] == 1
assert 53_760 == 745_434 - 691_674
assert 3_407_872 == 64 * (128 * 64 + 64 * 64 * 9 + 64 * 128)
assert 104_514_816 == 101_106_944 + 3_407_872

torch.manual_seed(4321)
accepted = train.WideResNet(2, 2, 10, refinement_width=None).to(device)
torch.manual_seed(4321)
candidate = train.WideResNet(2, 2, 10, refinement_width=64).to(device)
models = {"accepted": accepted, "candidate": candidate}

def make_optimizer(model):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": train.WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=train.MIN_LR, momentum=train.MOMENTUM, nesterov=True,
    )

optimizers = {name: make_optimizer(model) for name, model in models.items()}
distributions = {
    name: torch.distributions.Beta(
        torch.tensor(train.MIXUP_ALPHA, device=device),
        torch.tensor(train.MIXUP_ALPHA, device=device),
    ) for name in models
}
generator = torch.Generator().manual_seed(9876)
host_inputs = torch.randn(train.BATCH_SIZE, 3, 32, 32, generator=generator, pin_memory=True)
host_targets = torch.randint(0, 10, (train.BATCH_SIZE,), generator=generator, pin_memory=True)

def step(name, progress):
    model, optimizer = models[name], optimizers[name]
    t0 = time.perf_counter()
    inputs = host_inputs.to(device, non_blocking=True)
    targets = host_targets.to(device, non_blocking=True)
    lr = train.learning_rate(progress * train.TIME_BUDGET_S)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    if progress < train.MIXUP_END_FRACTION:
        mixed, a, b, mix = train.mixup_batch(inputs, targets, distributions[name])
        logits = model(mixed)
        loss = mix * F.cross_entropy(logits, a) + (1.0 - mix) * F.cross_entropy(logits, b)
    else:
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0), loss.detach().item(), tuple(logits.shape)

states = {}
for offset, name in enumerate(models):
    torch.manual_seed(20_000 + offset)
    torch.cuda.manual_seed(30_000 + offset)
    states[name] = (torch.get_rng_state(), torch.cuda.get_rng_state())

def window(name, progress, count):
    cpu, cuda = states[name]
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(cuda)
    values = [step(name, progress) for _ in range(count)]
    states[name] = (torch.get_rng_state(), torch.cuda.get_rng_state())
    assert values[-1][2] == (256, 10) and values[-1][1] == values[-1][1]
    return statistics.mean(value[0] for value in values)

torch.cuda.reset_peak_memory_stats()
for name in models:
    window(name, 0.5, 25)
order = (("accepted", "A"), ("candidate", "A"), ("candidate", "B"),
         ("accepted", "B"), ("accepted", "C"), ("candidate", "C"))
results = {regime: {name: [] for name in models} for regime in ("mixup", "hard")}
for regime, progress in (("mixup", 0.5), ("hard", 0.8)):
    for name, label in order:
        mean_ms = window(name, progress, 50)
        results[regime][name].append(mean_ms)
        print(f"{regime} {name}-{label}: {mean_ms:.6f} ms")

medians, cvs = {}, {}
for regime in results:
    medians[regime], cvs[regime] = {}, {}
    for name, values in results[regime].items():
        medians[regime][name] = statistics.median(values)
        cvs[regime][name] = statistics.pstdev(values) / statistics.mean(values)
        print(f"{regime} {name}: median={medians[regime][name]:.6f} ms cv_ratio={cvs[regime][name]:.6f}")
aggregates = {name: 0.65 * medians["mixup"][name] + 0.35 * medians["hard"][name] for name in models}
retention = aggregates["accepted"] / aggregates["candidate"]
projection = 141.9 * retention
print(f"accepted_aggregate_ms={aggregates['accepted']:.6f}")
print(f"candidate_aggregate_ms={aggregates['candidate']:.6f}")
print(f"retention={retention:.6f}")
print(f"projected_passes={projection:.6f}")
print(f"peak_memory_mb={torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")
assert all(value <= 0.05 for regime in cvs.values() for value in regime.values())
assert retention >= 0.92 and projection >= 130.5
print("PREFLIGHT PASS")
