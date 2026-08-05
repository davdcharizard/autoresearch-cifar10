import argparse
import importlib.util
import inspect
import multiprocessing
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import prepare


class GuardEval:
    constructions = 0

    def __init__(self):
        type(self).constructions += 1

    def evaluate(self, model, device):
        raise AssertionError("preflight may not evaluate")


prepare.Eval = GuardEval
import train


DEVICE = torch.device("cuda")
MP_CONTEXT = multiprocessing.get_context()
EXP026_PATH = ROOT / ".autoresearch/goals/maximize-cifar10-test-accuracy/experiments/026/preflight.py"
sys.path.insert(0, str(EXP026_PATH.parent))


def load_exp026_helpers():
    spec = importlib.util.spec_from_file_location("preflight", EXP026_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.train.NUM_BLOCKS = train.STAGE_BLOCKS
    return module


def optimizer_for(model):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return optim.SGD(
        [{"params": decay, "weight_decay": train.WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=train.MIN_LR, momentum=train.MOMENTUM, nesterov=True,
    )


def semantic_checks():
    assert GuardEval.constructions == 1
    assert train.STAGE_BLOCKS == (2, 2, 3)
    for invalid in ((2, 2), (2, 2, 0), (2, 2, -1), (2, 2, 3.0), (2, 2, True), [2, 2, 3]):
        try:
            train.WideResNet(invalid, train.WIDEN_FACTOR)
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)

    torch.empty(1, device=DEVICE)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    oracle_transform = train.make_train_transform(None)
    oracle = train.WideResNet((2, 2, 3), train.WIDEN_FACTOR)
    oracle_cpu = torch.random.get_rng_state().clone()
    oracle_cuda = torch.cuda.get_rng_state().clone()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    active = MP_CONTEXT.Value("b", 1, lock=False)
    candidate_transform = train.make_train_transform(active)
    candidate = train.WideResNet(train.STAGE_BLOCKS, train.WIDEN_FACTOR)
    assert torch.equal(torch.random.get_rng_state(), oracle_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), oracle_cuda)
    for name, tensor in oracle.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name]), name
    assert sum(p.numel() for p in candidate.parameters()) == 987_098
    assert tuple(len(layer) for layer in (candidate.layer1, candidate.layer2, candidate.layer3)) == (2, 2, 3)
    assert candidate.layer3[2].shortcut is None
    assert candidate.layer3[2].conv1.weight.shape == (128, 128, 3, 3)

    wrapper = candidate_transform.transforms[2]
    policy = wrapper.transform
    assert policy.num_ops == 1 and policy.magnitude == 5 and policy.num_magnitude_bins == 31
    assert policy.fill == [125, 123, 114]
    assert len(policy._augmentation_space(31, (32, 32))) == 14
    image = Image.fromarray(torch.arange(3072).remainder(256).byte().reshape(32, 32, 3).numpy())
    torch.manual_seed(27027)
    rng = torch.random.get_rng_state().clone()
    wrapper(image)
    assert torch.equal(torch.random.get_rng_state(), rng)

    with torch.inference_mode():
        probe = torch.zeros(8, 3, 32, 32)
        assert torch.equal(oracle(probe), candidate(probe))
    optimizer = optimizer_for(candidate)
    loss = F.cross_entropy(candidate(torch.zeros(8, 3, 32, 32)), torch.arange(8) % 10)
    assert torch.isfinite(loss)
    loss.backward()
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(candidate.parameters()))

    helpers = load_exp026_helpers()
    helpers.marker_cutoff_check()
    accepted_active, accepted_tail = helpers.trace_epochs(False)
    candidate_active, candidate_tail = helpers.trace_epochs(True)
    assert accepted_active != candidate_active and accepted_tail == candidate_tail
    source = inspect.getsource(train.main)
    assert source.count("randaugment_active.value = 0") == 1
    assert "and not budget_exhausted" in source and "iterator_exhausted=true" in source
    print("blocks=[2,2,3] params=987098 model_oracle=pass rng_tail_cutoff=pass")
    print("SEMANTICS PASS")


def distribution():
    return torch.distributions.Beta(
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
        torch.tensor(train.MIXUP_ALPHA, device=DEVICE),
    )


def timed_step(model, optimizer, host_x, host_y, dist, mixup, rng_state):
    torch.cuda.set_rng_state(rng_state)
    start = time.perf_counter()
    x = host_x.to(DEVICE, non_blocking=True)
    y = host_y.to(DEVICE, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    if mixup:
        mixed, a, b, coefficient = train.mixup_batch(x, y, dist)
        outputs = model(mixed)
        loss = coefficient * F.cross_entropy(outputs, a) + (1 - coefficient) * F.cross_entropy(outputs, b)
    else:
        loss = F.cross_entropy(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start), torch.cuda.get_rng_state().clone()


def throughput_checks():
    models = []
    for blocks in ((2, 2, 2), (2, 2, 3)):
        torch.manual_seed(42)
        models.append(train.WideResNet(blocks, train.WIDEN_FACTOR).to(DEVICE).train())
    assert [sum(p.numel() for p in model.parameters()) for model in models] == [691_674, 987_098]
    optimizers = [optimizer_for(model) for model in models]
    dists = [distribution(), distribution()]
    host_x = torch.randn(train.BATCH_SIZE, 3, 32, 32, pin_memory=True)
    host_y = (torch.arange(train.BATCH_SIZE) % 10).pin_memory()
    torch.cuda.manual_seed(27027)
    states = [torch.cuda.get_rng_state().clone() for _ in models]
    results = []
    for mixup in (True, False):
        for index in range(2):
            for _ in range(25):
                _, states[index] = timed_step(models[index], optimizers[index], host_x, host_y, dists[index], mixup, states[index])
        windows = [[], []]
        for window in range(3):
            for index in ((0, 1) if window % 2 == 0 else (1, 0)):
                values = []
                for _ in range(50):
                    elapsed, states[index] = timed_step(models[index], optimizers[index], host_x, host_y, dists[index], mixup, states[index])
                    values.append(elapsed)
                windows[index].append(statistics.mean(values))
        medians = [statistics.median(values) for values in windows]
        cvs = [statistics.pstdev(values) / statistics.mean(values) for values in windows]
        assert all(cv <= 0.02 for cv in cvs), (cvs, windows)
        results.append((medians, cvs, windows))
    accepted_ms = 0.65 * results[0][0][0] + 0.35 * results[1][0][0]
    candidate_ms = 0.65 * results[0][0][1] + 0.35 * results[1][0][1]
    projected = 141.9 * accepted_ms / candidate_ms
    print(f"mixup={results[0]} hard={results[1]}")
    print(f"accepted_ms={accepted_ms:.6f} candidate_ms={candidate_ms:.6f} projected_passes={projected:.6f}")
    assert projected >= 130.0
    print("THROUGHPUT PASS")


def loader_timing_checks():
    helpers = load_exp026_helpers()
    helpers.CONSUMER_SECONDS = 300.0 / 25_961
    arms = []
    for candidate in (False, True, True, False):
        arms.append((candidate, *helpers.timed_arm(candidate)))
    base = [value for kind, values, _ in arms if not kind for value in values]
    composed = [value for kind, values, _ in arms if kind for value in values]
    boundaries = [boundary for kind, _, boundary in arms if kind]
    base_median = statistics.median(base)
    composed_median = statistics.median(composed)
    base_cv = statistics.pstdev(base) / statistics.mean(base)
    composed_cv = statistics.pstdev(composed) / statistics.mean(composed)
    assert base_cv <= 0.05 and composed_cv <= 0.05
    assert all(value <= 1.20 * base_median for value in boundaries)
    differential = 338.5 + max(0, composed_median - base_median) * 134
    absolute = 38.5 + composed_median * 134
    print(f"arms={arms}")
    print(f"base_median_s={base_median:.6f} composed_median_s={composed_median:.6f} base_cv={base_cv:.6f} composed_cv={composed_cv:.6f} boundaries={boundaries} differential_s={differential:.3f} absolute_s={absolute:.3f}")
    assert differential <= 500 and absolute <= 500
    print("LOADER TIMING PASS")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--semantics", action="store_true")
    group.add_argument("--throughput", action="store_true")
    group.add_argument("--loader-timing", action="store_true")
    args = parser.parse_args()
    if args.semantics:
        semantic_checks()
    elif args.throughput:
        throughput_checks()
    else:
        loader_timing_checks()


if __name__ == "__main__":
    main()
