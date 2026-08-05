import ast
import importlib.util
import math
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
BASELINE = "a7c42dc"
EXPECTED_PARAMS = 1_003_482
RETENTION_FLOOR = 127.0 / 130.304


class BlockedEval:
    def evaluate(self, *_args, **_kwargs):
        raise AssertionError("evaluation forbidden")


def blocked_dataset(*_args, **_kwargs):
    raise AssertionError("dataset construction forbidden")


def load_modules():
    import prepare
    from torchvision import datasets

    prepare.Eval = BlockedEval
    datasets.CIFAR10 = blocked_dataset
    spec = importlib.util.spec_from_file_location("exp047_candidate", ROOT / "train.py")
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    source = subprocess.check_output(["git", "show", f"{BASELINE}:train.py"], cwd=ROOT, text=True)
    accepted = types.ModuleType("exp047_accepted")
    accepted.__file__ = f"git:{BASELINE}:train.py"
    exec(compile(source, accepted.__file__, "exec"), accepted.__dict__)
    return accepted, candidate, source, (ROOT / "train.py").read_text()


def rng_state():
    return torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def restore_rng(state):
    torch.random.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1])


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def optimizer(module, model, lr=0.037):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return optim.SGD(
        [{"params": decay, "weight_decay": module.WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        momentum=module.MOMENTUM,
        nesterov=True,
    )


def optimizer_signature(model, opt):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [([names[id(p)] for p in group["params"]], {k: v for k, v in group.items() if k != "params"}) for group in opt.param_groups]


def audit_source(accepted_source, candidate_source):
    diff = subprocess.check_output(["git", "diff", "--unified=0", BASELINE, "--", "train.py"], cwd=ROOT, text=True)
    print(diff, flush=True)
    assert "def mixup_pairing(targets, distribution):" in candidate_source
    assert "def forward(self, x, feature_mix=None):" in candidate_source
    assert "out = mix * out + (1.0 - mix) * out[permutation]" in candidate_source
    assert "outputs = model(inputs, feature_mix=(mix, permutation))" in candidate_source
    assert "mixup_pairing" not in accepted_source and "feature_mix" not in accepted_source
    assert subprocess.check_output(["git", "diff", "--name-only", BASELINE], cwd=ROOT, text=True).strip() == "train.py"
    subprocess.run(["git", "diff", "--exit-code", BASELINE, "--", "prepare.py", "pyproject.toml"], cwd=ROOT, check=True)
    tree = ast.parse(candidate_source)
    assert sum(isinstance(node, ast.FunctionDef) and node.name == "mixup_pairing" for node in ast.walk(tree)) == 1


def build_models(accepted, candidate, device="cuda"):
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    start = rng_state()
    restore_rng(start)
    model_a = accepted.WideResNet(accepted.STAGE_BLOCKS, accepted.WIDEN_FACTOR, accepted.NUM_CLASSES).to(device)
    after_a = rng_state()
    restore_rng(start)
    model_c = candidate.WideResNet(candidate.STAGE_BLOCKS, candidate.WIDEN_FACTOR, candidate.NUM_CLASSES).to(device)
    after_c = rng_state()
    assert torch.equal(after_a[0], after_c[0]) and torch.equal(after_a[1], after_c[1])
    state_a, state_c = model_a.state_dict(), model_c.state_dict()
    assert list(state_a) == list(state_c) and len(state_c) == 97
    assert len(list(model_c.parameters())) == 52
    assert sum(p.numel() for p in model_c.parameters()) == EXPECTED_PARAMS
    for name in state_a:
        assert torch.equal(state_a[name], state_c[name]), name
    assert optimizer_signature(model_a, optimizer(accepted, model_a)) == optimizer_signature(model_c, optimizer(candidate, model_c))
    return model_a, model_c


def fixed_input(device, batch=8, dtype=torch.float32):
    inputs = torch.linspace(-1, 1, batch * 3 * 32 * 32, device=device, dtype=dtype).reshape(batch, 3, 32, 32)
    targets = torch.arange(batch, device=device) % 10
    return inputs, targets


def default_path_identity(accepted, candidate, model_a, model_c):
    for training in (False, True):
        model_a.train(training); model_c.train(training)
        model_a.load_state_dict(model_c.state_dict())
        inputs, targets = fixed_input("cuda")
        model_a.zero_grad(set_to_none=True); model_c.zero_grad(set_to_none=True)
        torch.manual_seed(47_100 + int(training)); torch.cuda.manual_seed(47_100 + int(training))
        start = rng_state()
        restore_rng(start)
        logits_a = model_a(inputs)
        loss_a = F.cross_entropy(logits_a, targets)
        loss_a.backward()
        after_a = rng_state()
        restore_rng(start)
        logits_c = model_c(inputs)
        loss_c = F.cross_entropy(logits_c, targets)
        loss_c.backward()
        after_c = rng_state()
        assert torch.equal(logits_a, logits_c) and torch.equal(loss_a, loss_c)
        assert torch.equal(after_a[0], after_c[0]) and torch.equal(after_a[1], after_c[1])
        for (name_a, p_a), (name_c, p_c) in zip(model_a.named_parameters(), model_c.named_parameters()):
            assert name_a == name_c and torch.equal(p_a.grad, p_c.grad), name_a
        for name, value in model_a.state_dict().items():
            assert torch.equal(value, model_c.state_dict()[name]), name
    print("default_path exact")


class ActualCapture:
    def __init__(self, module, model):
        self.module = module
        self.model = model
        self.values = {}
        self.original_pool = module.F.adaptive_avg_pool2d

        def wrapped_pool(*args, **kwargs):
            output = self.original_pool(*args, **kwargs)
            output.retain_grad()
            self.values["pool"] = output
            return output

        module.F.adaptive_avg_pool2d = wrapped_pool
        self.handles = [
            model.pooled_head[0].register_forward_pre_hook(self._input("mixed")),
            model.pooled_head.register_forward_hook(self._output("head")),
            model.fc.register_forward_pre_hook(self._input("refined")),
            model.fc.register_forward_hook(self._output("logits")),
        ]

    def _input(self, name):
        def hook(_module, inputs):
            self.values[name] = inputs[0]
            inputs[0].retain_grad()
        return hook

    def _output(self, name):
        def hook(_module, _inputs, output):
            self.values[name] = output
        return hook

    def close(self):
        self.module.F.adaptive_avg_pool2d = self.original_pool
        for handle in self.handles:
            handle.remove()


def feature_oracle(candidate, model):
    model.eval()
    batch = 8
    inputs, targets = fixed_input("cuda", batch)
    permutation = torch.tensor([1, 2, 0, 4, 5, 3, 7, 6], device="cuda")
    mix = inputs.new_tensor(0.3)
    model.zero_grad(set_to_none=True)
    capture = ActualCapture(candidate, model)
    logits = model(inputs, feature_mix=(mix, permutation))
    z = capture.values["pool"].view(batch, -1)
    mixed_ref = mix * z + (1 - mix) * z[permutation]
    hidden_ref = F.relu(F.linear(mixed_ref, model.pooled_head[0].weight))
    head_ref = F.linear(hidden_ref, model.pooled_head[2].weight)
    refined_ref = mixed_ref + candidate.POOLED_HEAD_SCALE * head_ref
    logits_ref = F.linear(refined_ref, model.fc.weight, model.fc.bias)
    torch.testing.assert_close(capture.values["mixed"], mixed_ref, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(capture.values["head"], head_ref, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(capture.values["refined"], refined_ref, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(logits, logits_ref, rtol=2e-5, atol=2e-7)
    loss = mix * F.cross_entropy(logits, targets) + (1 - mix) * F.cross_entropy(logits, targets[permutation])
    loss.backward()
    q = capture.values["mixed"].grad
    expected = mix * q.clone()
    expected.index_add_(0, permutation, (1 - mix) * q)
    actual = capture.values["pool"].grad.view(batch, -1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    nonlinear_gap = (model.pooled_head(mixed_ref) - (mix * model.pooled_head(z) + (1 - mix) * model.pooled_head(z[permutation]))).norm().item()
    cosine = F.cosine_similarity(z, z[permutation], dim=1).mean().item()
    norm_ratio = mixed_ref.norm().item() / z.norm().item()
    print(f"feature_oracle loss={loss.item():.9f} jacobian_error={(actual-expected).abs().max().item():.9g} jensen_gap={nonlinear_gap:.9g} pair_cosine={cosine:.6f} norm_ratio={norm_ratio:.6f}")
    assert nonlinear_gap > 0
    capture.close()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


def draw_oracle(accepted, candidate):
    inputs, targets = fixed_input("cuda")
    distribution_a = torch.distributions.Beta(torch.tensor(accepted.MIXUP_ALPHA, device="cuda"), torch.tensor(accepted.MIXUP_ALPHA, device="cuda"))
    distribution_c = torch.distributions.Beta(torch.tensor(candidate.MIXUP_ALPHA, device="cuda"), torch.tensor(candidate.MIXUP_ALPHA, device="cuda"))
    torch.manual_seed(47_200); torch.cuda.manual_seed(47_200)
    start = rng_state()
    restore_rng(start)
    mixed_a, ta, tb, mix_a = accepted.mixup_batch(inputs, targets, distribution_a)
    after_a = rng_state()
    restore_rng(start)
    tc, td, mix_c, permutation = candidate.mixup_pairing(targets, distribution_c)
    after_c = rng_state()
    assert torch.equal(mix_a, mix_c) and torch.equal(ta, tc) and torch.equal(tb, td)
    assert torch.equal(mixed_a, mix_c * inputs + (1 - mix_c) * inputs[permutation])
    assert torch.equal(after_a[0], after_c[0]) and torch.equal(after_a[1], after_c[1])
    print(f"draw_oracle mix={mix_c.item():.9f} self_pairs={(permutation==torch.arange(len(permutation),device='cuda')).sum().item()}")


def update_oracle(candidate, state, gradients):
    for preseeded in (False, True):
        model = candidate.WideResNet(candidate.STAGE_BLOCKS, candidate.WIDEN_FACTOR, candidate.NUM_CLASSES).cuda()
        model.load_state_dict(state)
        opt = optimizer(candidate, model)
        expected, expected_buffers = {}, {}
        for index, (name, parameter) in enumerate(model.named_parameters()):
            parameter.grad = gradients[name].clone()
            decay = candidate.WEIGHT_DECAY if parameter.ndim >= 2 else 0.0
            direction = parameter.grad + decay * parameter.detach()
            if preseeded:
                b0 = torch.full_like(parameter, 1e-5 * (index + 1))
                opt.state[parameter]["momentum_buffer"] = b0.clone()
                b1 = candidate.MOMENTUM * b0 + direction
            else:
                b1 = direction
            expected[name] = parameter.detach().clone() - 0.037 * (direction + candidate.MOMENTUM * b1)
            expected_buffers[name] = b1
        opt.step()
        named = dict(model.named_parameters())
        max_p = max((named[name] - value).abs().max().item() for name, value in expected.items())
        max_b = max((opt.state[named[name]]["momentum_buffer"] - value).abs().max().item() for name, value in expected_buffers.items())
        print(f"update preseeded={preseeded} parameter_error={max_p:.9g} buffer_error={max_b:.9g}")
        assert max_p <= 2e-7 and max_b <= 2e-7
        del opt, model


def semantics():
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "NVIDIA H20"
    accepted, candidate, source_a, source_c = load_modules()
    audit_source(source_a, source_c)
    torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = True
    model_a, model_c = build_models(accepted, candidate)
    initial = clone_state(model_c)
    default_path_identity(accepted, candidate, model_a, model_c)
    model_c.load_state_dict(initial)
    gradients = feature_oracle(candidate, model_c)
    draw_oracle(accepted, candidate)
    update_oracle(candidate, initial, gradients)
    print("SEMANTICS PASS")


def timing_step(module, model, opt, host_inputs, host_targets, distribution, regime, candidate_arm):
    inputs = host_inputs.cuda(non_blocking=True); targets = host_targets.cuda(non_blocking=True)
    for group in opt.param_groups: group["lr"] = 0.037
    opt.zero_grad(set_to_none=True)
    if regime == "early":
        if candidate_arm:
            ta, tb, mix, permutation = module.mixup_pairing(targets, distribution)
            logits = model(inputs, feature_mix=(mix, permutation))
        else:
            mixed, ta, tb, mix = module.mixup_batch(inputs, targets, distribution)
            logits = model(mixed)
        loss = mix * F.cross_entropy(logits, ta) + (1 - mix) * F.cross_entropy(logits, tb)
    else:
        loss = F.cross_entropy(model(inputs), targets)
    if not torch.isfinite(loss): raise RuntimeError("nonfinite timing loss")
    loss.backward(); opt.step()


def run_window(module, state, window_rng, host_inputs, host_targets, regime, steps, candidate_arm, peak=False):
    model = module.WideResNet(module.STAGE_BLOCKS, module.WIDEN_FACTOR, module.NUM_CLASSES)
    model.load_state_dict(state); model = model.cuda().train()
    opt = optimizer(module, model)
    distribution = torch.distributions.Beta(torch.tensor(module.MIXUP_ALPHA, device="cuda"), torch.tensor(module.MIXUP_ALPHA, device="cuda"))
    restore_rng(window_rng); torch.cuda.synchronize()
    if peak: torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps): timing_step(module, model, opt, host_inputs, host_targets, distribution, regime, candidate_arm)
    torch.cuda.synchronize(); elapsed = 1000 * (time.perf_counter() - start) / steps
    allocated = torch.cuda.max_memory_allocated() if peak else 0
    del distribution, opt, model; torch.cuda.empty_cache()
    return elapsed, allocated


def timing():
    accepted, candidate, _sa, _sc = load_modules()
    torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = True
    model_a, model_c = build_models(accepted, candidate, device="cpu")
    states = {"A": clone_state(model_a), "C": clone_state(model_c)}
    del model_a, model_c
    host_inputs = torch.linspace(-1, 1, 256 * 3 * 32 * 32).reshape(256, 3, 32, 32).pin_memory()
    host_targets = (torch.arange(256) % 10).pin_memory()
    torch.manual_seed(47_300); torch.cuda.manual_seed(47_300); window_rng = rng_state()
    modules = {"A": accepted, "C": candidate}
    for arm, regime in (("A","early"),("C","early"),("A","hard"),("C","hard")):
        run_window(modules[arm], states[arm], window_rng, host_inputs, host_targets, regime, 20, arm == "C")
    schedule = [("A","early"),("C","early"),("A","hard"),("C","hard"),("C","hard"),("A","hard"),("C","early"),("A","early")]
    windows = {regime:{arm:[] for arm in ("A","C")} for regime in ("early","hard")}
    pairs = {"early":[],"hard":[]}; candidate_peak = 0
    for cycle in range(2):
        block = []
        for arm, regime in schedule:
            value, peak = run_window(modules[arm], states[arm], window_rng, host_inputs, host_targets, regime, 50, arm == "C", arm == "C")
            candidate_peak = max(candidate_peak, peak); windows[regime][arm].append(value); block.append((arm,regime,value))
            print(f"timing cycle={cycle} arm={arm} regime={regime} ms={value:.9f}")
        pairs["early"].extend([(block[0][2],block[1][2]),(block[7][2],block[6][2])])
        pairs["hard"].extend([(block[2][2],block[3][2]),(block[5][2],block[4][2])])
    for regime in ("early","hard"):
        ratios=[c/a for a,c in pairs[regime]]; ratio_cv=statistics.pstdev(ratios)/statistics.mean(ratios)
        for arm in ("A","C"):
            cv=statistics.pstdev(windows[regime][arm])/statistics.mean(windows[regime][arm])
            print(f"summary regime={regime} arm={arm} values={windows[regime][arm]} cv={cv:.9f}"); assert cv<=.05
        print(f"ratio regime={regime} values={ratios} cv={ratio_cv:.9f}"); assert ratio_cv<=.01
    retentions=[]
    for i in range(4):
        ae,ce=pairs["early"][i]; ah,ch=pairs["hard"][i]
        retentions.append((.65/ce+.35/ch)/(.65/ae+.35/ah))
    median=statistics.median(retentions); projected=130.304*median; peak_mb=candidate_peak/1024/1024
    print(f"timing_gate retentions={retentions} median={median:.9f} projected={projected:.6f} peak_mb={peak_mb:.3f}")
    assert all(value>=RETENTION_FLOOR for value in retentions) and projected>=127 and peak_mb<2048
    print("TIMING PASS")


if __name__ == "__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in {"semantics","timing"}: raise SystemExit("usage: preflight.py semantics|timing")
    semantics() if sys.argv[1]=="semantics" else timing()
