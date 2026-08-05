import importlib.util
import math
import subprocess
import sys
import types


class DummyEval:
    def evaluate(self, model, device):
        raise AssertionError("evaluator must not be used by preflight")


prepare = types.ModuleType("prepare")
prepare.DATASET_DIR = "./data"
prepare.NUM_WORKERS = 0
prepare.TIME_BUDGET_S = 300
prepare.Eval = DummyEval
sys.modules["prepare"] = prepare

accepted = types.ModuleType("accepted_train")
source = subprocess.check_output(["git", "show", "eb08811:train.py"], text=True)
exec(compile(source, "eb08811:train.py", "exec"), accepted.__dict__)

spec = importlib.util.spec_from_file_location("candidate_train", "train.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)

accepted_hparams = {
    name: value
    for name, value in accepted.__dict__.items()
    if name.isupper() and isinstance(value, (int, float))
}
candidate_hparams = {
    name: value
    for name, value in candidate.__dict__.items()
    if name.isupper() and isinstance(value, (int, float))
}
assert set(accepted_hparams) == set(candidate_hparams)
for name in accepted_hparams:
    if name == "MIXUP_END_FRACTION":
        assert accepted_hparams[name] == 0.65
        assert candidate_hparams[name] == 0.75
    else:
        assert accepted_hparams[name] == candidate_hparams[name], name

diff = subprocess.check_output(
    ["git", "diff", "--unified=0", "eb08811", "--", "train.py"], text=True
)
changed = [
    line
    for line in diff.splitlines()
    if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
]
assert changed == [
    "-MIXUP_END_FRACTION = 0.65",
    "+MIXUP_END_FRACTION = 0.75",
], changed

probes = [0.0, 0.749999, 0.75, 0.750001, 1.0]
observed = [
    progress < candidate.MIXUP_END_FRACTION for progress in probes
]
assert observed == [True, True, False, False, False]
states = [
    progress < candidate.MIXUP_END_FRACTION
    for progress in [index / 1000 for index in range(1001)]
]
transitions = sum(left and not right for left, right in zip(states, states[1:]))
assert transitions == 1

for training_time in [0.0, 14.9, 15.0, 195.0, 224.999, 225.0, 299.9, 300.0]:
    left = accepted.learning_rate(training_time)
    right = candidate.learning_rate(training_time)
    assert math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)

print("SEMANTICS PASS")
print("diff=MIXUP_END_FRACTION:0.65->0.75 only")
print("boundary=below:true equal:false above:false transitions=1")
print("other_hyperparameters=exact learning_rate=exact")

