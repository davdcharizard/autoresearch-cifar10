"""Weco evaluation harness for the CIFAR-10 training task.

This is the TRUSTED side of the optimization. Weco only edits `train.py`
(the `--source`); it has no access to this file or to `prepare.py`, so this
harness is where we keep the experiment honest AND comparable to the other
agent harnesses (Claude Code / Codex) that drive the same baseline.

What this harness does and why:

  1. Trusted metric (recompute, not self-report).
     Weco optimizes only the printed number, so it could `print("best_test_acc:
     99.9")`. We ignore everything train.py prints and recompute the score with
     the read-only evaluator in prepare.py, run on the model train() returns.
     This is numerically identical to what the agent harnesses report (same
     deterministic evaluator, same best checkpoint), so results stay comparable.

  2. Time budget == the SAME budget the agents get.
     The agent harnesses enforce the budget *inside* train.py: the training loop
     runs until `total_training_time` (sum of training-step time, excluding
     validation and startup) reaches TIME_BUDGET_S. We keep that exact mechanism
     -- train() honors `time_budget_s` -- so weco is held to the identical
     training-compute budget, not a stricter wall-clock one.

  3. Runaway guard == the agents' own kill rule.
     The agent TASK.md killed any run exceeding 10 minutes (2x the 300s budget)
     and treated it as a failure. We apply the same wall-clock ceiling as a
     backstop against a candidate that tampers with the in-loop budget. This is
     a comparable failure threshold, not a redefinition of the budget.

The reported `training_seconds` and measured wall-clock are printed as plain
[audit] lines (not metrics) so you can confirm in the run log / winning diff
that the candidate did not alter the budget loop.

Output: exactly one line `best_test_acc: <value>` (percent). A constraint
violation prints a plain message + `best_test_acc: 0.0` so weco avoids it.
"""

import importlib.util
import os
import sys
import time

# Repo root = two levels up from .weco/cifar10/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import prepare  # noqa: E402  read-only ground truth (evaluator + time budget)
from prepare import TIME_BUDGET_S  # noqa: E402

# Seed is fixed by the harness so the candidate cannot cherry-pick seeds.
SEED = int(os.environ.get("WECO_SEED", "42"))
# Wall-clock runaway guard, matching the agent harnesses' 10-minute kill rule
# (2x the 300s training budget). This is a backstop, not the budget itself.
WALL_CLOCK_LIMIT_S = float(os.environ.get("WECO_WALL_CLOCK_LIMIT_S", "600"))


def reject(message):
    print(f"Constraint violated: {message}")
    print("best_test_acc: 0.0")


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("candidate_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    candidate_path = os.path.join(REPO_ROOT, "train.py")
    candidate = load_candidate(candidate_path)

    if not hasattr(candidate, "train"):
        reject("train.py must expose a top-level train(seed, time_budget_s) function")
        return

    t0 = time.perf_counter()
    result = candidate.train(seed=SEED, time_budget_s=TIME_BUDGET_S)
    wall = time.perf_counter() - t0

    if not isinstance(result, dict):
        reject("train() must return a dict with 'model' and 'device'")
        return

    reported = result.get("training_seconds")
    print(
        f"[audit] wall_clock={wall:.1f}s "
        f"reported_training_seconds={reported} budget={TIME_BUDGET_S}s"
    )

    # Runaway guard: same wall-clock failure threshold the agent harnesses use.
    if wall > WALL_CLOCK_LIMIT_S:
        reject(
            f"wall-clock {wall:.1f}s exceeds the {WALL_CLOCK_LIMIT_S}s runaway limit"
        )
        return

    model = result.get("model")
    device = result.get("device")
    if model is None or device is None:
        reject("train() must return {'model': nn.Module, 'device': torch.device, ...}")
        return

    # Trusted recompute on the read-only evaluator (== the agent-harness metric).
    evaluator = prepare.Eval()
    _loss, acc = evaluator.evaluate(model, device)

    # acc is already a percentage (100 * correct / total).
    print(f"best_test_acc: {acc:.4f}")


if __name__ == "__main__":
    main()
