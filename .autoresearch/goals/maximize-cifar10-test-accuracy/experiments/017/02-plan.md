# Plan EXP-017: Neutral Stage-3 Squeeze-and-Excitation
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement exact-identity stage-3 gates
- [x] Modify only `train.py`: add `Stage3SE(128, reduction=16)` with biased `128->8->128` projections and `2*sigmoid` scaling; place it only on both existing stage-3 residual branches before shortcut addition.
- [x] Inside `WideResNet.__init__`, build/initialize the accepted WRN first, then before the caller's outer `.to(device)` attach both gates inside `torch.random.fork_rng(devices=[])`; seed only `torch.random.default_generator` with preregistered `ATTENTION_INIT_SEED=17017`, explicitly Kaiming-initialize first weights and zero all biases plus second weights.
- [x] Preserve accepted `[2,2,2]`, all existing state, training configuration, and global CPU/CUDA RNG; compile with `uv run python -m py_compile train.py`.

### Milestone 2: Pass evaluator-free semantics and throughput
- [x] Create ignored `experiments/017/preflight.py` with dummy evaluator; verify exactly two operational gates, 696,042 parameters, accepted-state/RNG equality, exact initial gate=1/logit equality, placement, optimizer grouping, and two-step opening gradients.
- [x] Run matched accepted/candidate mixup and hard-label timing; require each CV <=5%, weighted retention >=95%, and projected passes >=134.8.
- [x] Audit `git diff --name-only eb08811 --`, `git status --short --untracked-files=all`, root Python files, `git diff --check`, and compile; require only tracked `train.py` changed in production.

### Milestone 3: Run once and collect observational gate diagnostics
- [x] Confirm one H20, remove stale `run.log`, and execute `timeout 600s uv run train.py > run.log 2>&1` exactly once.
- [x] On training forwards only, update fixed-size on-device streaming scalars (never cache batch tensors); after completion synchronize once and print per-gate pooled mean/variance, across-example variance, saturation, and feature/bias RMS logits. Diagnostics must not inspect evaluator batches or affect training/control flow, and throughput must include them.
- [x] Require exit 0, finite complete summary, 300 counted seconds, total below 600, and one mixup transition near 195 seconds.

### Milestone 4: Verify
- [x] Evaluate `best_test_acc >=94.17`; observed 94.16, so the necessary metric condition failed; audit passed for 696,042 parameters, 27 unique evaluation epochs, transition, errors, 133.63712 passes, and production diff.
- [x] Record final loss and diagnostic values for mechanism interpretation, then complete `03-execute.md`.

## Code Changes
- **`train.py` / `Stage3SE`**: global-pool the signed residual output, apply ReLU `Linear(128,8)`, then `Linear(8,128)` and `2*sigmoid`; multiply the residual before the unchanged shortcut addition. Maintain only scalar device accumulators, updated under `no_grad()` during `self.training`: element count/sum/squared-sum; sum of per-forward per-channel batch variances (`unbiased=False`) and its channel-step count; count of scales `<=0.05` or `>=1.95`; feature-logit squared-sum, bias-logit squared-sum expanded over the batch, and logit element count. Pool all scored training forwards across both mixup and hard phases; evaluation forwards never contribute.
- **`train.py` / model construction**: inside `WideResNet.__init__`, leave accepted construction and whole-model initialization unchanged, then attach one gate to `layer3[0]` and `layer3[1]` before the main call's `.to(device)`. Use `torch.random.default_generator.manual_seed(17017)` inside a CPU-only restored fork; separately verify serialized CPU and CUDA states. Second projection weight/bias are exact zero; first weight uses accepted Kaiming-normal/ReLU semantics and first bias zero.
- **`train.py` / summary**: after all training, synchronize once and print per gate: `mean=sum/count`; pooled element variance `sq_sum/count-mean^2`; across-example variance `batch_var_sum/channel_step_count`; saturation fraction `saturated/count`; feature RMS `sqrt(feature_sq_sum/logit_count)`; bias RMS `sqrt(bias_sq_sum/logit_count)`. No value may influence loss, schedule, evaluation, early stopping, or verdict.

## Configuration Changes
- Attention: none -> two stage-3 ratio-16 exact-identity SE gates.
- Parameters: 691,674 -> 696,042; added 4,368 (0.6315%).
- Gate range/init: `2*sigmoid`, exactly one at initialization; seed 17017 fixed before implementation.
- Model widths/depths, Kaiming existing state, optimizer, FP32, LR/floor, decay, batch, crop/flip, batch-shared alpha-0.2 mixup through 65%, hard-label tail, seed 42, workers, and evaluation cadence remain unchanged.

## Execution Environment
- Method: offline local command; no remote, network, package install, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20, local CIFAR-10, existing `uv`, eight persistent workers.
- Estimated runtime: preflight under 3 minutes; scored run about 345 seconds wall, hard limit 600 seconds.
- Log output: scored stdout/stderr exclusively in project-root `run.log`, retained through analysis.
- Tool skill: none.

## Abort Criteria
- Abort before scoring on wrong gate count/placement/shapes/init/range/device/dtype; parameter count !=696,042; any accepted-state or separately serialized CPU/CUDA RNG mismatch; non-identity initial logits; shortcut gating; wrong optimizer groups; missing/non-finite expected first/second-step gradients; non-scalar/unbounded diagnostic state; evaluator access; syntax/scope error.
- Abort if timing CV >5%, retention <95%, projected passes <134.8, OOM, or non-finite state.
- During scoring abort/classify on nonzero exit, timeout, missing summary, wrong count, error/non-finite output, duplicate eval epoch, or missing/multiple transition. Never tune ratio, seed, scale, stage placement, or initialization and never rerun a valid score.

## Verification Protocol

### Verification Procedure

1. Query baseline via `exp-index.sh baseline`; require 94.07 at `eb08811`, threshold 94.17. Timeout 10 seconds.
2. Run `nvidia-smi --query-gpu=name --format=csv,noheader`, scope/status/untracked/root-Python audits, `git diff --check`, and `uv run python -m py_compile train.py`; require one H20 and only modified tracked `train.py`. Timeout 30 seconds.
3. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/017/preflight.py --semantics`. Require two `128->8->128` gates only on `layer3[0:2]`; 696,042 parameters; CPU-only seed 17017; bitwise accepted common state, separately equal CPU/CUDA post-construction RNG, and initial logits; exact scales of one; unchanged shortcut; all gate parameters matching model device/dtype after `.to`; second projection nonzero finite first-step gradients with first projection exactly zero, then nonzero finite first-projection gradient on step two; matrices in decay and biases in no-decay; only fixed-size scalar diagnostic accumulators whose oracle formulas match synthetic gates. Timeout 120 seconds.
4. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/017/preflight.py --throughput`. Warm >=25 steps and measure three >=50-step balanced windows in mixup and hard regimes using the exact instrumented production paths. Require CVs <=0.05, weighted retention >=0.95, projected passes >=134.8, finite state, and explicit pass. Timeout 180 seconds.
5. Remove stale log and run exactly `timeout 600s uv run train.py > run.log 2>&1`; require exit 0 and no rerun of a valid result.
6. Parse summary/evals/transition/errors/gate diagnostics using `rg`. Require one summary; `best_test_acc >=94.17`; 300 counted seconds; total <600; 696,042 parameters; one transition near 195 seconds; unique eval epochs; no errors; four finite diagnostic lines from training-only cached tensors. Stop immediately on necessary-condition failure.
7. Audit `git diff eb08811 -- train.py`; confirm only approved gate, attachment, and observational summary logic, with frozen evaluator and accepted training recipe.

### Informational Metrics (Optional)
- If necessary conditions pass, collect final accuracy/loss, timing, epochs, steps/passes, VRAM, parameters, plus per-gate mean/variance/saturation and feature/bias logit norms from `run.log`.
