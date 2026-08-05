# Plan EXP-025: Diagnostic-Free Full Two-Gate SE Closure
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Recreate EXP-017 gates without observation work
- [x] Create experiment branch `autoresearch/maximize-cifar10-test-accuracy-025` from integration commit `eb08811` and modify only `train.py` in production.
- [x] Add the exact EXP-017 `Stage3SE(128, reduction=16)` function and an opt-in `stage3_attention` constructor path: signed residual global pooling, biased `128->8->128` projections, ReLU, and `2*sigmoid` scaling on both stage-3 residual branches before shortcut addition.
- [x] Attach and initialize the two gates after accepted whole-model initialization, before the caller's `.to(device)`, inside a CPU-only RNG fork seeded with fixed `ATTENTION_INIT_SEED=17017`; omit every diagnostic buffer, reduction, method, and summary print from EXP-017.
- [x] Compile `train.py` and require a clean diff with no training or evaluation recipe changes.

### Milestone 2: Prove semantic identity and recovered exposure
- [x] Create ignored `experiments/025/preflight.py` with a guarded evaluator and instantiate the real production `WideResNet(..., stage3_attention=True)` path; verify exact accepted common state/RNG/logits, two gate states against the archived EXP-017 seed oracle, placement, initialization, device/dtype, parameter count, optimizer grouping, and two-step gradient opening.
- [x] Explicitly audit that the production gates contain no diagnostic buffers or observation operations and that only the residual branch is scaled.
- [ ] Run matched accepted/candidate timing in both mixup and hard-label regimes using balanced order, private RNG states, warmup, and three measured windows; require every CV <=5% and at least 137.0 projected data passes from the accepted 141.9-pass reference.

### Milestone 3: Run one scored experiment
- [ ] Confirm exactly one NVIDIA H20, remove stale `run.log`, then execute `timeout 600s uv run train.py > run.log 2>&1` exactly once.
- [ ] Monitor for non-finite loss, CUDA/resource failure, timeout, stalled output, or malformed control-flow events without launching a replacement run.
- [ ] Require exit 0, one complete finite summary, 300 counted training seconds, wall time below 600 seconds, at least 137.0 realized data passes, at most one evaluation per epoch, and exactly one mixup transition near 195 counted seconds.

### Milestone 4: Verify and close the hypothesis
- [ ] Parse `best_test_acc` and require `>=94.17%` against the indexed `94.07%` baseline.
- [ ] Record final loss/accuracy, timing, epochs, steps and passes, VRAM, and parameters in `03-execute.md`; audit the final production diff.
- [ ] Accept only on the metric threshold. Any valid sub-threshold result closes diagnostic overhead/exposure as the missing ingredient; do not alter seed, ratio, placement, initialization, gate count, or rerun.

## Code Changes
- **`train.py` / `Stage3SE` and `PreActBlock`**: add the exact EXP-017 conditional gate, globally pooling the signed residual output and applying `2*sigmoid(fc2(relu(fc1(pool))))`; multiply `out` immediately before the existing `out + shortcut`. `attention=None` remains the accepted path. No diagnostic state, `no_grad` reductions, hooks, counters, reporting, or evaluator interaction is allowed.
- **`train.py` / `WideResNet` construction**: add an opt-in `stage3_attention=False` constructor argument. Build and initialize the accepted WRN exactly as at `eb08811`, then on the true production opt-in path attach one gate to each existing `layer3` block inside `torch.random.fork_rng(devices=[])`. Seed only `torch.random.default_generator` with 17017, use Kaiming-normal fan-in/ReLU for `fc1.weight`, and zero both biases plus `fc2.weight`. Zero-valued diagnostic buffers in EXP-017 consumed no RNG; compare against its archived preflight oracle to prove that their removal does not shift either gate. This preserves post-construction global CPU/CUDA RNG and all accepted common tensors.
- **`.autoresearch/.../experiments/025/preflight.py`**: ignored verification-only harness; it may import `train.py` with a dummy evaluator but must not enter the scored path or modify evaluator code.

## Configuration Changes
- Attention: none -> two stage-3 ratio-16 full conditional SE gates.
- Parameters: 691,674 -> 696,042 (4,368 added, matching EXP-017).
- Gate initialization seed: fixed 17017, reused solely for exact EXP-017 treatment identity; this is not a seed reroll.
- Diagnostics: EXP-017 training-time gate observation -> none.
- Depth `[2,2,2]`, widths `[32,64,128]`, FP32, batch 256, optimizer/decay, time-based LR and 0.002 floor, crop/flip, batch-shared alpha-0.2 mixup through 65%, hard-label tail, seed 42, persistent workers, and evaluation cadence remain unchanged.

## Execution Environment
- Method: offline local execution; no remote, network, package installation, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20, local CIFAR-10, existing `uv` environment and persistent DataLoader workers.
- Estimated runtime: semantic and timing preflight under 4 minutes; scored run about 345 seconds wall, with a hard 600-second timeout.
- Log output: scored stdout/stderr exclusively to project-root `run.log`, retained until analysis and removed before the next experiment.
- Tool skill: none.

## Abort Criteria
- Abort before scoring for wrong gate count, placement, tensor shapes, state, seed oracle, exact-neutral scale, parameter count, device/dtype, accepted common tensor equality, separately serialized CPU/CUDA RNG equality, initial-logit equality, shortcut scaling, optimizer grouping, or two-step gradient behavior.
- Abort before scoring if any production diagnostic buffer/operation remains, the timing CV exceeds 5%, projected exposure is below 137.0 passes, or a syntax/scope audit fails. Do not lower the exposure gate.
- During scoring, abort/classify on nonzero exit, timeout, OOM/resource error, non-finite loss, missing final summary, fewer than 137.0 realized passes, duplicate evaluation in an epoch, or missing/multiple mixup transitions. Never rerun a valid score.

## Verification Protocol

### Verification Procedure

1. Run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; within 10 seconds require `baseline=94.07` and `baseline_commit=eb08811`, yielding threshold 94.17.
2. Run `nvidia-smi --query-gpu=name --format=csv,noheader`, `git diff --check`, `git diff --name-only eb08811 --`, `git status --short --untracked-files=all`, and `uv run python -m py_compile train.py`; within 30 seconds require one H20 and only tracked production change `train.py`.
3. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/025/preflight.py --semantics`; within 120 seconds require exactly two `128->8->128` gates on `layer3[0:2]`, no diagnostic state/operations, 696,042 parameters, fixed seed 17017, bitwise accepted common state, separately equal post-construction CPU/CUDA RNG, exact gate seed-oracle state, exact initial scales/logits, residual-only placement, correct device/dtype and optimizer grouping, nonzero finite `fc2` gradients on step one with exactly-zero `fc1` gradients, then nonzero finite `fc1` gradients on step two.
4. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/025/preflight.py --throughput`; within 240 seconds require balanced matched mixup/hard timing, all window CVs <=0.05, finite state, and `projected_passes >=137.0` using `141.9 * accepted_weighted_ms / candidate_weighted_ms`. Abort without scoring on failure.
5. Remove stale `run.log` and run exactly `timeout 600s uv run train.py > run.log 2>&1` once. Require exit 0; do not rerun a completed valid result.
6. Parse the summary, evaluations, transition, and errors with `rg`. Require one complete finite summary, `best_test_acc >=94.17`, `training_seconds` approximately 300, `total_seconds <600`, `num_params=696042`, and `num_steps * 256 / 50000 >=137.0`; also require one transition near 195 counted seconds, unique evaluation epochs, and no traceback/error/non-finite markers.
7. Audit `git diff eb08811 -- train.py` and confirm it contains only the approved two diagnostic-free gates and attachment logic, with frozen evaluator and accepted training recipe.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `num_epochs`, `num_steps`, `peak_vram_mb`, and `num_params`: collect from the single final summary in `run.log` after all necessary conditions pass.
- Effective data passes: compute `num_steps * 256 / 50000` from the same summary.
