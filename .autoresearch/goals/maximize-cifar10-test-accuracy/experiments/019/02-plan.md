# Plan EXP-019: Static First-Block Scale Plus Final SE
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement the interaction-preserving hybrid
- [x] Modify only `train.py`: add an exact-neutral 128-channel static residual scale to `layer3[0]` and a ratio-16 SE gate to `layer3[1]`, both after `conv2` and before shortcut addition.
- [x] Fully initialize accepted WRN state first. Inside a restored CPU RNG fork, initialize the final SE gate from the project's fixed seed 42 exactly as EXP-018. Attach a separately created all-ones static scale to block 0.
- [x] Preserve every accepted training/data/evaluation setting; add only a terminal, post-budget summary of the learned static scale mean/std/min/max.

### Milestone 2: Pass evaluator-free semantic preflight
- [x] Create ignored `experiments/019/preflight.py` with dummy evaluator; verify exact placement/types, 693,986 parameters, initial unit scales/logits, accepted common state, CPU/CUDA RNG, and fixed-seed-42 final-gate oracle.
- [x] Verify the static vector is 1D/no-decay, final gate matrices are decay and biases no-decay, first-step static/final-projection gradients open, and the SE first projection opens on step two.
- [x] Run compile, diff/scope, protected-file, and root-Python audits; require only production `train.py` changed.

### Milestone 3: Pass matched throughput preflight
- [x] Warm at least 25 steps and measure three balanced windows of at least 50 accepted/candidate steps in both mixup and hard-label regimes with identical synthetic inputs and optimizer logic.
- [x] Require all timing CVs <=5%, weighted throughput retention >=97%, finite state, and no persistent diagnostic buffers; observed 98.24% retention, worst CV 0.386%, finite synthetic projection 120.21 passes.

### Milestone 4: Execute once and verify
- [x] Confirm one H20, remove stale `run.log`, and execute exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require exit 0, finite complete summary, 300 counted seconds, total below 600, one transition near 195 seconds, unique evaluation epochs, 693,986 parameters, finite terminal scale statistics, and no error signature.
- [x] Compare `best_test_acc` to 94.17; observed 93.86, a valid no-improvement that will not be tuned or rerun.

## Code Changes
- **`train.py` / `StaticChannelScale`**: hold `nn.Parameter(torch.ones(128))` and multiply the signed residual by its broadcast channel vector. The 1D shape intentionally places it in the accepted no-decay optimizer group.
- **`train.py` / `Stage3SE`**: use the established global-pool, biased `128->8->128`, ReLU, and `2*sigmoid` design with Kaiming first weights, zero biases, and zero second weights for exact unit initial scale.
- **`train.py` / residual blocks**: add an optional residual transform applied after `conv2` and before the unchanged shortcut addition; attach static scaling only at `layer3[0]` and SE only at `layer3[1]`.
- **`train.py` / initialization**: after accepted whole-model initialization, enter `torch.random.fork_rng(devices=[])`, seed only `torch.random.default_generator` with the project seed 42, and instantiate/explicitly initialize the final gate exactly as EXP-018. Create the all-ones static vector without consuming RNG.
- **`train.py` / summary**: after training/evaluation has ended, print static scale mean, population std, min, and max from the parameter. This terminal observation does not affect gradients, schedule, evaluation, or counted training time.

## Configuration Changes
- First stage-3 residual selector: none -> learned static 128-channel vector, exact ones at initialization and no weight decay.
- Final stage-3 residual selector: none -> EXP-018's fixed-seed-42 ratio-16 SE gate.
- Parameters: 691,674 -> 693,986; added 128 static scales and 2,184 SE parameters.
- Accepted `[2,2,2]` topology/common state, global RNG, FP32, optimizer, LR/floor, decay rules, batch, crop/flip, batch-shared alpha-0.2 mixup through 65%, clean tail, fixed seed 42, workers, and evaluation cadence remain unchanged.

## Execution Environment
- Method: offline local execution; no remote, network, package install, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20, local CIFAR-10, existing `uv`, eight persistent workers.
- Estimated runtime: preflights under 4 minutes; scored run about 341 seconds wall, hard limit 600 seconds.
- Log output: scored stdout/stderr exclusively in project-root `run.log`, retained through analysis.
- Tool skill: none.

## Abort Criteria
- Abort before scoring on syntax/scope error; wrong transform placement/type; parameter count !=693,986; changed accepted common state; CPU/CUDA RNG mismatch; non-unit initial scales or non-identical logits; shortcut transformation; fixed-seed-42 gate oracle mismatch; wrong optimizer groups; failed gradient opening; or any persistent diagnostic state.
- Abort before scoring if any timing CV >5%, weighted retention <97%, H20 unavailable, OOM, or non-finite state.
- During scoring abort/classify on nonzero exit, timeout, OOM, error/non-finite output, missing summary, duplicate eval epoch, wrong count, or missing/multiple transition. Do not adjust seed, scale initialization, placement, decay grouping, or gate design; never rerun a valid result. Terminal scale statistics are explanatory only.

## Verification Protocol

### Verification Procedure

1. Query the results index baseline; require 94.07 at `eb08811`, threshold 94.17. Timeout 10 seconds.
2. Run H20, git scope/status/untracked/root-Python, `git diff --check`, and `uv run python -m py_compile train.py` audits; require exactly one H20 and only tracked production `train.py` changed. Timeout 30 seconds.
3. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/019/preflight.py --semantics`; require static scale only on `layer3[0]`, SE only on `layer3[1]`, 693,986 parameters, exact common state/RNG/logits/unit scales, fixed-seed-42 final-gate oracle, ungated shortcuts, correct device/dtype/groups, static gradient on step one, SE second projection on step one, and SE first projection on step two. Timeout 120 seconds.
4. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/019/preflight.py --throughput`; require balanced warmed windows, CV <=0.05, retention >=0.97, finite projection, and explicit pass. Timeout 180 seconds.
5. Remove stale `run.log`; execute exactly `timeout 600s uv run train.py > run.log 2>&1` once and never rerun after valid completion.
6. Parse with `rg`/`awk`; require one finite summary, `best_test_acc >=94.17`, 300 counted seconds, total <600, 693,986 parameters, one transition near 195 seconds, unique eval epochs, finite static-scale statistics, and no errors. Stop at first necessary-condition failure.
7. Audit `git diff eb08811 -- train.py`; confirm only the approved hybrid selector, controlled initialization, and terminal parameter summary changed.

### Informational Metrics (Optional)
- If necessary conditions pass, collect final accuracy/loss, timing, epochs, steps/passes, VRAM, parameters, and terminal static scale mean/std/min/max from `run.log`.
