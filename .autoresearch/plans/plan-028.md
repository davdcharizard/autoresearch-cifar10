# Plan EXP-028: SiLU/Swish activation (ReLU → SiLU everywhere)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-028.md

## Milestones

### Milestone 1: Code changes implemented and passing local checks
- [ ] Replace all three `F.relu(...)` calls in train.py with `F.silu(...)`: BasicBlock L89 (pre-residual), BasicBlock L92 (post-residual), stem L127. Use `F.silu` (functional, matching the existing `F.relu` call style) — no signature/module changes.
- [ ] `grep -n "relu\|silu" train.py` confirms ZERO remaining `relu` and exactly 3 `silu` calls.
- [ ] `git diff --name-only` shows ONLY `train.py`.
- [ ] Smoke check (`uv run python`): AST parse clean; build `ResNet(3,10,4)`, assert `num_params == 4,299,866` unchanged (activation is parameter-free); a forward pass on a (2,3,32,32) tensor returns (2,10) with no error.

### Milestone 2: Experiment launched and confirmed running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Within ~60s: `run.log` shows `params: 4,299,866`, clean compile (no graph break / error on SiLU), step lines appearing, no NaN.

### Milestone 3: Run completes; throughput-neutrality confirmed (CRITICAL)
- [ ] Run exits 0 and prints the summary block.
- [ ] **KEY: confirm throughput-neutral** — `num_epochs ≈ 91` and `dt ≈ 8ms`. Per the epoch-wall + FLOPs-neutral-≠-wall-clock-neutral High insights (EXP-015/024), even a pointwise op can cost wall-clock if torch.compile fails to fuse it. If `num_epochs < ~85`, the result is COMPUTE-CONFOUNDED (SiLU didn't fuse, dt rose) and any delta is NOT a clean test of the activation.
- [ ] `total_seconds < 600`.

## Code Changes
- **train.py — three `F.relu` → `F.silu` swaps (L89, L92, L127)**:
  - L89 `out = F.relu(self.bn1(self.conv1(x)))` → `out = F.silu(self.bn1(self.conv1(x)))`
  - L92 `return F.relu(out)` → `return F.silu(out)`
  - L127 `out = F.relu(self.bn1(self.conv1(x)))` → `out = F.silu(self.bn1(self.conv1(x)))`

  Tests the hypothesis that a smooth, non-monotonic activation (SiLU = `x·σ(x)`) improves generalization over ReLU by smoothing the optimization landscape and removing the hard dead-ReLU zero region. This is the single largest untried orthogonal lever (activation function, flagged in the EXP-009 goal-learning) and is NOT a regularizer (so not subject to the convergence-bound rejection), NOT capacity (so not the epoch wall, provided SiLU fuses), and NOT convergence-polish (it changes the representation, which can move top-1 unlike EMA/SWA).

  Risks/edge cases: (a) Throughput — SiLU is a pointwise op that torch.compile should fuse into the conv/BN epilogue, so at the launch-bound 8ms/step the added wall-clock should be ~0; the Milestone-3 epoch check guards against a fusion failure that would raise dt (the EXP-015 lesson). (b) Init — `_weights_init` (L110-115) uses `kaiming_normal_` whose gain assumes ReLU; the slight mismatch is absorbed by the BatchNorm after every conv (left unchanged to keep the test clean and isolate the activation effect). (c) Magnitude — on an already well-tuned shallow net the gain may be within the ~0.2pp noise floor → no-improvement.

## Configuration Changes
- (none — pure activation change. LR 0.2, batch 128, warmup 0.05, WD 1e-4, LS 0.1, Cutout 16, TA, cosine-to-0, Nesterov, channels_last, bf16, torch.compile(reduce-overhead), params 4,299,866, seed 42 all unchanged.)

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background (`run_in_background: true`), with a `Monitor` watch on run.log for progress + NaN/error.
- Resources: single NVIDIA H20 (GPU 0). VRAM ~0.5 GB (unchanged — activation is param-free; SiLU's intermediate `σ(x)` is the same shape as the ReLU mask, negligible).
- Estimated runtime: ~300s training compute + ~30s startup/compile + ~91 evals ≈ baseline ~380-405s total. Must stay < 600s.
- Log output: `run.log` in project root via redirection; source of truth.
- Tool skill: none (local run).

## Abort Criteria
- **NaN/inf loss** at any point → kill, treat as failed.
- **Loss not decreasing after warmup** → kill.
- **No output / hang**: no new step lines in `run.log` for >120s → kill.
- **Wall-clock runaway**: process running past ~580s → kill (<600s constraint).
- CUDA OOM (not expected) → kill, treat as failed.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22**, pass threshold **best_test_acc ≥ 96.32**. Run conditions in order; stop at first failure.

1. **Cond 1 — primary metric clears bar.** After completion:
   `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`
   PASS iff `best_test_acc ≥ 96.32`. Empty `best_test_acc:` ⇒ crash (`tail -n 50 run.log`) → crash verdict.
2. **Cond 2 — clean completion within budget.** PASS iff summary block printed, `grep -c Traceback run.log` == 0, `total_seconds < 600`.
3. **Cond 3 — no constraint violations.** PASS iff: `git diff --name-only` lists only `train.py`; `num_params == 4,299,866`; eval-count == num_epochs (`grep -c "eval ep" run.log` == num_epochs); no new deps (`F.silu` is core torch); seed 42 unchanged.

**MANDATORY attribution note (epoch-wall + FLOPs-neutral-≠-wall-clock-neutral, EXP-015/024 — High):** record `num_epochs` and mean `dt`. SiLU adds ~zero FLOPs and should fuse, so epochs SHOULD be ~91 / dt ~8ms.
- epochs ~91 & dt ~8ms → throughput-neutral → the accuracy delta is a FAIR test of the smooth-activation hypothesis (whatever the verdict).
- epochs < ~85 (dt risen) → COMPUTE-CONFOUNDED (SiLU didn't fuse) → a regression cannot be attributed to the activation's merit; note this explicitly and consider it a fusion/throughput finding, not an activation finding.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -a "^peak_vram_mb:" run.log` — expect ~0.5 GB.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 / ~35,500 (throughput-neutral check vs baseline).
- final_test_loss: `grep -a "^final_test_loss:" run.log` — compare to baseline 0.195.
