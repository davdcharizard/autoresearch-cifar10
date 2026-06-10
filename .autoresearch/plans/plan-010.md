# Plan EXP-010: SiLU (Swish) activation in place of ReLU
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md

## Milestones

### Milestone 1: Code changes implemented and parse-clean
- [ ] Edit `train.py` only: replace the 3 `F.relu` sites with `F.silu` — `BasicBlock.forward` (post-bn1 and
      post-residual) and `ResNet.forward` (stem). Add `compiled_model = torch.compile(model, mode="reduce-overhead")`
      after model on device + num_params printed; route training forward through `compiled_model`; keep eval eager.
- [ ] `python -c "import ast; ast.parse(open('train.py').read())"` passes; `uv run ruff check train.py` passes.
- [ ] Sanity: param count prints **4,299,866** (UNCHANGED — SiLU adds no params; confirms only the nonlinearity
      changed, not the architecture); `git diff --name-only autoresearch/dev` = only `train.py`; eval line still
      eager `model`; `grep -c "F.relu" train.py` → 0; `grep manual_seed train.py` → 42.

### Milestone 2: Run launched and confirmed training
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background.
- [ ] `run.log` shows `Device: cuda`, `num_params 4,299,866`, clean compile (no graph breaks/recompile spam),
      steady-state dt printed, loss decreasing, no NaN. Record steady-state dt + img/s + projected epochs.

### Milestone 3: Run completes within budget and summary emitted
- [ ] `run.log` contains the `best_test_acc:` summary; `total_seconds` < 600.
- [ ] Record `num_epochs` — expect ~80–89 (compiled, like EXP-007); ≥~75 ⇒ fair, converged test (the model
      converges by ~77 per EXP-007, so this isolates SiLU's effect cleanly).

### Milestone 4: Verification verdict
- [ ] Apply the Verification Protocol. PASS all three ⇒ improvement; Cond-2 fail (best_test_acc < 96.10) ⇒
      no-improvement; constraint breach ⇒ invalid; crash/empty summary ⇒ crash.

## Code Changes
All changes confined to `train.py` (only editable file; `prepare.py` hook-protected). The full EXP-003 recipe stays
FIXED (k=4, Cutout(16), PEAK_LR 0.2 / 5% warmup cosine, Nesterov, label smoothing 0.1, batch 128, WD 1e-4, bf16,
channels_last, seed 42). Two coupled changes — activation (the intervention) + compile (validated enabler):

- **train.py — activation swap (3 sites)**: `F.relu` → `F.silu` at `BasicBlock.forward` line ~89
  (`out = F.silu(self.bn1(self.conv1(x)))`) and line ~92 (`return F.silu(out)` after the residual add), and
  `ResNet.forward` line ~127 (`out = F.silu(self.bn1(self.conv1(x)))`). *Why*: SiLU (x·sigmoid(x)) is a smooth,
  non-monotonic activation with consistent small gains over ReLU on conv classifiers (EfficientNet default);
  nonzero gradient for small negatives + smoothness → marginally better generalization at fixed capacity. This is
  the one architectural axis never tried and is orthogonal to all six exhausted axes.

- **train.py — compile + eval split** (identical pattern to EXP-007/008, validated enabler): after the model is on
  device and `num_params` printed, add `compiled_model = torch.compile(model, mode="reduce-overhead")`; change the
  training forward to `outputs = compiled_model(inputs)`; keep eval on the eager `model` (UNCHANGED). *Why*: SiLU's
  extra elementwise sigmoid adds kernel-launch overhead on this launch-bound net which could mildly cut epochs and
  confound a fair test (the EXP-008/SE concern). Compile absorbs that so SiLU gets a fully-converged (~80–89 epoch)
  test. EXP-007 established compiled-k4 = 95.92 ≈ baseline (null standalone accuracy effect), so any gain over
  ~96.0 is attributable to SiLU, not compile.

**Risks/edge cases**: (a) compile cost ~20s charged to budget (already accounted; EXP-007/008 precedent);
(b) SiLU is the dominant suspect for any change — param count UNCHANGED (4,299,866) confirms no accidental
architecture change; (c) most likely outcome is a sub-0.2pp delta within the noise band (no-improvement) — the
model is converged/generalization-bound and activation swaps are typically small on compact CIFAR ResNets.

## Configuration Changes
- Activation: `F.relu` → `F.silu` at all 3 forward sites (no new hyperparameter; SiLU is parameter-free)
- Execution: training forward via `torch.compile(model, mode="reduce-overhead")`; eval on eager `model`
- ALL else UNCHANGED: WIDTH_MULT 4, NUM_BLOCKS 3, Cutout(16), PEAK_LR 0.2, WARMUP_FRAC 0.05, WD 1e-4, label
  smoothing 0.1, batch 128, MOMENTUM 0.9 + Nesterov, bf16, channels_last, cosine, MAX_STEPS 10_000_000, seed 42,
  eval harness frozen.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background via Bash run_in_background).
- Resources: 1× NVIDIA H20 (GPU 0; GPU 1 free). VRAM ≈ EXP-007 (~500 MB of 98 GB).
- Estimated runtime: ~300s training (incl. ~20s one-time compile) + ~60–90s startup/eval ≈ 6–8 min. Expect
  num_epochs ~80–89.
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^final_test_loss:|^num_params:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` or diverging after warmup → kill, treat as crash.
- Python traceback in `run.log` (compile failure, shape error, empty `best_test_acc:`) → crash; `tail -n 50 run.log`.
- Recompile spam / repeated multi-second stalls mid-training cutting throughput → record; if it pushes wall-clock
  toward the 10-min limit, treat as failure.
- No new output in `run.log` for > 3 min while training (allow for the one-time ~20s compile) → kill (hang).
- Total wall-clock > 10 min → kill, failure.
- num_epochs collapsing well below ~70 → NOT an abort, but record: SiLU+compile cost more than expected (note for
  analysis; unlikely given compile and SiLU's cheapness).

## Verification Protocol

### Verification Procedure
Run from project root after completion. Baseline = **96.00** (`exp-index.sh baseline`); success bar = **96.10**.

1. **Clean completion within budget** (necessary cond 2): `grep -aE "^best_test_acc:|^total_seconds:" run.log`;
   `tail -n 50 run.log` for tracebacks. PASS if `best_test_acc:` present, `total_seconds < 600`, no traceback.
   Timeout 10 min. FAIL → crash.
2. **Metric improvement** (necessary cond 1): parse `best_test_acc`. PASS if `best_test_acc >= 96.10`
   (= baseline 96.00 + 0.1). Else → no-improvement. Stop here on fail.
3. **No constraint violations** (necessary cond 3): `git diff --name-only autoresearch/dev` = only `train.py`;
   no `pyproject.toml`/`uv.lock` diff (SiLU + compile = core torch, no new dep); eval-line count
   (`grep -c "eval ep" run.log`) == num_epochs (eval once/epoch); seed unchanged (`grep manual_seed train.py` → 42);
   num_params == 4,299,866 (no architecture change). PASS if all hold, else → invalid. All necessary conditions
   must PASS; stop at first failure.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~80–89 (fair converged test).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — vs EXP-003's 0.204 / compiled-k4's 0.208 (does SiLU
  lower the loss?).
- num_params: `grep -aE "^num_params:" run.log` — expect 4,299,866 (confirms parameter-free swap).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` (soft-constraint awareness).
