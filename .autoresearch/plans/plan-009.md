# Plan EXP-009: Mixup (α=0.2) on the compiled 4x recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md

## Prior-Knowledge Check

Not a retry of any Failed Approaches entry: the capacity entries (High count 3 + Low) are architecture changes — this experiment keeps the proven 4x ResNet-20 topology untouched. The closest prior datapoint is "augmentation stacking diminishing returns" (Patterns, EXP-003/004 — never a failure), and the brainstorm documents why mixup is a different regularizer class (input-label space vs spatial) measured under better conditions (139 vs 114 epochs). Composes with all validated Patterns: time-keyed one-cycle, compile + warmup, TA, RE, selective WD. dt-gate protocol (Medium) not binding here — epoch count is not uncertain (architecture unchanged) — but a dt sanity bound is included anyway.

## Milestones

### Milestone 1: Experiment branch + mixup implemented
- [x] Create experiment branch `autoresearch/exp-009` from `autoresearch/dev`
- [x] Add `MIXUP_ALPHA = 0.2` constant; insert per-batch mixup in the training step (sample lam, permute, mix inputs, two-term lam-weighted CE — see Code Changes); architecture and all other hyperparameters unchanged
- [x] Static check passes: `uv run ruff check train.py`
- [x] Scope check: `git status --porcelain` shows only `train.py` modified (+13/−1 lines)

### Milestone 2: Experiment running on GPU 0 with early health checks passed
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [x] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Early signal: params 4,286,026 (unchanged); dt 23ms ≤ 24ms at steps 100–150; epoch-1 eval acc 33.87% ≥ 15%
- [x] Loss-scale sanity: smoothed train loss 2.13 at step 150, above prior runs as expected (mixed targets)

### Milestone 3: Run completed and metrics extracted
- [x] Process exited; `grep "^best_test_acc:" run.log` non-empty
- [x] `total_seconds:` ≤ 600
- [x] Full summary block recorded into logs/exp-log-009.md, including num_epochs (137 — within the 135–139 prediction)

## Code Changes

- **train.py** (only file modified — hard constraint): one new constant and ~8 lines in the training step. Conceptually:
  1. Constant block: add `MIXUP_ALPHA = 0.2` after `LABEL_SMOOTHING`.
  2. In the per-step loop, after `inputs`/`targets` land on device (channels_last) and BEFORE the autocast block:
     ```python
     lam = float(torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA).sample())
     perm = torch.randperm(inputs.size(0), device=device)
     inputs = torch.lerp(inputs[perm], inputs, lam).contiguous(memory_format=torch.channels_last)
     targets_perm = targets[perm]
     ```
     (`torch.lerp(b, a, lam) = lam*a + (1-lam)*b`; the explicit `.contiguous(channels_last)` guards against index_select dropping the memory format.)
  3. Loss inside autocast becomes the two-term mixup CE, each term keeping label smoothing:
     ```python
     loss = lam * F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING) \
          + (1 - lam) * F.cross_entropy(outputs, targets_perm, label_smoothing=LABEL_SMOOTHING)
     ```
  All mixing happens OUTSIDE the compiled module (model input shape/dtype unchanged → no recompilation; compile warmup block stays as-is). Eval path untouched (`base_model`, clean inputs).

  Why this tests the hypothesis: single regularizer addition in a new mechanism class on the otherwise-frozen best recipe; any delta vs 96.71 is attributable to mixup.

  Risks/edge cases: lam-permutation bug would silently degrade — guarded by epoch-1 health check and the expectation that mid-run evals track EXP-006's trajectory within a few pp; per-step Beta sample on CPU is ~µs; `inputs[perm]` allocates one extra batch (~6MB) — negligible VRAM; train-loss EMA not comparable to prior runs (expected).

## Configuration Changes
- MIXUP_ALPHA: (new) 0.2 — timm-style mild setting; the paper's α=1.0 risks over-regularization stacked on LS 0.1 + TA + RE at 4.29M params
- No other changes: NUM_BLOCKS 3, WIDTH_MULT 4, BATCH_SIZE 512, PEAK_LR 0.4, schedule/compile/augmentation identical to baseline 1990397

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM ~1.65GB expected (baseline +1 batch buffer)
- Estimated runtime: ~8.5 min total (300s training + ~23s compile startup + ~135–139 evals ≈ 120–140s eval+loader overhead; EXP-006 measured 491s total at the same epoch count → expect ~490–500s)
- Log output: stdout/stderr → `run.log`; metrics via grep; delete run.log after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill, treat as failure (hard constraint)
- No output in run.log within 120s of launch → kill and diagnose
- No `eval ep   1` line within 300s of launch → kill and diagnose
- Training loss NaN/inf, or epoch-1 eval accuracy < 15% → kill, diagnose (a mixup-bug signature would be epoch-1 acc near 10% = random)
- dt ≥ 26ms sustained at step ~100 → mixing is unexpectedly expensive; let the run finish (epochs ~120 still fine) but flag prominently in exp-log
- Empty `grep "^best_test_acc:" run.log` after exit → crash; read `tail -n 100 run.log`

## Verification Protocol

### Verification Procedure
Run from project root after process exit. Baseline via `exp-index.sh baseline` = **96.71** (commit 1990397) at planning time → pass threshold **≥ 96.81** (+0.1 pp).

1. **Run completes without crashing within budget (≤ 10 min total)**
   - Command: `grep "^best_test_acc:\|^total_seconds:" run.log`
   - Pass: `best_test_acc:` present AND `total_seconds:` ≤ 600. Timeout: kill if alive >10 min → fail.
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.81)**
   - Command: `grep "^best_test_acc:" run.log` → parse; compare against fresh `exp-index.sh baseline` at verification time.
   - Pass: value ≥ 96.81. Evaluation stops at first failure.
3. **Validation at most once per epoch**
   - Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log`
   - Pass: eval-line count ≤ num_epochs.

### Informational Metrics (Optional)
- num_epochs: `grep "^num_epochs:" run.log` (prediction 135–139 — confirms mixing is throughput-free)
- startup_seconds: `grep "^startup_seconds:" run.log` (expect ~23s, unchanged)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~1.65GB)
- num_params: `grep "^num_params:" run.log` (expect 4,286,026 — must be unchanged)
