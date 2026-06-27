# Plan EXP-007: Aligned width 6x on the compiled recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md

## Failed-Approach Retry Justification

This retries "capacity increase without throughput increase" (goal-learnings § Failed Approaches, count 2 — EXP-002, EXP-005). The retry is justified because both root causes are removed this time: (1) throughput is 1.22x higher (torch.compile, EXP-006), and EXP-006 demonstrated the exact mechanism this plan depends on — extra epochs convert into accuracy; (2) 6x stage widths 96/192/384 are all multiples of 32, complying with the project-insights High-Importance alignment rule that EXP-005's 80/160/320 violated. Projected epochs ~75–80 (139 / 2.25^0.76), above the ~70 starvation floor — vs 40 and 52 in the failures.

## Milestones

### Milestone 1: Experiment branch + width change implemented
- [x] Create experiment branch `autoresearch/exp-007` from `autoresearch/dev`
- [x] Change `WIDTH_MULT = 4` to `WIDTH_MULT = 6` in train.py (stage widths 96/192/384 — all 32-aligned); update the inline comment accordingly
- [x] Static check passes: `uv run ruff check train.py`
- [x] Scope check: `git status --porcelain` shows only `train.py` modified

### Milestone 2: Experiment running on GPU 0
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [x] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Early signal: params ~9.6M (quadratic-in-width from 4,286,026 at 4x); step dt ~47–55ms (aligned scaling ≈ 2.25^0.76 x 22ms — a dt ≥ ~70ms would signal an EXP-005-style kernel-efficiency problem); epoch-1 eval acc ≥ 15% — NOTE: dt measured 57ms, slightly above prediction window but below the 70ms anomaly bar; ~54 epochs projected

### Milestone 3: Run completed and metrics extracted
- [x] Process exited; `grep "^best_test_acc:" run.log` non-empty
- [x] `total_seconds:` ≤ 600
- [x] Full summary block recorded into logs/exp-log-007.md, including num_epochs (the width-epoch curve datapoint)

## Code Changes

- **train.py** (only file modified — hard constraint): single constant change, `WIDTH_MULT = 4` → `6` (comment updated to "-> (96,192,384)"). The compiled doubly-regularized recipe (torch.compile + warmup, TA, RE, time-keyed one-cycle peak 0.4, batch 512, selective WD, label smoothing) stays byte-identical to baseline 1990397.

  Why this tests the hypothesis: single-variable capacity step; any delta vs 96.71 is attributable to the width-epoch tradeoff under the unlocked throughput.

  Risks/edge cases: epochs land below projection (compile speedup may shrink at larger channel counts as the model becomes more tensor-core-saturated) → undertraining; PEAK_LR 0.4 known stable at 8x width (EXP-002), so no LR confound; VRAM ~3.5GB (quadratic-ish from 1.64GB) — far under limit; compile time may grow modestly with model size (lands in startup; watch startup_seconds).

## Configuration Changes
- WIDTH_MULT: 4 → 6 (~4.29M → ~9.6M params; stage widths 96/192/384, all multiples of 32 per project-insights alignment rule)
- Rationale: capacity was throughput-throttled (EXP-006); 6x is the canonical aligned step whose projected ~75–80 epochs clear the measured ~70-epoch starvation floor
- No other changes (single-variable experiment)

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM ~3.5GB expected
- Estimated runtime: ~7 min total (300s training + ~30–60s compile startup + ~75–80 evals ≈ 70s; total ≈ 430–460s)
- Log output: stdout/stderr → `run.log`; metrics via grep; delete run.log after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill, treat as failure (hard constraint)
- No output in run.log within 120s of launch → kill and diagnose
- No `eval ep   1` line within 300s of launch (compile hang at larger graph) → kill and diagnose
- Training loss NaN/inf, or epoch-1 eval accuracy < 15% → kill, diagnose
- dt ≥ ~70ms sustained in step lines → alignment-style throughput anomaly; let the run finish (the datapoint is still valuable) but flag prominently in exp-log
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
- num_epochs: `grep "^num_epochs:" run.log` (prediction 75–80; the width-epoch curve datapoint — record regardless of verdict)
- startup_seconds: `grep "^startup_seconds:" run.log` (compile cost at 6x; expect ≤ ~60s)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~3.5GB)
- num_params: `grep "^num_params:" run.log` (expect ~9.6M)
