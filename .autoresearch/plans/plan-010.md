# Plan EXP-010: PEAK_LR 0.4 → 0.6 on the compiled 4x recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md

## Prior-Knowledge Check

Not a retry of any Failed Approaches entry (capacity and fourth-regularizer entries are architecture/augmentation changes; this is the first optimization-hyperparameter experiment). Composes with every validated Pattern unchanged — in particular the time-keyed one-cycle (High), whose elapsed-fraction keying makes a hotter peak safe: the cosine anneal to ~0 always completes, so the failure mode is graceful under-recovery, not divergence. Consistent with the EXP-009 saturation learning, which names base-hyperparameter re-tuning as a remaining headroom direction. No High project-insights entry is contradicted (no architecture change → alignment rule moot; no cross-regime throughput projection involved). infra-errors.md: empty.

## Milestones

### Milestone 1: Experiment branch + LR change implemented
- [x] Create experiment branch `autoresearch/exp-010` from `autoresearch/dev`
- [x] In train.py change `PEAK_LR = 0.4` to `PEAK_LR = 0.6`; update the inline comment (no longer pure linear scaling — tuned upward for the augmented 4x recipe per super-convergence headroom)
- [x] Static check passes: `uv run ruff check train.py`
- [x] Scope check: `git status --porcelain` shows only `train.py` modified (1 line)

### Milestone 2: Experiment running on GPU 0 with early health checks passed
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [x] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Early signal: params 4,286,026 (unchanged); dt 22ms at steps 100–150; epoch-1 eval acc 34.39% ≥ 15%
- [ ] Mid-schedule expectation: eval accuracy MAY sit below EXP-006's trajectory at peak LR (hotter peak = stronger implicit regularization) — this is the predicted signature, NOT an abort signal; only NaN/collapse aborts

### Milestone 3: Run completed and metrics extracted
- [x] Process exited; `grep "^best_test_acc:" run.log` non-empty
- [x] `total_seconds:` ≤ 600
- [x] Full summary block recorded into logs/exp-log-010.md, including num_epochs (139) and the trajectory shape (incomplete recovery — still creeping at cutoff)

## Code Changes

- **train.py** (only file modified — hard constraint): single constant change, `PEAK_LR = 0.4` → `0.6`, comment updated. Architecture, schedule shape (WARMUP_FRAC 0.15, cosine to ~0), augmentation, WD, momentum, batch, compile — all byte-identical to baseline 1990397.

  Why this tests the hypothesis: peak LR is the dominant untouched optimization constant; a single-variable step isolates whether 0.4 (linearly scaled for the EXP-000 recipe) sits below the optimum for the current heavily-augmented 4x recipe.

  Risks/edge cases: too-hot peak → mid-schedule accuracy chaos that the anneal cannot fully recover (graceful −0.2 to −0.5pp, still brackets the search); genuine divergence (NaN) unlikely with BN + bf16 + nesterov at 0.6 but covered by abort criteria; no throughput/VRAM/epoch-count interaction (pure scalar in the LR closure).

## Configuration Changes
- PEAK_LR: 0.4 → 0.6 (1.5x). Rationale: super-convergence (arXiv 1708.07120) demonstrates stable one-cycle peaks ≥1.0 on CIFAR ResNets; heavy augmentation (TA+RE) and 4x width both shift the optimal peak upward; 0.4 was set by linear scaling in EXP-000 for an unaugmented 1x net and never re-tuned. 1.5x is a meaningful, safely-interior step that brackets the search in either outcome.
- No other changes (single-variable experiment)

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM ~1.62GB expected (unchanged)
- Estimated runtime: ~8.5 min total (300s training + ~10–25s startup + ~137 evals ≈ 120s + loader stalls; EXP-009 measured 482.8s at the same epoch count → expect ~480–500s)
- Log output: stdout/stderr → `run.log`; metrics via grep; delete run.log after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill, treat as failure (hard constraint)
- No output in run.log within 120s of launch → kill and diagnose
- No `eval ep   1` line within 300s of launch → kill and diagnose
- Training loss NaN/inf at any point → kill immediately (the one real hot-LR risk); record as research failure (LR 0.6 unstable), do NOT retry with tweaks
- Epoch-1 eval accuracy < 15% → kill, diagnose
- Sustained accuracy collapse mid-schedule (eval < 30% after epoch 20 while loss is finite) → kill, treat as LR-instability research failure
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
- num_epochs: `grep "^num_epochs:" run.log` (expect ~137–139 — must be unchanged; LR cannot affect throughput)
- startup_seconds: `grep "^startup_seconds:" run.log` (expect ~10–25s)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~1.62GB, unchanged)
- num_params: `grep "^num_params:" run.log` (expect 4,286,026 — must be unchanged)
