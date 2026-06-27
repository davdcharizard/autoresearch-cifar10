# Plan EXP-008: Wider-but-shallower — NUM_BLOCKS 2 + WIDTH_MULT 6 (ResNet-14)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md

## Failed-Approach Retry Justification

This touches "capacity increase without throughput increase / epoch starvation" (goal-learnings § Failed Approaches, **High, count 3** — EXP-002/005/007), whose callout forbids retries "unless measured compiled dt projects ≥70 epochs first". This plan differs structurally from all three failures: they widened at FIXED depth (FLOPs strictly up 2.25–4x), while this trades depth FOR width — net FLOPs only ~1.46x (≈937M vs ≈643M MACs/img) for 1.43x params. The epoch projection uses the *measured compiled* scaling from EXP-007 (not eager exponents, per the EXP-007 learning): dt ≈ 22ms x 1.46 x 1.15 inefficiency ≈ 36–40ms → ~80–85 epochs, above the ~70 floor with margin. A hard early-abort guard (dt ≥ 47ms at step ~100, projecting < 65 epochs) caps the residual estimate risk — none of the prior failures had this guard.

## Milestones

### Milestone 1: Experiment branch + topology change implemented
- [x] Create experiment branch `autoresearch/exp-008` from `autoresearch/dev`
- [x] In train.py change `NUM_BLOCKS = 3` to `NUM_BLOCKS = 2` and `WIDTH_MULT = 4` to `WIDTH_MULT = 6`; update both inline comments (ResNet-14 = 6*2+2; stage widths (96,192,384))
- [x] Static check passes: `uv run ruff check train.py`
- [x] Scope check: `git status --porcelain` shows only `train.py` modified

### Milestone 2: Experiment running on GPU 0 with early dt gate passed
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [x] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Early signal: params ≈ 6.1–6.2M printed (6,149,866); epoch-1 eval acc ≥ 15% (39.58%)
- [x] **dt gate (binding, from the EXP-007 High learning)**: steady-state dt at step ~100 ≤ 46ms — PASSED: dt 38ms at steps 100–150, projecting ~80 epochs

### Milestone 3: Run completed and metrics extracted
- [x] Process exited; `grep "^best_test_acc:" run.log` non-empty
- [x] `total_seconds:` ≤ 600
- [x] Full summary block recorded into logs/exp-log-008.md, including num_epochs and dt (the depth-for-width scaling datapoint)

## Code Changes

- **train.py** (only file modified — hard constraint): two constant changes, `NUM_BLOCKS = 3` → `2` (ResNet-14) and `WIDTH_MULT = 4` → `6` (stage widths 96/192/384, all 32-aligned per project-insights High alignment rule). The compiled doubly-regularized recipe (torch.compile + warmup, TA, RE, time-keyed one-cycle peak 0.4, batch 512, selective WD, label smoothing) stays byte-identical to baseline 1990397. The `_make_layer`/`BasicBlock` code paths already support any block count — no structural edits.

  Why this tests the hypothesis: single topology step that raises capacity 1.43x at ~1.46x FLOPs; any delta vs 96.71 is attributable to the depth-for-width trade at a non-starved epoch count.

  Risks/edge cases: dt lands above projection (compile efficiency at this exact shape unknown) → caught by the Milestone-2 dt gate; depth 14 may itself cost accuracy beyond width's recovery (WRN says no at this scale — WRN-16-8 within 0.3pp of WRN-28-10); PEAK_LR 0.4 known stable up to 8x width (EXP-002), shallower nets are easier to optimize, so no LR confound expected; VRAM ~2GB (under the 6x ResNet-20's 2.4GB) — far below limit.

## Configuration Changes
- NUM_BLOCKS: 3 → 2 (ResNet-20 → ResNet-14; 18 → 12 stage conv layers)
- WIDTH_MULT: 4 → 6 (~4.29M → ~6.15M params; stage widths 96/192/384, multiples of 32)
- Rationale: capacity is the highest-payoff axis (+2.07pp at the last non-starved step) but uniform widening is closed (High, count 3); spending depth FLOPs on width is the only capacity move whose measured-regime projection (~80–85 epochs) clears the ~70-epoch starvation floor. External: WRN (arXiv 1605.07146) — shallow-wide ≥ deep-thin on CIFAR-10.
- No other changes (single-variable experiment in the topology dimension)

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM ~2GB expected
- Estimated runtime: ~7.5 min total (300s training + ~21s compile startup + ~80–85 evals ≈ 75s; total ≈ 450–470s)
- Log output: stdout/stderr → `run.log`; metrics via grep; delete run.log after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill, treat as failure (hard constraint)
- No output in run.log within 120s of launch → kill and diagnose
- No `eval ep   1` line within 300s of launch (compile hang) → kill and diagnose
- Training loss NaN/inf, or epoch-1 eval accuracy < 15% → kill, diagnose
- **dt ≥ 47ms sustained at step ~100 → KILL** (projects < ~65 epochs — foregone starvation per the High count-3 failed approach; the measured dt is the datapoint, the accuracy result would add nothing). This is deliberately stricter than EXP-007's flag-only policy: the width-epoch curve is already established, so a starved run has no information value left.
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
- num_epochs: `grep "^num_epochs:" run.log` (prediction 72–95; the depth-for-width throughput datapoint — record regardless of verdict)
- startup_seconds: `grep "^startup_seconds:" run.log` (expect ~20–25s; smaller graph than EXP-007)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~2GB)
- num_params: `grep "^num_params:" run.log` (expect ~6.1–6.2M)
