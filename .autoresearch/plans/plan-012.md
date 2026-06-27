# Plan EXP-012: BATCH_SIZE 1024 + PEAK_LR 0.8 (linear scaling — last throughput lever)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md

## Milestones

### Milestone 1: Constants changed, passing static checks
- [x] Branch `autoresearch/exp-012` created from `autoresearch/dev`
- [x] `BATCH_SIZE = 512` → `1024`; `PEAK_LR = 0.4` → `0.8` (comment updated to `0.1 x (1024/128)`); NOTHING else touched (2-line diff confirmed via git diff --stat)
- [x] `uv run python -m py_compile train.py` exits 0 (SYNTAX_OK)

### Milestone 2: Run launched into a clean window, early signals healthy
- [x] GPU 0 confirmed free immediately before launch (both GPUs 0 compute apps, 0% util — fully idle node)
- [x] `uv run train.py > run.log 2>&1` started in background (task bm70wed4e); contention detector armed (bhxorltu2, SLOW threshold 55ms)
- [x] Batches per epoch printed as 48
- [x] dt-gate: 41–42ms at steps 350–500 → img/s 24.7k (+8% vs baseline) → projected ~150 epochs. PASSED (above no-gain band 142, far above kill floor 65); pct/print cross-check clean
- [x] Epoch-1 eval 31.62% — sane raw-eval range, no LR-0.8 instability signature

### Milestone 3: Run completed within caps, clean-run sanity passed
- [x] Run exits cleanly; total_seconds 560.8 ≤ 600 (tightest yet: 151 evals + 23.8s cold-cache compile for the new shape)
- [x] CONTENTION SANITY: 151 epochs vs ~150 projected (1%); zero SLOW events — clean run
- [x] `grep "^best_test_acc:\|^peak_vram_mb:" run.log` → 96.66 / 3134.6

### Milestone 4: Verification rendered
- [x] Conditions: pre-check CLEAN, 1 PASS (560.8s), 2 FAIL (96.66 < 96.81), 3 skipped per first-failure stop; recorded in exp-log-012.md

## Code Changes
- **train.py** (only file; all other paths hook-protected): two constants in the hyperparameter block.
  1. `BATCH_SIZE = 512` → `BATCH_SIZE = 1024`. Steps/epoch drops 97 → 48 (drop_last). The compile-warmup synthetic batch, loader, and time-keyed LR schedule all key off the constant — no other code needs to change.
  2. `PEAK_LR = 0.4  # linear scaling: 0.1 x (512/128)` → `PEAK_LR = 0.8  # linear scaling: 0.1 x (1024/128)`.

  Why this tests the hypothesis: img/s = BATCH_SIZE/dt. At 512 the H20 runs 22ms (≈23k img/s) on tiny 32x32 kernels with utilization headroom; if a 1024-batch step costs < 2x (≤ ~44ms), img/s rises and the fixed 300s holds MORE epochs — the only mechanism with a 100% conversion record (EXP-000/006). Scaling LR with batch (Goyal linear rule — the same rule that set 0.4@512 in this file) preserves gradient-noise scale, so this is NOT a rerun of EXP-010's fixed-batch LR raise: there the noise scale changed; here it is held constant by construction.

  Risks/edge cases: (a) torch.compile recompiles for the new shape — absorbed by the pre-loop warmup (startup, not budget; possibly +10–20s startup on a cold cache for this shape); (b) cudnn.benchmark re-autotunes once per new shape at warmup — same absorption; (c) VRAM roughly doubles activations (~1.6 → ~3.2GB) — trivial on H20, soft constraint; (d) loader demand rises to ~27k img/s vs ~23–25k worker production — stalls grow but land OUTSIDE the timed budget (fetch precedes t0), costing wall clock only (~120s headroom); (e) large-batch generalization gap could offset the epoch gain — that is the research question.

## Configuration Changes
- BATCH_SIZE: 512 -> 1024 (utilization headroom at 22ms/512; airbench trains a similar-scale net at 1024 — knowledge/README.md, arXiv 2404.00498; H20 channel-alignment insight untouched since widths don't change)
- PEAK_LR: 0.4 -> 0.8 (linear scaling 0.1 × 1024/128, preserving gradient-noise scale; distinct from EXP-010's fixed-batch raise — that failure's mechanism, deferred progress from a hotter noise regime, does not apply when batch scales too)
- Everything else byte-identical to baseline 1990397 (schedule shape, augmentation, WD, compile, eval path).

## Execution Environment
- Method: local — `uv run train.py > run.log 2>&1` from project root, GPU 0 only (CUDA_VISIBLE_DEVICES=0; confirm free first, wait if busy — never GPU 1)
- Resources: 1x H20 (GPU 0); VRAM ~3.2GB
- Estimated runtime: ~500s total (300s budget + ~150 evals × ~0.85s + ~11s startup + loader stalls); hard cap 600s
- Log output: all stdout/stderr to `run.log` (no tee/stream); deleted after the experiment concludes
- Tool skill: none (background Bash + Monitor watchers per established practice; contention protocol per infra-errors.md § Important)

## Abort Criteria
- Loss NaN/inf at any step (large-batch warmup instability would appear in the first ~2 epochs)
- dt-gate: projected epochs < 65 at step ~100 (dt ≥ ~96ms — not expected; protocol floor from goal-learnings)
- Contention: SLOW events (windowed step time > 55ms from pct_done deltas) or epoch cadence collapsing mid-run → kill, wait for a clean window, relaunch (infrastructure retry, max 2; per infra-errors.md the metric of a contended run is never analyzed)
- No new output in run.log for > 3 minutes
- Total wall clock reaching 590s with the run still going (the 10-minute cap is a hard failure boundary)

## Verification Protocol

### Verification Procedure
Run from project root after the run exits. Baseline from `exp-index.sh baseline` = **96.71** (commit 1990397), direction higher; bar = 96.71 + 0.10 = **96.81**.

0. **Pre-condition (contention sanity, Protocol Findings EXP-011)**: num_epochs from the summary block must be within ~10% of the step-100 projection 300/(48 × measured_dt), and the SLOW detector must have stayed silent. If violated, the run is infrastructure-contaminated: do NOT evaluate conditions on it; rerun instead.
1. **Run completed within budget (≤ 600s total)** — `grep "^total_seconds:" run.log`; PASS if ≤ 600 and the summary block exists (clean exit). Empty grep ⇒ crash ⇒ `tail -n 50 run.log`. Timeout: n/a (post-hoc parse).
2. **best_test_acc ≥ 96.81** — `grep "^best_test_acc:" run.log`; PASS if parsed value ≥ 96.81. Timeout: n/a.
3. **Validation at most once per epoch** — `grep -c "eval ep" run.log` ≤ `grep "^num_epochs:" run.log` value. Structurally guaranteed; checked mechanically. Timeout: n/a.

First failed condition stops evaluation. On all-pass: verdict improvement; index insert; commit the two-constant diff on `autoresearch/exp-012`; merge to `autoresearch/dev` (no remote — skip push/PR per TASK.md). On condition-2 failure with a clean run: no-improvement; discard changes. Delete run.log after analysis either way.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~3.2GB (soft constraint, document)
- num_epochs: `grep "^num_epochs:" run.log` — THE mechanism check: ≥ ~148 means the throughput hypothesis held; ~139–142 means break-even (pure linear-scaling test); < 139 means dt scaled super-linearly
- num_params: `grep "^num_params:" run.log` — must equal 4,286,026 (architecture untouched)
