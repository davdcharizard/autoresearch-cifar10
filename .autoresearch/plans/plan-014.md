# Plan EXP-014: Shorten LR warmup (WARMUP_FRAC 0.15 → 0.08)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md

## Milestones

### Milestone 1: One-constant change implemented, passing static checks
- [x] Branch `autoresearch/exp-014` created from `autoresearch/dev`
- [x] `WARMUP_FRAC = 0.15` → `WARMUP_FRAC = 0.08` in train.py; 1-line diff confirmed via `git diff --stat` (1 file, +1/−1)
- [x] `uv run python -m py_compile train.py` exits 0 (SYNTAX_OK)

### Milestone 2: Run launched into a clean window, early signals healthy
- [x] GPU 0 confirmed free immediately before launch — twice: Run 1 launched into a window that later CLOSED (foreign job returned mid-run, 48ms time-slicing, 92 epochs — killed by protocol, quarantined, see exp-log Errors & Dead Ends); Run 2 relaunched clean with the watchdog inline in the launch chain
- [x] Run 2 started (task bh32axche); inline watchdog armed in the same command chain (Run-1 lesson)
- [x] dt-gate: windowed 22.4ms mean from the start ⇒ ~139 epochs projected
- [x] LR ramp confirmed faster (peak reached ~step 1075 region per eval-trail shape)
- [x] Trajectory check: mid-schedule ran BELOW baseline trail (ep 20: 78.5 vs ~88; ep 60: 87.8 vs ~92) — hypothesis failing in direction; not an abort per plan, run allowed to converge

### Milestone 3: Run completed within caps, clean-run sanity passed
- [x] Run 2 exited cleanly (rc=0); summary present; total_seconds 482.9 ≤ 600
- [x] CONTENTION SANITY: 139/139 epochs exact, zero SLOW watchdog events, post-hoc profile 0/267 windows > 30ms — cleanest run on record
- [x] Metrics extracted: best_test_acc 96.49, peak_vram_mb 1613.0

### Milestone 4: Verification rendered
- [x] Pre-condition CLEAN; condition 1 FAILED (96.49 < 96.81); conditions 2–3 skipped per first-failure stop; recorded in exp-log-014.md

## Code Changes
- **train.py** (only editable file; prepare.py/pyproject.toml/uv.lock hook-protected): line 24, `WARMUP_FRAC = 0.15` → `WARMUP_FRAC = 0.08`. No other edits.

  Why this tests the hypothesis: `lr_at(progress)` keys the one-cycle schedule to elapsed-budget-fraction; WARMUP_FRAC is the linear-ramp endpoint. Cutting 0.15 → 0.08 moves ~21s (≈9 epochs) of the 300s timed budget from sub-peak ramp into the near-peak/cosine-anneal phase, strictly raising the LR-time integral with zero effect on throughput, memory, data pipeline, or regularization. The stability margin is established in-project: EXP-010 ran peak 0.6 (1.5x) with zero instability, and the NeurIPS 2024 warmup study (brainstorm-014 § Web Search) says warmup's only first-order role is peak-LR tolerance, typical length 1–5% of training.

  Risks/edge cases: (a) the first ~24s now sweep LR 0→0.4 twice as fast — per-step LR increments are still tiny (~4e-4/step) so no instability expected; loss spikes would show in the smoothed-loss prints; (b) cosine phase normalization `q = (progress − WARMUP_FRAC)/(1 − WARMUP_FRAC)` adapts automatically — no other constant depends on WARMUP_FRAC; (c) interaction with the max-statistic is neutral (no variance manipulation — this is a mean-shift lever).

## Configuration Changes
- WARMUP_FRAC: 0.15 -> 0.08 (arXiv 2406.09405: warmup beyond peak-LR-tolerance is waste, 1–5% typical; EXP-010 proves tolerance at 0.4 has ≥1.5x margin; conservative halving rather than jump to 0.02–0.05 so a clean single-constant dose point is added to the schedule-shape curve)

## Execution Environment
- Method: local — `uv run train.py > run.log 2>&1` from project root on GPU 0 only (confirm free first via nvidia-smi; if busy, wait — never use GPU 1)
- Resources: 1x H20 (GPU 0); VRAM ~1613MB expected (must be identical to baseline — pure scalar change)
- Estimated runtime: ~481s total (300s timed budget + ~139 evals + ~11s warm-cache startup; ~23s if inductor cache cold); hard cap 600s
- Log output: all stdout/stderr to `run.log` (no tee/stream per goal procedure); deleted after the experiment concludes
- Tool skill: none (background Bash + Monitor watcher; contention protocol per infra-errors.md § Important — detect via pct_done deltas between step prints, NOT printed dt, NOT PID matching)

## Abort Criteria
- Loss NaN/inf or smoothed loss spiking upward persistently during the faster ramp (first ~25s) — instability the brainstorm assumed away
- dt-gate: projected epochs < 65 at step ~100 (impossible for a scalar change unless infra is broken — treat as infra, not research)
- Contention: windowed step time > ~30ms (pct_done-delta method) or collapsing epoch cadence → kill, wait for clean window, relaunch (infrastructure retry, max 2)
- No new output in run.log for > 3 minutes
- Total wall clock reaching 590s with the run still going

## Verification Protocol

### Verification Procedure
Run from project root after the run exits. Baseline from `exp-index.sh baseline` = **96.71** (commit 1990397), direction higher; bar = 96.71 + 0.10 = **96.81**.

0. **Pre-condition (contention sanity, Protocol Findings EXP-011)**: num_epochs within ~10% of the step-100 projection (~139 expected) AND zero SLOW detector events. Otherwise the run is infrastructure-contaminated — do NOT evaluate conditions; rerun (max 2 retries).
1. **best_test_acc ≥ 96.81** — `grep "^best_test_acc:" run.log`; PASS if parsed value ≥ 96.81. Empty grep ⇒ crash ⇒ `tail -n 50 run.log`. Timeout: n/a (post-hoc parse).
2. **Run completed within budget (≤ 600s total)** — `grep "^total_seconds:" run.log`; PASS if ≤ 600 and the summary block exists (clean exit). Timeout: n/a.
3. **Validation at most once per epoch** — `grep -c "eval ep" run.log` ≤ num_epochs from the summary. Structurally guaranteed by the loop (one eval per epoch); checked mechanically. Timeout: n/a.

First failed condition stops evaluation. On all-pass: verdict improvement; `exp-index.sh insert`; commit the one-line diff on `autoresearch/exp-014`; merge to `autoresearch/dev` (no remote — skip fetch/push/PR per TASK.md, PR field stays "(pending)"/"N/A"). On condition-1 failure with a clean run: no-improvement; discard changes (`git checkout -- .`), return to `autoresearch/dev`. Delete run.log after analysis either way.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1613MB (must match baseline; any change is a red flag for the "pure scalar" claim)
- num_epochs: `grep "^num_epochs:" run.log` — expect ~139 (throughput must be untouched)
- num_params: `grep "^num_params:" run.log` — must equal 4,286,026 (unchanged architecture)
