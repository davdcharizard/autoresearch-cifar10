# Plan EXP-015: Halve weight decay (WEIGHT_DECAY 5e-4 → 2.5e-4)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-015.md

## Milestones

### Milestone 1: One-constant change implemented, passing static checks
- [x] Branch `autoresearch/exp-015` created from `autoresearch/dev`
- [x] `WEIGHT_DECAY = 5e-4` → `WEIGHT_DECAY = 2.5e-4` in train.py; 1-line diff confirmed via `git diff --stat` (1 file, +1/−1)
- [x] `uv run python -m py_compile train.py` exits 0 (SYNTAX_OK)

### Milestone 2: Run launched into a clean window with inline watchdog
- [x] GPU 0 confirmed free (zero compute apps) IN THE SAME command chain as the launch (EXP-014 lesson: windows close — the check, launch, and watchdog must be one script)
- [x] Composite background launcher started (task b4a4n6lud): pre-check passed → training launched → inline watchdog armed (15s sampling, auto-kill on 4 consecutive >30ms windows)
- [x] Early trajectory noted from run.log on completion (download lines at start — `git clean` had removed `data/`; absorbed in startup, see exp-log)

### Milestone 3: Run completed within caps, clean-run sanity passed
- [x] Run exited cleanly (rc=0); summary present; total_seconds 509.8 ≤ 600
- [x] CONTENTION SANITY: 139/139 epochs, zero watchdog SLOW events, post-hoc profile 0/266 windows > 30ms (mean 22.4ms) — clean
- [x] Metrics extracted: best_test_acc 96.41, peak_vram_mb 1613.0
- [x] Diagnostic trail captured: ep 1/20/60/100/120/130/139 evals + final_test_loss 0.1901 — classification: "pressure was AT optimum" (flat converged tail, no over-fit decay)

### Milestone 4: Verification rendered
- [x] Pre-condition CLEAN; condition 1 FAILED (96.41 < 96.81); conditions 2–3 skipped per first-failure stop; recorded in exp-log-015.md

## Code Changes
- **train.py** (only editable file; prepare.py/pyproject.toml/uv.lock hook-protected): line 26, `WEIGHT_DECAY = 5e-4` → `WEIGHT_DECAY = 2.5e-4`. No other edits.

  Why this tests the hypothesis: WD applies only to the decay_params group (conv/linear weights, ndim>1); halving it reduces total regularization pressure — the direction the saturated four-point augmentation dose-response points (goal-learnings § Patterns High) — and, per the WD-with-BN literature (brainstorm-015 § Web Search), lets weight norms grow, mildly cooling the late-schedule EFFECTIVE learning rate on the heat curve's untested cold side. The optimizer param-group structure already isolates the knob; no code-path change.

  Risks/edge cases: (a) under-regularization shows as an over-fitting tail (test_acc peaks mid/late then decays while train loss keeps falling) — the max-statistic tolerates a mid-run peak, and the trajectory shape is the diagnostic we want; (b) effective-LR cooling could shave eval variance at convergence (EXP-011 max-statistic lesson) — netted in the result either way; (c) zero interaction with throughput/memory/data pipeline — signatures must be byte-identical (1613MB, ~139 epochs, 4,286,026 params); any deviation is a red flag for contamination, not the change.

## Configuration Changes
- WEIGHT_DECAY: 5e-4 -> 2.5e-4 (halving = standard one-octave probe; 5e-4 is 2016 step-decay folklore never re-tuned for this 139-epoch one-cycle + LS+TA+RE recipe; literature says the right dose is setup-specific — arXiv 2310.04415, 1810.12281, 1706.05350)

## Execution Environment
- Method: local — composite background script (GPU-0 pre-check → `uv run train.py > run.log 2>&1` → inline watchdog) from project root, GPU 0 only; if GPU 0 busy at pre-check, the script aborts and we re-arm a free-window watcher (`until` loop on zero compute apps), never GPU 1
- Resources: 1x H20 (GPU 0); VRAM ~1613MB expected (must equal baseline)
- Estimated runtime: ~483s total (300s timed budget + ~139 evals + ~12s warm-cache startup); hard cap 600s
- Log output: all stdout/stderr to `run.log` (no tee/stream per goal procedure); deleted after the experiment concludes
- Tool skill: none (background Bash; contention per infra-errors.md § Important — windowed pct_done-delta detection, inline watchdog auto-kill, post-hoc profile; PID matching forbidden)

## Abort Criteria
- Loss NaN/inf (not expected — WD-down cannot destabilize; would indicate something else broke)
- Watchdog auto-kill: 4 consecutive windowed samples > 30ms/step (contention) → wait for clean window, relaunch (infrastructure retry, max 2)
- dt-gate: projected epochs < 65 from early windowed samples (impossible for a scalar change — treat as infra)
- No new output in run.log for > 3 minutes
- Total wall clock reaching 590s with the run still going

## Verification Protocol

### Verification Procedure
Run from project root after the run exits. Baseline from `exp-index.sh baseline` = **96.71** (commit 1990397), direction higher; bar = 96.71 + 0.10 = **96.81**.

0. **Pre-condition (contention sanity, Protocol Findings EXP-011/EXP-014)**: num_epochs within ~10% of ~139 AND zero watchdog SLOW events AND post-hoc windowed profile ≈ 0 windows > 30ms (mean ~22.4ms). Otherwise infrastructure-contaminated — do NOT evaluate conditions; rerun (max 2 retries).
1. **best_test_acc ≥ 96.81** — `grep "^best_test_acc:" run.log`; PASS if parsed value ≥ 96.81. Empty grep ⇒ crash ⇒ `tail -n 50 run.log`. Timeout: n/a (post-hoc parse).
2. **Run completed within budget (≤ 600s total)** — `grep "^total_seconds:" run.log`; PASS if ≤ 600 and summary block exists (clean exit). Timeout: n/a.
3. **Validation at most once per epoch** — `grep -c "eval ep" run.log` ≤ num_epochs from the summary. Structurally guaranteed; checked mechanically. Timeout: n/a.

First failed condition stops evaluation. On all-pass: verdict improvement; `exp-index.sh insert`; commit the one-line diff on `autoresearch/exp-015`; merge to `autoresearch/dev` (no remote — skip fetch/push/PR per TASK.md, fields stay "(pending)"/"N/A"). On condition-1 failure with a clean run: no-improvement; capture the diagnostic trail (Milestone 3), discard changes (`git checkout -- .`), return to `autoresearch/dev`. Delete run.log after analysis either way.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1613MB (byte-identical claim check)
- num_epochs: `grep "^num_epochs:" run.log` — expect ~139 (throughput untouched)
- num_params: `grep "^num_params:" run.log` — must equal 4,286,026 (architecture unchanged)
- final_test_loss: `grep "^final_test_loss:" run.log` — over/under-fit diagnostic vs baseline region ~0.19
