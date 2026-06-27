# Plan EXP-016: Linear-to-zero anneal (replace cosine post-warmup branch)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md

## Milestones

### Milestone 1: One-line schedule change implemented, passing static + shape checks
- [x] Branch `autoresearch/exp-016` created from `autoresearch/dev`
- [x] In `lr_at()`: cosine return → `return PEAK_LR * (1 - q)` (+ comment line); diff confirmed via `git diff --stat` (1 file, +2/−2)
- [x] `uv run python -m py_compile train.py` exits 0 (SYNTAX_OK)
- [x] Shape spot-check: lr_at(0.075)=0.2, lr_at(0.15)=0.4, lr_at(0.575)=0.2, lr_at(1.0)=0.0 — all exact (SHAPE_OK)

### Milestone 2: Run launched into a clean window with inline watchdog
- [x] GPU 0 confirmed free at pre-check, in the same command chain as the launch
- [x] Composite background launcher started (task bnkebo0fk): pre-check → train → inline watchdog (15s sampling, auto-kill on 4 consecutive >30ms)

### Milestone 3: Run completed within caps, clean-run sanity passed
- [x] Run exited cleanly (rc=0); summary present; total_seconds 510.9 ≤ 600
- [x] CONTENTION SANITY: 138 epochs (~projection), zero watchdog SLOW events, post-hoc profile 0/266 windows > 30ms (mean 22.4ms) — clean
- [x] Metrics extracted: best_test_acc 96.21, peak_vram_mb 1613.0
- [x] Diagnostic trail captured: behind baseline ALL schedule, NO crossover, STILL CLIMBING at cutoff (best at final epoch 138; last-8 evals 95.52→96.21) — cosine's cold tail is load-bearing for the max-statistic

### Milestone 4: Verification rendered
- [x] Pre-condition CLEAN; condition 1 FAILED (96.21 < 96.81); conditions 2–3 skipped per first-failure stop; recorded in exp-log-016.md

## Code Changes
- **train.py** (only editable file; prepare.py/pyproject.toml/uv.lock hook-protected): in `lr_at()` line 38, replace the cosine anneal return with `return PEAK_LR * (1 - q)`. Optionally update the function's 2-line comment to say "linear to 0" instead of "cosine to ~0" (comment-only, keeps the file honest). PEAK_LR, WARMUP_FRAC, warmup branch, and `q` normalization untouched.

  Why this tests the hypothesis: `q = (progress − WARMUP_FRAC)/(1 − WARMUP_FRAC)` already maps the anneal to [0,1]; swapping `0.5(1+cos πq)` for `(1−q)` changes ONLY the anneal family. Both shapes integrate to 0.5 over q∈[0,1], so total LR-time heat — certified at optimum from both sides (EXP-010/014/015) — is held exactly constant; what changes is distribution: linear is cooler than cosine for q<0.5 (e.g. 0.75 vs 0.85 of peak at q=0.25) and hotter for q>0.5 (0.25 vs 0.146 at q=0.75). Defazio et al. (knowledge/papers/optimal-linear-decay.md) show this shape beats cosine across 10 problems with theory; cifar10-fast used it in-domain.

  Risks/edge cases: (a) lr reaches exactly 0 only at progress=1.0, which the loop clamps via `min(total_training_time / TIME_BUDGET_S, 1.0)` — same as cosine, no division or domain hazard; (b) the early-mid schedule runs ~12% cooler locally — per-step LR is still far below the 0.6-stable ceiling, no instability possible; (c) signatures must be byte-identical (1613MB, ~139 epochs, 4,286,026 params) — the change is a pure scalar formula.

## Configuration Changes
- LR anneal family: cosine → linear-to-zero (same peak 0.4, same warmup 0.15, same time-keying; rationale: arXiv 2310.07831 theory + 10-problem evaluation, arXiv 2502.15938 replication, cifar10-fast in-domain precedent; integral-preserving by construction so the closed heat axis is untouched)

## Execution Environment
- Method: local — composite background script (GPU-0 pre-check → `uv run train.py > run.log 2>&1` → inline watchdog) from project root, GPU 0 only; if busy, wait via `until`-loop watcher, never GPU 1
- Resources: 1x H20 (GPU 0); VRAM ~1613MB expected (must equal baseline)
- Estimated runtime: ~480–510s total (300s timed budget + ~139 evals + ~12s warm-cache startup; data/ cached); hard cap 600s
- Log output: all stdout/stderr to `run.log` (no tee/stream); deleted after the experiment concludes
- Tool skill: none (background Bash; contention protocol per infra-errors.md § Important; cleanup per § Warning: `git clean -fd -e .autoresearch/ -e data/`)

## Abort Criteria
- Loss NaN/inf or persistent upward loss spikes (not expected — the schedule is pointwise ≤ peak 0.4, proven stable at 0.6)
- Watchdog auto-kill: 4 consecutive windowed samples > 30ms/step (contention) → wait for clean window, relaunch (infrastructure retry, max 2)
- dt-gate: projected epochs < 65 from early windowed samples (impossible for a scalar-formula change — treat as infra)
- No new output in run.log for > 3 minutes
- Total wall clock reaching 590s with the run still going

## Verification Protocol

### Verification Procedure
Run from project root after the run exits. Baseline from `exp-index.sh baseline` = **96.71** (commit 1990397), direction higher; bar = 96.71 + 0.10 = **96.81**.

0. **Pre-condition (contention sanity, Protocol Findings EXP-011/014)**: num_epochs within ~10% of ~139 AND zero watchdog SLOW events AND post-hoc windowed profile ≈ 0 windows > 30ms. Otherwise infrastructure-contaminated — do NOT evaluate conditions; rerun (max 2 retries).
1. **best_test_acc ≥ 96.81** — `grep "^best_test_acc:" run.log`; PASS if parsed value ≥ 96.81. Empty grep ⇒ crash ⇒ `tail -n 50 run.log`. Timeout: n/a (post-hoc parse).
2. **Run completed within budget (≤ 600s total)** — `grep "^total_seconds:" run.log`; PASS if ≤ 600 and summary block exists (clean exit). Timeout: n/a.
3. **Validation at most once per epoch** — `grep -c "eval ep" run.log` ≤ num_epochs from the summary. Structurally guaranteed; checked mechanically. Timeout: n/a.

First failed condition stops evaluation. On all-pass: verdict improvement; `exp-index.sh insert`; commit the diff on `autoresearch/exp-016`; merge to `autoresearch/dev` (no remote — skip fetch/push/PR per TASK.md, fields stay "(pending)"/"N/A"). On condition-1 failure with a clean run: no-improvement; capture the diagnostic trail (Milestone 3), discard changes (`git checkout -- .` + `git clean -fd -e .autoresearch/ -e data/`), return to `autoresearch/dev`. Delete run.log after analysis either way.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1613MB (pure-scalar claim check)
- num_epochs: `grep "^num_epochs:" run.log` — expect ~139 (throughput untouched)
- num_params: `grep "^num_params:" run.log` — must equal 4,286,026 (architecture unchanged)
- final_test_loss: `grep "^final_test_loss:" run.log` — tail-quality diagnostic vs baseline region ~0.19
