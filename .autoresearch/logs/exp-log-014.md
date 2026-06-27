# EXP-014: Shorten LR warmup (WARMUP_FRAC 0.15 → 0.08)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md
- **Plan**: plans/plan-014.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-014
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (necessary condition 1 failed: best_test_acc 96.49 < 96.81 bar; clean valid run — research no-improvement, not infra)

## Implementation Notes

### Summary

Milestone 1 executed exactly as planned: branched `autoresearch/exp-014` from `autoresearch/dev` (clean @ 1990397), changed line 24 of train.py `WARMUP_FRAC = 0.15` → `0.08` (git diff --stat confirms 1 file, 1 insertion, 1 deletion), `uv run python -m py_compile train.py` exited 0. No other file touched. The cosine phase needs no edit — `lr_at()` normalizes by `(1 - WARMUP_FRAC)` so the anneal automatically stretches from 85% to 92% of the budget.

### Surprises & Discoveries

- At launch-prep time GPU 0 was occupied by a foreign compute app (PID 3197114, 13.7GB, 96% util; GPU 1 also busy). Per the hard constraint (always GPU 0; wait if busy) launch is gated on a background watcher (task bsb09b8os) that fires when GPU 0 has zero compute apps. This is the pre-launch half of the EXP-011 contention protocol doing its job — launching now would have produced a contaminated run.

### Decisions

- Kept the plan's conservative halving (0.08) rather than the literature's 1–5% floor: adds a clean second dose point on the warmup axis (0.15 → 0.08) so a follow-up probe (e.g. 0.03) can be sequenced if this one moves the metric.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: task bm5zfzrxw (local background)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log (deleted post-triage)
- **WandB**: N/A
- **Status**: failed (infrastructure — GPU contention; metrics quarantined)
- **Started**: 2026-06-10 08:15
- **Ended**: 2026-06-10 08:23

Description:
- Single training run of the baseline recipe with warmup halved from 15% to 8% of the 300s timed budget — the last untouched schedule-shape constant. Expected: byte-identical throughput signatures (dt ~22.3ms, ~139 epochs, ~1613MB VRAM, num_params 4,286,026), LR reaching peak 0.4 by ~step 1075 instead of ~2018, mid-schedule evals running at-or-ahead of the baseline trail, and best_test_acc ≥ 96.81 if the hypothesis (sub-peak ramp time was wasted budget) holds.

Observations:
- Run exited cleanly but contaminated: 92 epochs / 8889 steps (34% below clean projection); windowed step-time profile showed 43% of windows > 30ms with sustained 48ms stretches (steps ~800–2550, ~5300+) — EXP-011 time-slicing signature (source: Run-1 run.log windowed profile; Errors & Dead Ends entry)
- VRAM 1613.0MB and num_params 4,286,026 matched expectations; startup 13.5s warm-cache (source: Run-1 run.log summary block)
- Per Protocol Findings (EXP-011): contaminated metrics are NOT analyzed; rerun in a clean window (this is infrastructure retry 1 of 2)

Key Metrics:
- quarantined (contaminated run — not evidence for or against the hypothesis)

### Run 2

Metadata:
- **Job ID**: task bh32axche (local background; composite launcher + inline watchdog, kills run on 4 consecutive >30ms windows)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: completed (clean)
- **Started**: 2026-06-10 08:54
- **Ended**: 2026-06-10 09:02

Description:
- Identical relaunch of Run 1 (no code change — the diff is untouched) into a verified-clean GPU-0 window. Detector armed in the SAME command chain as the launch this time (Run-1 lesson: a Monitor armed in a later turn can first poll a finished run).

Observations:
- Cleanest possible execution: inline watchdog emitted ZERO SLOW events; post-hoc windowed profile 0 of 267 windows > 30ms, mean 22.4ms/step; 139 epochs / 13410 steps — exact clean projection; total 482.9s, startup 12.4s warm-cache (source: task bh32axche output; run.log windowed profile)
- HYPOTHESIS REFUTED in direction: trajectory ran BEHIND baseline the whole mid-schedule, not ahead — ep 20: 78.49 (vs ~88 baseline trail), ep 60: 87.82 (vs ~92), ep 100: 93.11 (vs ~96); tail converged (best 96.49 first reached ep 132, final 96.45 ≈ best — flat plateau) (source: run.log eval lines, sampled ep 1/5/10/20/40/60/80/100/120/139)
- Mechanism: with a time-keyed cosine, shortening warmup makes LR STRICTLY HIGHER at every progress point p<1 (anneal starts earlier, q_new(p) < q_old(p) ⇒ lr_new(p) > lr_old(p) for all p in (0.08, 1)). The change is not "less wasted ramp" but "hotter everywhere" — a scaled-down EXP-010 (peak 0.6, −0.57pp): heat defers progress and the fixed-length tail cannot fully repay it (−0.22pp)
- final_test_loss 0.1851 — BETTER than baseline's (0.1907 region) despite lower best acc, echoing the EXP-011 pattern that mean/loss improvements do not move the max-statistic

Key Metrics:
- best_test_acc: 96.49% @ ep 132 (source: run.log summary; bar was 96.81)
- total_seconds: 482.9; training_seconds: 300.0; num_epochs: 139; num_steps: 13410; peak_vram_mb: 1613.0; num_params: 4,286,026 (source: run.log summary block)

## Verification Results

<!-- Filled after the experiment completes successfully.
     If ANY necessary condition fails, remaining conditions are not evaluated. -->

### Conditions Checked

- **Pre-condition — contention sanity (Protocol Findings EXP-011)**: num_epochs 139 = clean projection exactly; inline watchdog zero SLOW events; post-hoc windowed profile 0/267 windows > 30ms (mean 22.4ms). **CLEAN — conditions evaluable.** (source: run.log windowed profile; task bh32axche output)
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: parsed 96.49 from `grep "^best_test_acc:" run.log`. **FAILED** (96.49 < 96.81; −0.22pp vs baseline). (source: run.log summary block)
- **Condition 2 — total ≤ 600s**: skipped — aborted after prior failure (observed informally: 482.9s would have passed)
- **Condition 3 — validation ≤ once/epoch**: skipped — aborted after prior failure (observed informally: 139 eval lines = 139 epochs would have passed)

### Informational Metrics

- Not collected per protocol (necessary condition failed). Informal: peak_vram_mb 1613.0 (= baseline), num_epochs 139 (= baseline), num_params 4,286,026 (= baseline) — the "pure scalar change" claim held on every signature.

## Errors & Dead Ends

### 2026-06-10 — Run 1 GPU contention (infrastructure, not research)
- Error: `no error — exit 0, plausible summary; 92 epochs / 8889 steps vs ~139/13418 clean; 43% of windowed step-time samples > 30ms, sustained 48ms stretches at steps ~800–2550 and ~5300+ (24/48ms time-slicing signature)`
- Root cause: foreign GPU-0 job (PID 3197114 lineage — it occupied GPU 0 before launch, freed it, then compute returned mid-run); run launched into a window that closed. Windowed dt from pct_done deltas is the canonical detector (printed dt samples 2% and read 22–23ms clean).
- Source: run.log (Run 1, deleted post-triage) — windowed profile computed via tr '\r' '\n' + awk; task bm5zfzrxw
- Do NOT retry: analyzing contaminated metrics (best 96.00 @ 92 epochs is NOT a data point for the warmup hypothesis); also do not trust a live Monitor armed in a separate turn after launch — turn-scheduling delay meant it first polled a finished run and could never emit SLOW. Arm watchers in the same command chain as the launch, and always run the post-hoc windowed profile regardless.

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
