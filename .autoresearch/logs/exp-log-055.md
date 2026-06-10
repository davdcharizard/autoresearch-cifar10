# EXP-055: Richer per-image AugMix chains on a subset — RandomApply([AugMix(mixture_width=4)], p=0.35)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-055.md
- **Plan**: plans/plan-055.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-055
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
One-line change in `train_tf` (train.py:171): `RandomApply([AugMix()], p=0.5)` (EXP-054 winner, w3/50%) → `RandomApply([AugMix(mixture_width=4, chain_depth=-1)], p=0.35)` — richer 4-chain AugMix on ~35% of images. Built on the EXP-054 winner. GPU Cutout, model, optimizer, schedule, seed 42, batch 128, compile unchanged. Comment updated to reflect w4/p=0.35 and the wall rationale. Smoke tests passed: AST parse OK; `git diff --name-only` = train.py only; AugMix(w4) instantiates (no new dep); 10 augmented samples all shape (3,32,32).

### Surprises & Discoveries
The p choice was driven by a wall-feasibility subtlety: this session's isolated dataloader probe put `RandomApply(AugMix w4) p=0.4` at 12.2ms/batch — essentially identical to EXP-054's `w3 p=0.5` reference (12.1ms), which shipped at the tight 593.0s wall. Per the EXP-054 protocol finding (stochastic-application aug under-projects the wall; target ≤540s, don't trust a tight wall), p=0.4 was rejected as too risky and p=0.35 (11.4ms, real buffer) chosen as the lead.

### Decisions
Lead with p=0.35 (not the brainstorm's nominal p=0.4) to respect the conservative-wall protocol finding; p=0.3 held as the gated contingency. Accepted the width↑/coverage↓ confound (documented for analysis) as acceptable for a within-lever richness probe with a graceful-null failure mode.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash ID b3ggxifeo (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (exit 0, 543.0s wall)

Description:
- Runs `RandomApply([AugMix(mixture_width=4, chain_depth=-1)], p=0.35)` training on idle GPU 0 within the 300s Σdt budget. Tests whether richer 4-chain AugMix on ~35% of images (pushing the validated chain-COUNT lever from width 3→4) beats the EXP-054 winner (w3/50%, 96.45). Expect dt steady ~8ms (GPU step unchanged, aug is CPU-side); wall ~560s projected at p=0.35 (early real-load gate at projected+60s buffer > 595s → p=0.3 fallback). Bar = 96.55.

Observations:
- FEASIBILITY GATE PASSED at p=0.35 (no fallback needed). Two-point steady-state wall measurement (window +34s / +2300 steps): **14.8 ms/step wall → projected total ~534s**, +60s conservatism buffer = 594s ≤ 595 gate, and ≤540s protocol target → proceed. (source: ps etimes + run.log step deltas, in-conversation)
- dt steady 8-10ms (GPU step unchanged — AugMix is CPU-side, Σdt budget unaffected). ep4 test_acc 73.57% @ step ~1550 (normal early trajectory). No NaN. (source: run.log)
- CAVEAT: raw projection 534s equals EXP-054's early read (535s), which drifted +58s to a 593s finish — expect this run ~580-590s, tight but under the 600s limit. Hard-kill watch if wall approaches ~595s without a summary.

Key Metrics:
- best_test_acc: **96.01%** @ ep (summary) — **REGRESSION**, −0.44pp vs baseline 96.45, far below bar 96.55. (source: run.log summary)
- final_test_acc: 96.00%; final_test_loss: **0.2000** (worse than EXP-054's 0.1968 — loss AND top-1 both regressed). (source: run.log summary)
- total_seconds: **543.0s** (wall — comfortable, 57s margin; p=0.35 projection 534s was accurate, NO EXP-054-style +58s drift this run). num_epochs: 92; num_steps: 35,779; num_params: 4,299,866; peak_vram: 453.8 MB. (source: run.log summary)
- dt dist: 669×8ms, 45×9ms, 1×10ms (steady 8ms — GPU step unchanged, Σdt budget unaffected by the CPU-side aug change). (source: run.log)

## Verification Results

### Conditions Checked
- **Cond 1 — best_test_acc ≥ 96.55 (baseline 96.45 + 0.1)**: 96.01% → **FAIL** (−0.44pp vs baseline; −0.54pp below bar). Verdict → no-improvement. (source: run.log summary)
- **Cond 2 — clean completion within budget**: NOT EVALUATED (skipped — aborted after Cond 1 failure). [Informationally: would PASS — summary printed, total_seconds 543.0 < 600, num_params 4,299,866, no NaN/traceback (grep 0).]
- **Cond 3 — no hard-constraint violations**: NOT EVALUATED (skipped — aborted after Cond 1 failure). [Informationally: would PASS — `git diff --name-only` = train.py only; eval/prepare untouched; AugMix/RandomApply torchvision-native; seed 42 unchanged.]
- **Cond 1 (necessary) FAILED → Outcome: completed (clean run); verdict no-improvement (regression).**

### Informational Metrics
- delta vs baseline 96.45: **−0.44pp** (regression). final_test_loss 0.2000 > EXP-054's 0.1968 (both metrics worse). num_epochs 92 (Σdt budget intact). peak_vram 453.8 MB unchanged. Wall 543.0s — the p=0.35 wall was as projected (no drift), confirming p=0.35 is the safer config than EXP-054's p=0.5 (593s); but it cost too much coverage.

## Errors & Dead Ends

## Human Notes

> {none — autopilot}
