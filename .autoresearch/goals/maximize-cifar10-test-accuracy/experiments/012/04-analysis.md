# Report EXP-012: Recipe-scalar refresh — weight-decay shaping + label-smoothing retune
- **Created**: 2026-06-29

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within the fixed 300s training
budget, editing only `train.py`. Baseline **96.38%** (EXP-008, commit 07c3760); improvement
bar **≥96.48** (+0.10pp, clearly above the ~0.1pp throughput-jitter noise floor).

## Idea & Hypothesis
Chosen idea (quick brainstorm, Codex-reviewed): **Recipe-scalar refresh**, with **weight-decay
shaping** as the headline and a **label-smoothing retune** as a secondary, clean isolation.
After input-space augmentation saturated (EXP-011 CutMix tied) and the optimizer axis exhausted
(EXP-009/010 Muon ties SGD), the next move was a regularization change on a genuinely different
axis. WD-shaping removes weight decay (5e-4) from BN γ/β and the ReZero α scalar (ndim≤1) while
keeping it on conv/fc weight matrices (ndim≥2) — throughput-free (cannot under-anneal, the failure
that sank every capacity experiment) and with a net-specific untested mechanism: uniform 5e-4
applies a restoring force toward 0 on the ReZero α capacity gate, so decoupling α should let the
layer2 block reach a larger steady-state capacity.

**Hypothesis**: removing wd from BN γ/β + α (cell-A, LS 0.2 held) raises `best_test_acc` ≥0.10pp
over a same-session baseline and clears 96.48, with the final ReZero α *measurably larger* than
baseline (mechanism corroboration) and `num_epochs` unchanged (~142–150).

## Approach
All changes in `train.py` only. Added `import os`; made `LABEL_SMOOTHING` env-overridable and
added a `WD_SHAPING` env flag (default OFF → unmodified invocation reproduces baseline). Replaced
the single-group SGD with a branch: when `WD_SHAPING`, split params into `decay` (ndim≥2, wd=5e-4)
and `no_decay` (ndim≤1, wd=0), guarded by an exact in-run assert `len(no_decay)==2·#BN+1`; else the
exact original single-group call. Added five summary prints (`wd_shaping`, `label_smoothing`,
`no_decay_params`, `rezero_alpha` raw + EMA). Reviewer-mandated 4-cell **same-session** design,
env-toggled so no file edit happens between cells; PEAK_LR held at 0.4 throughout to keep WD-shaping
attribution clean. Smoke test confirmed an exact id-set partition (10 BN layers → 21 no-decay tensors
= 5,505 params; 11 decay tensors; frozen whitening conv excluded from both). num_params invariant.

| cell | WD_SHAPING | LS | role |
|------|-----------|----|------|
| 0 | off | 0.2 | same-session baseline (reproduces EXP-008) |
| A | **on** | 0.2 | **headline** — isolates WD-shaping |
| B | on | 0.1 | bundle (shaping + LS retune) |
| C | off | 0.1 | isolates LS (clean, CutMix-free) |

## Execution
All four cells ran back-to-back in one background process (exit 0), GPU 1 uncontended (3 MiB at
launch), ~26.5k img/s throughout, ~30 min total wall. No retries, no errors, no divergence. Clean
throughput band for every cell (148–150 epochs, spread 2 < the 5-epoch sequential-drift threshold →
same-session comparison sound). Anti-bookkeeping passed for all cells (max per-epoch test_acc ==
printed summary best).

## Results
- **Primary metric**: 96.29% (headline cell-A) (baseline: 96.38, delta: −0.09, −0.09%). Same-session
  cell-0 reproduced baseline at 96.32.
- **Full table** (best_test_acc / num_epochs / rezero_α raw):
  - cell-0 (off, LS0.2): **96.32** / 150 / −1.2533
  - cell-A (on, LS0.2): **96.29** / 150 / −0.7296  → **−0.03pp vs same-session c0**
  - cell-B (on, LS0.1): **96.16** / 148 / −0.7150
  - cell-C (off, LS0.1): **96.09** / 150 / −1.2108
- **Observations**: WD-shaping *did* fire mechanically — the no-decay cells' ReZero α magnitude
  shrank to ≈0.72 from ≈1.25 in the decayed cells. But this falsified the hypothesis's directional
  prediction: removing the wd restoring force made |α| **smaller**, not larger. Uniform weight decay
  was actually *inflating* the steady-state |α|; the data-gradient equilibrium for α sits near
  ∓0.72, and the +5e-4·α decay term was pushing it further from zero, not toward it. Crucially, the
  halved |α| had **no accuracy cost** (96.29 ≈ 96.32) — the capacity gate is not operating at an
  accuracy-limiting magnitude either way. LS 0.1 clearly *hurt* in a clean (CutMix-free) setting:
  −0.23pp (cell-C) and −0.16pp (cell-B), confirming LS 0.2 is the better operating point and that
  EXP-011's "LS-insensitive" read was masked by CutMix.
- **Analysis**: hypothesis rejected on all three falsifiable predictions — (a) α moved but in the
  opposite direction; (b) all WD-shaping cells landed within noise of same-session cell-0, so the
  regularization *allocation* is already near-optimal on this saturated net; (c) clean LS-0.1
  underperformed baseline, slightly worse than "≈baseline" expected. The regularization-scalar axis
  (weight-space allocation + target-space LS) is now empirically exhausted alongside input-space aug,
  optimizer, eval-TTA, and capacity.
- **Key Learning**: WD-shaping ties baseline (96.29 vs same-session 96.32) despite halving |ReZero
  α| (uniform wd was *inflating* α, not suppressing it); the capacity gate is not accuracy-limiting
  and clean LS 0.1 degrades — SGD regularization scalars are already tuned for this 300s recipe.

## Verification
- **Conditions**: NC1 passed (all cells 300s training, <600s wall, valid metrics, exit 0); **NC2
  failed** (best cell 96.32 < 96.48 bar; headline cell-A 96.29 is −0.03pp vs same-session cell-0,
  within noise); NC3 passed (only `M train.py`, prepare.py unchanged, num_params 7,784,627, seeds
  intact, 1 eval/epoch).
- **Review Notes**: results confirmed trustworthy — same-session control reproduced the stored
  baseline (96.32 vs 96.38, within the ~0.1pp floor), clean throughput, anti-bookkeeping passed,
  mechanism instrumentation (α, no_decay_params) internally consistent. No false-pass/false-fail risk.
- **Verdict**: no-improvement
- **Verdict Basis**: valid result, NC2 (result quality gate) failed — no cell cleared the bar or beat
  the same-session baseline above noise.

## Unexplored Avenues
- **Schedule shape (cosine decay / pct_start / peak-LR)** — the one remaining throughput-free lever
  not yet refreshed (one-cycle shape set in EXP-001, never revisited across 4 recipe generations).
  A cosine approach to 0 gives the weight-EMA a lower-variance tail; distinct mechanism from
  regularization. Most likely small (≤0.1–0.2pp) but cannot under-anneal. (idea-03, deferred.)
- **Mild capacity step (widen layer2 256→320)** — higher ceiling (a *different* axis: capacity, not
  regularization), the pre-registered EXP-007 follow-up after 256→384 overshot (cut epochs to 94).
  The 1.25× step targets ~120–135 epochs; under-anneal risk is real but it is the only untried lever
  with a ceiling clearly above the noise floor. Pre-register num_epochs as the first decision metric.
- **WD magnitude sweep on conv/fc** (not allocation but level) — EXP-012 tested allocation at fixed
  5e-4; the conv/fc decay *level* itself (e.g., 3e-4 / 8e-4) was never swept post-EXP-008-aug-change.
  Plausibly sub-noise like allocation, but untested.

## Next Steps
1. **Mild capacity widen layer2 256→320** (confidence: medium) — the highest-ceiling untried lever;
   the regularization axes are now exhausted, so accept the under-anneal risk with num_epochs
   pre-registered as the first decision gate. The natural next experiment.
2. **One-cycle cosine schedule reshape** (confidence: low-medium) — cheapest throughput-free probe;
   fold pct_start 0.15→0.10 and a cosine tail together, read against a same-session baseline.
3. If both stall, **SAM / sharpness-aware** (confidence: low) — a loss-geometry moonshot at ~2× step
   cost (would roughly halve epochs → under-anneal risk), only if the cheap levers are confirmed dead.

## Exit Action Results
- None defined (autopilot goal) — section intentionally empty.
