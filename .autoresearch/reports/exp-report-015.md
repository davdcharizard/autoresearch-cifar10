# Report EXP-015: Halve weight decay (WEIGHT_DECAY 5e-4 → 2.5e-4)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-015.md
- **Plan**: plans/plan-015.md
- **Log**: logs/exp-log-015.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: is the recipe over-regularized — does halving the last never-probed constant (WD 5e-4, inherited unchanged from EXP-000) recover accuracy by reducing total pressure?

## Idea & Hypothesis

Chosen idea: single-constant change WEIGHT_DECAY 5e-4 → 2.5e-4 on the selective decay group (conv/linear weights only). Three converging arguments: (a) last unmeasured axis after 15 experiments; (b) the saturated four-point augmentation dose-response pointed to pressure-DOWN as the only untried direction; (c) the WD-with-BN literature (van Laarhoven 2017; Zhang et al. ICLR 2019; NeurIPS 2024 — brainstorm-015 § Web Search) reframes conv-weight WD as an effective-LR knob, making this also a mild cold-side probe of the heat curve. Hypothesis: best_test_acc ≥ 96.81 with train loss below baseline mid-schedule; pre-registered alternative readings: over-fit tail ⇒ "was at pressure optimum"; flat-lower tail ⇒ same conclusion via dynamics. Runners-up: cold peak 0.35, heat-compensated schedule reshape.

## Approach

1-line diff in train.py: `WEIGHT_DECAY = 5e-4` → `2.5e-4`. No deviations. Launch used the EXP-014-derived composite protocol: GPU-0 pre-check + training + inline contention watchdog in one background command chain.

## Execution

One run, no retries (task b4a4n6lud, launched 09:11:44 into a free GPU 0). One pre-noted wrinkle: the post-EXP-014 `git clean -fd` had deleted the untracked `data/` dir, so the run paid a one-time CIFAR-10 re-download in startup — pre-registered as infrastructure if it busted the cap; it did not (total 509.8s ≤ 600; the footgun and the `-e data/` fix are now in infra-errors § Warning). Execution was pristine: watchdog zero SLOW events, post-hoc windowed profile 0 of 266 windows > 30ms (mean 22.4ms), 139 epochs / 13393 steps on projection, VRAM 1613.0MB, params 4,286,026 — signatures byte-identical to baseline as the "pure scalar" claim requires.

## Results

- **Primary metric**: best_test_acc = 96.41% (baseline: 96.71, delta: −0.30pp, −0.31%); bar was 96.81
- **Observations**: The pre-registered diagnostic disambiguated cleanly — and chose the closing reading. The trajectory ran slightly BEHIND baseline through the whole schedule (ep 20: 81.5; ep 60: 89.9; ep 100: 94.2) and the tail CONVERGED FLAT: 96.38 @ ep 130 → best 96.41 first reached ep 133, final = best, final_test_loss 0.1901 ≈ baseline. There is NO over-fitting signature (no peak-then-decay, no loss divergence) — the failure mode is not "too little regularization let it memorize"; it is "the dynamics trained marginally worse at every stage". This matches the literature's effective-LR account: halving WD lets weight norms grow, lowering the effective LR and gradient noise throughout — a mild cooling that, like every other heat perturbation, moved the metric down.
- **Analysis**: This was the last never-probed constant, and its failure completes a remarkable picture: the regularization axis is now measured on BOTH sides of the current dose (WD-half −0.30 below; reflect −0.14 and mixup −0.46 above; TA +0.17 was the last positive increment) and the heat axis on both sides too (peak +50% −0.57, warmup-halving −0.22 hot; WD-half's effective-LR cooling −0.30 cold). Every single-constant neighborhood of the EXP-006 recipe — capacity (both directions), augmentation dose (both directions), peak LR (both directions via EXP-010/this), warmup, batch, eval smoothing — is now a measured local maximum. Ten consecutive no-improvements with clean, converged, mutually-consistent mechanisms is no longer a streak of bad luck; it is evidence the recipe sits at a genuine local optimum of the single-change landscape at this architecture/budget. Remaining unprobed space is necessarily MULTI-constant (compensated combinations that hold one quantity fixed while trading others, e.g. warmup 0.08 + peak 0.35) or STRUCTURAL (different schedule family, different optimizer, architectural variants beyond width/depth scaling — e.g. stem/downsampling changes that alter FLOPs distribution rather than scale).
- **Key Learning**: The recipe is at a measured local optimum: ten clean single-constant probes in every direction (pressure ±, heat ±, capacity ±, batch, smoothing) all converged below baseline with consistent mechanisms — single-knob tuning of this recipe is exhausted; only multi-constant trades or structural changes remain.

## Verification

- **Conditions**: pre-condition contention sanity CLEAN (139/139 epochs, watchdog silent, 0/266 slow windows); condition 1 FAILED (best_test_acc 96.41 < 96.81); conditions 2–3 skipped per first-failure stop (observed informally: 509.8s ≤ 600 and 139 evals = 139 epochs would have passed)
- **Review Notes**: trustworthy — metric consistent with the eval trail; throughput signatures byte-identical to baseline so the deficit is attributable to the WD change alone; the download wrinkle was absorbed in startup and did not contaminate anything
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid clean run; primary-metric necessary condition not met)

## Unexplored Avenues

- **Smaller WD step (e.g. 4e-4) or WD up (7.5e-4)**: the −0.30 at a half-octave-equivalent step suggests the optimum is close to 5e-4; finer probes are within-noise territory (±0.1pp resolution) — low value.
- **Decoupled WD on the FC layer only / norm-based tuning**: second-order variants of the same axis; the flat-tail result gives them no mechanism to exploit.
- **Heat-compensated schedule reshape (warmup 0.08 + peak 0.35)**: still the cleanest untested MULTI-constant trade — isolates anneal length at ~constant heat (exp-report-014 § Unexplored Avenues).

## Next Steps

1. **Structural change: schedule family swap (cosine → linear-to-zero anneal at same peak/warmup)** — changes the heat DISTRIBUTION shape (more mid-schedule LR, less tail) without monotonically raising or lowering it; the speedrun reference (cifar10-fast, knowledge/README References) used piecewise-linear and beat cosine-era baselines. Confidence: low-medium.
2. **Compensated schedule reshape (WARMUP 0.08 + PEAK 0.35)** — the cleanest two-constant trade; tests anneal-length at ~constant heat using EXP-014's mechanism insight. Confidence: low.
3. **Structural architecture probe outside the width/depth axis** — e.g. a 2x-width stem conv (channels stay 32-aligned) or moving one block from stage 1 to stage 3 at constant params: redistributes FLOPs rather than scaling them, untouched territory per goal-learnings (capacity closed only for UNIFORM scaling). Confidence: low.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
