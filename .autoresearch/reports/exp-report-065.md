# Report EXP-065: Warmup-phase augmentation lightening — head quadrant completes the pressure-profile law
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-065.md
- **Plan**: plans/plan-065.md
- **Log**: logs/exp-log-065.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged budget by modernizing train.py only. Baseline: 96.71 @ 1990397 (bar 96.81; family mean 96.57, σ 0.16).

## Idea & Hypothesis

The pressure-profile law was three-quarters mapped: lightening the TAIL loses (EXP-025 clean −0.87, EXP-033 light-aug −0.46), and the parameter-side mirror holds (EXP-055). The HEAD side — whether the LR warmup phase needs full augmentation pressure — was unmeasured. EfficientNetV2's reg-ramp ablation (knowledge/papers/efficientnetv2-progressive-learning.md) plus EXP-033's measured banked-alignment gain (+0.48 from a light phase) argued the warmup might be over-regularized; EXP-018's "turns on during peak heat loses" precedent argued the opposite. **Hypothesis**: TA+RE off during p < 0.15 (~21 epochs, crop+flip floor retained), on for the entire anneal, banks early alignment that survives — pair-mean ≥ 96.81; alternatively ≤ 96.41 confirms the EXP-018 transition signature on the data side. Zero dt toll (aug is worker-side) → launch-certain, family bands binding.

## Approach

train.py (+46/−4): shared-memory `AUG_GATE` tensor (EXP-041 fork-propagation pattern); `GatedTA`/`GatedErase` wrappers around the byte-identical baseline transform instances at the same pipeline positions; one-time flip at the first epoch boundary with progress ≥ WARMUP_FRAC, with an `AUG_ON` marker. All constants byte-identical. CPU sanity 10/10 — including a live persistent-workers propagation test (flip visible after 5 batches of prefetch drain).

## Execution

Single composite run, gates clear at poll 1, fully pristine: RC=0, D0 and all 30 watchdog windows 21.7–22.7ms, 13,474 steps (family band), 139 epochs/139 evals, charged 300.0s, total 486.4s, VRAM 1,613MB, zero NaN. Mechanism engagement proven by the pre-registered physical signature, not the marker alone: `AUG_ON ep 22 (p=0.152)` (exactly the first boundary past 0.15), debiased train loss 0.76–0.79 over ep 15–20 (far below family's ~1.1 at full aug) jumping to 1.14–1.21 within 2 epochs of the flip. Eval transient at the flip (ep 21–24 dipped to 70–77) recovered by ep 25.

## Results

- **Primary metric**: best_test_acc = 96.38 (baseline 96.71, delta −0.33; family mean 96.57, −1.2σ; marginally below the family band floor 96.41)
- **Observations**: The light phase delivered its premise — the model trained visibly faster for 21 epochs (loss 0.76 vs family ~1.1 at the same epochs). The advantage then washed out and mildly inverted: the plateau landed at 96.25–96.38 (family plateaus 96.4–96.7), best == final == 96.38.
- **Analysis**: Pre-registered branch (iii). The banked-progress mechanism is real but worthless here: what the light epochs bank is alignment to the EASY distribution, and the model must re-learn the full-aug distribution from a worse-than-scratch position relative to a model that lived under full pressure from step 0 (the EXP-018 signature, data side — "capacity that turns on during peak heat loses" generalizes to "distributions that switch during peak heat lose"). Single-draw caveat: −1.2σ is suggestive rather than conclusive of active damage, but the ≥-bar outcome is decisively excluded, and the law's shape is now four-quadrant complete: head light −0.19 (this), tail light −0.46, tail clean −0.87, full-on = measured optimum. Notably the curriculum intuition ("easy first, hard later") is now directly refuted in this regime — the recipe's pressure schedule has no exploitable time structure in either direction.
- **Key Learning**: Banked progress from an easier early distribution does not transfer forward — augmentation pressure, like LR heat and parameter trainability, must be constant-on from step 0; "curriculum" time-structure is a measured negative in heavy-aug fixed-budget training.

## Verification

- **Conditions**: Integrity PASS (clean ledger 13,474 steps, byte-clean signatures, AUG_ON marker + loss-rise engagement signature per the EXP-055 law); Condition 1 FAIL (96.38 < 96.81); Conditions 2–3 not evaluated per first-failure-stop (informationally: 486.4s ≤ 600; cadence structural pass).
- **Review Notes**: Results trustworthy — the engagement proof rules out "mechanism never ran" false negatives; the step ledger rules out contamination; the value is plausible (smooth plateau, family-adjacent).
- **Verdict**: no-improvement
- **Verdict Basis**: Condition 1 failure on a valid, integrity-clean run (pre-registered branch (iii)).

## Unexplored Avenues

- **Gradual ramp instead of a step flip** (e.g., TA probability ramping 0→1 over ep 15–30): would soften the transition shock — but the result shows the light phase's banked progress is the liability, not just the shock; a ramp shortens the light phase's benefit AND keeps a (smaller) mismatch. Bounded by this result and EXP-033's taper (the tail-side gradual variant, also negative). Not worth a loop.
- **Reverse direction (heavy early, light late)**: that is exactly EXP-033's tail taper — measured −0.46. Both time-orderings now measured.
- **Phase-matched BN re-warm at the flip**: the eval transient recovered in 3 epochs on its own; BN handling is not the bottleneck (the deficit persisted to budget end).

## Next Steps

1. **Adversarial audit of the measured-ceiling hypothesis as the brainstorm's primary task** (medium confidence): with the pressure-profile law closed, every mechanism class AND every time-profile of the recipe is measured; the next brainstorm should re-derive the residual unpriced degrees of freedom from first principles before reaching outward (per exp-report-064 next-step 1).
2. **Targeted lit excavation, budget-creating mechanisms only** (low-medium): the excavation pattern keeps producing clean closures; restrict candidates to those whose mechanism survives both the absorption screen AND the cost-landing screen (project-insights EXP-064 entry).
3. **σ-tightening replicate pair** (low): defer unless ideation yields nothing above the effect-size screen.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
