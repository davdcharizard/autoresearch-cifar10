# Report EXP-025: Identity-Initialized Final-Stage ECA
- **Created**: 2026-08-06

## Goal

Raise CIFAR-10 `best_test_acc` above the 94.15% frontier under the fixed seed-42, one-H20, 300-second, `train.py`-only protocol. Acceptance required at least 94.25% plus successful completion and wall-time integrity.

## Idea & Hypothesis

Add three tiny ECA gates only to `layer3` residual branches. Zero length-5 kernels make `2*sigmoid(0)=1`, preserving the accepted initial function, depth, widths, and Option-A ratios while allowing input-conditional semantic-channel allocation. The hypothesis predicted safe gradual recruitment, at least 26,000 projected updates, healthy switch fit, and `best_test_acc>=94.25%`.

## Approach

Added `ECAGate`, optional `use_eca` plumbing, and three pre-add final-stage gates for 15 new parameters (1,073,977 total). A pre-edit `7c1e7d8` state/RNG oracle proved exact shared initialization and logits. The immutable EXP-024 corpus supplied 100 strong-hard, 100 strong-soft CutMix, and 100 weak-hard batches. First-update and 200-step controllers measured weights, hard/soft gradient recruitment, gate min/quantiles/max/means, class concentration, finite state, and loss EMA before timing.

## Execution

Static, scope, state, RNG, logit, BN, and first-update checks passed. Cold cross-graph CUDA backward differed by at most `9.6858e-08` absolute and `3.5529e-07` relative; an independent critic approved fixed `1e-7`/`1e-6` bounds after a separately warmed diagnostic was bitwise exact. The actual recruitment run then failed its pre-registered bounds. Gates began near one after separate hard/soft updates, but reached sigmoid endpoints `[0,2]` by hard step 19. No timing or production run followed, and no operating-point rescue was attempted.

## Results

- **Primary metric**: unavailable (baseline: 94.15%; delta: N/A)
- **Observations**: First-update maximum kernel changes were only 0.007176 hard and 0.005448 soft, with gate ranges roughly 0.94-1.04 and 0.97-1.02. In the continuing trajectory, recurrent momentum/global LR rapidly amplified the kernels: hard recorded gates reached exact 0/2 endpoints, while soft step 200 had block means 1.0398/1.1516/1.3604 and maxima 1.7092/1.9135/1.9997. State stayed finite, no candidate-only class concentration occurred, and candidate/control terminal loss-EMA ratio was 1.083684.
- **Analysis**: Exact function and first-update continuity were insufficient for stable multi-step recruitment. The tiny gate parameters share LR 0.1 and momentum 0.9 with convolution weights, but their descriptor-driven gradients operate on a very different scale; momentum pushed logits into sigmoid saturation within tens of steps. Saturated gates can fully suppress or double channels, defeating the intended bounded near-identity adjustment and making later timing/accuracy uninterpretable under the reviewed safety contract. This invalidates the exact three-gate/global-optimizer point, not all channel attention: a separately reviewed optimizer treatment, parameterization, or inherently smaller output range would be a new hypothesis. Lower loss or absence of class concentration cannot rescue this mechanism failure.
- **Key Learning**: Unit-start ECA gates saturated toward 0/2 within 20 steps under global LR 0.1; exact function identity did not ensure stable recruitment.

## Verification

- **Conditions**: Goal necessary conditions skipped; production was blocked before a scored metric existed.
- **Review Notes**: The research veto is trustworthy. It followed an independently reviewed diagnostic amendment, used exact shared data/state, echoed immutable thresholds, and serialized the complete report. The gate failures were repeated across blocks/steps and are not a marginal parsing event.
- **Verdict**: invalid
- **Verdict Basis**: Mandatory recruitment integrity failed before timing/production; metric is `NaN`.

## Unexplored Avenues

- Parameterize a much narrower multiplicative range around one so saturation cannot suppress/double channels, but this changes the hypothesis and effect ceiling.
- Give attention parameters a separately justified learning-rate/decay treatment; this is explicitly outside EXP-025 and requires optimizer-state safety review.
- Apply one endpoint gate rather than three branch gates to reduce launch cost and recurrent gate feedback, at the cost of weaker conditional capacity.

## Next Steps

- **Medium confidence**: Return to an orthogonal data/optimization lever that preserves the accepted graph and standard momentum path.
- **Medium-low confidence**: Develop a structurally bounded, narrow-range endpoint recalibration only if its gradient scale can be derived before testing.
- **Low confidence**: Run channels-last only as a cheap timing probe, not as a scored accuracy bet without an exposure mechanism.
