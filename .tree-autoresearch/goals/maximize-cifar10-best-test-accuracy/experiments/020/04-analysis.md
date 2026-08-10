# Report EXP-020: Convolution-only official-order Gradient Centralization
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the frozen 300-second charged training budget, with higher better and only `train.py` mutable. EXP020 grew from EXP002 at 95.23%, so formal improvement required at least 95.33%. The goal-wide best remains EXP011 at 95.61%.

## Idea & Hypothesis

EXP019's full convolution-plus-classifier GC scored 95.07% and removed 93.21% of classifier direction norm versus 41.89% for convolutions. The ECCV reference also states convolution-only eligibility is sufficient for small-resolution CIFAR. EXP020 therefore retained the official `data gradient + coupled L2 -> projection -> momentum/Nesterov` order but excluded the classifier from projection. The hypothesis predicted that preserving unconstrained class-boundary motion would produce `best_test_acc >=95.33%` without meaningful exposure loss.

## Approach

Only `train.py` changed. External `1e-4` coupled L2 was applied to all 44 FP32 gradients, exactly preserving ordinary decay on excluded tensors, while row-mean subtraction applied on every step only to 16 convolution weights containing 2,742,704 elements and 2,256 rows. The classifier, BN affine parameters, and biases remained unprojected; inherited SGD then applied unchanged momentum and Nesterov with internal decay disabled.

Cadence-512 fixed FP64 device scalars audited applied energy, orthogonal decomposition, post-projection residual, nonfiniteness, and phase buckets. Additional read-only scalars decomposed removed row-mean energy into raw-gradient, effective-L2, and cross terms. Exact dose/inventory reconciliation and final-16 context were appended without changing evaluation. A plan review narrowed the closure claim to official-order rules and made the raw/L2 diagnostic non-fatal, since magnitude cannot establish causal accuracy for untested variants.

## Execution

Deterministic GPU smoke passed exact inventory, bitwise audit-on/off applied updates, two-step excluded-parameter/momentum parity, zero evaluator calls, and numerical bounds. The first decisive preflight passed without repair: median overhead 1.019440x, p90 1.023965x, MAD/median 0.003304, parent drift 0.008854, projected 27,417 steps / 141 epochs, zero live-allocation growth, and zero evaluator calls.

Exactly one metric run executed on physical GPU 0. It exited 0 after 300.0 charged and 461.1 total seconds, completed 28,090 steps over 145 epochs, and emitted a complete integrity-valid summary. There were no code repairs, metric-driven changes, retries, tracebacks, OOMs, or nonfinite values.

## Results

- **Primary metric**: 95.24% (parent: 95.23%, delta vs parent: +0.01 points, +0.01%; global best: 95.61%, delta: -0.37 points)
- **Observations**: GC executed on all 28,090 steps, split 10,292 CutMix, 10,453 early-clean, and 7,345 late-clean, with 55 exact audits. It removed 4.2625 of 24.4448 convolution squared-energy units, a 41.7580% norm ratio. Phase ratios were 40.7105% CutMix, 42.7511% early-clean, and 39.8024% late-clean. Raw-gradient means accounted for essentially all removal: L2-only plus cross energy was about `2.01e-5`, just `4.71e-6` of total removed energy. Applied decomposition error was `3.04e-9`, maximum residual `3.73e-9`, and integrity passed. The final-16 mean was 95.12125%, range 95.03-95.24%, and final accuracy 95.19%; final loss was 0.2029.
- **Analysis**: The formal hypothesis is not validated: +0.01 point is below the 0.10-point gate and well inside observed single-run selection noise. Exposure exceeded the parent (28,090 vs 27,950 steps), the projection remained strongly active, and the stable tail ended exactly at the parent's 95.19 final accuracy, so neither underdose nor an inert mechanism explains the absence of gain. Relative to EXP019, excluding classifier projection recovered +0.17 best and +0.17 final points while retaining nearly the same measured convolution norm removal (41.76% vs 41.89%). This supports disproportionate output-layer sensitivity as the cause of full-GC's harm, despite the classifier's tiny parameter share. Convolution projection itself appears accuracy-neutral on EXP002 rather than beneficial. The negligible L2 contribution makes raw-order GC numerically redundant here, and the late phase had the lowest—not highest—removed norm ratio, weakening the specific case for a late-only exemption; neither audit alone proves causal accuracy for those untested rules.
- **Key Learning**: Excluding classifier projection restored full-GC's loss but improved EXP002 only 0.01 points; convolution GC remained strongly active and accuracy-neutral.

## Verification

- **Conditions**: Execution integrity passed, but the primary metric condition failed: 95.24% was below the required 95.33%.
- **Review Notes**: Results are trustworthy. Claude independently returned `AUDIT_VERDICT: PASS`, rechecking log freshness, scope, all 145 evaluations, charged timing, unchanged evaluator/seeds, exact inventory/dose/audit arithmetic, energy and tail calculations, and absence of reward hacking (`04-result-review.md`).
- **Verdict**: no-improvement
- **Verdict Basis**: The run was complete, valid, full-dose, and mechanism-active, but exceeded its parent by only 0.01 point. `tree.sh insert` recorded EXP020 as a terminal failed leaf on `br-000`; global best remained 95.61% at EXP011.

## Unexplored Avenues

- **Raw-gradient GC**: remains formally untested, but effective L2/cross terms were only `4.71e-6` of removed energy, making its applied projection numerically almost identical and too weakly distinct to justify a metric run without new evidence.
- **Phase-limited convolution GC**: remains causally untested, but late-clean removal was the smallest phase ratio and full-run convolution GC was already accuracy-neutral. A timing variant should require evidence beyond magnitude before consuming another run.
- **Classifier sensitivity as a diagnostic**: the +0.17 recovery from EXP019 to EXP020 suggests output-layer geometry matters, but the useful next intervention should improve class boundaries rather than retry another GC eligibility variant.

## Next Steps

- **High confidence**: Move away from EXP002 optimizer projections and pursue an orthogonal representation or classifier-geometry intervention with plausible stable upside above 0.25 points.
- **Medium-high confidence**: Revisit the EXP011 global-best branch only with a mechanism capable of lifting its 95.49 EMA plateau, not another generic optimizer smoother.
- **Medium confidence**: Use the raw/L2 and phase audits to retire numerically redundant GC variants unless new causal or literature evidence supplies a genuinely different mechanism.

## Exit Action Results

No exit actions were defined for this goal.
