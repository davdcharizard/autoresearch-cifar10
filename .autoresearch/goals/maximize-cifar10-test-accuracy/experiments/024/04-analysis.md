# Report EXP-024: Two Diagonal Conditional Stage-3 Gates
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.07% baseline, requiring at least 94.17%, while testing whether cheap per-channel input dependence can retain EXP-017's stage-3 attention signal.

## Idea & Hypothesis

Place exact-neutral diagonal gates on both 128-channel stage-3 residual branches. Each channel responds only to its own signed pooled residual. The hypothesis predicted at least 138 projected passes and acceptance if both placements plus per-example self-gating were sufficient without global cross-channel mixing.

## Approach

Each gate registered zero weight/bias vectors and applied `2*sigmoid(weight*pooled+bias)` before unchanged shortcut addition. Gates were attached after accepted model initialization, consuming no RNG. All vectors followed the existing one-dimensional no-decay policy; no diagnostics or training-loop changes were added.

## Execution

Semantic preflight passed exact accepted state/RNG/logits, unit scales, correct placement, open aggregate first-step gradients, optimizer grouping, and 692,186 parameters. Timing retained 97.8872% and projected 138.9019 passes. The sole score completed on one H20 in 341.7 wall seconds, disabled mixup once at 195.0 counted seconds, and produced no error or duplicate evaluation.

## Results

- **Primary metric**: 93.91% (baseline: 94.07%, delta: -0.16 points, -0.17%)
- **Observations**: The run completed 27,141 steps, 140 epochs, and 138.961920 effective passes. Best and final accuracy were both 93.91%; final loss was 0.2379, close to EXP-017's 0.2321 but below its 94.16 top-1.
- **Analysis**: The intervention achieved every local premise except accuracy: exact-neutral initialization, both gate placements, per-example channel response, open gradients, and near-baseline exposure. It still lost 0.16 points and trailed full SE by 0.25. Compared with EXP-019's static 93.86, restoring diagonal input dependence recovered only 0.05, far short of the full cross-channel gate. This strongly indicates that per-channel self-response is insufficient and global cross-channel interaction, not merely conditionality, drove EXP-017's near-positive signal. The single fixed-seed result strengthens rather than proves that causal conclusion.
- **Key Learning**: Both placements and input dependence are insufficient: diagonal self-gating scores 93.91, leaving global cross-channel mixing as the distinguishing SE mechanism.

## Verification

- **Conditions**: Completion/runtime passed; primary metric failed at 93.91% versus required 94.17%.
- **Review Notes**: Results are trustworthy: exact semantics, fixed seed, one H20/run, 300 counted seconds, 138.96 passes, correct transition/count, unique evaluations, and scoped production diff.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid complete result below baseline and acceptance threshold.

## Unexplored Avenues

- **Diagnostic-free full two-gate SE**: preserves the sole near-positive full interaction and removes only observation overhead, but its ceiling is noise-scale and seed-selected; useful mainly as closure.
- **Alternative cheap global mixing**: a structured cross-channel operator might preserve interaction with less overhead, but proposing one now requires new evidence rather than another adjacent simplification.

## Next Steps

- **Medium confidence - diagnostic-free full SE closure**: measure and score only if removing observation projects materially more exposure than EXP-017 while fixing seed 17017 and all semantics.
- **Low confidence - alpha-0.1 mixup closure**: test the last isolated strength side despite the under-regularization prior.
- **Low confidence - non-compounding input invariance**: develop an early substitution schedule that preserves the accepted clean tail.
