# Report EXP-007: Width-2 ResNet-20
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the moving baseline of 92.30% at `11f8469`. A valid improvement had to reach at least 92.40%, modify only `train.py`, run once on one idle H20 under the fixed 300-second counted budget, and finish within 600 seconds.

## Idea & Hypothesis

Double post-activation ResNet-20 stage widths from 16/32/64 to 32/64/128 while preserving depth, raw Option-A shortcuts, and the complete accepted EXP-004 N1/M7-through-80% plus weak-tail recipe. External Claude adversarial review selected width as the highest-ceiling measured-feasibility candidate, while challenging the unproven capacity diagnosis and the expected 31% update loss. The narrowed hypothesis asked whether capacity specifically under strong views was worth that fixed-time exposure cost.

## Approach

Changed only `train.py` at commit `8faf0f3`. Added `WIDTH_MULTIPLIER = 2`, made `ResNet` accept a multiplier defaulting to 1, derived only stem/stage/classifier dimensions from the existing widths, and instantiated training with width 2. `BasicBlock`, post-activation order, Option-A slicing/padding, initialization, batch 128, hard-label loss, SGD at `1e-4` decay, LR schedule, transforms, worker lifecycle, seed, and evaluator remained unchanged. The model grew from 269,722 to exactly 1,073,962 parameters.

## Execution

Mandatory external Claude idea and plan reviews both completed with exit code 0; no fallback reviewer was used. Static and architecture checks passed. Three paired H20 timing trials measured width-1/width-2 median trial means of 7.541/10.928 ms, 0.452%/0.299% trial CV, a 1.4492 ratio, 598.7 MB peak allocation, and 26,469 calibrated steps. Claude's plan review prevented a noisy 26,500 point-estimate gate from aborting this valid candidate. One fixed-seed full run then exited 0 without retry in 333.0 seconds total.

## Results

- **Primary metric**: `93.55%` (baseline: `92.30%`, delta: `+1.25` percentage points, `+1.35%` relative)
- **Observations**: The final strong checkpoint was 90.08% at epoch 56 with train-loss EMA 0.2283, versus EXP-004's 84.60% strong checkpoint. After the exact 80.0% switch, the first weak epoch reached 92.96% versus EXP-004's 91.43%. Accuracy peaked at 93.55% on epoch 67 and finished at 93.49%/0.2196 on epoch 71; EXP-004 finished at 92.23%/0.2574. The final three evaluations were effectively flat (93.48, 93.51, 93.49), so the shorter tail was not visibly still climbing.
- **Analysis**: The hypothesis is strongly supported as a net fixed-time result. Width 2 retained only 27,143 steps, 70.76% of EXP-004's 38,358, yet improved the strong-view checkpoint by 5.48 points and the peak by 1.25. This localizes the benefit to representation capacity under the accepted hard-view regime rather than extra optimization exposure. The weak tail still supplied 15 evaluated epochs, converted the strong representation immediately, and plateaued before termination; fewer evaluations did not prevent observing a clear gain. Capacity at equal compute was not tested, and wider initialization changes the inherent fixed-seed augmentation stream, so the exact effect size is not purely causal. The 1.25-point margin, trajectory separation, exact scope, and clean process make the improvement robust under the declared protocol.
- **Key Learning**: Width 2 gained 1.25 points despite 29.2% fewer updates; capacity under the accepted strong-view recipe was the dominant trade.

## Verification

- **Conditions**: All passed: 93.55% >= 92.40%, exit 0, ten finite summary fields, 300.0 counted seconds, 333.0 total seconds, one 80.0% switch with eight stopped workers, 21 unique evaluator epochs, and exactly 1,073,962 parameters.
- **Review Notes**: Results confirmed trustworthy. Only the reviewed model-width diff changed; one idle H20 was used; evaluator, seed, data recipe, optimizer, time budget, and evaluation cadence were preserved; the preflight predicted exposure accurately; no retry or seed selection occurred. The improvement is the net width/update effect under fixed time, not a claim about width at equal steps.
- **Verdict**: improvement
- **Verdict Basis**: Every integrity and runtime condition passed, and the primary metric exceeded the moving baseline by 1.25 points and the acceptance threshold by 1.15.

## Unexplored Avenues

- Increase weight decay on the accepted width-2 model. The tail train loss fell near 0.04 while test accuracy plateaued, and the wider model has more overfitting headroom; `5e-4` is already a reviewed canonical candidate but must be tested as a separate intervention.
- Benchmark width 3 or another larger multiplier. Width 2's strong gain supports a capacity trend, but additional quadratic work may shorten the tail too far and requires a new paired timing gate.
- Test preactivation on width 2. The shallow-depth generic case was weak, but its interaction with a now-capable strong-view model remains orthogonal; expected upside is lower than regularization tuning.

## Next Steps

- **High confidence**: preserve width 2 and the full EXP-004 data/schedule recipe, then adversarially review stronger weight decay as the cleanest regularization follow-up.
- **Medium confidence**: compare a larger width multiplier against the measured capacity/exposure frontier if H20 timing preserves a usable weak tail.
- **Low-medium confidence**: revisit preactivation only after width-2 regularization is tested; do not stack CutMix on the already hard strong phase.

## Exit Action Results

- None defined.
