# Adversarial Idea Review: EXP-004

**Reviewer**: Claude Code cross-model harness

## Prioritized Feedback

1. **All predicted effects overlap the measured noise floor.** EXP-003 showed 0.14-0.29-point search-to-confirmation movement, so effect ceiling is more important than implementation simplicity. EMA and label smoothing plausibly produce effects near the 0.10-point gate; SAM has materially larger demonstrated upside.
2. **Sparse late EMA targets the wrong variance.** EXP-003 exposed between-run selection variance, while EMA reduces within-run trajectory variance. EXP-002's best/final gap was only 0.04 points, and classical SWA relies on a constant or cyclic LR tail that the current decaying-cosine schedule lacks. Starting earlier or using a conventional full-run EMA would be more defensible than the proposed 75% start.
3. **Period-four SAM may be too dilute.** Three plain Nesterov updates between SAM pulses can wash the sharpness-aware gradient out of the momentum buffer. Period two in the final quarter should double the SAM dose while retaining roughly 25,500-25,900 total steps, only about a 7-8% exposure loss.
4. **SAM's correctness surface needs explicit cadence auditing.** The plan must define whether cadence uses the pre- or post-increment step, and must concretely test the intended pulse sequence in addition to RNG replay, BatchNorm suppression, exact restoration, and one momentum update.
5. **Clean-only label smoothing likely adds to a saturated regularization regime.** EXP-003 showed that stronger CutMix exposure did not confirm and lower drop path gained only 0.05 points. Label smoothing is cleanly implementable but has the lowest plausible ceiling on top of existing CutMix and drop path.
6. **EMA's evaluation-source switch can hide online improvement.** Evaluating only EMA after activation respects the one-evaluation-per-epoch rule but makes a failure ambiguous if EMA lags the online model.
7. **Label-smoothed training loss is not directly comparable to EXP-002.** Any loss-trajectory analysis would need to account for the discontinuity when smoothing turns off.

No hard-constraint, evaluator-integrity, seed-hacking, or reward-hacking violation was found in any finalist.

## Scored Verdict

### Sparse Late State EMA
- **Evidence and reasoning: 4/10** - the cited SWA mechanism expects a more diverse LR tail, while the parent shows only 0.04 points of late checkpoint drift.
- **Potential impact: 3/10** - averaging near-identical low-LR iterates is likely too small to clear the gate.

### Clean-Only Two-Stage Label Smoothing
- **Evidence and reasoning: 6/10** - strong literature grounding and clean implementation, but it underweights EXP-003's regularization-saturation evidence.
- **Potential impact: 4/10** - plausible upside exists, but half of early batches already use soft CutMix targets and the marginal effect may be near zero.

### Clean-Finish Periodic SAM
- **Evidence and reasoning: 6/10** - full SAM has the strongest CIFAR evidence; the late periodic adaptation is unvalidated, but its compute model and correctness plan are rigorous.
- **Potential impact: 7/10** - it is the only finalist whose documented full-strength effect is large enough for a diluted fixed-budget form to plausibly clear the gate.

## Pick

**Clean-Finish Periodic SAM** wins because it directly targets generalization from a well-fitted solution and has the highest effect ceiling. Refine the chosen version from period four to period two in the final clean quarter, and pin cadence to an explicit post-update step convention before planning.
