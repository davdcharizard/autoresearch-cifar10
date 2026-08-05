# Report EXP-028: Freeze the High-Resolution Prefix for the Hard Tail
- **Created**: 2026-07-26

## Goal

Increase fixed-seed CIFAR-10 `best_test_acc` by at least 0.10 points over the 94.32% baseline within the fixed 300-second counted budget, while testing whether late high-resolution gradients are less valuable than additional upper-layer hard-label updates.

## Idea & Hypothesis

Train the accepted EXP-027 model identically through the early mixup/RandAugment interval, then freeze exactly the stem and stage-1 parameters after the first eligible exhausted epoch. The full prefix forward path and live BN statistics remain. The hypothesis required at least 145 passes and both best and final accuracy at least 94.42% if early prefix features were already sufficiently established.

## Approach

`train.py` gained a strict 33,424-parameter prefix-freeze helper and a testable one-way post-iterator controller. The controller distinguished normal exhaustion from a budget break, disabled RandAugment and froze the prefix at the same boundary, retained all optimizer groups and momentum buffers, and left prefix BN buffers in train mode. The ignored verifier loaded accepted source independently from `git show 67c8e98:train.py`, proved exact pre-boundary identity and post-freeze state semantics, and calibrated timing to EXP-027's observed boundary/tail steps.

## Execution

The first semantic preflight exposed missing cuDNN deterministic flags in the verifier; after matching the production setting, the retry passed. Balanced hard-tail timing measured 11.213227 ms accepted versus 7.191277 ms frozen, a 1.559282x speed ratio projecting 159.374673 passes. One fixed-seed H20 score completed without retry. Mixup stopped at step 16,551 / 195.0 seconds; RandAugment and prefix freezing occurred together after epoch 85 exhausted at step 16,575 / 195.3 seconds.

## Results

- **Primary metric**: 93.99% (baseline: 94.32%, delta: -0.33 points, -0.350%)
- **Observations**: The run completed 31,074 steps / 159.09888 passes, 5,096 steps and 19.6% more exposure than accepted, in 300.0 counted / 348.4 wall seconds. Final accuracy was 93.92%; final loss 0.2804 was 0.0281 worse than accepted 0.2523. Peak VRAM stayed 1096.3 MiB.
- **Analysis**: The systems mechanism worked more strongly than projected: freezing only 3.4% of parameters removed 35.9% of complete hard-step time because they occupy the high-resolution prefix. The accuracy mechanism failed decisively. Extra upper-layer decisions could not replace continued low-level affine/filter adaptation after the input and label distributions became clean, and continued BN statistics alone were insufficient. The worse test loss despite near-zero training loss and much greater exposure rules out premature convergence; temporal representational drift or stopped prefix decay/momentum is the binding cost. This closes exact whole-prefix freezing at the 65% boundary.
- **Key Learning**: Late high-resolution adaptation remains essential: freezing 3.4% of parameters saves 35.9% hard-step time but loses 0.33 accuracy points.

## Verification

- **Conditions**: process, semantics, exposure, transitions, cadence, and wall time passed; best accuracy failed 94.42%, so the final-accuracy gate was skipped.
- **Review Notes**: Results confirmed trustworthy: independent accepted oracle, one H20, one fixed-seed score, exact boundary/counts, 159.10 passes, 32 unique evaluations, one finite summary, and `train.py`-only production scope.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid 93.99% best accuracy is 0.33 points below baseline and 0.43 below the required threshold.

## Unexplored Avenues

- A much later or narrower prefix freeze would trade less adaptation for less speed, but selecting its boundary/subset from this result would be post-hoc tuning; do not retry without an independent gradient-stability measurement.
- Batch 128 with the fully scaled LR curve remains a distinct whole-run operating-point test; it does not freeze any representation and was the reviewed runner-up.

## Next Steps

- **Medium confidence**: test batch 128 with the fully scaled LR curve using the existing strict exposure/update gates.
- **Medium confidence**: preserve gradients in every stage and seek generalization improvements at near-zero backward overhead.
- **Low confidence**: revisit temporal freezing only after direct layer-wise late-update or gradient-drift evidence identifies a genuinely dormant subset.

