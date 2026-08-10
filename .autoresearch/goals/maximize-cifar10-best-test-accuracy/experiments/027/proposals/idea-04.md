# Proposal: Short-Horizon Weak-Tail EMA Evaluator

## Mechanism and hypothesis

Preserve the accepted training path and maintain a detached exponential moving average of parameters only after the 80% weak-tail switch. Use a short effective window, for example decay `0.995` per optimizer step (roughly 200-step memory), so the evaluated model follows recent refinements rather than uniformly averaging the entire annealed tail. Copy ordinary BatchNorm running buffers from the online model at evaluation time, swap EMA parameters into the same model for the one allowed epoch evaluation, then restore online parameters and continue SGD.

The hypothesis is that recent-state EMA will suppress the observed tail checkpoint variance without EXP018's backward bias, raising `best_test_acc` from 94.15% to at least 94.25% while retaining at least 99% of accepted updates and keeping total wall time below 600 seconds.

## Evidence and limits

The persistent knowledge entry `knowledge/papers/weight-averaging.md` reports that late averaging can improve generalization cheaply and works well with annealing. EXP018 explicitly did not test EMA: its uniform eight-snapshot average covered 87.14-97.30%, had substantial parameter spread, and ended 0.17 below its own online best after BN recalibration. EMA's exponential weighting directly addresses that lag mechanism. However, accepted EXP010 finished at its best with only 0.01-0.12 point late fluctuations, so the available improvement ceiling may be smaller than the required 0.10 margin. Copying online BN buffers is expedient rather than a principled EMA of the full function.

## Exact production design

Add a flat FP32 EMA tensor list matching named parameters, initialized from the online model at the weak switch. After each weak-tail `optimizer.step`, update under `torch.no_grad()` with in-place `lerp_`. At each existing dense-tail evaluation, charge and synchronize a reversible transaction: save online parameters, install EMA parameters, keep the online model's current BN buffers, call the evaluator once, then restore online parameters before the next epoch. Track EMA age, normalized online/EMA distance, transaction time, and evaluated provenance. Do not evaluate online and EMA models in the same epoch, change evaluator cadence, recalibrate BN, alter SGD, or extend the counted budget.

## Verification

- Static mapping and round-trip tests must prove parameter ordering, exact online restoration, untouched optimizer/momentum and BN buffers, and no aliasing.
- Exact real-batch paired training must show identical online logits/loss/state with and without passive EMA tracking through at least 200 steps, plus finite EMA values and a nonzero bounded distance after weak updates.
- Five alternating H20 timing pairs must show counted-step ratio <=1.01, each pair <=1.03, projected steps >=26,629, peak allocation <650 MiB, and total projection <540 seconds.
- A synthetic moving-sequence oracle must match an FP64 EMA reference. Evaluation swap/restore must be reversible to registered tolerance and preserve RNG.
- Production must log one evaluator call per epoch, EMA age/distance, one 80% worker switch, accepted CutMix proportions, and ten finite summary fields.

## Risks and abort criteria

The main scientific risk is insufficient ceiling: EXP010's online tail is already smooth and ends at its best. A short EMA may simply lag useful cosine refinement, while a longer EMA recreates EXP018's failure. The main integrity risk is using max-over-checkpoints to tune decay after observing results; `0.995` and the start point must be fixed before preflight. Abort on any online-state drift, incomplete restoration, BN mutation, EMA non-finiteness, timing/exposure miss, extra evaluation, or preflight EMA underactivity. This is lower-evidence than mechanisms that directly address the 89.73% switch-fit constraint.
