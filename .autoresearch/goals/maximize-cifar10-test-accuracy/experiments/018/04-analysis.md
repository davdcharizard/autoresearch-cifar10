# Report EXP-018: Final-Block-Only Neutral SE
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.07% WRN-16-2 baseline within 300 counted training seconds. Success required at least 94.17% with all local H20, scope, timing, seed, and evaluation constraints intact.

## Idea & Hypothesis

EXP-017's two stage-3 SE gates reached 94.16% and improved final loss, but cost 4.6% step throughput. Its final gate showed 7.8 times the first gate's across-example variance. EXP-018 therefore retained only the final-block ratio-16 gate and removed scored diagnostics, hypothesizing that it carried the useful conditional selection while the weakly conditional first gate mainly consumed runtime.

## Approach

One `128->8->128` SE gate globally pooled the signed residual of `layer3[1]` and applied `2*sigmoid` channel scales before the unchanged shortcut addition. The accepted WRN was fully initialized first. Gate initialization then reused fixed project seed 42 inside a CPU-only restored RNG fork, with Kaiming first-projection weights and an exact-zero second projection for unit initial scale. No diagnostic buffers or summary work were included in production.

## Execution

Semantic preflight passed exact single-gate placement, 693,858 parameters, common state/logits, CPU/CUDA RNG preservation, shortcut isolation, optimizer coverage, and two-step opening. Matched H20 timing retained 98.58% throughput with worst window CV 0.078%. The sole scored run completed cleanly, disabled mixup exactly once at step 17,245 and 195.0 seconds, evaluated 28 unique epochs, and produced no error signature.

## Results

- **Primary metric**: 93.67% (baseline: 94.07%, delta: -0.40 points, -0.43%)
- **Observations**: The candidate completed 26,920 steps, or 137.8304 dataset-equivalent passes, in 300.0 training seconds and 340.9 wall seconds. Final accuracy/loss were 93.67%/0.2468, with the best accuracy still rising at the final epoch. It recovered 4.19 passes over EXP-017 but lost 0.49 accuracy points relative to that two-gate treatment.
- **Analysis**: The operational hypothesis succeeded: removing one gate and all diagnostics raised matched retention to 98.58% and realized exposure above 137 passes. The accuracy hypothesis failed decisively. Gate 1's greater example dependence did not mean it caused EXP-017's gain in isolation. The removed first gate had learned a strong mean residual attenuation near 0.65, and either that transformation or interaction between both gates was essential. Because extra exposure coincided with worse accuracy and slightly worse loss than accepted, undertraining is not the limiting explanation. Conditionality magnitude alone is therefore not a sound component-selection criterion.
- **Key Learning**: Final-only SE recovers exposure but destroys the two-gate accuracy signal; the first gate's attenuation or gate interaction is essential.

## Verification

- **Conditions**: Completion and integrity passed; the primary metric condition failed because 93.67% is below 94.17%.
- **Review Notes**: Results are trustworthy. The run used one H20, exact accepted common state/RNG, fixed seed 42, one scored execution, correct timing/evaluation behavior, and only the planned `train.py` diff.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid complete run with no hard-constraint violation, but accuracy fell 0.40 points below baseline and 0.50 below the acceptance threshold.

## Unexplored Avenues

- **Two gates without observational diagnostics**: preserve EXP-017's exact gate placement and initialization while removing only diagnostic accumulation. This directly tests whether recovered exposure can move the 94.16 near miss over the threshold without deleting the essential first gate.
- **Cheap learnable first-block residual scale plus final SE**: replace the weakly conditional first SE MLP with an exact-neutral static channel scale, retaining learned attenuation and the conditional final gate at lower cost.
- **Early-only mild RandAugment**: stop augmentation with mixup to preserve the validated clean tail; this remains orthogonal but has weaker local evidence.

## Next Steps

- **High confidence - two-gate SE without diagnostics**: isolate instrumentation overhead while preserving the only complete attention treatment with positive top-1 and loss signals.
- **Medium confidence - static first-block scale plus final SE**: test whether cheap learned attenuation recovers the interaction exposed by EXP-018.
- **Low confidence - early-only mild RandAugment**: pursue only after the attention neighborhood is exhausted.

## Exit Action Results

