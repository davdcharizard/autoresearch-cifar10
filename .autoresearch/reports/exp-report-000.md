# Report EXP-000: Cutout, Label Smoothing, and Cosine LR
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md
- **Plan**: plans/plan-000.md
- **Log**: logs/exp-log-000.md

## Goal

Maximize CIFAR-10 `best_test_acc` in the higher-is-better direction while modifying only `train.py` and preserving the fixed `prepare.py` evaluation harness. The current baseline from `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` is 91.52%.

## Idea & Hypothesis

The chosen idea was a cheap training-recipe upgrade for the existing ResNet-20: add cutout-style masking, modest label smoothing, Nesterov SGD, and a cosine LR schedule. The hypothesis was that stronger regularization and smoother learning-rate decay would improve generalization without materially reducing throughput.

## Approach

`train.py` was updated to add `transforms.RandomErasing` with an exact 16x16 square area at probability 0.5, use `label_smoothing=0.05` in cross entropy, enable Nesterov momentum, and replace the step LR schedule with `CosineAnnealingLR(T_max=MAX_STEPS, eta_min=0.001)`. The only notable implementation decision was using exact cutout area bounds rather than a variable RandomErasing scale range, matching the intended CIFAR cutout size more closely.

## Execution

One local single-GPU run was launched with output redirected to `run.log`. The run completed normally without tracebacks, OOM, or timeout. It used 300.0 training seconds and 360.7 total seconds, completing 91 epochs and 35,279 optimizer steps.

## Results

- **Primary metric**: 90.45 (baseline: 91.52, delta: -1.07, -1.17%)
- **Observations**: Accuracy climbed steadily but remained below baseline; by epoch 90 the best value was 90.45%, and the final summary preserved that as the run best. Runtime and VRAM were normal: 330.1 MB peak VRAM and the same parameter count scale as baseline.
- **Analysis**: The combined recipe likely over-regularized or slowed effective optimization for this small ResNet-20 under the fixed 300 second budget. Cosine annealing also decayed more slowly than the baseline step schedule because only 35,279 of 64,000 planned steps were reached, leaving LR relatively high late in the wall-clock run. This does not discredit cutout or cosine individually, but this combined configuration is worse than the baseline.
- **Key Learning**: Combining exact 16x16 cutout, label smoothing, Nesterov, and slow cosine decay undertrained ResNet-20 within 300 seconds.

## Verification

- **Conditions**: primary metric condition failed.
- **Review Notes**: Results are trustworthy. The run completed, reported a numeric `best_test_acc`, modified only `train.py`, and used the fixed evaluation harness.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=90.45%` did not exceed the 91.52% baseline.

## Unexplored Avenues

- Isolate the recipe components: try Nesterov plus a baseline-like schedule, or cutout without label smoothing, to identify which component caused most of the regression.
- Retune cosine for the observed 35k-step budget rather than `MAX_STEPS=64000`, or use an epoch/observed-step horizon so LR reaches its low regime during the 300 second run.
- Move to a compact WRN-style architecture after isolating cheap recipe changes, since this ResNet-20 recipe appears capacity or optimization limited.

## Next Steps

- High confidence: test a schedule-only intervention using the baseline augmentation and no label smoothing, with LR decay calibrated to the observed 35k steps.
- Medium confidence: try mild cutout only, with lower probability or smaller area, to see whether image-space regularization helps without undertraining.
- Medium confidence: try a compact WRN-16-2 architecture with baseline optimizer first, using the saved WRN knowledge entry as the architecture reference.

## Exit Action Results
