# Report EXP-009: Isolated BF16 Autocast at Batch 256
- **Created**: 2026-07-24

## Goal

Maximize CIFAR-10 `best_test_acc` above the accepted 94.07% baseline, with at least 94.17% required for improvement, by testing whether H20 BF16 autocast can trade lower-precision training compute for more useful updates inside the fixed 300-second budget.

## Idea & Hypothesis

Keep batch 256, FP32 master parameters/gradients/SGD state, the accepted WRN, schedule, mixup, decay, seed, and FP32 evaluator, while executing only training forward and loss under BF16 autocast. The joint hypothesis was that BF16 numerical quality plus at least 10% denser time-scheduled updates would reach 94.17% with final accuracy near best and loss near or below 0.2432.

## Approach

Added `TRAIN_AUTOCAST_DTYPE = torch.bfloat16`, logged it once, kept mixup interpolation FP32, and wrapped model forward plus cross entropy in training-only autocast. Backward and SGD stayed outside autocast; no GradScaler, model/state cast, batch/LR change, layout, fusion, compilation, or evaluator change was made. A fail-closed, evaluator-free preflight benchmarked the full production timed body with cloned models, matched RNG streams, pinned host copies, and a preregistered six-window order.

## Execution

The preflight passed with FP32/BF16 median step times of 10.877337/9.482360 ms, population CVs 1.1004%/0.4270%, a 1.147113x throughput ratio, and 162.7753 projected passes. All parameters, gradients, momentum buffers, and loss were FP32; treatment logits were BF16. One fixed-seed scored run then completed without retry or adjustment. Mixup disabled exactly once at epoch 102, step 19,814, 195.0 seconds with LR 0.0612; the run completed 31,069 steps / 160 epochs in 300.0 counted / 344.3 total seconds.

## Results

- **Primary metric**: 93.81% (baseline: 94.07%, delta: -0.26 percentage points, -0.28%)
- **Observations**: The treatment realized 159.07328 passes, 12.1% above accepted 141.9 and safely above the 156.09 mechanism gate, though 2.3% below the preflight projection. Best accuracy occurred at epoch 150; final accuracy was 93.78%, a small 0.03-point gap. Final test loss worsened from accepted 0.2432 to 0.2634. Peak allocation fell from about 1,094 MiB to 540 MiB.
- **Analysis**: The operational mechanism occurred: BF16 materially increased full-loop exposure and reduced memory while preserving stability, schedule, and cadence. Accuracy and loss nevertheless regressed. Because BF16 changes convolution numerics from the first update and maps each batch to a denser wall-time LR trajectory, this run cannot isolate whether extra FP32 updates would help. It does establish that the deployed BF16-plus-denser-update tradeoff is inferior for this recipe; neither additional legal evaluations nor 12.1% more data passes recovered BF16's numerical/trajectory cost. Batch 512, LR scaling, or GradScaler would be distinct confounded treatments and are not justified as result-conditioned rescues.
- **Key Learning**: BF16 buys 12.1% more exposure and halves VRAM, but its combined numerical/update trajectory worsens loss and accuracy for the accepted WRN.

## Verification

- **Conditions**: Completion and process integrity passed; the required 94.17% accuracy threshold failed.
- **Review Notes**: Results confirmed trustworthy. One H20, fail-closed preflight, exit 0, complete summary, 300.0 counted seconds, 344.3 total seconds, 32 unique accepted-cadence evaluations, 159.07 passes, one correct transition, and a `train.py`-only allowlisted diff were verified.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid run scored 0.26 points below baseline and 0.36 below the required threshold despite achieving its exposure gate, with worse final test loss.

## Unexplored Avenues

- An FP32-only throughput optimization could test extra exposure without BF16 rounding, but likely gains from fused SGD or layout changes must be measured before consuming a scored run.
- BF16 plus batch 512 is not attractive: local development timing added only 1.5% image throughput while changing update count, BatchNorm statistics, mixup grouping, and LR policy.
- Mixed policies that force selected layers back to FP32 could improve numerical behavior, but layer selection would be a new precision search without local evidence and risks post-hoc overfitting.

## Next Steps

- **High confidence**: return to accepted FP32 and test the already-developed selective 160-channel final stage, contingent on its strict matched throughput/exposure gate.
- **Medium confidence**: measure a narrow FP32-only fused-SGD or memory-format optimization; score it only if it creates meaningful exposure without changing model numerics.
- **Low confidence**: retain low-magnitude RandAugment as an exploratory fallback, but repeated additive-regularization regressions make its expected value lower.

## Exit Action Results

No exit actions were defined for this local-only goal.
