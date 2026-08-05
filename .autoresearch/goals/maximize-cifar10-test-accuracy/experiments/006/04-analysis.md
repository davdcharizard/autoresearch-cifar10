# Report EXP-006: Early p=0.10 WRN Block Dropout
- **Created**: 2026-07-24

## Goal

Maximize CIFAR-10 `best_test_acc` above the accepted 94.07% baseline, with at least 94.17% required for a meaningful improvement, by testing whether weak internal WRN feature regularization adds value beyond early mixup.

## Idea & Hypothesis

Apply p=0.10 dropout inside each learned residual branch during the first 65% of counted training, then disable it together with mixup for the validated hard-label tail. The idea was selected over lower-ceiling EMA and LR-floor probes because internal feature regularization had a plausible multi-tenth upside. The hypothesis predicted at least 94.17% while retaining 95% of EXP-002's exposure; lower accuracy and higher loss at normal exposure would indicate additive over-regularization.

## Approach

Added a probability field to all six `PreActBlock` modules and applied guarded `F.dropout` after the second BN/ReLU and before `conv2`, leaving identity shortcuts deterministic. The existing 65% transition calls `WideResNet.set_block_dropout(0.0)` once, after which the guard bypasses dropout and consumes no mask RNG. WRN-16-2, alpha-0.2 mixup, batch 256, optimizer, time-cosine schedule, seed, loader, evaluator, and evaluation cadence remained unchanged. Separate preflights stubbed module-scope `prepare.Eval`, so they tested the real model code without constructing or inspecting the test set.

## Execution

One preregistered fixed-seed run completed without retry or adjustment. Semantic checks confirmed six active blocks, eval-mode determinism, train-mode stochastic masks, an RNG-free p=0 path, finite logits/loss, and the unchanged 691,674 parameters. An order-balanced H20 benchmark measured 12.567 ms for p=0 and 12.763 ms for p=0.10, retaining 98.47% throughput and projecting 139.73 passes. The scored run disabled mixup and dropout exactly once at epoch 90, step 17,422, 195.0 seconds (65.0%), then completed 27,361 steps and 141 epochs in 341.9 total seconds.

## Results

- **Primary metric**: 93.52% (baseline: 94.07%, delta: -0.55 percentage points, -0.58%)
- **Observations**: Final accuracy equaled best accuracy at 93.52%. The run realized 140.09 passes, 98.7% of EXP-002's 141.9 and above the 134.8-pass attribution floor. Final test loss worsened from 0.2432 to 0.2718, while peak VRAM rose modestly from about 1,094 to 1,214 MiB.
- **Analysis**: The intervention had its intended local behavior and its 1.53% measured overhead did not materially reduce optimization exposure. Accuracy improved steadily through the dropout-free tail but converged to a stable lower endpoint with worse loss. This supports the preregistered additive-over-regularization diagnosis: alpha-0.2 mixup already provides enough early regularization for WRN-16-2, and p=0.10 residual-branch masking weakens representation learning rather than improving feature diversity.
- **Key Learning**: Early p=0.10 block dropout compounds mixup regularization and degrades this small WRN even when throughput and the clean tail are preserved.

## Verification

- **Conditions**: Run completion and process conditions passed; the required 94.17% accuracy threshold failed.
- **Review Notes**: Results confirmed trustworthy. The run used one H20, completed 300.0 counted seconds and 341.9 total seconds, evaluated 29 unique epochs at the allowed cadence, crossed the exposure floor, changed only `train.py`, and retained the frozen evaluator and seed value.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid completed run scored 0.55 percentage points below baseline and 0.65 points below the required threshold, with normal exposure and worse test loss.

## Unexplored Avenues

- A lower p=0.05 or earlier dropout cutoff could reduce the additive regularization, but the clear p=0.10 regression and prior alpha-0.4 failure make further stacked-regularizer tuning low priority.
- Replacing mixup with block dropout would isolate dropout as an alternative rather than stacking both, but it would discard the validated +0.69-point mixup gain and has weak supporting evidence.

## Next Steps

- **High confidence**: pivot away from additional regularization and isolate an optimization lever, such as a true late cosine-to-zero floor with the accepted 0.002 warmup start preserved.
- **Medium confidence**: test a low-overhead conditioning change such as evaluator-consistent in-model channel standardization.
- **Low confidence**: revisit late iterate averaging only if its BatchNorm-state policy and small expected ceiling can be justified against the 0.10-point threshold.

## Exit Action Results

No exit actions were defined for this local-only goal.
