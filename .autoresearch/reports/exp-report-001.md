# Report EXP-001: Baseline-Preserving Throughput Acceleration
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-001.md
- **Plan**: plans/plan-001.md
- **Log**: logs/exp-log-001.md

## Goal

Maximize CIFAR-10 `best_test_acc` in the higher-is-better direction while modifying only `train.py` and preserving the fixed `prepare.py` evaluation harness. The current baseline is 91.52%.

## Idea & Hypothesis

The chosen idea was to preserve the baseline statistical recipe while improving training throughput. The hypothesis was that channels-last layout, cuDNN benchmarking, BF16 autocast, and `torch.compile` would increase optimizer steps inside the fixed 300 second budget enough to improve `best_test_acc`.

## Approach

`train.py` kept the baseline model, crop/flip augmentation, cross-entropy target semantics, SGD hyperparameters, `MultiStepLR([32000, 48000])`, and once-per-epoch evaluation cadence. The implementation added feature flags for cuDNN benchmarking, channels-last model/input layout, CUDA BF16 autocast around forward/loss, and `torch.compile` wrapping of the model. `num_params` is computed before compile wrapping so the metric reports the underlying ResNet-20 size.

## Execution

One local single-GPU run completed normally with no traceback, OOM, compile failure, or timeout. The run used 300.0 training seconds and 394.0 total seconds, including compile/startup overhead. It completed 102 epochs and 39,558 optimizer steps.

## Results

- **Primary metric**: 91.48 (baseline: 91.52, delta: -0.04, -0.04%)
- **Observations**: Throughput improved relative to EXP-000: 39,558 steps versus 35,279, with 102 epochs and lower reported peak VRAM. Accuracy recovered sharply after the first LR drop and came within 0.04 percentage points of the baseline, but did not exceed it.
- **Analysis**: The throughput hypothesis was partially validated on systems behavior but not on the target metric. More steps helped relative to EXP-000 and nearly matched baseline, but BF16 autocast and/or compile/channels-last numerics may have introduced enough change to miss the strict target. The result suggests throughput is useful, but the next test should isolate non-numeric speedups or combine throughput with a small LR schedule adjustment.
- **Key Learning**: Throughput flags increased steps by ~12% over EXP-000 but BF16/compile reached only 91.48%, just below baseline.

## Verification

- **Conditions**: primary metric condition failed.
- **Review Notes**: Results are trustworthy. The run completed, reported a numeric metric, modified only `train.py`, and preserved the model, augmentation, optimizer, scheduler milestones, and evaluation cadence.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=91.48%` did not exceed the 91.52% baseline.

## Unexplored Avenues

- Disable BF16 while keeping cuDNN benchmark, channels-last, and possibly compile to test whether numeric precision caused the 0.04-point miss.
- Keep throughput flags and move the first LR milestone slightly earlier, since the run achieved more steps and spent meaningful time at LR 0.01 but still did not reach the second drop.
- Try a very narrow Nesterov-only or schedule-only change on top of baseline, avoiding BF16 and strong regularization.

## Next Steps

- High confidence: run a precision-preserving throughput variant with `USE_AMP=False`, keeping cuDNN benchmark/channels-last/compile if stable.
- Medium confidence: retune LR milestones to the observed 39k-step budget after precision-preserving throughput is isolated.
- Medium confidence: move to compact WRN-16-2 if baseline-preserving implementation changes keep plateauing below the target.

## Exit Action Results
