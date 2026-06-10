# Report EXP-003: Earlier Second LR Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md
- **Plan**: plans/plan-003.md
- **Log**: logs/exp-log-003.md

## Goal

Maximize CIFAR-10 `best_test_acc` in the higher-is-better direction. The current baseline entering this experiment was EXP-002 at 91.95%.

## Idea & Hypothesis

The idea was to keep the successful FP32 throughput setup and move the second LR milestone from 48,000 to 40,000 steps. The hypothesis was that a reachable LR 0.001 phase would provide final refinement beyond 91.95%.

## Approach

Only `train.py` changed, and only the second scheduler milestone changed: `milestones=[32000, 48000]` became `milestones=[32000, 40000]`. The model, augmentation, optimizer, loss, throughput flags, and evaluation cadence stayed unchanged from EXP-002.

## Execution

One local single-GPU run completed normally. It reached LR 0.001 after step 40,000, ran for 300.0 training seconds, and completed 45,279 optimizer steps over 117 epochs.

## Results

- **Primary metric**: 91.85 (baseline: 91.95, delta: -0.10, -0.11%)
- **Observations**: The retuned milestone was reached and produced a final LR 0.001 phase, but peak accuracy stayed below EXP-002. Final accuracy was 91.66%.
- **Analysis**: The second drop at 40k was too early or unnecessary for this setup. EXP-002 benefited from staying at LR 0.01 through the end, while EXP-003's low-LR phase reduced the peak by 0.10 points.
- **Key Learning**: Moving the second LR drop to 40k reached LR 0.001 but reduced peak accuracy to 91.85%.

## Verification

- **Conditions**: primary metric condition failed.
- **Review Notes**: Results are trustworthy. The run completed, reported metrics, reached LR 0.001, and changed only the intended scheduler milestone.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but 91.85% did not exceed the 91.95% baseline.

## Unexplored Avenues

- Try a later second drop such as 42k or 44k, though the small regression suggests the original unreachable 48k may already be close to optimal for this budget.
- Test TF32 matmul precision warning from EXP-002 for possible additional throughput without schedule changes.
- Move to a compact WRN architecture with the EXP-002 FP32 throughput setup.

## Next Steps

- High confidence: avoid a 40k second milestone for the current ResNet-20 throughput setup.
- Medium confidence: test `torch.set_float32_matmul_precision("high")` as a speed-only follow-up.
- Medium confidence: begin WRN-16-2 architecture exploration using the successful FP32 throughput baseline.

## Exit Action Results
