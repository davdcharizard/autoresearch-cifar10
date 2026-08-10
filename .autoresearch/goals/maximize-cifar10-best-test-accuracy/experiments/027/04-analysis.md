# Report EXP-027: CutMix-Off, RandAugment-On Refinement Window
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the `94.15%` frontier at `7c1e7d8`. A valid improvement required at least `94.25%` under the fixed seed-42, one-H20, 300-counted-second, `train.py`-only protocol.

## Idea & Hypothesis

Preserve accepted N1/M7 plus 50% alpha-1 CutMix through 70%, then disable only CutMix while retaining hard-label N1/M7 until the unchanged 80% weak-tail/LR transition. EXP005 argued against dropping RandAugment early, while EXP011/026 showed rapid recovery after mixed targets disappeared. The hypothesis predicted that a 10%-budget hard strong-view bridge would repair adaptation debt, preserve at least 26,629 steps, restore the switch to at least 89.73%, and reach at least 94.25% best accuracy.

## Approach

Only tracked `train.py` changed. A module-level collator read one locked forkserver-shared boolean snapshot per batch, applied the accepted fork-RNG-isolated CutMix policy while enabled, and returned explicit provenance. The parent flipped the flag once after the first completed step at 70% without breaking or rebuilding the strong loader. It counted eligible hard/CutMix, disabled hard, and prefetched drain batches; N1/M7, workers, model, optimizer, schedule, evaluator, and weak tail otherwise remained unchanged.

Preflight created an EXP027-owned 200-batch immutable pre-policy corpus, passed exact common-prefix continuation, exercised 20,000 deliveries with real epoch iterator recreation, and verified all worker/lifecycle semantics. Five alternating H20 timing pairs preceded the single scored run.

## Execution

The corpus SHA was `3af2f47731c1184b18cad3d51849619cdda08fe30c2a0a73f1ba8ad08d70d70e`; semantic replay produced 97/200 CutMix decisions. Exact continuation passed with loss-EMA ratio `0.939069` and no candidate-only concentration. In 20,000 live deliveries, CutMix was 49.967%, the flag drained after eight batches, every worker observed it, and both loaders stopped cleanly. Timing passed at mean ratio `1.006886`, worst pair `1.016473`, projected 26,714 steps, and 332.98 seconds total.

One scored run started at 2026-08-06T17:41:29Z and exited zero without retry at 2026-08-06T17:47:28Z. The request fired at 70.002%; 13 prefetched eligible batches drained through 70.051%, followed by 2,688 hard N1/M7 batches. The run log SHA is `6341b24361945bece4d70b22877a7dfe09b002f63099483442050774f6dbb2aa`.

## Results

- **Primary metric**: `93.69%` (baseline: `94.15%`, delta: `-0.46` percentage points, `-0.49%` relative)
- **Observations**: Accuracy was 88.35% at 60%. At the request epoch it collapsed to 82.96%, then recovered only to 88.76% at the 80% switch—0.97 below EXP010. The first weak checkpoint was 93.00%, best reached 93.69% in epoch 69, and final accuracy/NLL were 93.57%/0.2031 versus EXP010's 94.15%/0.1934. Exposure remained healthy at 26,986 steps, with 49.702% CutMix while eligible and unchanged 598.7 MiB peak memory.
- **Analysis**: The proposed debt-repair mechanism is discredited at this operating point. Hard labels did not sharpen continuing N1/M7 training under LR 0.1; they abruptly made the distorted-view objective harder to reconcile, producing a much larger transient than ordinary trajectory noise and leaving a persistent switch/NLL deficit. Because throughput, lifecycle, proportions, and exposure all passed, the miss is not a systems confound. Together with EXP005, the result indicates that removing either broad-view strength or mixed-target regularization before the 80% LR drop harms this short recipe. The accepted simultaneous transition works because hard labels arrive with both weak views and low LR, not because any hard-label bridge is inherently helpful.
- **Key Learning**: Hard N1/M7 at LR 0.1 caused a sharp fit collapse and 0.46-point loss; keep mixed targets until the LR/weak-tail boundary.

## Verification

- **Conditions**: Primary accuracy failed (`93.69% < 94.25%`). All scope, semantic, corpus, continuation, timing, lifecycle, exposure, completion, summary, time, transition, proportion, and evaluator-cadence conditions passed.
- **Review Notes**: Results are trustworthy. Exactly one scored seed-42 run used one idle H20; only `train.py` changed; the process exited zero with ten finite fields; times were 300.0/333.5 seconds; all 19 evaluation epochs were unique; and request/drain provenance proves the intended 2,688-batch bridge occurred.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid result fell 0.46 points below the moving baseline and 0.56 below the required gate.

## Unexplored Avenues

- A CutMix-off bridge beginning simultaneously with the LR drop but retaining N1/M7 would separate low LR from weak views. It is distinct, but the accepted simultaneous switch is strong and another phase transition needs compelling evidence.
- Removing only soft targets while preserving pasted pixels would isolate classifier adaptation from spatial composition, but requires a custom target policy with weak literature support.
- A shorter bridge nearer 80% might reduce damage, but tuning the failed 70% boundary in place would be low-information parameter chasing.

## Next Steps

- **High confidence**: preserve the complete accepted N1/M7+CutMix phase until the simultaneous 80% low-LR/weak/hard transition.
- **Medium confidence**: return to a bounded representation mechanism that does not suppress or retarget high-LR strong training.
- **Low confidence**: revisit phase decoupling only with direct evidence that low LR, rather than view strength, makes hard strong labels viable.

## Exit Action Results

- None defined.
