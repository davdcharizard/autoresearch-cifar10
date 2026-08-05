# Report EXP-029: Batch 128 With a Fully Scaled LR Curve
- **Created**: 2026-07-26

## Goal

Increase CIFAR-10 `best_test_acc` by at least 0.10 points over the 94.32% accepted baseline within 300 counted seconds. This experiment first required the exact batch-128 operating point to retain at least 120 passes and 46,875 full-model updates.

## Idea & Hypothesis

Halve batch size and the complete LR curve while doubling only the nonbinding step cap. The hypothesis was that finer full-model optimizer, BN, and batch-shared mixup decisions could improve boundary quality without freezing any representation, provided H20 image throughput retained at least 90.22% of accepted.

## Approach

The only production changes were `BATCH_SIZE=128`, `LR=0.1`, `MIN_LR=0.001`, and `MAX_STEPS=128000`. The ignored verifier loaded accepted source independently, proved exact initialization/optimizer/LR semantics and paired batch-128 clean-tail replay, then timed fresh deterministic accepted/candidate fixtures across complete pinned-copy-through-synchronize mixup and hard steps.

## Execution

Semantic verification passed: 987,098 parameters, 390 batches / 49,920 images per epoch, exact half-LR curve, finite update, and worker-safe RandAugment cutoff replay. The balanced GPU benchmark passed every <=5% timing-CV assertion, then failed `retention >=0.9022`. Because the payload print followed the assertion, exact windows were not emitted; the failed inequality proves projected passes were below 119.99924019 and updates below 46,874.703. Per the fixed gate, loader timing and scoring were not run and timing was not repeated.

## Results

- **Primary metric**: NaN (baseline: 94.32%; no score)
- **Observations**: Stable complete-body image retention was below 90.22%, so the candidate missed both the 120-pass and 46,875-update floors before loader/scored execution.
- **Analysis**: Batch 128 did not halve full-step latency sufficiently on the H20. Doubling update frequency therefore costs more than 9.78% image exposure, placing the intended operating point outside its preregistered joint exposure/update regime. The mechanism remains accuracy-unmeasured, but the exact four-constant treatment is operationally closed: scoring below the fixed regime would confound any result with excessive image loss, and repairing the LR floor or momentum would define another treatment.
- **Key Learning**: Batch 128 loses more than 9.78% image throughput on this H20 and misses the fixed joint exposure/update regime before scoring.

## Verification

- **Conditions**: semantic conditions passed; GPU exposure/update feasibility failed; loader, score, and accuracy verification were skipped.
- **Review Notes**: The failure is trustworthy because every CV assertion preceded the retention assertion. Exact timing values are unavailable due to fail-before-print ordering, but the threshold-derived upper bounds are deterministic.
- **Verdict**: crash
- **Verdict Basis**: No scored metric was produced after the mandatory pre-score feasibility gate failed.

## Unexplored Avenues

- Batch 128 with a different floor, momentum, or LR scaling remains mathematically possible but is not supported as an immediate repair; it would change the indivisible operating point and require new evidence.
- An intermediate batch could trade fewer extra updates for better image efficiency, but selecting it from this threshold miss would be adjacent post-hoc tuning rather than a principled next experiment.

## Next Steps

- **Low confidence**: complete the remaining alpha-0.1 one-constant mixup bracket if a low-cost controlled run is preferred.
- **Low confidence**: consider targeted drop-path only with the existing private-RNG and normal-exposure gates; local regularization evidence remains negative.
- **Medium confidence**: return to broader offline ideation for a new full-gradient, near-zero-overhead generalization mechanism rather than repairing batch 128.

