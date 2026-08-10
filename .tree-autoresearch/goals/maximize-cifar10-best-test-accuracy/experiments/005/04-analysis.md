# Report EXP-005: Clean-Gated Last-Mini-Batch Self-Distillation
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc` under the frozen 300-second charged training budget by adding a low-cost temporal consistency mechanism to EXP-004. The parent and global best were EXP-004 at 95.40%; improvement required at least 95.50%. EXP-005 reached 95.28% and becomes a terminal no-improvement leaf.

## Idea & Hypothesis

Use Self-Distillation from the Last Mini-Batch (DLB): repeat half of each batch one update later with a fresh crop/flip and train it against detached temperature-three logits from its prior occurrence. Claude selected DLB over HMix and ECA because it had matched CIFAR-10 and CutMix-composition evidence with the largest plausible effect. The preregistered hypothesis expected 95.50% or better while retaining at least 24,500 optimizer steps. The effect was explicitly attributed to the combined repeat-view sampler plus clean-gated KL mechanism, because no sampler-only control was run.

## Approach

Only `train.py` changed. A private seed-42 sampler permutes 50,000 identities, retains 49,920, forms 390 chunks of 128, and emits 389 overlapping 256-example batches. The dataset returns raw indices so every adjacent overlap is asserted before KL, while repeated occurrences independently rerun crop/flip transforms. Clean-clean transitions add `tau^2 * KL(softmax(cached/tau) || softmax(current/tau))` with `tau=3` and `alpha=1`; mixed batches keep the exact parent CutMix objective and invalidate the cache. The clean-tail SAM path uses CE+KL in both passes with the same teacher, but publishes only primary-forward logits after exact restore and the sole optimizer update. Separate sampler, worker, and CutMix generators, plus final transition counters, make the changed data schedule auditable.

## Execution

One run launched on physical GPU 0 with fixed seed 42 and a 600-second timeout. Static, full-epoch DataLoader, KL, and full-WRN GPU smokes passed before launch. They verified 49,920 unique identities, exact overlap, 128/128 fresh repeated views, temperature scaling, detached targets, 0.05 SAM perturbation, CUDA RNG replay, one BatchNorm update, exact restoration, and primary-only cache publication. The full run completed exit 0 without retry, metric-driven adjustment, or protocol error.

## Results

- **Primary metric**: 95.28% (parent: 95.40%, delta vs parent: -0.12 points, -0.13%; global best: 95.40%)
- **Observations**: The run retained 25,492 steps, only 68 fewer than the parent, so arithmetic overhead was negligible. DLB was active on 10,037 batches (39.4% of all steps), with 1,284,736 repeated examples, 15,247 clean teacher publications, 5,173 mixed-batch invalidations, 66 epoch resets, and zero mismatches. CutMix remained 0.4962 and SAM remained 0.5001, beginning at step 20,646/progress 0.7500. Final accuracy was 95.22%, only 0.06 below best, while final test loss improved from the parent's 0.1654 to 0.1455. Peak allocation stayed 1,190.5 MiB.
- **Analysis**: The throughput component of the hypothesis passed but the accuracy hypothesis failed. Each 389-step DLB epoch presented 99,584 transformed images while introducing only 49,920 unique identities, approximately half the parent's new-identity rate, and produced 66 evaluations versus 132. The nearly identical step count rules out compute loss as the explanation. Lower test loss with lower top-1 accuracy is consistent with the repeated-view/KL recipe improving probabilistic smoothness without improving enough class decisions, while reduced identity diversity plausibly offsets its consistency benefit. The preregistered simple over-regularized-tail diagnostic did not trigger because final loss improved. Without an `alpha=0` overlapping-sampler control, the result cannot separate repeat-view sampling from KL and therefore discredits only the combined recipe on this lineage.
- **Key Learning**: Half-overlap DLB preserved step throughput and lowered test loss, but halved new-image exposure and reduced accuracy by 0.12 points.

## Verification

- **Conditions**: Parent/scope, static correctness, sampler/KL semantics, GPU/SAM integration, hardware, timing, complete summary, evaluation cadence, and every protocol-integrity check passed. The primary metric condition failed: 95.28% was 0.22 below the required 95.50%.
- **Review Notes**: Results are trustworthy. Physical GPU 0 was the 97,871 MiB H20; 66 evaluations occurred for 66 natural epochs; only `train.py` changed; model size remained 2,748,890; charged time was 300.0 seconds and total time 382.4 seconds; DLB mismatches were zero. The lower total time than the parent reflects half as many excluded validation runs, not more charged training.
- **Verdict**: no-improvement
- **Verdict Basis**: The run was valid and respected every hard constraint, but the metric fell 0.12 points below its parent and failed the required parent-plus-0.10 gate.

## Unexplored Avenues

- Run an `alpha=0` half-overlap control only if causal isolation becomes valuable; it would distinguish the sampler effect from KL but is a diagnostic rather than the highest-value accuracy experiment.
- Replace forced immediate overlap with a per-example logits memory bank under the parent's independent 256-image batches. This preserves new-identity throughput but uses older, variable-age teachers and needs a clearly fixed cache policy.
- Revisit temporal consistency on a branch without the already strong CutMix/SAM stack. The paper evidence may transfer better where consistency regularization is not competing with two validated generalization mechanisms.

## Next Steps

- **High confidence**: Navigate back to EXP-004 and test a representation or architecture lever that preserves its independent-image stream and validated SAM tail.
- **Medium confidence**: Profile a memory-bank self-distillation design before proposing it, focusing on teacher age, cache coverage, and whether its expected effect clears the 0.10-point gate.
- **Medium confidence**: Prefer mechanisms with a plausible effect above the 0.14-0.29-point run variability observed in EXP-003 rather than narrow augmentation or regularization scalars.
