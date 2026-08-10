# Report EXP-008: Paired Early RandAugment Preflight Reject
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc` under the fixed 300-second charged training budget by adding a stronger input-invariance mechanism to EXP-004. The parent and global best were EXP-004 at 95.40%; formal improvement required 95.50% and the preregistered mechanism target was 95.70%.

## Idea & Hypothesis

Claude's randomized idea review selected RandAugment over identity-initialized ECA and full-state EMA because only RandAugment had direct CIFAR WRN evidence and a plausible effect beyond the roughly 0.30-point resolution floor. The chosen package applied one mild operation only during the first 75% so the validated clean SAM tail remained unchanged. It predicted at least 25,000 steps and at least 95.70%.

Planning discovered that the paper's magnitude-2 scalar is not semantically portable to torchvision 0.24.1: several 32x32 operations quantize to no-ops. Before any training or accuracy output, the package was locked at magnitude 5, the lowest audited mild bin with nonzero two-pixel translation and seven-bit posterization.

## Approach

Only `train.py` changed. Each DataLoader worker applies the exact parent crop/flip once and returns two views: a parent-identical clean normalized FP32 tensor and an augmented uint8 tensor. RandAugment uses a worker-seed-keyed private torch generator, with global state exchanged and restored in `try/finally`. The main loop computes progress once, selects the augmented view only below 0.75, normalizes it on GPU, and selects clean input for every SAM batch. CutMix, independent identities, drop path, Euclidean SAM, model, optimizer, and evaluator remain unchanged.

This paired construction solves the worker-prefetch boundary problem exactly, but it requires workers and the pinning path to produce and transport both views throughout training.

## Execution

Static checks and transform correctness passed. Clean tensors, target order, and post-crop RNG matched the parent across every batch of a shuffled 195-batch epoch; a second worker recreation replayed the candidate stream exactly. Eight workers used distinct private keys. Across 10,240 samples, 91.1914% changed pixels, mean absolute uint8 delta was 14.0395, and p99 was 185.

The preregistered loader feasibility gate then failed, so no GPU training or evaluation was launched and no `run.log` was created. Five parent epochs took 1.444-1.693 seconds; five candidate epochs took 2.684-3.210 seconds. Candidate throughput was 15,549.8-18,601.0 images/s versus parent 29,485.7-34,578.3.

Claude's result audit found the 30,720-image/s absolute headroom floor miscalibrated because the parent's worst epoch also missed it. The stop remains robust without that floor: candidate best throughput was below parent worst, every candidate epoch was below the roughly 25,600-image/s early GPU demand, and even a 25% discount to that demand leaves 19,200, still above candidate best. The planned full-runtime projection was not performed and is not claimed.

## Results

- **Primary metric**: unavailable (parent/global best: 95.40%; delta: N/A)
- **Correctness**: static checks, parent-clean parity, target/order parity, exception restoration, private-stream distinction, and deterministic replay passed
- **Pixel effect**: 10,240 samples, 91.1914% changed, mean absolute uint8 delta 14.0395, p99 185
- **Parent loader**: worst / median 29,485.7 / 32,969.6 images/s; p90 inter-arrival 33.386 ms
- **Candidate loader**: worst / median 15,549.8 / 17,326.5 images/s; p90 inter-arrival 78.819 ms
- **Package cost**: about 2.1x parent epoch time; candidate best remained below parent worst
- **Training/evaluation**: not launched; zero charged training seconds and no accuracy observed

The failure localizes to the complete paired CPU package, not to RandAugment's statistical effect. The preflight did not separate augmentation compute from second-view materialization, serialization, IPC, and pinning, so none can be declared the sole bottleneck. The exact phase-control mechanism is technically valid but consumes more CPU headroom than this pipeline has.

- **Key Learning**: Exact paired-view phase gating preserves parent semantics but costs about 2.1x loader time; this node provides no evidence about RandAugment accuracy.

## Verification

- **Conditions**: Scope and transform integrity passed; fixed-package loader feasibility failed; GPU integration, full-run integrity, and accuracy were skipped.
- **Review Notes**: The absolute 1.20x loader floor was flawed because the parent did not always pass it. Independent raw comparisons still justify the stop. No hard constraint was violated and no untrustworthy metric exists because no metric was produced.
- **Verdict**: crash
- **Verdict Basis**: The tree schema requires `crash`/`NaN` when no result is produced. This is a mechanical encoding of a preflight reject, not a code crash or evidence that RandAugment reduces accuracy.
- **Tree placement**: failed leaf on `br-000`, parent EXP-004, commit `9625222`; global best remains EXP-004 at 95.40%.

## Unexplored Avenues

- Apply augmentation on-device to the parent single-view batch, retaining the exact parent loader and gating on measured GPU step latency. This removes dual-view worker transport but needs a production-faithful batched operation implementation.
- Return only uint8 source data and construct both clean/augmented semantics on GPU. This can remove clean-FP32 IPC but would need a proof that crop/flip and normalization remain faithful.
- Measure transport-only and operation-only loader variants before another CPU augmentation proposal. That would attribute the 2.1x slowdown rather than assuming RandAugment itself caused it.

## Next Steps

- **High confidence**: Return to EXP-004 and test the already-developed identity-preserving ECA representation mechanism, which avoids the newly measured CPU bottleneck.
- **Medium confidence**: Develop a parent-relative feasibility protocol whose parent passes by construction and which reports dispersion, not one approximate absolute demand.
- **Low confidence**: Revisit early RandAugment only through a single-view GPU-native batch path; do not tune this paired CPU node.
