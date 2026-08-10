# Claude Adversarial Plan Review: EXP-008

The first read-only Claude Opus attempt timed out after 600 seconds without output. Per the user requirement, no fallback reviewer was used. A second Claude Opus attempt reviewed the full executable design embedded in the prompt and returned the following concerns.

## Critical

1. **Torchvision magnitude 2 is near-identity on 32x32 images.** In torchvision 0.24.1, translations quantize from 0.967 pixels to zero and posterization remains eight bits; Identity is also in the operation set. The paper's scalar used a different operation mapping, so calling this a faithful transfer is incorrect. Resolve and audit the actual operation table, and require a nondegenerate pixel-change rate before launch.
2. **Use one progress scalar per step.** Compute charged progress exactly once after batch yield and use that same value for input selection, CutMix, and SAM. Otherwise a boundary crossing between reads could expose augmented inputs to SAM.
3. **Paired float tensors double CPU normalization, shared-memory IPC, and pinning.** Return the augmented view as uint8 and normalize it on GPU only when selected; keep the clean float tensor exact for parent parity. Gate on worst-case loader headroom, not a mean step projection.

## High

4. **Lazy private-generator initialization must be keyed by worker seed.** A generator created in the parent process could otherwise be inherited by every forked worker. Reinitialize whenever the current worker seed differs; test distinct worker keys and epoch recreation.
5. **Time-based schedules make realized step/SAM dose part of the package.** Audit candidate and parent steps, epochs, CutMix, and SAM counts. A large exposure difference limits causal attribution but is not a protocol-integrity failure.
6. **The frozen max-over-epoch metric has selection noise.** Keep it as primary, while reporting final accuracy, last-five mean, and evaluation count. The protocol cannot be changed mid-tree.
7. **Preflight and full-run timeouts must be separate.** Use a conservative worst-case total projection rather than only 20 seconds of margin.
8. **Low step count is an intervention-cost outcome, not invalid protocol by itself.** Completion, scope, timer, evaluation, and audit invariants determine validity; step exposure determines whether the mechanism hypothesis was adequately tested.

## Medium And Low

9. Define an explicit `num_workers=0` seed path, but require production-path checks with eight workers.
10. On a RandAugment exception, capture the advanced private RNG state in `finally` before restoring global state; test this chosen replay semantics.
11. Add direct aggregate-only audits for augmented/SAM overlap, cutoff ordering, pixel deltas, worker stream distinction, and exposure. Avoid per-step I/O.
12. Parent/candidate parity must restore any explicit loader generator as well as global RNG. The current parent has no explicit generator, but the test should assert that assumption.
13. Default collation retains both views through the loop iteration; promise only that the unselected view is never transferred to GPU, not immediate CPU release.

Claude judged the paired-view main-process selection and global-RNG swap fundamentally sound once these corrections are made. It also required the final analysis to disclose that RandAugment and CutMix compose throughout the same regularized phase.
