# Report EXP-017: Full-Run Eligible-Weight Gradient Centralization
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the fixed seed-42, 300-second charged training protocol while modifying only `train.py`. EXP017 grew from parent EXP002 at 95.23%, so local improvement required at least 95.33%. The goal-wide best before and after this experiment remains EXP011 at 95.61%.

## Idea & Hypothesis

Apply coefficient-free Gradient Centralization on all 16 convolution weights and the classifier weight throughout training. Coupled L2 is added before centralization, matching the official CIFAR `SGD_GC` ordering, so the regularized eligible direction entering Nesterov momentum is zero-row-mean. The hypothesis was that this low-forward-cost geometry change would improve stable generalization to at least 95.33% without materially reducing optimizer exposure.

## Approach

Only `train.py` changed. The implementation asserts an exact 17-tensor, 2,745,264-element, 2,266-row eligible inventory and reconciles 3,626 excluded BN-affine/bias elements. It materializes coupled L2 once across all 44 gradients, centralizes eligible directions with broadcast foreach subtraction, and runs PyTorch SGD with unchanged momentum/Nesterov behavior and internal decay disabled. Sparse charged FP64 energy/residual audits, path-dose counters, final state finiteness, and additive final-16 evaluation context were added without changing evaluation cadence or max selection.

Claude's plan review identified that the initial raw-gradient-only ordering differed from the official optimizer; this was corrected before code implementation. Deterministic CPU/GPU smokes confirmed the final ordering, excluded update parity, RNG neutrality, and FP32 gradients under BF16 autocast.

## Execution

Static scope, syntax, GPU identity, and deterministic smoke checks passed. The standalone smoke needed one import-path repair outside the decisive-preflight ledger. Its full-model decomposition error was `9.856645908319e-10` and maximum post-GC row-mean residual was `1.103789726287e-09`.

The first accuracy-blind preflight stopped after its 1,024-step candidate trace because the harness expected two audit samples instead of the correct three at steps 1, 512, and 1,024. The one permitted pre-vector repair corrected that assertion. The repaired preflight then stopped on its allocation-stability assertion because the harness retained all 1,024 detached CUDA loss scalars while comparing final allocation with a step-32 baseline. No paired timing vector or metric was produced. The repair allowance was exhausted, so no further repair and no metric launch were authorized.

Module import constructed inherited evaluator/test-loader objects, but guards replaced both evaluator globals before any trace. Neither test loader was iterated, no evaluator was called, and no accuracy was computed. Charged metric-training time was `0 s`.

## Results

- **Primary metric**: `NaN` (parent: 95.23; delta vs parent: N/A; global best: 95.61)
- **Observations**: The GC implementation passed deterministic mechanism checks, but the feasibility protocol failed twice before its numeric timing stage. The second failure measures retained diagnostic scalars, not evidence of candidate model/optimizer allocation growth.
- **Analysis**: There is no accuracy or paired-latency evidence for the hypothesis. EXP017 is a protocol/harness crash, not a negative Gradient Centralization result. A future experiment may test the same scientific idea only with a newly preregistered harness that reduces finiteness into a scalar and releases per-step tensors before allocation baselining.
- **Key Learning**: Retaining per-step CUDA diagnostics can invalidate allocation-stability gates and consume a preflight repair without testing the research mechanism.

## Verification

- **Conditions**: Parent/scope/syntax, GPU identity, and deterministic mechanism checks passed; the accuracy-blind paired preflight failed operationally after its one permitted repair; metric and primary-condition checks were skipped.
- **Review Notes**: Claude Opus independently returned `AUDIT_VERDICT: PASS` for `crash/NaN`. It confirmed zero test iteration/accuracy, corrected wording around loader construction, and judged GC untested (`04-result-review.md`).
- **Verdict**: crash
- **Verdict Basis**: No result was produced because the temporary preflight harness failed before a complete numeric vector and the sole repair allowance was exhausted. No hard constraint was violated, so `invalid` is not appropriate.

## Unexplored Avenues

- Retry the same reference-ordered GC mechanism in a future experiment with a clean preregistration and a corrected allocation harness that accumulates finite status into one device scalar rather than retaining losses.
- Measure allocation immediately after an audit step and again after deleting temporaries, separating allocator behavior from persistent model/optimizer state.
- If a corrected GC run passes locally, test composition on EXP011 only after treating one-seed gains as noise-limited evidence.

## Next Steps

- **High confidence**: keep reference-ordered GC eligible for future navigation because EXP017 provides no research result against it.
- **High confidence**: enforce scalar-only diagnostic accumulation in every future GPU memory-stability preflight.
- **Medium confidence**: explore another one-backward optimizer or representation intervention from EXP002 or the frontier while preserving the validated CutMix/SAM/EMA mechanisms.
