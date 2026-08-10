# Report EXP-018: Direct Canonical Lookahead on EXP002
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the frozen 300-second charged training budget, with higher being better and improvements coming only from genuine `train.py` changes. EXP018 grew from parent EXP002 at 95.23%, so its formal threshold was 95.33%. The goal-wide best before and after this experiment remains EXP011 at 95.61%.

## Idea & Hypothesis

The chosen idea wrapped EXP002's unchanged Nesterov SGD trajectory with canonical parameter-only Lookahead at fixed `k=5`, `alpha=0.5`. One FP32 slow copy interpolated toward fast parameters every fifth update and fed that state back into training while retaining momentum and excluding BatchNorm buffers. It was selected because SAM and EMA had already shown same-lineage value from optimizer geometry and trajectory control, while Lookahead added no extra forward and had canonical CIFAR evidence. The hypothesis predicted a valid `best_test_acc >=95.33%`, with 95.53% or higher providing stronger composability evidence.

## Approach

Only `train.py` changed. All 44 trainable parameters received detached FP32 slow state; after each inherited optimizer step, completed steps divisible by five used fused foreach interpolation and copy-back under `no_grad`. The operation, sparse cadence-128 distance audits, and path counters were charged inside the existing step timer. Momentum, weight decay, LR, architecture, initialization, CutMix/drop-path streams, BF16/channels-last execution, and evaluator/max-selection semantics stayed inherited.

Evaluation always used slow parameters. Full 195-step epochs naturally ended synchronized; the truncated final epoch used a preallocated fast snapshot, installed slow parameters for the sole evaluator call, and restored fast parameters/module flags exactly in `finally`. Runtime assertions reconciled first/last sync, phase counts, synchronized/swapped evaluations, finiteness, state ownership, and restoration. The adversarial plan review prompted two useful hardenings: define cadence from `completed_step=step+1`, and swap at any unsynchronized evaluation rather than assuming only the final boundary.

## Execution

Deterministic tiny/full-model smokes passed exact step-5 reference behavior, parent equality through step 4, retained momentum, RNG-neutral cloning/swaps, BF16/channels-last gradients, BN-buffer exclusion, and the 44-tensor/2,748,890-element inventory. The first complete accuracy-blind GPU-0 preflight passed without repair: median overhead was 1.005970x, worst round 1.016019x, MAD/median 0.009195, projected dose 27,784 steps / 143 epochs, allocation stayed exactly 125,040,640 bytes across the long trace, and evaluator calls were zero.

Exactly one metric run was launched on physical GPU 0. It exited 0 after 300.0 charged and 450.1 total seconds, completed 28,341 steps across 146 epochs, and emitted the complete inherited summary plus Lookahead diagnostics. There were no code, infrastructure, CUDA/OOM, assertion, NaN, or Inf failures and no retry or metric-driven adjustment.

## Results

- **Primary metric**: 94.82% (parent: 95.23%, delta vs parent: -0.41 points, -0.43%; global best: 95.61%, delta: -0.79 points)
- **Observations**: Lookahead was active and exact: 5,668/5,668 syncs from step 5 through 28,340, split 2,030 early-CutMix, 2,154 early-clean, and 1,484 late-clean. The 45 sparse audits measured normalized pre-sync slow-fast distance of 1.327% mean and 2.369% maximum, so feedback did not collapse into an identity. All 145 synchronized evaluations and the one swapped final evaluation reconciled with zero restore/nonfinite failures. The final-16 plateau was 94.7219% mean, 94.61-94.82% range, and 94.69% final; final CE was 0.2246 versus EXP002's 0.2044. EXP018 also completed 391 more steps than EXP002 (28,341 vs 27,950), excluding lost exposure as the cause.
- **Analysis**: The hypothesis is rejected for this fixed composition. Canonical frequent feedback materially changed the path at negligible compute cost but produced a uniformly lower late solution in both accuracy and CE. The stable 0.41-point best deficit and roughly 0.47-point parent-best versus EXP018-tail-mean gap exceed the observed ~0.17-point selection band. The most plausible mechanism is over-damping or mismatch between retained Nesterov momentum and parameters pulled halfway back every five updates: the intervention preserved throughput yet constrained useful progress. This discredits direct full-run canonical Lookahead on EXP002, not all parameter averaging; sparse evaluation-only EMA and late SAM remain empirically distinct successful mechanisms.
- **Key Learning**: Canonical `k=5`, `alpha=0.5` Lookahead over-damped EXP002, lowering a stable tail despite higher step exposure and exact low-overhead execution.

## Verification

- **Conditions**: Execution integrity passed, but the primary metric condition failed: 94.82% was below the 95.33% parent-relative threshold.
- **Review Notes**: Results are trustworthy. Claude independently returned `AUDIT_VERDICT: PASS`, rechecked freshness and scope, counted 146 evaluations, recomputed all cadence/phase/audit/tail identities, and found no reward-hacking or stale-output path (`04-result-review.md`).
- **Verdict**: no-improvement
- **Verdict Basis**: The run was valid and constraint-compliant, but missed the necessary threshold by 0.51 points and underperformed its parent by 0.41 points. `tree.sh insert` recorded EXP018 as a terminal failed leaf on `br-000`; the global best remained 95.61% at EXP011.

## Unexplored Avenues

- A clean-tail-only or much lower-frequency feedback schedule could avoid suppressing early representation learning while testing whether Lookahead helps near the low-LR endpoint. This would be a new preregistered mechanism, not a retry of canonical full-run settings, and currently has only low-confidence support.
- A momentum-aware synchronization variant that interpolates or resets the fast momentum buffer could remove the observed parameter/momentum mismatch, but it would no longer be canonical Lookahead and needs independent evidence before consuming a run.
- Lookahead might interact differently with the SAM/EMA tip, but nested smoothing risks duplicating the successful EMA mechanism; composition is not justified by this negative direct result without a sharper causal proposal.

## Next Steps

- **High confidence**: Return to EXP002 for the clean reference-ordered Gradient Centralization retry; EXP017 established implementation correctness but produced no metric, and EXP018 validated the corrected fixed-scalar harness pattern.
- **Medium confidence**: Prefer an orthogonal low-overhead representation or calibration intervention over another trajectory smoother, since direct Lookahead harmed both the stable tail and final CE.
- **Medium confidence**: Revisit the EXP011 global tip only with a mechanism expected to lift its 95.49% stable EMA plateau rather than adding another averaging layer.

## Exit Action Results

No exit actions were defined for this goal.
