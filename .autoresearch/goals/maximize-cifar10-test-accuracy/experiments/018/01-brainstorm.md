# Brainstorm EXP-018
**Created**: 2026-07-26

## Web Search & Literature Review

- **RandAugment** (`knowledge/papers/randaugment.md`): a one-operation, low-magnitude policy can add CIFAR invariances without a new dependency, but CPU cost and excessive strength remain fixed-budget risks.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): preserve the accepted moderate-depth wide backbone; local experiments show that deleting early transforms or compressing late transforms loses accuracy.
- **EXP-017 gate diagnostics** (`experiments/017/04-analysis.md`): both stage-3 gates became feature-driven, but the final gate had 0.02431 across-example variance versus 0.00312 for the first and the two-gate treatment peaked at 94.16%.

No network source was consulted; this local-only run uses the persistent offline knowledge base and completed experiment artifacts.

## Experimental History Review

- The accepted `[2,2,2]` WRN with early batch-shared mixup remains at 94.07%. Stronger/altered regularization, schedules, precision, averaging, initialization, coefficient diversity, and early-to-late block exchange all regressed.
- Late dense width, depth, and two-gate attention are the only repeated positive neighborhood: EXP-010/011/017 scored 94.11/94.15/94.16, but each missed the 94.17 threshold after reducing exposure to about 132-134 passes.
- EXP-017 isolates the current quality/cost gap. Feature-driven late selection improved final loss from 0.2432 to 0.2321, yet two gates imposed 4.6% step overhead. The final gate carried nearly 7.8x the first gate's across-example variance, while the first mostly learned near-static attenuation.
- The limiting error mode is therefore not raw fit or update count alone: the accepted model needs more useful late representation/generalization without sacrificing the early blocks or enough fixed-time exposure. Final-block-only attention, a kernel-cheaper conditional selector, and mild input invariance remain materially distinct gaps.

## Collected Ideas

## Combinations

## Candidate Ideas

### One-Operation Mild RandAugment
**Summary**: Add `RandAugment(num_ops=1, magnitude=5)` after crop/flip and before tensor conversion for all training samples, keeping accepted mixup, model, optimizer, and schedule unchanged.

**What it targets**: Image invariances absent from crop/flip, an orthogonal generalization gap that does not consume GPU model compute.

**Reasoning**: The persistent knowledge base records direct CIFAR evidence and torchvision support. One mild operation limits the risk relative to a full strong policy, and DataLoader workers can overlap much of its CPU work.

**Sources**: `knowledge/papers/randaugment.md`; `knowledge/papers/time-matters-regularization.md`; EXP-002/003/005/006/015 in `04-results.tsv`.

**Estimated Effort**: low

**Risk Assessment**: Always-on augmentation conflicts with the validated clean hard-label tail and the history of additive regularization failures. Worker-side randomness complicates semantic oracles, and CPU transforms may reduce feed rate.

### Final-Block Efficient Channel Gate
**Summary**: Place an exact-neutral lightweight channel gate only on `layer3[1]`: global-average-pool the residual, apply a zero-initialized short 1D convolution across channels, and scale with `2*sigmoid`. This removes the ratio-16 MLP and tests whether local cross-channel conditioning is sufficient.

**What it targets**: The same late feature-selection signal as EXP-017 with much less arithmetic and parameter traffic, aiming to recover almost all accepted exposure.

**Reasoning**: Feature-driven logits mattered more than learned bias in EXP-017, so an input-conditioned selector remains justified. A short channel convolution uses only a few parameters and should be cheaper than two 128-to-8-to-128 MLP gates. Exact-neutral initialization protects accepted logits at step zero.

**Sources**: `experiments/017/04-analysis.md`; accepted `train.py`; H20 shape-sensitivity insight in `.autoresearch/project-notes/project-insights.md`.

**Estimated Effort**: medium

**Risk Assessment**: The MLP's global channel mixing may be essential; local channel adjacency has no semantic ordering guarantee. A tiny GPU kernel can be latency-bound despite low FLOPs, and this mechanism has no direct local score.

### Final-Block-Only Neutral SE
**Summary**: Attach the exact identity-initialized ratio-16 SE treatment from EXP-017 only to `layer3[1]`, after its second convolution and before shortcut addition. Preserve accepted initialization/RNG and deterministically initialize the gate from the project's fixed seed 42, never varied. Verify neutrality and two-step opening in an evaluator-free preflight, with no scored diagnostic instrumentation.

**What it targets**: The measured fixed-budget mismatch in EXP-017: retain the gate with strong per-example conditional behavior while eliminating the weakly conditional first gate and roughly half its attention overhead.

**Reasoning**: EXP-017 reached 94.16%, improved final loss, and showed gate 1 across-example variance of 0.02431 versus 0.00312 for gate 0. This is a controlled, materially different placement hypothesis supported by post-run diagnostics rather than a ratio/seed rerun. Halving gate work should recover exposure while preserving the most clearly input-dependent selector.

**Sources**: `experiments/017/04-analysis.md`; EXP-010/011/012/016 in `04-results.tsv`; `03-experiment-learnings.md`.

**Estimated Effort**: medium

**Risk Assessment**: Gate 0's mostly static attenuation may still have contributed to the loss improvement, so removing it can erase the gain. Kernel-launch overhead may not halve cleanly. The diagnostic association is observational, not proof that gate 1 caused the accuracy signal.

## Review

The blind review selected **Final-Block-Only Neutral SE** at 4.5/5 evidence and 4/5 impact. I adopt its warning that gate 0's mean 0.6468 attenuation may have contributed even though it was weakly conditional; removal is the experiment's causal risk, not a free optimization. I also adopt stripping runtime diagnostics from scored code: EXP-017 already established gate behavior, while preflight identity/two-step-gradient oracles can verify implementation without consuming the fixed budget. The review rejected the efficient gate as a placement/mechanism confound with dubious tiny-kernel latency, and found always-on RandAugment inconsistent with the validated clean tail. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Final-block-only SE is the only candidate with a direct 94.16% precursor and a measured reason to remove one component. It preserves the proven selector and makes one treatment change, whereas the efficient gate changes mechanism too and RandAugment enters a locally unfavorable regularization regime.

## Chosen Idea
**Selected**: Final-Block-Only Neutral SE

**Why this idea**:
Retain only EXP-017's more example-dependent final-stage gate and recover the runtime spent on the first gate and observational diagnostics. This directly tests whether the final gate carried enough of the +0.09 signal to cross the margin at higher fixed-time exposure, while explicitly accepting that the removed first gate's static attenuation may have mattered.

**Hypothesis**:
One exactly neutral ratio-16 SE gate on `layer3[1]`, with no scored diagnostics, will retain at least 97% matched throughput, realize at least 137 dataset-equivalent passes, and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17% by preserving the strongest conditional late selector while recovering exposure.
