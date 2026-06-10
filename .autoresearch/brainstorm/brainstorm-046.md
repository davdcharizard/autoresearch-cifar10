# Brainstorm EXP-046
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **PyTorch `torch.compile` (compile-warmup pattern)** (pytorch.org docs): compilation is lazy (first forward); a warmup call before the timed region moves the one-time compile cost off a per-step-timed budget. Validated in EXP-045 (startup_seconds 2→79, ~14s reclaimed from the 300s budget → epochs 91→96). `mode="reduce-overhead"` reaches the conv dt floor (EXP-040) and — unlike max-autotune — uses the SAME conv kernels as the project baseline, so it adds epochs without changing numerics.
- **project-insights.md / goal-learnings.md** (local): EXP-045 — adding ~5 net epochs via warmup + max-autotune did NOT raise top-1 (95.71, −0.51pp), BUT max-autotune's faster Triton kernels appear to cost ~0.3-0.5pp vs the reduce-overhead baseline (mirrors cudnn.benchmark EXP-040). The epoch effect and the kernel-numerics penalty are entangled → the saturation question is not yet cleanly answered.
- **Ghost BatchNorm** (Hoffer et al., NeurIPS 2017): BN stats over small sub-batches inject noise → implicit regularizer; benefit strongest at large batch. **Stochastic depth / drop-path** (Huang et al., ECCV 2016): randomly drop residual blocks during training (linear survival schedule) — a structured regularizer whose benefit is depth-driven and which also reduces train compute.

## Experimental History Review

- **Current best / baseline**: 96.22% (EXP-012, 6c417a4), k=4 ResNet-20, ~91 ep @ 8ms. **35 consecutive no-improvements**.
- **ALL accuracy axes closed**: capacity (all 3 directions — width EXP-004/009, FLOP-realloc EXP-038, depth EXP-044), augmentation, LR schedule, regularizer-adds, classifier head, intermediate-feature-routing, activations, weight-averaging, optimizer family+grad/objective, bag-of-tricks, large-batch, cheap-throughput (EXP-040), and now **throughput→buy-epochs (EXP-045, max-autotune+warmup, 96 ep → 95.71)**.
- **The one not-cleanly-resolved question**: is the TrivialAugment recipe convergence-bound or epoch-saturated at ~91 ep? EXP-045 added epochs (91→96) but FELL to 95.71 — however that run swapped in max-autotune kernels that themselves cost ~0.3-0.5pp (per the EXP-040 parallel). So we cannot tell whether the ~5 extra epochs were neutral (saturated) or actually helpful but masked by the kernel penalty.
- **Untried/under-tested gaps**: (a) the off-budget compile-warmup with the BASELINE reduce-overhead kernels (clean +5-epoch test — never run); (b) normalization axis (Ghost BN — completely untouched); (c) stochastic depth (untouched). Note: depth-driven regularizers (dropout EXP-022, deep-supervision EXP-042, zero-γ EXP-026) have all been null/regressive on this shallow 9-block net.

## Candidate Ideas

### 1. Clean +5-epoch test: off-budget compile-warmup, reduce-overhead kernels (de-confound EXP-045)
**Summary**: Apply ONLY the seed-safe compile-warmup from EXP-045 (one fwd+bwd on a zeros batch before the timed loop, `zero_grad`, reset BN buffers, no optimizer step) while KEEPING `mode="reduce-overhead"` (the project baseline's exact conv kernels). This reclaims the ~14s one-time compile cost from the per-step-timed budget → ~+5 net epochs (91→~96) at byte-identical numerics and recipe. It isolates the pure epoch effect that EXP-045 entangled with max-autotune's kernel-numerics penalty.
**Reasoning**: EXP-045 left a real ambiguity: epochs rose to 96 but accuracy fell to 95.71, and that run also changed conv kernels (which independently cost ~0.3-0.5pp per the EXP-040 parallel). With baseline kernels, the ONLY change is +5 epochs. Two clean outcomes: (a) lands ≥96.32 → the TA recipe was convergence-bound and the warmup is a free recipe-preserving win (IMPROVEMENT); (b) lands ~96.0-96.2 → epoch-saturated, confirmed kernel-independently, fully closing the last open question. There is a genuine (if modest) improvement path because strong augmentation (TrivialAugment, added post-EXP-007's saturation finding) rewards longer schedules.
**Sources**: EXP-045 report/log (warmup mechanism validated, startup 2→79s); goal-learnings EXP-007/EXP-045 saturation entry; project-insights EXP-040 kernel-numerics finding.
**Estimated Effort**: low — the EXP-045 warmup block (already debugged) minus the mode change; ~10 lines, recipe untouched.
**Risk Assessment**: Lowest risk of any candidate — baseline kernels (no numerics change), reduce-overhead compile is ~14s (no wall-clock risk), warmup is seed-safe (proven in EXP-045 Run 2). Worst case: a clean no-improvement that definitively closes the saturation question. Main limitation: +5 epochs is small, so even a convergence-bound net might gain < the +0.1 bar (the gain could be within the ±0.25pp jitter band) → possibly inconclusive-but-leaning-saturated.

### 2. Ghost BatchNorm — implicit regularization via small-sub-batch BN statistics
**Summary**: Replace `nn.BatchNorm2d` with a Ghost-BN computing batch stats over sub-groups (e.g. ghost 32 of 128); eval unchanged (population running stats). Noisier stats act as an implicit regularizer.
**Reasoning**: Normalization is the one completely-untouched axis in 46 experiments; a genuine generalization lever distinct from the closed penalty-style regularizers.
**Sources**: Hoffer et al. 2017; contrast EXP-022 dropout.
**Estimated Effort**: medium — custom GhostBN over 18 BN layers + reshape logic.
**Risk Assessment**: Ghost BN's benefit is documented for LARGE batch (>256); at batch 128 ghost-32 adds modest noise, uncertain payoff. HIGH torch.compile dt-confound risk: per-block reshapes can break the reduce-overhead CUDA graph → dt doubles (EXP-042 gotcha) → epoch wall → uninterpretable regression. Likely confounded.

### 3. Stochastic depth (drop-path) with linear survival schedule
**Summary**: Randomly drop each residual block during training with linearly-decaying survival probability (1.0 → ~0.8 by the last block), scaling at test time; dropped blocks skip compute.
**Reasoning**: A structured regularizer that also frees compute (could partly self-fund its own epochs). Untouched here.
**Sources**: Huang et al. 2016 (stochastic depth).
**Estimated Effort**: medium — per-block Bernoulli gating in the forward.
**Risk Assessment**: Two strong negatives: (a) stochastic depth's benefit is DEPTH-driven (helps 50-1000-layer nets); on this shallow 9-block net depth-driven tricks have been null/regressive (dropout EXP-022 −1.37pp, deep-supervision EXP-042, zero-γ EXP-026). (b) Data-dependent block dropping (`if random`) breaks reduce-overhead CUDA graphs → dt doubles (EXP-042 gotcha). High chance of a confounded regression.

## Idea Evaluation

The strategic reality: 35 no-improvements, every accuracy axis closed, and EXP-045 left exactly one loose thread — the saturation question is confounded by max-autotune's kernel penalty. The highest-value move is the one that (a) cleanly resolves that thread and (b) still carries a real improvement path, at low risk.

- **Evidence strength**: #1 builds directly on a just-validated mechanism (the EXP-045 warmup, debugged and working) and a concrete confound to remove (max-autotune kernels). #2/#3 rest on literature for regimes that don't match this net (large-batch / deep-net) and that the project has repeatedly found don't transfer to the shallow 32×32 net.
- **Mechanism clarity**: #1 is crystal — single variable (+5 epochs at baseline kernels). #2/#3 have murky payoff and known dt-confound failure modes on this compile-sensitive net.
- **Expected impact / risk**: #1 fails gracefully (clean no-improvement closes the question) and has a genuine small improvement path; near-zero dt risk. #2 and #3 both carry the recurring CUDA-graph-break dt-confound (EXP-042) that has produced uninterpretable regressions before, plus weak priors on this net.
- **Feasibility**: #1 is the lowest-effort (reuse the debugged warmup, drop the mode change).

#1 wins: it is the principled, low-risk completion of the EXP-045 investigation, the only candidate with a clean single-variable test, and it still offers a real (if modest) chance to clear the bar. The radical alternatives (#2/#3) are deprioritized because their mechanisms are depth/large-batch-specific (repeatedly null here) and both risk a dt-confound that would waste the loop.

## Chosen Idea
**Selected**: Clean +5-epoch test — off-budget compile-warmup with reduce-overhead kernels

**Why this idea**:
EXP-045 demonstrated the warmup mechanism works (91→96 epochs) but entangled it with max-autotune's ~0.3-0.5pp kernel-numerics penalty, leaving the saturation question unresolved. Keeping the baseline reduce-overhead kernels and applying only the warmup isolates the pure +5-epoch effect at byte-identical numerics — the cleanest possible single-variable test. It is the lowest-risk experiment available (baseline kernels, ~14s compile, seed-safe proven warmup) and uniquely both (a) carries a real improvement path (if the strongly-augmented TA recipe is convergence-bound, the free epochs could clear 96.32) and (b) definitively closes the last open strategic question if null. The radical alternatives (Ghost BN, stochastic depth) target depth/large-batch regimes that have repeatedly failed to transfer to this shallow net and carry the recurring CUDA-graph dt-confound.

**Hypothesis**:
The seed-safe compile-warmup with `mode="reduce-overhead"` will raise realized epochs from ~91 to ~96 at byte-identical kernels/recipe. IF the TrivialAugment recipe is convergence-bound at 91 epochs, best_test_acc rises ≥0.1pp over 96.22 (≥96.32). The more likely outcome — given EXP-045 (96 ep → 95.71) and EXP-040 (94 ep → 95.91) — is that it lands within ±0.25pp of 96.22 (no improvement), which, combined with baseline-kernel numerics, would CLOSE the saturation question kernel-independently: the net is epoch-saturated at ~91 and the EXP-045 regression was the max-autotune kernel penalty. Either way the 96.22 k=4/300s ceiling is fully mapped.
