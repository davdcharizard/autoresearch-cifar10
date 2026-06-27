# Brainstorm EXP-009
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search this loop — the decision rests on in-project measurements plus already-distilled sources:
- **mixup (arXiv 1710.09412)** (standard knowledge; repeatedly promoted in exp-report-006/007/008 § Next Steps): vicinal risk minimization by convex input/label blending. Published CIFAR-10: WRN-28-10 error 3.8% → 2.7% (α=1.0, 200 epochs); gains hold across architectures. Known to need longer schedules to pay off and to interact with label smoothing (modern timm recipes nevertheless combine them).
- **knowledge/papers/trivialaugment.md**: TA composes with occlusion erasing; depressed mid-schedule accuracy is normal under strong augmentation — the same caution applies when adding mixup (interpret mid-run evals carefully).
- **goal-learnings § Failed Approaches (High count 3 + Low)**: capacity closed bidirectionally — wider starves epochs, shallower-wider converges lower. The remaining axes are per-epoch signal (regularization/augmentation), optimization quality (EMA), and throughput.

## Experimental History Review

- **Trajectory**: 91.97 → 93.16 → 95.23 (4x) → 94.41 (8x fail) → 96.06 (RE) → 96.23 (TA) → 95.12 (5x fail) → 96.71 (compile) → 96.00 (6x fail) → 95.76 (ResNet-14 6x fail). Baseline: **96.71 @ 1990397** (4x ResNet-20, compiled TA+RE+LS recipe, 139 epochs, dt 22ms).
- **Patterns**: time-keyed one-cycle, compile, RE (+0.83), TA (+0.17) validated and kept; EXP-008's dt-gate protocol (Medium) is now standard for uncertain-epoch runs; plateau-vs-climbing at cutoff distinguishes converged from starved.
- **Failed Approaches**: capacity is closed in BOTH directions (High count 3: starvation; Low: ResNet-14 6x convergence ceiling). ResNet-20 4x @ 139 epochs is the topology optimum.
- **Where the bottleneck sits now**: EXP-006 showed the 139-epoch run still gained from more epochs (final 96.65 vs best 96.71, still slightly climbing) — but throughput levers worth >1.05x are exhausted. The most under-exploited resource is per-epoch signal: the augmentation stack last moved in EXP-004 and its diminishing trend (0.83 → 0.17) was measured at 114 epochs PRE-compile; the schedule is now 22% longer, which shifts the regularization optimum upward.
- **Untried gaps**: mixup/CutMix (input-label-space regularization — a class distinct from the spatial/occlusion transforms tried so far), EMA weights for eval, compile max-autotune, non-uniform stage widths (foreclosed-adjacent, low expectation).

## Candidate Ideas

### 1. Mixup (α = 0.2) on the 139-epoch compiled 4x recipe
**Summary**: Per-batch mixup applied on the GPU batch before the compiled forward: sample `lam ~ Beta(0.2, 0.2)` once per step, mix `x = lam*x + (1-lam)*x[perm]`, and compute `loss = lam*CE(out, y) + (1-lam)*CE(out, y[perm])` (each CE keeping label_smoothing=0.1). TA/RE/LS and all hyperparameters unchanged.

**Reasoning**: This is the strongest remaining lever by published evidence. It is NOT a retry of the diminishing-returns augmentation stacking: RE and TA are spatial/policy transforms on single images, while mixup regularizes in input-label space across images — a different mechanism (vicinal risk minimization, linear-behavior prior) with documented +0.5–1pp on CIFAR WRNs. The two conditions that made TA's gain small (+0.17) have both improved: the schedule is 139 epochs (vs 114), and capacity-side alternatives are now provably closed, meaning the train/test gap is the only remaining source of headroom. α=0.2 (timm-style mild setting) rather than the paper's α=1.0 hedges the over-regularization risk on a 4.29M-param net carrying three co-regularizers. Implementation is ~8 lines in the training step, all outside the compiled graph (mixing is elementwise on the batch; the model call is unchanged — no recompilation).

**Sources**: arXiv 1710.09412 §3.1 + Table 1 (CIFAR); reports/exp-report-008.md § Next Steps (lead, medium); reports/exp-report-004.md (diminishing stacking caveat); goal-learnings § Patterns (compile/TA/RE recipes to keep).

**Estimated Effort**: low-medium (~8-line loss-path change; main risk is a subtle bug, mitigated by checking epoch-1 health and loss scale)

**Risk Assessment**: Moderate probability the gain lands under the +0.1pp bar at this capacity (over-regularization with LS+TA+RE). Failure is clean (no-improvement) but causally ambiguous (too much regularization vs too-short schedule). Zero throughput cost (mixing is ~0.1ms on GPU); epoch count unchanged at ~139; no VRAM impact. Train-loss EMA becomes incomparable to prior runs (mixed targets) — expected, not a bug.

### 2. EMA weight averaging for evaluation
**Summary**: Keep an exponential moving average of all parameters (decay 0.995, updated each step via `torch.lerp_` on a parallel copy), evaluate the EMA copy once per epoch via the existing eager `base_model` pathway, and report its accuracy.

**Reasoning**: Optimization-quality lever, orthogonal to augmentation. But our one-cycle LR anneals to ~0, which already performs implicit late-weight averaging — published EMA gains shrink to ~0–0.2pp under cosine-to-zero schedules. The per-step EMA update costs ~0.5–1ms inside the timed budget (~3–6 epochs lost), partially offsetting gains. Best-epoch tracking also already harvests some of the same variance.

**Sources**: standard practice (PyTorch ImageNet recipes, timm); no in-project datapoint.

**Estimated Effort**: low-medium (parallel weight copy + per-step update + eval switch; must keep the EMA copy out of the compiled graph)

**Risk Assessment**: Expected gain likely below the +0.1pp bar; costs real epochs to find out. Clean failure. Slight integrity care: the evaluated object must still be the plain eager model weights (EMA of them) passed through the frozen Eval — compliant.

### 3. compile mode="max-autotune"
**Summary**: `torch.compile(model, mode="max-autotune")` — deeper kernel search for a few % more img/s at 1–3 min extra startup.

**Reasoning**: Pure throughput on a lever already banked. At 139 epochs, +5% img/s ≈ +7 epochs ≈ well under +0.1pp by the measured epochs→accuracy slope near convergence (EXP-006: +25 epochs ≈ +0.48pp at a steeper point of the curve; near 139 the curve is flatter). Startup risk: autotune compile could take minutes, eating the 600s cap margin (489s total at EXP-006 + extra startup is tight if epochs increase too).

**Sources**: PyTorch compile docs; exp-report-006/008 § Unexplored Avenues.

**Estimated Effort**: trivial

**Risk Assessment**: Standalone gain almost certainly sub-bar; wall-clock margin shrinks. Low information value.

## Idea Evaluation

**Evidence strength**: Idea 1 dominates — the only candidate with published ≥0.5pp gains in a directly comparable setting (CIFAR-10 WRN, heavy aug), and the in-project counterargument (diminishing stacking) was measured under conditions that have since improved (114→139 epochs) and applies to a different regularizer class. Idea 2's external evidence is strong for constant-LR/step schedules but weak precisely for our annealed schedule. Idea 3's expected effect is below measurement resolution.

**Mechanism clarity**: Idea 1 — reduces the train/test gap via input-label interpolation, the one headroom source left now that capacity is closed; the 139-epoch schedule absorbs the slower convergence mixup induces. Idea 2 — variance reduction largely duplicated by the annealed schedule + best-epoch tracking. Idea 3 — marginal epochs on a flat region of the epochs-accuracy curve.

**Expected impact**: Idea 1: +0.1–0.5pp (published gain attenuated by LS/TA/RE overlap and 0.7x schedule length); downside −0.3pp. Idea 2: 0–0.2pp minus ~4 epochs. Idea 3: <0.1pp.

**Risk profile**: all fail cleanly to no-improvement; Idea 1's bug risk is mitigated by its tiny diff surface and an epoch-1 health check; its failure would close the regularization axis at this capacity (informative, since regularization is the last big axis).

**Feasibility**: Idea 3 trivial, Ideas 1–2 small loss/eval-path changes. Idea 1's evidence-impact profile dominates regardless.

## Chosen Idea
**Selected**: Idea 1 — Mixup (α = 0.2) on the 139-epoch compiled 4x recipe

**Why this idea**:
Capacity is provably closed, making the train/test gap the only remaining headroom; mixup is the strongest published lever against that gap not yet in the recipe, acts in a mechanism class (input-label interpolation) distinct from the spatial transforms whose stacking showed diminishing returns, and costs zero throughput. The mild α=0.2 hedges the over-regularization risk of a fourth regularizer.

**Hypothesis**:
Adding per-batch mixup with α=0.2 (Beta-sampled lam, mixed inputs, lam-weighted label-smoothed CE) to the compiled 4x TA+RE recipe will raise best_test_acc from 96.71% to ≥96.81%, because input-label-space regularization attacks the residual train/test gap that capacity changes can no longer address, and the 139-epoch schedule is long enough to absorb mixup's slower convergence; predicted execution signatures are unchanged dt (~22ms) and epochs (~139), a HIGHER final_test_loss-comparable train loss trace (mixed targets), epoch-1 eval ≥ 15%, and final-epochs evals within noise of best (converged).
