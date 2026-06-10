# Brainstorm EXP-063
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- (none consulted — no new external technique. This loop combines a documented project near-miss (aug cooldown, EXP-034) with the current best recipe (AugMix-p0.5, EXP-054), per the standing directive to combine previous near-misses on a deeply-mapped plateau. The cooldown / "FixRes"-style train-test distribution-matching mechanism is well-understood from the project's own EXP-033/034/035/061 history; no high-signal external source adds beyond it.)

## Experimental History Review
- **Current best: 96.45 (EXP-054)** = k=4 WideResNet-20 + `RandomApply([AugMix() w3], p=0.5)` + GPU Cutout16 + cosine peak0.2/warmup0.05/Nesterov/LS0.1/WD1e-4 + compile, 91 ep, dt 8ms, 593s wall.
- **63 experiments, 8 improvements. EVERY major lever is mapped** (project-insights High): augmentation strength/policy/coverage/delivery (CPU+GPU, EXP-012..060), capacity ×4 directions (EXP-004/009/038/044/058), optimizer family+gradient-dynamics+objective (EXP-031/036/041/043), LR peak+shape+**warmup (EXP-062, just closed)**, normalization-as-regularizer (GhostBN EXP-047), eval-BN-statistics (EXP-061), residual scaling (EXP-051), head (EXP-032/039), batch (EXP-025/050, 128 optimal), activation (EXP-010/028), regularizers (dropout/Mixup/CutMix/SE), weight-averaging (EMA/SWA), throughput→epochs (saturated ~91).
- **Aug cooldown — a documented near-miss NEVER combined with the current best**: EXP-033 (@0.15, 96.10), **EXP-034 (@0.10, 96.26 — +0.04 over its TA base 96.22, the best cooldown result)**, EXP-035 (@0.10 + LR-reheat, 96.12), EXP-049 (@0.10 + GC, 96.13). **ALL FOUR ran on the OLD TrivialAugment recipe (commit 6c417a4).** Cooldown has NEVER been applied to the EXP-054 AugMix-p0.5 best (commit 86161d9). This is the one un-combined near-miss.
- **EXP-061 insight (key to cooldown's mechanism)**: clean-BN recalibration ALONE hurt −1.6pp (desyncs BN running stats from the affine params). But cooldown re-adapts weights AND BN JOINTLY to the clean tail distribution — a DISTINCT mechanism, and the reason cooldown produced a real tail-climb (95.43→96.10 in EXP-033) while BN-recalib-alone collapsed.
- **Genuinely UNTESTED scalars** (all low-ceiling): gradient clipping (never tried), BN momentum/eps (never tuned).

## Candidate Ideas

### 1. Augmentation cooldown @0.10 on the AugMix-p0.5 best recipe
**Summary**: Port the EXP-034 best cooldown (disable train augmentation for the final 10% of the time budget, keeping only RandomCrop+Flip) onto the current EXP-054 AugMix-p0.5 best. Concretely: in the training loop, once `total_training_time/TIME_BUDGET_S > 0.90`, switch the dataloader's transform so AugMix (and GPU Cutout) are disabled and only crop+flip remain, letting the model + BN running stats jointly re-adapt to the near-clean distribution that `Eval.evaluate()` sees. Everything else byte-identical to EXP-054.

**Reasoning**: This is the single un-combined near-miss flagged across multiple reports. EXP-034 showed cooldown @0.10 produces a real clean-tail climb (pre-cooldown base ~96.05 → 96.26, +0.21 tail-climb) and was the best of the cooldown family. EXP-061 clarified the mechanism — JOINT weight+BN re-adaptation to the clean eval distribution (not BN-recalib-alone, which hurts). The AugMix recipe has a meaningful train→eval distribution gap (AugMix on 50% + Cutout on 100% of train images; eval is clean), so the joint-adaptation tail-climb should transfer. If it adds a similar +0.1–0.2 to the 96.45 base, it clears the 96.55 bar.

**Sources**: EXP-033/034/035 (cooldown family, TA recipe); EXP-049 (cooldown+GC); EXP-061 (joint vs BN-only adaptation); EXP-054 (current best recipe); train.py train loop (L221-285) + train_tf (L156-175).

**Estimated Effort**: Low (a tail-epoch transform switch on the dataloader; the cooldown-firing logic is the EXP-034 pattern).

**Risk Assessment**: Low. Wall-NEUTRAL-to-faster — cooldown REMOVES augmentation work in the tail (lighter CPU dataloader), so unlike EXP-061 there is NO wall-overrun risk on the wall-tight AugMix recipe. Failure mode is no-improvement: cooldown's TA lift was within-noise (+0.04 net), and on AugMix-p0.5 only 50% of images carry AugMix so the train→eval gap (hence the cooldown headroom) may be smaller → marginal/null likely. Edge case: switching the dataloader transform mid-run must not break the compiled forward (it doesn't — the transform is CPU-side in the dataloader, outside the compiled graph; the model forward is unchanged). Must re-init the DataLoader iterator cleanly when switching transforms.

### 2. Gradient-norm clipping at a permissive threshold
**Summary**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)` between `loss.backward()` and `optimizer.step()`, clipping only outlier gradient spikes. Single-line addition, threshold a constant.

**Reasoning**: AugMix distorts ~50% of images via multi-chain mixing; heavily-distorted batches can produce large-loss → large-gradient spikes during the long high-LR (0.2) plateau that perturb converging weights. A permissive clip tames only those spikes while leaving normal steps untouched — potentially smoother convergence to a marginally better minimum. Untested knob.

**Sources**: train.py L245-246 (backward/step); standard practice; EXP-016/017 (high-LR regime).

**Estimated Effort**: Trivial (one line; clip is on eager param grads, outside the compiled graph → cudagraph-safe per EXP-042).

**Risk Assessment**: Low, but low-evidence. This net already trains stably (no divergence in 63 runs), so there are likely no harmful spikes to clip → most probable outcome is an exact null (project-insights: optimizer/gradient-dynamics polish is closed). A too-low threshold would under-step and hurt; 2.0 is permissive. Near-noise ceiling.

### 3. BN momentum reduction (0.1 → 0.03) for smoother eval-time running stats
**Summary**: Lower `BatchNorm2d` momentum from the default 0.1 to 0.03 (longer EMA window for running mean/var), so eval-time running stats are a smoother average over more recent batches. Set at module construction. Single-variable.

**Reasoning**: With heavy AugMix augmentation, per-batch BN statistics are noisy; a longer EMA window could give running stats that better represent the steady-state distribution the eval forward uses. Untested axis.

**Sources**: train.py BasicBlock/ResNet BN construction (L71/75/83/103); EXP-061 (BN-stat operating point).

**Estimated Effort**: Trivial (momentum kwarg on BN constructors).

**Risk Assessment**: Low-to-moderate, low-evidence — and EXP-061 weakly argues AGAINST: it showed the augmented running stats ARE the correct trained-in operating point the affine params expect. Lowering momentum just smooths those same augmented stats; with a cosine-annealing LR the distribution shifts through training, and a longer EMA window lags that shift → could slightly hurt at eval. Near-noise, mild-regression-possible.

## Idea Evaluation
- **Evidence strength**: Idea 1 has by far the most concrete project evidence — a documented +0.21 tail-climb (EXP-034) and a clarified mechanism (EXP-061 joint adaptation), never combined with the current best. Ideas 2 and 3 are low-evidence untested scalars on an exhausted plateau; Idea 3 is weakly contraindicated by EXP-061.
- **Mechanism clarity**: Idea 1 clear (joint weight+BN clean re-adaptation in the tail closes the train→eval distribution gap); Idea 2 plausible-but-this-net-is-stable; Idea 3 plausible-but-EXP-061-suggests-the-augmented-stats-are-already-correct.
- **Expected impact**: all near-noise on a 63-experiment plateau. Idea 1 has the only documented positive precedent (+0.04 net on TA, +0.21 tail-climb) and a real distribution gap to close on AugMix.
- **Risk profile**: Idea 1 is wall-NEUTRAL-to-faster (removes tail work — strictly safer on the wall-tight AugMix recipe than EXP-061's eval-side overhead), throughput-neutral, no scope risk. Cleanest failure mode (no-improvement).
- **Feasibility**: Idea 1 low effort; 2 and 3 trivial.
- **Conclusion**: Lead with **Idea 1 (aug cooldown @0.10 on the AugMix recipe)** — the one un-combined documented near-miss, with the clearest mechanism and only positive precedent, and a wall-safe profile. Honest expectation: near-noise on a deeply-mapped plateau (cooldown's TA net-lift was within-noise, and the 50%-subset AugMix has a smaller train→eval gap than full-coverage TA), but it is the most defensible principled probe remaining. Ideas 2/3 are weaker fallbacks for later loops.

## Chosen Idea
**Selected**: Augmentation cooldown @0.10 on the AugMix-p0.5 best recipe.

**Why this idea**: It is the single documented near-miss (EXP-034, the best cooldown result) that was NEVER combined with the current best recipe (all cooldown experiments ran on the superseded TrivialAugment recipe). It has the clearest causal mechanism on the project's own evidence — joint weight+BN re-adaptation to the clean eval distribution in the tail (EXP-061 showed this joint adaptation is real and distinct from BN-recalib-alone, which hurts). It is wall-neutral-to-faster (cooldown REMOVES tail augmentation work, so it carries none of the EXP-061 wall-overrun risk on the wall-tight AugMix recipe), throughput-neutral, and a clean single-variable probe. On a plateau where every strength/policy/capacity/schedule lever is closed, combining the strongest near-miss with the best recipe is the most defensible move per the standing "combine near-misses" directive.

**Hypothesis**: Disabling AugMix+Cutout for the final 10% of the time budget (crop+flip only) will let the model and BN stats jointly re-adapt to the near-clean eval distribution, producing a tail-climb that raises best_test_acc to ≥ 96.55 (baseline 96.45 + 0.1pp). Given that cooldown's net lift on TA was within-noise (+0.04) and the AugMix-p0.5 recipe augments only 50% of images (smaller train→eval gap), the most likely outcome is a within-noise null, but the probe is wall-safe, throughput-neutral, and the best-evidenced untried combination.
