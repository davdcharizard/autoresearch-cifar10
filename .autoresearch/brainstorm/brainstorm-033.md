# Brainstorm EXP-033
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **YOLOX (Ge et al. 2021), "close mosaic" schedule** (well-established detection result; WebSearch 2026-06-09)
  Strong augmentation (Mosaic + MixUp) is DISABLED for the final ~15 epochs of training. This consistently lifts final mAP — the model fine-tunes on the un-augmented, real data distribution after the heavy-aug exploration phase. The general principle ("turn off the strongest, most distribution-shifting augmentation near the end") is the core evidence for an augmentation-cooldown schedule.
- **"Tradeoffs in Data Augmentation: An Empirical Study" (openreview ZcKPWuhG6wy)** (WebSearch 2026-06-09)
  For detrimental/strong transforms, the test accuracy gained by turning the augmentation off "merely recovers the clean baseline" — i.e. strong aug shifts the training distribution away from the test distribution, and removing it at the right time recovers clean-distribution performance. Supports a TIME-SCHEDULED removal rather than a global reduction.
- **Train–test distribution gap + BatchNorm recalibration** (mechanism, standard)
  TrivialAugmentWide applies aggressive ops (extreme shear/solarize/posterize/contrast) → the training pixel distribution and BN running statistics are computed on heavily-distorted images, but evaluation runs on clean images. A clean-data cooldown both (a) lets the weights settle onto the test manifold under the low-LR tail and (b) re-aligns BN running mean/var toward the clean distribution. This is a compounding, top-1-relevant mechanism (not just a loss/calibration effect).
- Internal: knowledge/papers/trivialaugment.md (TA is the augmentation ceiling here, EXP-012), project-insights.md (convergence-bound; polish-vs-top1 cluster).

## Experimental History Review

- **Current best = baseline 96.22% (EXP-012, commit 6c417a4)**; bar = 96.32. 33 experiments done; ~24 axes closed.
- **Diagnosis (firmly established)**: the k=4 / 300s net is **generalization-bound at fixed capacity** AND **convergence-bound at the short budget**. CLOSED axes: capacity/width (k=4 optimum, k≥5 epoch-wall EXP-004/009), LR schedule peak+floor+shape (EXP-016/017/019/020/029), augmentation family (TA+Cutout(16) ceiling; Mixup/CutMix/dropout regress EXP-011/018/022; policy saturated EXP-014; Cutout-size interior optimum EXP-013/021), weight-averaging (EXP-006/019/020), optimizer/gradient-dynamics→polish (GC EXP-030/031), activation (EXP-010/028), downsampling/anti-aliasing (EXP-024/027), block micro-arch (EXP-015), SE/attention (EXP-008), batch size (EXP-025), bag-of-tricks (EXP-026), WD (EXP-005), LS (EXP-023), and now the classifier HEAD / feature-aggregation axis (multi-scale head, EXP-032, −1.50pp).
- **Key meta-patterns**: (a) ADDING or globally tuning a regularizer fails — the recipe is convergence-bound, not overfit-bound (project-insights Medium). (b) Compute-neutral convergence-POLISH moves loss/calibration, not top-1 (polish cluster). (c) Any compute-adder hits the epoch wall.
- **Untried gap**: NObody has touched the augmentation *SCHEDULE over time*. Every run applied the SAME full augmentation (RandomCrop+Flip+TA+Cutout) for 100% of training. A time-varying augmentation schedule — specifically removing the strong distribution-shifting augs in the final low-LR phase — is a distinct lever that is neither "add a regularizer" (it keeps full aug for 85% of training) nor "globally reduce regularization" (EXP-023/005). It directly targets the train-test distribution gap, which no prior experiment addressed.

## Candidate Ideas

### 1. Augmentation cooldown (disable TrivialAugment + Cutout for the final ~15% of training)
**Summary**: Keep the full augmentation pipeline (RandomCrop pad-4 + HorizontalFlip + TrivialAugmentWide + GPU Cutout(16)) for the first ~85% of the time budget, then for the final `COOLDOWN_FRAC` (probe 0.15) switch to a CLEAN/light pipeline that keeps only the mild, label-preserving geometric augs (RandomCrop + Flip) and drops the two strong distribution-shifting augs (TA and Cutout). Implementation: build two CPU transform pipelines; at each epoch boundary, once `total_training_time/TIME_BUDGET_S ≥ 1 − COOLDOWN_FRAC`, mutate `train_set.transform` to the clean pipeline (picked up by freshly-forked dataloader workers on the next epoch, since persistent_workers defaults to False) and gate `cutout_batch` off by the same fraction. Everything else (architecture, LR schedule, optimizer, WD, LS, seed, batch, torch.compile) is unchanged.

**Reasoning**: The cooldown window coincides with the low-LR tail of the cosine schedule (lr ≲ 0.012 over the last 15%), so it is a clean-data fine-tuning phase: the model settles onto the test (clean) distribution and BN running stats re-align from the augmented distribution toward the clean one — both directly improve top-1, not just loss. This is the YOLOX "close mosaic" principle ported to CIFAR, and the openreview empirical study shows turning off strong aug recovers clean-distribution accuracy. Crucially it sidesteps every closed-axis trap: it is **compute-neutral** (toggling a CPU transform off; if anything the dataloader gets slightly cheaper → no epoch wall — the binding constraint per project-insights High), it **preserves the tuned coarse-to-fine hierarchy and the full recipe**, and it is **NOT** a global regularizer add/reduce (full aug is retained through the entire high-LR learning phase, so no convergence penalty — the failure mode of EXP-011/018/022).

**Sources**: YOLOX close-mosaic (WebSearch); openreview ZcKPWuhG6wy; knowledge/papers/trivialaugment.md; project-insights.md (epoch-wall High, convergence-bound Medium, polish-vs-top1 Medium); train.py L156-167 (train_tf), L223 (cutout_batch), L213-217 (epoch loop).

**Estimated Effort**: low — ~15 lines in train.py (second transform + an epoch-boundary switch + a fraction gate on cutout). No new deps.

**Risk Assessment**: (a) Most likely outcome is a small effect within the ±0.2pp noise floor (no-improvement) — the net is already regularization-saturated and the gap may be small. (b) Over-long cooldown could let the net slightly overfit clean data in the tail; mitigated by the tiny tail LR and keeping Crop+Flip. (c) Mutating `train_set.transform` mid-run must actually propagate to workers — verify in the smoke test that cooldown epochs show the clean transform (e.g. via an epoch log line). Worst case is a mild regression if clean-tail overfitting dominates; bounded and safe (no crash/invalid risk).

### 2. Stage-wise capacity redistribution at ~constant compute
**Summary**: Keep total width multiplier and depth but change HOW capacity is distributed across the three stages — e.g. shift channels toward the high-level stage ({64,128,256} → {64,128,320} with a compensating trim elsewhere) or rebalance blocks-per-stage — while holding params/FLOPs (and thus dt/epochs) ≈ constant. Tests whether the standard ResNet-20 width allocation is optimal for CIFAR-10 at k=4.

**Reasoning**: Overall capacity (k=4) is settled, but the DISTRIBUTION of capacity across stages has never been varied in 33 experiments. More high-level (8×8) capacity is cheap in FLOPs (small spatial size) and could add discriminative power where class-level features live, while preserving the coarse-to-fine hierarchy (unlike the EXP-032 head change).

**Sources**: WRN paper (width allocation), train.py L101-107 (stage widths); project-insights (capacity sweet-spot is overall, not distributional).

**Estimated Effort**: medium — change stage widths + verify param/throughput neutrality.

**Risk Assessment**: Hard to keep genuinely compute-neutral (changing channel counts shifts dt and epoch count → compute confound, the EXP-015/024 trap). Capacity is "settled," so evidence it helps is weak. Medium risk of an unattributable regression.

### 3. Sharpness-Aware Minimization (SAM), efficient/periodic variant
**Summary**: Replace the plain SGD step with SAM — compute a gradient at a perturbed (worst-case-in-neighborhood) weight point to bias optimization toward flat minima. Use an efficient variant (apply the SAM perturbation every k steps, or LookSAM) to limit the 2× forward-backward cost.

**Reasoning**: SAM is the one optimizer change specifically documented to improve TOP-1 GENERALIZATION (flatter minima → better test acc), distinct from the polish cluster that only moves loss. CIFAR WRN is SAM's home turf.

**Sources**: SAM (Foret et al. 2021); project-insights (polish-vs-top1, epoch-wall High).

**Estimated Effort**: medium — custom two-step optimizer in the loop.

**Risk Assessment**: HIGH — full SAM doubles compute → ~halves epochs → epoch wall (project-insights High: every compute-adder regressed). Even periodic SAM adds dt and risks under-training at 300s. Likely compute-confounded regression; the budget is the enemy. Deprioritized.

## Idea Evaluation

**Idea 3 (SAM)** has the most appealing mechanism (genuine top-1 generalization, not polish) but the worst fit for THIS project's binding constraint: the 300s budget. Every compute-adding change in 33 experiments hit the epoch wall (k≥5, BlurPool, pre-act, b256). SAM's 2× cost makes it the textbook epoch-wall victim; even periodic SAM adds dt. High risk of an unattributable, compute-confounded regression — not worth a loop yet.

**Idea 2 (capacity redistribution)** preserves the hierarchy (its one advantage over EXP-032) but suffers two problems: capacity is the most-confirmed-settled axis, and any channel-count change risks shifting dt/epochs → compute confound (the EXP-015/024/032 attribution trap). Weak evidence, medium risk.

**Idea 1 (augmentation cooldown)** is the strongest. It is the ONLY candidate that (a) targets a genuinely UNTRIED lever (the augmentation *schedule over time* — every prior run used a static pipeline), (b) has a clear, compounding, top-1-relevant causal mechanism (clean-distribution settling + BN recalibration in the low-LR tail), (c) is backed by an established result (YOLOX close-mosaic) and a supporting empirical study, and critically (d) sidesteps EVERY closed-axis trap simultaneously: compute-neutral (no epoch wall — the dominant failure mode), hierarchy- and recipe-preserving, and NOT a global regularizer add/reduce (full aug retained through the high-LR learning phase, so no convergence penalty). It is also the lowest-effort and has the safest failure mode (bounded mild regression at worst, no crash/invalid path). It directly attacks the one diagnosis nothing has addressed: the train-test distribution gap of a convergence-bound, generalization-bound net.

## Chosen Idea
**Selected**: Augmentation cooldown — disable TrivialAugment + Cutout for the final ~15% of training, keeping RandomCrop + HorizontalFlip.

**Why this idea**: Among all surviving candidates it has the best evidence-to-risk ratio and is the only one that opens a genuinely untried axis (time-varying augmentation) while avoiding the project's three established failure modes (epoch wall, hierarchy disruption, regularizer add/reduce). The mechanism is concrete and top-1-relevant — the low-LR tail becomes a clean-data fine-tune that aligns both the weights and BN running statistics with the test distribution — and it is cheap to implement and compute-neutral, guaranteeing a fair, cleanly-attributable test.

**Hypothesis**: Disabling TrivialAugment + Cutout for the final 15% of the time budget (keeping Crop+Flip) will raise `best_test_acc` above the 96.32 bar at an unchanged ~91 epochs / dt ~8ms / 4,299,866 params, by closing the train-test distribution gap during the low-LR annealing tail. If the effect is real but sub-threshold we expect a small positive delta within noise (no-improvement); a regression would indicate clean-tail overfitting dominates the distribution-alignment benefit at this budget.
