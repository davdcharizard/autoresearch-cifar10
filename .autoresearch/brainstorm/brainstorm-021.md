# Brainstorm EXP-021
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new sources needed. Key learnings from experimental history are the primary driver for hyperparameter tuning ideas.

## Experimental History Review

- **22 experiments** (BASE through EXP-020), baseline 96.39%. Four consecutive failures (017-020).
- **Over-regularization signal**: EXP-011 (Dropout) confirmed model is already well-regularized — adding more regularization hurts. Multiple regularizers present: CutMix(p=0.5), label smoothing 0.1, WD 5e-4, EMA 0.999, random crop/flip.
- **TTA/architecture space exhausted**: SE blocks, spatial-shift TTA, pre-activation, stochastic depth, deeper models, different architectures — all failed. Hflip TTA is the only eval improvement that worked.
- **Training recipe space undertried**: LR, CutMix prob, label smoothing, CutMix alpha, gradient clipping — never individually tuned since the winning recipe was established at EXP-007.
- **Bottleneck analysis**: Training accuracy is 95.73%. With ~54 epochs, every regularizer that reduces effective sample utility (CutMix makes half the batches harder to learn from) costs convergence speed.

## Candidate Ideas

### 1. CutMix Probability Reduction (0.5 → 0.3)
**Summary**: Reduce CUTMIX_PROB from 0.5 to 0.3. This means 70% of training batches use clean images and 30% use CutMix-augmented images, vs the current 50/50 split. All other hyperparameters unchanged.

**Reasoning**: The model trains for only ~54 epochs. CutMix at p=0.5 means half the training samples are mixed images — harder to learn from. EXP-006 showed that stacking augmentation hurts at ~60 epochs, confirming the model is convergence-limited. Reducing CutMix probability gives the model more "easy" training samples, accelerating convergence while retaining CutMix's regularization benefit on the remaining 30%. The model has 4 other regularizers (label smoothing, WD, EMA, data augmentation) providing sufficient regularization even with less CutMix.

**Sources**: EXP-006 (augmentation stacking hurts), EXP-011 (over-regularization confirmed), convergence budget analysis

**Estimated Effort**: low — change one constant

**Risk Assessment**: Low. Reducing from 0.5 to 0.3 is conservative. Worst case: slight under-regularization leading to ~0.1% accuracy drop. The model has ample remaining regularization.

### 2. Label Smoothing Reduction (0.1 → 0.05)
**Summary**: Reduce LABEL_SMOOTHING from 0.1 to 0.05. This gives the true class a target probability of 0.95 (vs 0.9 currently) and reduces the "soft" regularization from the loss function.

**Reasoning**: Label smoothing 0.1 is moderately aggressive. Combined with CutMix (which also provides soft targets via mixed labels) and EMA, the total target-softening effect may be excessive. Reducing to 0.05 allows the model to be more confident on correct predictions, which can improve accuracy. The 0.05 value is still meaningful regularization (not 0).

**Sources**: EXP-011 (over-regularization), general practice for label smoothing tuning

**Estimated Effort**: low — change one constant

**Risk Assessment**: Low. 0.05 is still non-zero smoothing. Worst case: model becomes slightly overconfident, losing ~0.1%.

### 3. Gradient Clipping (max_norm=1.0)
**Summary**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before the optimizer step. This clips large gradient spikes from CutMix-augmented batches, stabilizing training.

**Reasoning**: CutMix can create confusing mixed images that produce large gradients. Gradient clipping prevents these outlier updates from destabilizing the learned representation. This is especially valuable in late training (low LR) where a single large gradient can undo many small fine-tuning updates.

**Sources**: Standard practice in modern training pipelines, theoretical justification in https://arxiv.org/pdf/1905.11881

**Estimated Effort**: low — add 1 line after backward()

**Risk Assessment**: Very low. Max_norm=1.0 is standard. If gradients are already small, clipping has no effect. Only activates on outlier gradients.

## Idea Evaluation

**Evidence strength**: CutMix probability reduction has the strongest evidence — EXP-006 and EXP-011 both point to over-regularization as a problem. Label smoothing reduction is supported by the same evidence but is a weaker signal (label smoothing interacts more subtly). Gradient clipping has general support but no specific evidence it's needed for this model.

**Mechanism clarity**: CutMix probability has the clearest mechanism — fewer mixed images → more "easy" samples → faster convergence in limited epochs. Label smoothing has a clear mechanism too (softer targets → model can be more confident). Gradient clipping's mechanism requires that large gradients actually occur, which we haven't verified.

**Expected impact**: CutMix prob reduction targets the most significant regularizer (CutMix affects 50% of all samples, much more impactful than label smoothing which affects all samples subtly). If the model is convergence-limited, giving it 20% more clean samples should help.

**Risk profile**: All three are very safe. CutMix prob reduction has the safest failure mode (model is still well-regularized with 4 other mechanisms).

## Chosen Idea
**Selected**: CutMix Probability Reduction (0.5 → 0.3)

**Why this idea**: Directly targets the strongest regularizer in a convergence-limited regime. The over-regularization hypothesis is well-supported by EXP-006 and EXP-011. Reducing CutMix prob from 0.5 to 0.3 is conservative and gives the model 40% more clean training batches.

**Hypothesis**: Reducing CutMix probability from 0.5 to 0.3 will improve convergence in the ~54 epoch budget, yielding best_test_acc > 96.49% (training accuracy > 95.83%, boosted by hflip TTA).
