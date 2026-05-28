# Brainstorm EXP-019
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Super-Convergence / OneCycleLR** (Smith & Topin 2018, web search)
  OneCycleLR with large max LR achieves higher accuracy in fewer epochs via superconvergence. hlb-CIFAR10 speedrun uses this to reach ~94% in 30 epochs. However, aggressive LR cycling requires careful tuning and our EXP-000 cosine failure + High Importance pattern on MultiStepLR suggest caution.

- **hlb-CIFAR10 speedrun techniques** (https://github.com/tysam-code/hlb-CIFAR10, web search)
  Achieves ~95.79% in ~110s with base_depth=128, 90 epochs, EMA, cutmix_size=10. Key: uses a custom 8-layer ConvNet, not ResNet. Relevant techniques: EMA (we tried — marginal), cutmix (we tried — over-regularized). Architecture is fundamentally different from ours.

- **cifar10-airbench** (https://github.com/KellerJordan/cifar10-airbench, web search)
  94% in 3.29s, 96% in 27s on A100. Uses Muon optimizer, extra conv layers, 12-pixel cutout, data filtering, progressive augmentation scheduling. Many techniques are architecture-specific but Muon optimizer and augmentation scheduling are portable ideas.

- **Test-Time Augmentation (TTA)** (Shanmugam et al. 2021, Krizhevsky 2012, web search)
  Horizontal flip TTA gives +0.72% on ResNet18/ImageNet (Shanmugam et al.). Original AlexNet paper used 10-crop TTA. For CIFAR-10, horizontal flip is the natural choice since it's used in training augmentation. Zero training cost — only adds inference time which doesn't count against the 300s budget.

- **Self-distillation / Knowledge Distillation** (Zhang et al. 2019 "Be Your Own Teacher", Hinton et al. 2015, web search)
  Self-distillation on ResNet improves accuracy 0.61-4.07% across architectures without a separate teacher. Born-Again Networks show student matching teacher then exceeding via re-distillation. Knowledge distillation lowers top-1 error by 1.1% for ResNet-110 on CIFAR-10. However, requires multi-stage training which is hard to fit in 300s.

- **Deeper ResNet variants for CIFAR-10** (He et al. 2015, web search)
  ResNet-32 gains ~1% over ResNet-20 at standard widths. ResNet-56/110 show further gains but with diminishing returns. At WIDTH_MULT=4, going deeper reduces epoch count — tradeoff between capacity and training time.

## Experimental History Review

- **Current best**: 95.57% (EXP-015, label smoothing 0.2 on WIDTH_MULT=4 baseline)
- **Improvement threshold**: >95.67% (baseline + 0.1pp)
- **Total experiments**: 19 (BASE + EXP-000 through EXP-018), 8 improvements

**What worked (improvement trajectory)**:
BASE 91.72 → EXP-001 92.29 (width-2x + wall-clock LR) → EXP-002 92.92 (TrivialAugmentWide + RandomErasing) → EXP-003 93.33 (WD=5e-4) → EXP-005 94.44 (AMP) → EXP-007 94.82 (WIDTH_MULT=4) → EXP-009 95.39 (batch 256 + warmup) → EXP-015 95.57 (label smoothing 0.2)

**What didn't work**:
- Regularization additions all failed: CutMix (EXP-010, -0.36pp), Mixup (EXP-017, -0.04pp), DropPath (EXP-018, -0.33pp) — regularization stack is saturated
- SE blocks exhausted (EXP-011/012): ~9ms/step overhead intrinsic to computation, no implementation fix
- EMA marginal (EXP-013 broken BN, EXP-014 only +0.05pp)
- torch.compile zero speedup (EXP-008), BN momentum negligible (EXP-016)
- LR schedule changes hurt: earlier drops (EXP-006), cosine with wrong T_max (EXP-000)

**What hasn't been tried**:
- Test-time augmentation (zero training cost)
- Different optimizer (AdamW, Muon, LAMB)
- Deeper architecture (NUM_BLOCKS=4 or 5)
- OneCycleLR / cosine with correct T_max
- Progressive augmentation scheduling
- Self-distillation within the single training run

## Candidate Ideas

### 1. Test-Time Augmentation (Horizontal Flip)
**Summary**: Add a custom TTA evaluation function in train.py that, for each test image, runs the model on both the original and horizontally-flipped versions, averages the logits, and takes argmax. This replaces the standard single-crop `evaluator.evaluate()` call for the final best_test_acc measurement. The training loop itself is completely unchanged — TTA is applied only at evaluation time. Since evaluation time does not count against the 300s training budget (only training step time is accumulated in `total_training_time`), this is a zero-cost improvement.

**Reasoning**: Horizontal flip is already used as a training augmentation (RandomHorizontalFlip in train_tf), so the model has learned flip-equivariant features. Averaging predictions over the original and flipped image reduces variance in the softmax output and corrects for cases where the model is marginally wrong on one orientation but correct on the other. Literature shows +0.2-0.7pp improvement on ImageNet with simple horizontal flip TTA (Shanmugam et al. 2021). On CIFAR-10 with a well-trained model near 95.5%, a conservative estimate of +0.1-0.3pp is reasonable. The implementation is straightforward: iterate `evaluator.loader`, for each batch apply `torch.flip(inputs, dims=[3])`, run both through the model, average logits, compute accuracy. The `evaluator.loader` attribute is public and accessible from train.py.

**Sources**: Shanmugam et al. 2021 (TTA survey), Krizhevsky 2012 (AlexNet 10-crop TTA), prepare.py (evaluator.loader is a public DataLoader)

**Estimated Effort**: low — ~20 lines of code, no hyperparameter tuning needed

**Risk Assessment**: Very low risk. Worst case: TTA adds noise rather than signal and accuracy stays flat or drops marginally. The training loop is completely untouched so there's no risk of destabilizing training. The only risk is that evaluation takes ~2x longer per epoch (iterating test set twice), but since eval is outside the training time budget this doesn't matter. The improvement might be below the 0.1pp threshold if the model's predictions are already very stable across flips.

### 2. OneCycleLR with Tuned Max LR
**Summary**: Replace the wall-clock-fractional MultiStepLR schedule with PyTorch's OneCycleLR using a triangular-then-anneal shape. The schedule ramps LR from a low value up to a max LR (e.g., 0.4-0.8) in the first ~30-40% of training, then anneals down to near zero. This exploits superconvergence: large learning rates in mid-training act as regularizers and find wider minima, while the final anneal enables sharp convergence. Total training time and all other hyperparameters remain fixed.

**Reasoning**: OneCycleLR is the dominant schedule in CIFAR-10 speedrun codebases (hlb-CIFAR10, cifar10-airbench). The current MultiStepLR spends 50% of training at full LR=0.2, then drops to 0.02 and 0.002 — but superconvergence literature shows that briefly visiting much higher LRs (2-5x the MultiStepLR base) and then aggressively annealing can reach higher accuracy. The wall-clock-fractional MultiStepLR is well-validated (High Importance pattern), but it was validated at earlier accuracy levels (92-95%). At 95.5%+, the optimization landscape may benefit from the wider minima that OneCycleLR's high-LR phase provides.

**Sources**: Smith & Topin 2018 (superconvergence), hlb-CIFAR10 codebase, cifar10-airbench codebase, EXP-000 (cosine failure — wrong T_max, not wrong idea)

**Estimated Effort**: medium — schedule replacement is simple code, but max_LR tuning may need adjustment

**Risk Assessment**: Medium-high risk. The current MultiStepLR is a High Importance validated pattern. EXP-000 showed that cosine annealing with wrong T_max failed badly. OneCycleLR is fundamentally different from cosine (it ramps UP first), but the risk is that the max LR is wrong for this model width — too high causes divergence, too low wastes the superconvergence benefit. AMP instability at intermediate LR values (EXP-005 pattern) could also interact badly with the high-LR phase. If it fails, the failure mode is likely a significant accuracy drop (not crash), so it's recoverable but wastes a loop iteration.

### 3. Deeper Architecture (NUM_BLOCKS=5, ResNet-32)
**Summary**: Increase NUM_BLOCKS from 3 to 5, creating a ResNet-32 (6*5+2=32 layers) with WIDTH_MULT=4. This adds 4 more BasicBlocks (2 per layer group that gains blocks), increasing model capacity from ~4.3M to ~7.1M parameters. The wider channels (64/128/256) are preserved. With more parameters, the model can learn more complex feature hierarchies. The tradeoff is fewer epochs in the 300s budget — estimated ~58-65 epochs at 25-27ms/step vs current ~92-98 epochs at 16-17ms/step.

**Reasoning**: The improvement trajectory shows that capacity increases have been the most reliable source of gains: WIDTH_MULT=2 (+0.57pp over BASE), WIDTH_MULT=4 (+0.38pp over WIDTH_MULT=2), batch 256 throughput (+0.57pp via more epochs). The regularization stack is saturated — deeper architecture provides more capacity to absorb existing regularization rather than adding more. He et al. 2015 reports ~1% gain from ResNet-20 to ResNet-32 at standard widths; at WIDTH_MULT=4, the absolute gain may be smaller but the capacity headroom is real. The risk is that ~60 epochs may be insufficient — but EXP-005 showed AMP's jump from 69→106 epochs was critical, suggesting the model needs ~80+ epochs to converge fully.

**Sources**: He et al. 2015 (ResNet paper, CIFAR-10 depth experiments), EXP-007/EXP-009 (current epoch counts), goal-learnings patterns on epoch count

**Estimated Effort**: low — single hyperparameter change (NUM_BLOCKS=3→5)

**Risk Assessment**: Medium risk. The main concern is epoch count: going from ~95 epochs to ~60 epochs is a 37% reduction. The wall-clock-fractional LR schedule will adapt correctly (drops at 50%/75% of 300s), but 60 epochs may not be enough for the model to converge, especially with the current regularization stack that was calibrated for ~95 epochs. EXP-005 showed AMP instability at intermediate LR values with fewer epochs. If it fails, the learning is clear: at WIDTH_MULT=4, the throughput cost of depth is too high for the 300s budget.

## Idea Evaluation

**Evidence strength**: TTA has the strongest direct evidence — horizontal flip TTA is universally reported as beneficial for models trained with horizontal flip augmentation, and the mechanism is well-understood (variance reduction over flip-equivariant features). OneCycleLR has strong evidence from speedrun codebases but in architecturally different settings; its applicability to our specific ResNet-20 setup is uncertain. Deeper architecture has solid theoretical backing (He et al. 2015) but the evidence is at standard widths, not WIDTH_MULT=4.

**Mechanism clarity**: TTA has the clearest mechanism — averaging logits from original and flipped images reduces prediction variance, period. OneCycleLR's mechanism (wider minima via high-LR exploration) is well-established but harder to predict the magnitude of improvement. Deeper architecture's mechanism (more capacity) is clear but its interaction with the 300s budget (fewer epochs) creates uncertainty.

**Expected impact**: OneCycleLR has the highest ceiling if it works (could be +0.3-0.5pp). TTA has a moderate but reliable ceiling (+0.1-0.3pp). Deeper architecture is the most uncertain due to the epoch count tradeoff.

**Risk profile**: TTA has the lowest risk — training is completely untouched, worst case is no improvement. OneCycleLR has the highest risk — it replaces a well-validated High Importance pattern. Deeper architecture has medium risk — epoch count reduction could negate capacity gains.

**Feasibility**: TTA is the easiest to implement (~20 lines) and requires no hyperparameter tuning. OneCycleLR requires max_LR tuning. Deeper architecture is a one-line change but the outcome is most uncertain.

**Recommendation**: TTA is the clear winner — strongest evidence, clearest mechanism, lowest risk, easiest implementation, and zero training cost. It targets the metric through a completely orthogonal axis (evaluation quality) rather than competing with the saturated training regularization. If TTA yields +0.1-0.3pp, it brings the baseline to 95.67-95.87% before any training changes are needed, creating headroom for future optimization experiments.

## Chosen Idea
**Selected**: Test-Time Augmentation (Horizontal Flip)

**Why this idea**:
TTA has the strongest evidence-to-risk ratio of all candidates. It operates on a completely orthogonal axis (evaluation improvement) compared to all 18 prior experiments (which modified training). The mechanism is well-understood: averaging logits from original and horizontally-flipped test images reduces prediction variance. The implementation is minimal (~20 lines), requires no hyperparameter tuning, and carries zero risk to training stability since the training loop is completely untouched. Evaluation time does not count against the 300s budget, making this a genuinely free improvement.

**Hypothesis**:
Replacing single-crop evaluation with horizontal-flip TTA (averaging logits from original and flipped images) will improve best_test_acc by +0.1-0.3pp, from 95.57% to ≥95.67%, because the model was trained with RandomHorizontalFlip and has learned flip-equivariant features whose prediction variance is reduced by averaging over both orientations.
