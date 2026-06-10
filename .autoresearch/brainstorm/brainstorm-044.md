# Brainstorm EXP-044
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Torchvision RandAugment documentation** (https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html)
  `torchvision.transforms.RandAugment` is available in this environment and exposes the standard `num_ops`, `magnitude`, and `num_magnitude_bins` parameters without adding dependencies. Local API check confirmed `transforms.RandAugment` resolves to `torchvision.transforms.autoaugment.RandAugment`.
- **RandAugment paper** (https://papers.neurips.cc/paper_files/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf)
  RandAugment is designed as a reduced-search automated augmentation method. For this fixed-budget CIFAR loop, the useful adaptation is not the default strong policy, but a conservative low-magnitude probe that can test whether policy augmentation adds generalization beyond crop/flip/reflection.
- **Existing knowledge: SGDR cosine scheduling** (`.autoresearch/knowledge/papers/sgdr-cosine-schedule.md`)
  Cosine annealing remains a possible schedule alternative, but the current 21k step-drop trajectory has strong local support.
- **Existing knowledge: PyTorch EMA averaging** (`.autoresearch/knowledge/references/pytorch-ema-averaging.md`)
  EMA remains a possible late-stability lever, but prior EMA/averaging experiments showed overhead and collapse risks that require careful containment.

## Experimental History Review

- Current baseline is `best_test_acc=93.97%` from EXP-038 / commit `755be2c`; the active goal requires `best_test_acc >= 94.07%`.
- The current anchor is `STAGE_WIDTHS=(28,56,112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `WEIGHT_DECAY=2e-4`, `LR=0.1`, `MOMENTUM=0.9`, and `LR_MILESTONES=[21000,64000]`.
- EXP-038 validated the current stronger-decay anchor; EXP-039 and EXP-041 bracketed nearby weight decay values as worse, so isolated weight-decay retuning is low priority.
- EXP-040 and EXP-043 now bracket initial LR around `0.1`: both `0.12` and `0.08` underperformed, so isolated LR scalar retuning should stop.
- EXP-042's mixup attempt crashed before final metrics. This does not disprove augmentation, but it argues against immediately trying another mixed-sample method with run-control risk.
- Cutout-style masking failed twice, but crop-boundary augmentation via reflection padding was a strong validated improvement. This suggests augmentation can work when it is low overhead and not too destructive.
- EMA/averaging has a mixed history: EXP-004 per-step EMA lost too many steps for a tiny +0.03 point gain, and EXP-021 long equal averaging collapsed. A bounded late EMA is still possible, but implementation risk is higher than a transform-only probe.
- Full schedule swaps remain untested on the final `2e-4` anchor, but schedule-only second drops and adjacent first-drop brackets have repeatedly underperformed. Cosine should be considered a later clean schedule experiment, not the immediate best candidate.

## Candidate Ideas

### 1. Mild RandAugment After Crop/Flip
**Summary**: Insert `transforms.RandAugment(num_ops=1, magnitude=5)` into the training transform after `RandomHorizontalFlip()` and before `ToTensor()`, preserving every optimizer, schedule, architecture, and validation setting.

**Reasoning**: The current best recipe has benefited from no-overhead or low-overhead regularization: reflection crop padding, mild label smoothing, and stronger weight decay. RandAugment is a standard policy-augmentation tool available directly in torchvision. A mild setting avoids the default stronger policy and tests whether non-erasing color/geometric perturbations add generalization where cutout masking was too destructive. Data augmentation work occurs in the data pipeline and should not change the fixed training-time budget; total wall-clock remains subject to the 10-minute cap.

**Sources**: Torchvision RandAugment docs; RandAugment paper; EXP-029 reflection-padding improvement; EXP-032 label-smoothing improvement; EXP-038 weight-decay improvement; cutout failures EXP-005/EXP-009.

**Estimated Effort**: low

**Risk Assessment**: RandAugment may be too strong even at mild settings or may increase dataloader overhead enough to raise total wall-clock. Failure mode should be a valid no-improvement, and the implementation is a single transform insertion in `train.py`.

### 2. Bounded Late EMA Evaluation
**Summary**: Maintain an EMA copy only after the first LR drop, update it at low frequency, and evaluate the EMA model once per epoch instead of the raw model after EMA activation.

**Reasoning**: Several runs show late peak/final drift, so averaged weights could improve the evaluated checkpoint without changing optimizer dynamics. Prior EMA/averaging failures were not definitive for this bounded variant: EXP-004 updated every step across the full run and lost too many steps, while EXP-021 used long-horizon equal averaging that collapsed. A late low-frequency EMA could target the same mechanism with less overhead.

**Sources**: `.autoresearch/knowledge/references/pytorch-ema-averaging.md`; EXP-004 report; EXP-021 report; EXP-043 report showing peak 93.49 then final 93.19.

**Estimated Effort**: medium

**Risk Assessment**: EMA implementation is more fragile because of compiled-model wrappers, BatchNorm buffers, and previous averaging collapse. A wrong averaging policy could produce an invalid or very low metric even if the raw model trains normally.

### 3. No-Restart Cosine Schedule Over Reachable Steps
**Summary**: Replace the current `MultiStepLR([21000,64000])` with a no-restart `CosineAnnealingLR` over the expected reachable step horizon, preserving all other current anchor settings.

**Reasoning**: Cosine annealing is a CIFAR-relevant schedule and could smooth the abrupt 21k transition. The final anchor has not seen a clean cosine-only experiment. However, the current step schedule is unusually well calibrated: the first drop consistently drives the main post-drop accuracy jump, and isolated schedule retunes have failed repeatedly.

**Sources**: `.autoresearch/knowledge/papers/sgdr-cosine-schedule.md`; failed schedule entries in `goal-learnings`; EXP-016 and EXP-038 reports.

**Estimated Effort**: medium

**Risk Assessment**: A full schedule swap may spend too long away from the proven low-LR refinement regime and underperform. It is clean to run, but the evidence is weaker than the transform-only augmentation probe.

## Idea Evaluation

Mild RandAugment has the best evidence-to-risk profile for the next loop. The local optimizer scalar space is now bracketed: `WEIGHT_DECAY=2e-4` is locally best and `LR=0.1` is locally best. That makes another scalar optimizer probe low value. RandAugment targets a different mechanism, generalization through non-erasing policy perturbations, and uses an installed torchvision transform with a narrow code change. It also composes with the currently validated reflection, smoothing, and weight-decay anchor rather than replacing them.

Bounded late EMA is plausible because late accuracy drifts below peak, but it is more implementation-sensitive. The project has already seen both EMA overhead and equal-averaging collapse. It deserves a future careful plan, but after EXP-042's crash and several scalar no-improvements, the immediate next experiment should prefer a simple run-control path.

Cosine scheduling has external support but weaker local support. The 21k step drop is central to the current anchor's behavior, and schedule-only modifications have repeatedly failed. A clean cosine-only test may become useful once augmentation and late-stability probes are exhausted, but it is not the strongest next move.

## Chosen Idea
**Selected**: Mild RandAugment After Crop/Flip

**Why this idea**:
It is a low-risk, single-file, transform-only probe that tests a new mechanism after optimizer scalar brackets have closed. A conservative `num_ops=1, magnitude=5` setting gives policy augmentation a chance to improve generalization without repeating destructive cutout or unstable mixup behavior.

**Hypothesis**:
Adding mild `RandAugment` after crop/flip and before tensor conversion will improve generalization of the current `2e-4` label-smoothed reflection anchor enough to reach `best_test_acc >= 94.07%` while preserving the fixed harness, schedule geometry, parameter count, and single-GPU constraints.
