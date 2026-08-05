# Brainstorm EXP-001
**Created**: 2026-07-24

## Web Search & Literature Review

- **Wide Residual Networks** (`experiments/001/papers/wide-residual-networks.md`): wider, shallower residual networks can improve both CIFAR accuracy and efficiency relative to very deep thin networks.
- **mixup: Beyond Empirical Risk Minimization** (`experiments/001/papers/mixup.md`): convex image/label interpolation improves CIFAR-10 generalization with low implementation and compute overhead.
- **RandAugment** (`experiments/001/papers/randaugment.md`): a compact augmentation search space can improve CIFAR-10 accuracy without a separate policy-search phase.
- **Time Matters in Regularizing Deep Networks** (`experiments/001/papers/time-matters-regularization.md`): augmentation, mixup, and weight decay have their largest generalization effect early, supporting clean late fine-tuning.

## Experimental History Review

- First experiment under this goal; no prior experimental failures or patterns exist.
- The untouched baseline reached `best_test_acc=91.54%` in 38,254 steps across 99 epochs, using 330.1 MiB peak VRAM.
- Untried space includes budget-aligned schedules, wider/pre-activation residual models, modern augmentation, mixed-label losses, EMA, and fused/compiled execution.

## Objective Diagnosis

The baseline is schedule-limited and compute-underutilized. Its fixed milestones are 32,000 and 48,000 steps, but the 300-second run completed only 38,254 steps: the first decay therefore begins after about 84% of training and the second is unreachable. The accuracy trace jumped from a pre-decay best of 87.46% to 91.54% after the first decay, showing that low-LR refinement is valuable but receives too little budget. The H20 uses only 330.1 MiB and processes a small ResNet-20, leaving substantial capacity headroom. Generalization is the other gap: the code uses only crop/flip and weight decay, while established CIFAR methods support wider residual representations and stronger early regularization. Any larger model must preserve enough steps and complete under the 10-minute total limit; the baseline already consumed 592.7 seconds because per-epoch evaluation is outside the training budget.

## Collected Ideas

- **Budget-aligned cosine schedule** — replace unreachable fixed step milestones with cosine decay driven by measured training seconds, reaching a low learning rate within the 300-second budget. This directly targets the observed late-decay mismatch and should allocate much more time to refinement.
- **One-cycle schedule with brief warmup** — ramp quickly to a higher SGD learning rate and anneal to near zero by the budget boundary. This targets slow early optimization and the missing second decay while remaining a small, interpretable change.
- **Wider pre-activation residual network** — use fewer or similar blocks with 2-4x channel width and pre-activation shortcuts. It spends the H20's large idle capacity on richer features; WRN evidence suggests width can outperform additional depth efficiently on CIFAR.
- **Early mixup, late clean fine-tuning** — use mixup during the early critical period and disable it during the final low-LR phase. This targets overfitting without allowing soft-target training to suppress late hard-label accuracy.
- **Cutout or modest RandAugment** — add low-cost occlusion or a conservative torchvision RandAugment policy on top of crop/flip. This targets limited input diversity, but strength and CPU overhead must be controlled for a small model and short budget.
- **Exponential moving average weights** — maintain an EMA of model parameters and evaluate the averaged model late in training. This targets noisy SGD iterates and could improve peak test accuracy at small memory cost, though evaluation handling must stay once per epoch.
- **Modernized residual block and optimizer details** — use pre-activation blocks, learnable projection shortcuts, corrected CIFAR normalization in the training transform only if compatible with the frozen test transform, Nesterov momentum, and weight decay near 5e-4. This targets architectural and regularization weaknesses of the 2016 baseline.
- **Compiled high-capacity moonshot** — use `torch.compile` plus a substantially wider network or multi-branch Shake-Shake-style block to trade compile startup for more useful model capacity within the training timer. Upside is high, but compiler stability and per-step cost make this risky for a first experiment.

## Combinations

- **Wider pre-activation network + budget-aligned cosine**: width uses the idle H20 capacity while cosine guarantees that the slower model still traverses a complete optimization schedule. The combination is stronger than width with unreachable milestones or a new schedule on an under-capacity model.
- **Early mixup/Cutout + late clean cosine refinement**: strong early regularization shapes generalizable features, then disabling mixed targets while cosine reaches low LR allows hard-label convergence. This directly follows the critical-period evidence and should outperform either always-on regularization or schedule correction alone.
- **Modest width + early mixup + EMA**: increased capacity raises fit potential, mixup controls its generalization gap, and EMA stabilizes late checkpoints. The mechanisms are complementary but combined complexity makes attribution harder for EXP-001.

## Candidate Ideas

### Critical-Period Augmentation With Clean Fine-Tuning
**Summary**: Preserve ResNet-20 and its optimizer, but use moderate RandAugment, 8x8 Cutout, and mixup for the first 72% of counted training time. Switch at an epoch boundary to baseline crop/flip and hard labels, while applying time-aligned LR drops at 72% and 90% for clean low-LR refinement.

**What it targets**: The baseline's limited crop/flip-only generalization and schedule mismatch, while retaining its known model throughput and isolating a training-policy intervention.

**Reasoning**: Mixup and RandAugment improve CIFAR generalization, and critical-period evidence supports removing regularization late. The clean tail directly mitigates soft-target under-convergence within a short budget.

**Sources**: `proposals/idea-02.md`; `papers/mixup.md`; `papers/randaugment.md`; `papers/time-matters-regularization.md`; baseline row in `04-results.tsv`.

**Estimated Effort**: medium

**Risk Assessment**: Three regularizers may compound into underfitting, and CPU RandAugment could push the already 592.7-second baseline beyond the 10-minute ceiling. The abrupt phase handoff and fixed 72% choice are unvalidated.

### Time-Aligned Pre-Activation WRN-16-2
**Summary**: Replace the thin post-activation ResNet-20 with a pre-activation WRN-16-2, use batch 256, Nesterov SGD, selective 5e-4 weight decay, and a five-percent warmup followed by cosine decay keyed to counted training time. The moderate width increase and reduced depth aim to use the H20 more effectively without collapsing the number of data passes.

**What it targets**: The model's low capacity and 330.1 MiB H20 footprint, plus the fixed schedule whose first decay occurs at about 84% of realized steps and whose second decay is unreachable.

**Reasoning**: The baseline's largest accuracy gain occurred only after its late first LR drop. WRN evidence supports shifting compute from thin depth to width on CIFAR, while a time-derived schedule guarantees a complete convergence phase despite changed throughput.

**Sources**: `proposals/idea-01.md`; `papers/wide-residual-networks.md`; `papers/time-matters-regularization.md`; baseline row in `04-results.tsv`.

**Estimated Effort**: medium

**Risk Assessment**: Simultaneous architecture, batch, LR, and decay changes weaken attribution. LR 0.2 or 5e-4 decay may over-regularize a short run, and a slower model may complete too few data passes.

### Capacity-Matched Pre-Activation ResNet With Early Cutout
**Summary**: Keep ResNet-20 depth but use pre-activation blocks and 1.5x stage widths, batch 256, LR 0.2 warmup/cosine decay tied to training time, 5e-4 weight decay, and vectorized on-device 8x8 Cutout for the first 70% of the run. This is a more conservative capacity increase than WRN-16-2 and adds only one augmentation.

**What it targets**: H20 underutilization, narrow-model capacity, late/unreachable LR milestones, and limited data regularization.

**Reasoning**: Moderate width reduces throughput risk while time-aligned cosine fixes the clearest measured defect. Early-only Cutout counterbalances extra capacity without changing targets or permanently slowing hard-label convergence.

**Sources**: `proposals/idea-03.md`; `papers/wide-residual-networks.md`; `papers/time-matters-regularization.md`; baseline row in `04-results.tsv`.

**Estimated Effort**: medium

**Risk Assessment**: Bundling block ordering, width, batch, schedule, decay, and Cutout makes failures hard to attribute. The claimed 70-epoch minimum and 92.3-93.2% target are estimates rather than measured guarantees.

## Review

The reviewer identified the time-aligned LR schedule as the highest-confidence lever and selected the canonical WRN-16-2 proposal as the strongest way to pair it with unused H20 capacity. Significant feedback adopted: enable `torch.backends.cudnn.benchmark`, use `zero_grad(set_to_none=True)`, retain selective weight decay that excludes normalization and bias parameters, remove the legacy scheduler completely, and log realized epochs, steps, LR, and throughput. Full concerns and scores are in `01-idea-review.md`. The main residual risk is completing too few useful data passes, which the moderate WRN-16-2 size and batch 256 bound and the run metrics will expose.

## Idea Evaluation

Adopt the reviewer verdict in `01-idea-review.md`: **Time-Aligned Pre-Activation WRN-16-2** scored 8.5/10 for both evidence/reasoning and impact, ahead of the conservative width-plus-Cutout bundle and the higher-risk augmentation stack. No hard-constraint or reward-hacking issue was found.

## Chosen Idea
**Selected**: Time-Aligned Pre-Activation WRN-16-2

**Why this idea**:
It directly corrects the baseline's measured schedule failure while using the H20's large capacity headroom through a canonical shallow-wide CIFAR architecture. Relative to the other finalists, it has the strongest literature grounding, fewer sequential blocks, a clean late convergence mechanism, and lower wall-clock risk than CPU-heavy RandAugment.

**Hypothesis**:
A pre-activation WRN-16-2 trained with batch 256, selective `5e-4` weight decay, Nesterov SGD, and a five-percent warmup plus training-time cosine decay will exceed the 91.54% baseline by at least 0.1 percentage points within the fixed 300-second training budget and 10-minute total limit. A target of at least 92.0% is plausible if the model retains roughly 45 or more dataset-equivalent passes.
