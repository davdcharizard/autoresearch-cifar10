# Knowledge Base Index

This directory contains persistent external knowledge that informs brainstorming and planning across all research loops. Unlike experiment logs (which are loop-specific), entries here represent standing background context pulled in throughout the research process: relevant paper distillations, reference implementation notes, and domain knowledge.

## How to use
- `research-brainstorm` loads this README as a lightweight index, then loads specific entries on demand
- Entries are added by researchers (manually), by `/lit-search` (automated paper distillations), or proposed by the agent after consulting external sources
- Add entries for any paper, reference implementation, or domain insight that is likely to inform future experiments

## Venues
- `venues.md` in this directory defines which academic proceedings `/lit-search` searches for this goal — seeded with the default top CS/ML conferences (NeurIPS, ICML, ICLR)
- Edit it to match this goal's research domain; venues are per-goal, so different goals can search different domains

## Papers

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add paper summaries here)_ | | |

## References

| File | Topic | Key Takeaway |
|------|-------|-------------|
| references/fast-cifar10-recipes.md | Fast CIFAR-10 training (DavidNet/hlb/airbench) | Wide-shallow ResNet-9 + one-cycle + Cutout/LS/bf16 → 94–96%; whitening conv & flip-TTA untried next steps. Validated base: EXP-001 → 95.22%. |
| references/rezero-identity-init.md | ReZero trainable identity-init (arXiv:2003.04887) | Gate residual with learnable α=0 → identity at init, live gradient. Use for adding depth (NOT zeroed-BN-γ: dead under post-BN ReLU). Validated EXP-004 → 96.00%. |
| references/muon-optimizer.md | Muon (Newton-Schulz orthogonalized momentum; airbench94_muon) | Orthogonalize conv-weight momentum via NS quintic; airbench peak LR 0.24 + weight-renorm. EXP-009: 0.24 too hot for our long one-cycle (diverged→94.11); lower ~2-3× or use update-scale+decoupled-WD. |
| references/mixing-augmentation.md | Label-mixing aug (CutMix/mixup) at short schedules | CutMix (region mix + area-split label) beats mixup at short/fast schedules (~96.2 @200ep ResNet-18); mixup needs 800+ ep. Complementary to occlusion aug; draw box on CPU (no CUDA sync); watch LS interaction. First tested EXP-011. |
| references/sam-sharpness-aware.md | SAM — Sharpness-Aware Minimization (arXiv:2010.01412) | Flat-minima loss-geometry: ascent e_w=ρ·g/‖g‖ then descent with perturbed grad; ρ=0.05 CIFAR default; +0.3–1.0pp at matched epochs. 2× fwd-bwd → under-anneal risk at 300s; tail-only gate `progress>=0.65`. BN-stats 1st-pass-only, fp32 perturbation. First tested EXP-013. |
| references/torch-compile-throughput.md | torch.compile (TorchInductor) as a throughput meta-lever | Fuses BN/ReLU/ReZero/autocast glue → ~7–15% img/s (airbench arXiv:2404.00498 got ~14%, math-equiv). Off-budget compile-warmup (before t_start_training) keeps compile cost off the 300s budget. Traps: BN-buffer restore around warmup, uncompiled eval/EMA path, param aliasing via `_orig_mod`. Spend throughput on capacity (320-width) to beat EXP-007 under-anneal. Chosen EXP-014. |
| references/ghost-batchnorm.md | Ghost BatchNorm (regularizing activation-statistic noise; Hoffer 2017 / DavidNet) | Split batch-512 BN train-stats into ghosts of 64–128 → regularization noise, a DavidNet trick on a DIFFERENT axis than the saturated input-aug/wd/LS. Near-throughput-free. CRITICAL: update eval running-stats from FULL-batch moments (not noisy per-ghost) so EMA(use_buffers) stays clean; g=512≡nn.BatchNorm2d smoke. Chosen EXP-016. |
| references/policy-augmentation.md | Transform-based policy aug (AutoAugment/RandAugment/TrivialAugment) | 3rd aug mechanism (geometric+photometric), distinct from occlusion/mixing; torchvision built-in (no new dep). Canonical ~96→97% CIFAR lever BUT gains are at 200–2000ep; use mild magnitude + REPLACE-not-stack to avoid under-fit at ~150ep. persistent_workers blocks mid-train curriculum mutation. Chosen EXP-015. |
| references/blurpool-antialiasing.md | BlurPool anti-aliased downsampling (Zhang 2019 ICML, arXiv:1904.11486) | FIRST architectural-inductive-bias lever: fixed binomial blur between dense max and stride-2 subsample restores shift-equivariance; "anti-aliasing as regularization" raises clean accuracy. No params/deps, standard convs (no fused-BN break, unlike GhostBN). #1 risk = throughput of depthwise blurs → num_epochs≥135 gate; layer1/2-only & ksize=2 fallbacks. Chosen EXP-018. |
| references/lr-schedule-shape.md | LR schedule SHAPE: cosine decay vs cyclic/linear one-cycle (CIFAR) | MosaicML: cyclic/linear underperform cosine by ~0.5% on CIFAR-class CNNs. EXP-020 NO-IMPROVEMENT: cosine TIED linear-triangular (confirm +0.04pp; session-1 +0.32 was a low-control-draw artifact), <96.48 — schedule-shape axis closed. MosaicML's ResNet-50 figure shrank to noise on our 150ep heavily-augmented net. |
| references/squeeze-excitation.md | Squeeze-Excitation channel attention (Hu 2018, CVPR) | Content-adaptive per-channel recalibration (GAP→bottleneck→gate); a NEW functional form (channel attention), <1% params, ~throughput-free. CRITICAL: identity-init the gate via 2*sigmoid + zero-init fc2 (plain sigmoid→0.5 gate disturbs un-gated Residual blocks). EXP-019 no-improvement: ties control (+0.28pp s1 didn't replicate, +0.02pp confirm); redundant on this saturated net; layer1 SE net-negative. |
| references/self-distillation-kd.md | EMA self-distillation / KD (Hinton 2015, Born-Again 2018, Mean Teacher 2017, Müller 2019) | The LOSS/learning-signal axis (chosen EXP-023, first attack after EXP-022 showed ceiling is backbone-independent). KD from the FREE already-maintained EMA teacher: `L=(1−α)CE_LS + α·T²·KL(p_teacher^T ‖ p_student^T)`, teacher detached as TARGET (forward-KL), T≈4, α tail-gated. KD carries inter-class structure LS erases (non-redundant). Risks: under-anneal (1 extra fwd→compile+tail-gate+smoke≥12610), over-soften w/ LS0.2 (reduced-LS arm). |
| references/wide-resnet.md | Wide ResNet (Zagoruyko & Komodakis 2016, pre-act He 2016) | FIRST wholesale-different backbone (EXP-022). EXP-022 NO-IMPROVEMENT: budget WRN-22-4 (96.31@133ep) & WRN-16-4 (96.34@196ep) both TIE compiled DavidNet c0 96.32 at full anneal; WRN-16-8 under-anneals (68ep). The 97.1% is WRN-28-10@200ep — budget-feasible WRNs (≤4.3M) don't inherit it. Ceiling is backbone-family-INDEPENDENT → limiter is recipe/data/compute, not topology. Do NOT re-test budget WRN. |

## Domain Notes

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add domain-specific notes here)_ | | |
