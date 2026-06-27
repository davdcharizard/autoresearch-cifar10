# Knowledge Base Index

This directory contains persistent external knowledge that informs brainstorming and planning across all research loops. Unlike experiment logs (which are loop-specific), entries here represent standing background context pulled in throughout the research process: relevant paper distillations, reference implementation notes, and domain knowledge.

## How to use
- `research-brainstorm` loads this README as a lightweight index, then loads specific entries on demand
- Entries are added by researchers (manually), by `/lit-search` (automated paper distillations), or proposed by the agent after consulting external sources
- Add entries for any paper, reference implementation, or domain insight that is likely to inform future experiments

## Venues
- `venues.md` (if it exists in this directory) defines which academic proceedings to search via `/lit-search`
- Copy the template from `${CLAUDE_PLUGIN_ROOT}/skills/lit-search/templates/venues-template.md` and edit to match your research domain
- Without a venues file, `/lit-search` defaults to top CS/ML conferences (NeurIPS, ICML, ICLR)

## Papers

| File | Topic | Key Takeaway |
|------|-------|-------------|
| papers/trivialaugment.md | TrivialAugment (arXiv 2103.10158, ICCV 2021) | Tuning-free policy augmentation; composes with occlusion erasing (TA-then-cutout); validated in-project +0.17pp at 114 epochs (EXP-004); PIL-stage op, insert before ToTensor |
| papers/why-warmup-lr.md | Why Warmup the LR? (arXiv 2406.09405, NeurIPS 2024) | Warmup's only first-order role is peak-LR tolerance (1–5% typical) — BUT this fixed-iteration intuition INVERTS under our time-keyed schedule: shorter warmup = hotter everywhere, −0.22pp (EXP-014) |
| papers/optimal-linear-decay.md | Optimal Linear Decay LR Schedules (arXiv 2310.07831; + 2502.15938) | Warmup + linear-to-zero beats cosine across 10 problems, theory-backed; SAME LR-time integral as cosine ⇒ pure shape change that holds the closed heat axis constant (EXP-016 probe) |
| papers/regnet-design-spaces.md | RegNet: Designing Network Design Spaces (arXiv 2003.13678, CVPR 2020) | Optimized design spaces are third-stage-heavy/first-stage-light — but did NOT transfer to depth 20: [2,3,4] lost 0.28pp despite equal FLOPs and more epochs (EXP-017); shallow-extreme allocation is uniform |
| papers/bag-of-tricks-zero-gamma.md | Bag of Tricks zero-γ init (arXiv 1812.01187; + 1706.02677) | Identity-at-init eases early optimization at fixed epochs — but INVERTED under fixed wall clock: blocks turn on during peak heat, −0.99pp (EXP-018); init tricks must ADD early learning, not stability headroom |
| papers/sam-sharpness-aware-minimization.md | SAM (arXiv 2010.01412, ICLR 2021) + LookSAM (2203.02714) | Flat-minima LEVEL gains on CIFAR WRN at fixed epochs; periodic-k retains most gain at ~1.2x cost; BN-freeze on perturbed pass + eager-second-pass compile pattern; project arithmetic in note |
| papers/squeeze-excitation-senet.md | SENet (arXiv 1709.01507, CVPR 2018) + ECA-Net (CVPR 2020) | Channel-attention LEVEL gains on CIFAR ResNets (+0.5-1.2 at fixed epochs); near-identity init pattern (fc2 zero-weight, bias 2.0) vs deferral law; launch-bound dt pricing; project arithmetic in note |
| papers/deep-ensembles-function-space.md | Deep ensembles family (arXiv 1612.01474, 1912.02757, 2005.00570, 2002.06715, 2010.06610) | Function-space (multi-mode) prediction averaging gains where weight-space (SWA/EMA, single-mode) measured zero here; matched-FLOPs small-net ensembles beat single big nets; grouped-conv/alternating-step project arithmetic in note — measured EXP-042 |
| papers/freezeout-brock-2017.md | FreezeOut progressive layer freezing (arXiv 1706.04983) — measured EXP-055 | Fixed-epoch "freeze is free" INVERTS under fixed time: 31% step saving fully recycled into +1,550 tail steps still lost −1.6σ (tail-pressure law is parameter-side too); torch.compile gotcha: mid-run requires_grad flips are silent no-ops — use graph-visible flag + detach + dual-variant warmup |
| papers/dont-decay-lr-increase-batch.md | Smith et al. batch-size-as-LR-decay (arXiv 1711.00489, ICLR 2018) — measured EXP-059, REFUTED | Late 512→1024 step at fixed LR delivered both mechanisms (tail noise halved + ~6% dividend, +2 ep) yet read exact family null — the cosine anneal already saturates late noise reduction; ramp and noise-up mirror inherit the null; multi-shape compile requires dynamic=False + per-shape warmup |
| papers/schedule-free-road-less-scheduled.md | Schedule-Free optimization (arXiv 2405.15682, NeurIPS 2024) — measured EXP-062, REFUTED | Any-horizon claim fails at the 300s/13.5k-step horizon: eval-at-x read 94.87 (−1.84) with a monotone x-curve still climbing at budget end — averaging hot iterates does not reproduce the anneal's basin refinement; exact SGDScheduleFree math + BN-refresh-at-x machinery verified and reusable |
| papers/cutmix-yun-2019.md | CutMix (arXiv 1905.04899, ICCV 2019) — measured EXP-060, REFUTED | Substituted for RE at matched dose p=0.5: precise family null (96.69) at byte-clean signatures — under TA, occlusion is pure information deletion (fill content + label mixing irrelevant); absorption extends to augmentation TYPE; substitution at constant count is safe where stacking (EXP-009 mixup) was over-pressure |
| papers/efficientnetv2-progressive-learning.md | EfficientNetV2 progressive learning (arXiv 2104.00298, ICML 2021) — measured EXP-031+EXP-065, REFUTED both halves | Image-size half zero conversion (EXP-031); reg-ramp half 96.38 = mean −1.2σ (EXP-065) — light warmup trains faster but banked progress inverts at the full-pressure transition; pressure-profile law four-quadrant complete |
| papers/acnet-structural-reparameterization.md | ACNet asymmetric conv blocks (arXiv 1908.03930, ICCV 2019) — measured EXP-064, NO LAUNCH | Family cost-closure at zero charged seconds: full ACB tolls 1.93× dt (1D convs hit slow odd-shape kernels), minimum 1x1-branch variant still 28.6ms vs required-gain arithmetic — "free at inference" inverts under a train-time budget; internal-control probe pattern validated |
| papers/cifar10-label-errors-confident-learning.md | CIFAR-10 label errors / confident learning (arXiv 1911.00068, 2103.14749) — closed EXP-069, NO LAUNCH | Natural error rate ~0.54% confirmed; the "+0.9pp cleaning" figure is a 20–40% ADDED-noise artifact ⇒ ≤0.1pp ceiling at natural rate under LS+TA+RE; regime-check NOISE RATE like augmentation regime before trusting cleaning gains |

## References

| File | Topic | Key Takeaway |
|------|-------|-------------|
| (URL) https://github.com/davidcpage/cifar10-fast | Fast CIFAR-10 training (ResNet-9, OneCycle, batch 512, mixed precision) | Budget-matched one-cycle + large-batch mixed precision dominates step decay in short-budget regimes |
| (URL) https://arxiv.org/abs/2404.00498 + https://github.com/KellerJordan/cifar10-airbench | CIFAR-10 speedrun, 94% in 3.29s (A100) | GPU-resident dataset, channels_last + half precision, derandomized flip, whitening first-layer init — data pipeline dominates cost for tiny models. CAVEAT: whitening init did NOT transfer to our recipe (EXP-019, −0.26pp) — bn1 right after the stem undoes the scale structure, and ~139 epochs learn the basis anyway; it pays only in BN-free-stem, ~10-epoch regimes |
| (URL) https://arxiv.org/abs/1708.07120 | Super-convergence (one-cycle LR) | Large-peak-LR one-cycle schedules reach high accuracy in far fewer iterations than step decay |
| (URL) https://arxiv.org/abs/1605.07146 + https://github.com/szagoruyko/wide-residual-networks | Wide Residual Networks (WRN) | Width beats depth on CIFAR-10: consistent gains at 1-12x width for 16/22/40-layer nets; WRN-40-4 matches ResNet-1001 while training 8x faster. CAVEAT: WRN's projection shortcuts did NOT transfer to our fixed wall clock (EXP-020, −0.13pp) — they cost early heat (learned during peak LR) plus 0.6ms/step (4 epochs); fixed-epoch architecture evidence needs to be free in both early heat AND epochs here |

| references/muon-optimizer.md | Muon optimizer (Jordan; airbench CIFAR-10 record holder) | NS-5 orthogonalized momentum for 2D/conv weights; airbench anchor lr 0.24 / momentum 0.6 nesterov, convs only; ~285 small eager matmuls/step cost — gate-price before use; benefit class = per-step sample efficiency |
| references/progressive-resizing.md | Progressive resizing (fastai/MosaicML) — measured EXP-031 | Early phase at reduced resolution under wall-clock budgets; quality-neutral on ImageNet but ZERO conversion on CIFAR (24px discards signal, not redundancy); dual-shape compile warmup + phase-aware watchdog patterns; wall levers: eval thinning + 2x workers |
| references/swa-stochastic-weight-averaging.md | SWA (Izmailov 2018; torch.optim.swa_utils) — measured EXP-032 | Constant-tail-LR iterate averaging + mandatory augmented-loader update_bn; ZERO accuracy gain under fixed wall clock (SWA phase replaces the anneal instead of extending training); improved loss, unchanged argmax — see EXP-011/029/032 |

## Domain Notes

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add domain-specific notes here)_ | | |
