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
| papers/trivialaugment.md | Data augmentation (TrivialAugment, ICCV 2021) | Parameter-free strong auto-aug; TA+Cutout lifted CIFAR-10 k=4 WRN 96.00→96.22 (EXP-012); orthogonal to occlusion/interpolation. |
| papers/cutmix.md | Data augmentation (CutMix, ICCV 2019) | Regional label-mixing aug (paste box + area-weighted label mix); GPU-vectorizable/throughput-neutral; wants long schedules (200-300 ep) so may underfit at our ~84-91 ep; judge on acc not loss (EXP-018). |
| papers/swa.md | Stochastic Weight Averaging (SWA, UAI 2018) | Average SGD trajectory points under a CONSTANT/cyclic LR floor → flatter, better-generalizing optimum (~0.5–1.3pp on CIFAR WRN); needs a terminal-LR floor (cosine-to-0 makes it a no-op, cf. EXP-006) + BN-recompute; core torch.optim.swa_utils, throughput-neutral (EXP-019). |
| papers/wrn-dropout.md | Wide ResNets — in-block dropout (WRN, BMVC 2016) | Our k=4 net is a WRN; the paper recommends dropout BETWEEN the two block convs (after first ReLU, before conv2), plain nn.Dropout, p≈0.3 for 200-ep schedules; param-free fair test, but under-fit risk at our ~92-ep budget → probe mild p first (EXP-022). |
| papers/blurpool.md | Anti-aliased downsampling (BlurPool, ICML 2019) | Blur (fixed binomial, param-free depthwise) before every stride-2 subsample → shift-invariance/generalization; but moving the downsample conv to stride-1 ~4×'s its FLOPs at the 2 heaviest convs → epoch-wall risk at 300s (k=4 is launch-bound so maybe absorbed); CHECK epoch count (EXP-024). |
| papers/large-batch-scaling.md | Large-batch training (Smith ICLR 2018; Goyal 2017) | Batch↑ ≡ LR-decay: same accuracy at same epochs with fewer updates; linear LR scaling + warmup. Corollary: under a compute-`dt`-gated budget on a launch-bound net, batch↑ buys MORE effective epochs (free) iff dt stays flat & recipe is epoch-hungry. FALSIFIED here — b256 is compute-bound, collapsed updates, −2.38pp (EXP-025). |
| papers/bag-of-tricks.md | Bag of Tricks (He et al., CVPR 2019) | Free compute-neutral convergence tricks: zero-init residual γ (last BN γ=0 → block starts as identity) and no-bias-decay (WD only on conv/linear weights, not BN/bias). Right class for our convergence-bound recipe (no epoch-wall risk); magnitude may be small on shallow ResNet-20 (EXP-026). Also covers ResNet-D downsample — REGRESSED (EXP-027). |
| papers/smooth-activations.md | Smooth activations (Swish/SiLU 2017, Mish 2019) | Drop-in ReLU→SiLU (`x·σ(x)`) / Mish (`x·tanh(softplus(x))`): smooth non-monotonic, removes dead-ReLU zero region; small consistent ResNet/CIFAR top-1 gains. Pointwise → fuses under torch.compile so ~throughput-neutral at launch-bound (sidesteps epoch wall). NULL here both recipes (SiLU EXP-010/028); also cost real dt (didn't fully fuse). Activation axis CLOSED. |
| papers/sgdr-warm-restarts.md | SGDR warm restarts (Loshchilov & Hutter, ICLR 2017) | Cosine annealing with LR restarts to PEAK each cycle; re-exploration + snapshot-ensemble benefits need LONG schedules. REGRESSED here (EXP-029, 2-cycle, −0.67pp, fair 91-ep test): short budget → each cycle under-converges, single full-budget cosine-to-0 is optimal. LR-schedule axis (peak+floor+shape) fully CLOSED. |
| papers/gradient-centralization.md | Gradient Centralization (Yong et al., ECCV 2020) | Centralize each ndim>1 weight grad (per-output-unit mean over fan-in) between backward & step; constrains weight space + standardizes gradient → claims to BOTH accelerate convergence AND regularize. Opens the optimizer/gradient-dynamics axis. STRONGEST NEAR-MISS (EXP-030): tied baseline + improved loss DESPITE a 3-epoch handicap from the un-fused Python loop. LEAD: vectorize via `torch._foreach_` to test throughput-neutral (EXP-031). |
| papers/sam.md | Sharpness-Aware Minimization (SAM, Foret et al. ICLR 2021) | Minimize worst-case loss in an ε-ball → flat minima → genuine generalization gain (NOT polish); 2 fwd-bwd/step (~2×), ρ≈0.05; sparse/LookSAM variants cut cost. NEGATIVE here (EXP-036): even sparse (every-5th-step) cost 1.27× → 76 ep → 95.89 (−0.33pp); flat-minima benefit doesn't transfer to shallow ResNet-20 at 300s + epoch-wall cost. Optimizer-objective axis CLOSED. |
| papers/polyloss.md | PolyLoss (Leng et al., ICLR 2022) | Poly-1 reshapes CE by adding ε·(1−p_t): ε>0 amplifies hard-example gradients (convergence accelerator), compute-free/convergence-neutral, composes with label smoothing (may partially cancel); ε is dataset-tuned (ImageNet ResNet ≈+1..+2), no published CIFAR value (EXP-041). |
| papers/deep-supervision.md | Deep supervision / aux classifiers (Lee et al. 2015; GoogLeNet 2015) | Auxiliary intermediate-layer CE loss (discarded at inference); benefit is DEPTH-driven (signal propagation in very deep nets). NULL→regression on the shallow 9-block net: −0.31pp, throughput-neutral (EXP-042). Intermediate-feature-routing family CLOSED (with EXP-032). torch.compile gotcha: keep a stable forward output structure or CUDA graphs break → dt doubles. |
| papers/adamw.md | AdamW — decoupled weight decay (Loshchilov & Hutter, ICLR 2019) | Decoupled wd fixes Adam's generalization gap; lr~1e-3–3e-3, wd~0.05 from scratch. On this net: stable + throughput-neutral (8ms/91ep) but REGRESSED −0.35pp (95.87) via the adaptive generalization gap — tuned SGD+Nesterov wins (EXP-043). Optimizer axis (family + grad/objective mods) fully CLOSED. |
| papers/ghost-batchnorm.md | Ghost BatchNorm (Hoffer et al., NeurIPS 2017) | BN stats over small ghost sub-batches → noise → implicit regularizer (strongest at large batch; milder at batch 128). The one untouched accuracy axis (normalization). dt-safe IFF static shapes + split OUTER batch dim only (channels_last view trap) + in-place running-stat update; eval = standard BN. Used EXP-047. |
| papers/gridmask.md | GridMask (Chen et al., 2020) | Delete a regular GRID of squares (vs Cutout's one hole / Random-Erasing's scatter); reported to beat Cutout on CIFAR. GPU-vectorizable like `cutout_batch` (coord-grid mask, dt-neutral). Match removed-area to Cutout (~25%, side=0.5·d) to isolate PATTERN from strength on this saturated recipe. Occlusion-pattern = the one untested aug sub-lever. Used EXP-048. |

## References

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add reference notes here)_ | | |

## Domain Notes

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add domain-specific notes here)_ | | |
