# Brainstorm EXP-001
**Created**: 2026-06-28

<!-- Goal/metric/constraints live in goals/maximize-cifar10-test-accuracy/01-definition.md; baseline (91.57%) in 04-results.tsv. -->

## Web Search & Literature Review

- **David Page, "How to Train Your ResNet" / cifar10-fast** (DAWNBench winner; davidcpage/cifar10-fast; 99991/cifar10-fast-simple; johanwind "94% in 94 lines/94s" https://johanwind.github.io/2022/12/28/cifar_94.html): a wide-shallow 9-layer residual net ("DavidNet"/ResNet-9, ~6.5M params) trained 24 epochs with **one-cycle (triangular) LR**, SGD+Nesterov, **Cutout 8×8** + pad-4-crop + flip, **label smoothing**, output logits scaled by ~1/8, reaches **~94%** in seconds. Core lesson: a schedule that *completes its anneal* + better architecture is the path from ~91% to ~94%.
- **Keller Jordan, cifar10-airbench / tysam hlb-CIFAR10** (https://github.com/KellerJordan/cifar10-airbench; arXiv:2404.00498 "94% on CIFAR-10 in 3.29s"): record-holding family. Key trick = a **fixed whitening initial conv** (weights from eigendecomposition of ~5000 training patches, 3→24 ch, kernel 2, frozen) → input decorrelation → convergence in ~10–40 epochs. **GELU ConvGroup** blocks (Conv→MaxPool→BN→GELU), flip **TTA inside forward**. Reaches **94% (~10ep), 95% (~15ep), 96% (~37ep)** on A100 in seconds-to-tens-of-seconds.
- **"Bag of Tricks for Image Classification"** (He et al., CVPR 2019, arXiv:1812.01187): cosine-decay-to-zero + warmup, **zero-init residual BN γ**, **label smoothing 0.1**, **no-weight-decay on bias/BN**. Each ~+0.1–0.8pp, additive.
- **"Accurate, Large Minibatch SGD"** (Goyal et al. 2017, arXiv:1706.02677): linear LR scaling + warmup preserves SGD accuracy at larger batch — basis for throughput-via-bigger-batch.
- **PyTorch Performance Tuning Guide / NVIDIA Hopper docs**: bf16 autocast (no GradScaler) + channels_last + cudnn.benchmark + TF32 are the standard CNN throughput stack; H20 has strong bf16. Throughput → more steps in the fixed 300s budget.

Full developed proposals (disposable retrieval): `proposals/idea-01.md` … `idea-04.md`.

## Experimental History Review

First experiment under this goal — no prior experiment history. Baseline established this loop:
- **BASE (commit 1f69af5): 91.57%** — unmodified CIFAR ResNet-20. KEY DIAGNOSIS below.

## Diagnosis — what limits the objective

The metric is `best_test_acc` under a **fixed 300s training-time budget** (not a fixed step/epoch count). The baseline limiter is two-fold and measured:
1. **Schedule never completes.** `train.py` uses `MultiStepLR(milestones=[32000,48000])` over a nominal `MAX_STEPS=64000`, but the baseline run fit only **~37,431 steps / 96 epochs** in 300s. So only the *first* LR drop (32k) ever happens; the final low-LR annealing phase — where CIFAR ResNets gain most of their accuracy — never runs. The model is read **under-annealed**.
2. **Compute massively underutilized.** ResNet-20 is tiny (270k params), fp32, batch 128, ~8 ms/step, using 330 MB of the H20's 98 GB. The whole modern fast-CIFAR toolkit (one-cycle schedule, better architecture, whitening front-end, bf16, augmentation/regularization) is unused. Known recipes reach **94–96%** on a single GPU in seconds-to-minutes, so there is **~3–4.5pp of headroom** under this budget.

The objective is advanced by "genuinely better training code": fix the schedule, raise throughput to fit more effective training, and/or upgrade the architecture — within the only-edit-`train.py`, no-new-deps, no-seed-hacking constraints.

## Collected Ideas

- (schedule/bottleneck) Replace the never-completing MultiStepLR with a one-cycle/cosine schedule sized to the *achievable* step budget.
- (throughput) bf16 autocast + channels_last + cudnn.benchmark (+ optional torch.compile warmed in startup) → more steps in 300s.
- (architecture, literature) Replace ResNet-20 with wide-shallow ResNet-9 / DavidNet (proven ~94% one-cycle recipe).
- (architecture, moonshot) Airbench-style net: frozen whitening initial conv (eigendecomposition init) + GELU ConvGroups (proven 95–96%).
- (regularization) Label smoothing + Cutout/RandomErasing + zero-init residual BN γ + no-decay-on-bias/BN.
- (optimizer) SGD + Nesterov + LR warmup + linear LR scaling for larger batch.
- (eval booster) Horizontal-flip TTA inside `model.forward` (legit: frozen eval calls `model(inputs)` directly).
- (algorithm) Bolt a frozen whitening conv front-end onto the existing ResNet.
- (simplification) Shallower-wider net (fewer stages, more channels) trains faster-per-accuracy under a time budget.

## Combinations

- **ResNet-9 + one-cycle + Cutout + label smoothing + bf16** (idea-02): the canonical fast recipe — each piece addresses a different axis (capacity, schedule completion, generalization, throughput); jointly proven to ~94%.
- **Whitening front-end + GELU ConvGroups + one-cycle + flip-TTA + bf16** (idea-03): whitening accelerates convergence so a higher-capacity net saturates within budget; TTA adds a free eval boost — jointly proven to ~96%.
- **Schedule fix + bf16/channels_last + zero-γ BN + label smoothing + RandomErasing on ResNet-20** (idea-01): keeps the proven-stable architecture, stacks low-risk additive tricks for a high floor.

## Candidate Ideas

### 1. Modernized bag-of-tricks recipe on ResNet-20
**Summary**: Keep ResNet-20; replace the recipe. Primary change: swap MultiStepLR for a **one-cycle/cosine schedule sized to the achievable ~37k+ steps** (runtime-calibrated from measured step time) so the LR fully anneals before the clock ends. Stack additive low-risk tricks: bf16 autocast + channels_last (more steps in budget), SGD+Nesterov, no-decay-on-bias/BN with wd 5e-4, **zero-init residual BN γ**, **label smoothing 0.1**, light **RandomErasing**. (`proposals/idea-01.md`)

**What it targets**: The under-annealing limiter head-on (schedule completes), plus throughput (more steps) and generalization. Lowest-risk, high-floor.

**Reasoning**: Bag of Tricks (arXiv:1812.01187) shows these tricks are individually evidenced and additive; cosine-to-zero is the textbook fix for a schedule frozen mid-high-LR. The schedule fix alone typically buys ResNet ~0.7–1.2pp on CIFAR.

**Sources**: `proposals/idea-01.md`; Bag of Tricks (arXiv:1812.01187); David Page cifar10-fast; OneCycle (Smith & Topin, arXiv:1708.07198).

**Estimated Effort**: low–medium (localized train.py edits, no architecture surgery).

**Risk Assessment**: Schedule horizon mis-estimate (mitigated by 0.97 margin + best-across-epochs scoring); over-regularization on a short schedule (RandomErasing is the droppable trick); high one-cycle peak LR could diverge (cosine+0.1 fallback is near-certain to beat baseline). Worst case: only the schedule fix matters → still a real gain. Expected **~92.3–93.8%**.

### 2. Replace ResNet-20 with a fast-CIFAR ResNet-9 (DavidNet) + one-cycle
**Summary**: Swap the deep-thin ResNet-20 for the wide-shallow 9-layer residual net (DavidNet, ~6.5M params: prep 3→64, stages 64→128→256→512 with MaxPool, residual blocks on the 128 & 512 stages, global max-pool, linear 512→10, logits ×0.125). Train with batch 512, SGD+Nesterov, **triangular one-cycle** (peak LR ~0.4 mean-loss convention), wd 5e-4, **label smoothing 0.2**, **Cutout 8×8** + pad-4-crop + flip, bf16 autocast + channels_last. Size one cycle to the budget (calibrate step time, pick epochs to fill ~280s). (`proposals/idea-02.md`)

**What it targets**: Fixes the under-annealing limiter (a short one-cycle that *completes*) AND upgrades capacity/convergence-per-epoch — the proven path to ~94%.

**Reasoning**: Multiple independent reimplementations hit **94.0–94.3%** with this exact recipe in <94s on weaker GPUs; our 300s H20 budget is far more generous, so the full anneal completes. The proposal pins the subtle LR/loss-reduction/output-scale conventions (mean-loss peak 0.4, wd 5e-4, scale_out 0.125).

**Sources**: `proposals/idea-02.md`; johanwind (94 lines/94s); Myrtle.ai "How to Train Your ResNet" 3 & 5; 99991/cifar10-fast-simple.

**Estimated Effort**: medium (~80–100 line rewrite of model+optimizer+schedule+aug; loop scaffold/timing/eval reused).

**Risk Assessment**: Main risk is faithful porting of LR/loss-reduction/output-scale (mis-set → divergence, detectable in epoch 1; safe LR sweep {0.2,0.4,0.6}); OneCycle step-count overrun (guard `if step<total_sched_steps`); budget under-use if cycle too short (size to budget). Method risk low (very well replicated). Expected **~93.5–94.3%** (central ~93.7%).

### 3. Airbench-style whitening + GELU-ConvGroup net (path to 95–96%)
**Summary**: Replace ResNet-20 with the airbench/hlb family: a **frozen whitening initial conv** (3→24 ch, kernel 2, weights = eigendecomposition of ~5000 normalized training patches via `torch.linalg.eigh`, `cat(scaled,−scaled)`, `requires_grad=False`) → **GELU ConvGroups** (Conv→MaxPool→BN→GELU; widths 64/256/256) → flatten → bias-free linear, logit scaling, **label smoothing 0.2**, SGD+Nesterov triangular one-cycle, bf16 + channels_last, flip **TTA inside forward** (gated on `not self.training`). Whitening + normalization computed in the *exact* frozen-eval space (mean=(0.4914,0.4822,0.4465), std=(1,1,1)). Get the 94-net clean first; flip to the residual "96-net" if budget remains. (`proposals/idea-03.md`)

**What it targets**: The slow-convergence limiter — whitening removes low-level decorrelation work so loss drops in epoch 1 and a higher-capacity net saturates within budget. Only candidate with documented headroom to ~96%.

**Reasoning**: Airbench reaches 94% (~10ep)/95% (~15ep)/96% (~37ep) on A100 in seconds-to-tens-of-seconds; our 300s H20 budget projects to fit even the 96-net. Whitening init is the documented load-bearing accelerator. Whitening math verified implementable in pure torch (`torch.linalg.eigh`), no new deps.

**Sources**: `proposals/idea-03.md`; KellerJordan/cifar10-airbench; tysam hlb-CIFAR10; arXiv:2404.00498.

**Estimated Effort**: high (full model+loop rewrite: whitening setup pass, new modules, param grouping, budget-aware schedule, TTA gate; many details independently break the run).

**Risk Assessment**: Highest implementation risk — schedule/horizon mismatch, **train/whitening/eval normalization desync** (silently tanks accuracy), high-LR divergence if a recipe piece is dropped, H20 slower than projected (then only 94-net fits — still beats baseline). Worst case: a subtly-broken port trains <91.57% / crashes, burning the loop. Floor of a correct port **~92.5%**; conservative 94-net+TTA **~93–94%**; full 96-net **~95–96%**.

## Review

Cross-model adversarial review by **Codex** (full text: `01-idea-review.md`). Scored verdict: Idea 02 evidence **9/10**, impact **8/10** (the winner); Idea 03 impact 9.5/10 but evidence 7.5/10 (recipe-fidelity traps, ambiguous logit scaling) → best *follow-up*; Idea 01 evidence 7/10, impact 5.5/10 (ceiling-limited) → fallback. **Pick: Idea 02** — "combines high upside with the cleanest path to a correct implementation under the constraints."

Top concerns and how they are resolved in the chosen idea:
1. **Calibration must not do off-budget training updates** (would violate the 300s training-time constraint, since `train.py` only starts the budget meter inside the step loop). → **Resolved by a time-based one-cycle schedule**: set LR from `progress = total_training_time / TIME_BUDGET_S` (a quantity the loop already tracks), not from a pre-counted step total. No step calibration, no `OneCycleLR` total_steps overrun, and the anneal is *guaranteed* to complete exactly at the budget regardless of throughput. This is strictly cleaner than the proposal's runtime-calibration sketch.
2. **Normalization consistency with frozen eval** → centralize `mean=(0.4914,0.4822,0.4465), std=(1,1,1)` in one constant used by the train transform; Cutout fills 0.0 (= dataset mean post-subtraction). Assert it matches `prepare.py`.
3. **Don't blindly stretch one cycle to fill 300s / keep LR high too long** → time-based one-cycle with a modest warmup fraction (`pct_start≈0.15`) fully anneals by 300s by construction; `best_test_acc` (best across epochs) captures the peak. Filling the budget with one *completing* cycle is the intended design.
4. **LR/loss-reduction/output-scale convention is the main hazard** → pin together: **mean** reduction, `PEAK_LR=0.4`, `weight_decay=5e-4`, `scale_out=0.125`, `label_smoothing=0.2`. Watch epoch-1 loss for divergence; safe LR sweep {0.2,0.4,0.6} if needed.

## Idea Evaluation

Adopting the reviewer's pick (Idea 02) without override. It directly fixes the diagnosed under-annealing limiter (a one-cycle that *completes* within budget) while upgrading capacity/convergence-per-epoch via a fast-CIFAR architecture that multiple independent implementations replicate at 94.0–94.3%. Idea 03's higher ceiling is real but carries recipe-fidelity/normalization-desync traps better tackled as a follow-up once a stable fast-CIFAR rewrite of `train.py` exists; Idea 01 is the lower-ceiling fallback whose best part (the schedule fix) is already subsumed by Idea 02's completing one-cycle. Scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: ResNet-9 (DavidNet) + time-based one-cycle (Idea 02, `proposals/idea-02.md`), refined per review.

**Why this idea**:
Highest expected value with the cleanest correct path under the hard constraints. It attacks the diagnosed limiter on two axes at once — (a) a one-cycle LR that actually *completes* its anneal within 300s (the baseline's milestone schedule never reaches its 2nd drop), and (b) a wide-shallow architecture with far higher convergence-per-epoch than ResNet-20 — and the recipe is replicated to 94.0–94.3% across multiple independent implementations on weaker GPUs with a much tighter wall-clock than our 300s H20 budget. All changes live in `train.py`, use only torch/torchvision, and respect the no-seed-hacking / ≤1-val-per-epoch / frozen-eval rules. The review's one feasibility risk (off-budget calibration) is engineered away by keying the schedule to elapsed training time.

**Hypothesis**:
Replacing ResNet-20 + MultiStepLR with the DavidNet/ResNet-9 architecture trained under a **time-based one-cycle** schedule (peak LR 0.4 mean-loss, SGD+Nesterov, wd 5e-4, label smoothing 0.2, Cutout 8×8 + pad-4-crop + flip, bf16 autocast + channels_last), all within the fixed 300s training budget on one H20, will raise `best_test_acc` from the 91.57% baseline to **~93.5–94.3%** — clearing the +0.1pp improvement bar with large margin. Falsified if the run crashes, diverges (epoch-1 loss explosion), or `best_test_acc` ≤ 91.67%.
