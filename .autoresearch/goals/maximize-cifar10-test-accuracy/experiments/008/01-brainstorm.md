# Brainstorm EXP-008
**Created**: 2026-06-28

<!-- Goal/metric/constraints/baseline live in goals/maximize-cifar10-test-accuracy/01-definition.md + 04-results.tsv. Baseline = 96.00% (EXP-004). Bar = ≥96.10% AND clearly above the ~0.1pp noise floor. -->

## Web Search & Literature Review

- **airbench / cifar10-airbench (Keller Jordan, arXiv:2404.00498)** (https://github.com/KellerJordan/cifar10-airbench): reaches **96.03% in ~37 epochs / 27–35 A100-s**. Architecture is now ~10 conv layers with residuals over the last two convs of each block (≈ our structure already). Augmentation = flip + translate4 + **cutout12** (we run cutout8). Newest records use the **Muon optimizer** and **per-update conv weight renormalization** (high implementation risk). FLOP-reducing tweaks (3×3→2×2 first conv) do NOT improve wallclock (corroborates EXP-005). → **Key takeaway: airbench gets the SAME ~96% in ~4× FEWER epochs than us → we have a large EPOCH SURPLUS on a saturated/regularization-bound net.**
- **torch.compile for fast CIFAR** (arXiv:2404.00498 §speed; PyTorch tuning guide): in the controlled airbench benchmark torch.compile gives only **~14% throughput**, with **multi-minute compile overhead**. On a saturated net 14% more epochs ≈ ~0 accuracy, and multi-minute compile risks the 10-min wall cap (our total wall is already ~440s). → **Deprioritized as a standalone lever.**
- **"Bag of Tricks for Image Classification" (He et al., CVPR 2019, arXiv:1812.01187)** (cited in `knowledge/references/fast-cifar10-recipes.md:27`): "No bias decay" — exclude BN γ/β and biases from weight decay; isolated effect ~0.1–0.3pp, one of the more reliable "free" tricks. We currently apply wd=5e-4 uniformly to ALL params.
- **"Random Erasing Data Augmentation" (Zhong et al., AAAI 2020)**: label-preserving regularizer, first-class in `torchvision.transforms` (no new dep); pairs with Cutout to add regularization on a net with epoch surplus.

## Experimental History Review

- **What's been tried**: 001 DavidNet+one-cycle (95.22, base recipe) → 002 EMA+flip-TTA (95.72) → 003 frozen ZCA whitening conv (95.87) → 004 ReZero Residual block @layer2/8×8 (**96.00, current best**) → 005 2nd ReZero @layer3/4×4 (95.90, no-imp) → 006 multi-crop TTA (95.93, no-imp) → 007 widen layer2 256→384 (95.85, no-imp).
- **What worked**: capacity at the proven full-speed **8×8 stage** (EXP-004, +0.13pp), eval-side EMA+TTA (EXP-002), whitening front-end (EXP-003). All composed into the 96.00 recipe.
- **What didn't (approach-specific)**: (a) **capacity that costs too many epochs UNDER-ANNEALS** — EXP-005 (4×4 deepen, −11 epochs, also cuDNN-slow) and EXP-007 (384 widen, −48 epochs, still climbing at budget end) both lost purely to lost epochs, NOT to capacity being useless (count:2, Medium). (b) **eval-side TTA is near-exhausted** vs the noise floor (EXP-006). 
- **Key constraint — the ~0.1pp NOISE FLOOR** (High): the time-budgeted loop fits a host-throughput-dependent epoch count (142/150/131/94 across runs), so best_test_acc varies ±~0.1pp with the seed fixed. The +0.1pp bar sits AT the floor → only changes with clearly >0.1pp headroom register; seed is fixed (no re-rolling).
- **What hasn't been tried**: any **throughput-free regularization/recipe lever** to exploit the epoch surplus — augmentation strength is untouched since EXP-001 (still cutout8); weight decay is applied uniformly (no BN/bias/α decoupling); activation is still ReLU (lineage uses GELU). The whole "use the wasted epochs via stronger regularization" axis is unexplored.

## Collected Ideas

1. (History / bottleneck) **Milder widen layer2 256→320** — the pre-registered EXP-007 fallback; half the capacity at ~half the throughput cost, aiming for ~120–135 epochs.
2. (Literature / Bag-of-Tricks) **Decoupled weight decay** — split optimizer into two param groups; wd=5e-4 on conv/fc weights, wd=0 on BN γ/β and the ReZero α scalar. Throughput-free.
3. (Orthogonal lever / regularization) **Stronger augmentation** — Cutout 8→12 (airbench value) + light RandomErasing, to spend the epoch surplus on a higher annealed ceiling. CPU-side → throughput-free.
4. (Algorithm / representation) **GELU activations** replacing ReLU (hlb/airbench lineage). Throughput-near-neutral, modest expected effect.
5. (Simplification) **Reduce label smoothing 0.2→0.1** — a saturated net may be over-smoothed; sweep-like single-knob change.
6. (Moonshot) **Muon optimizer / per-update conv weight renormalization** (airbench newest record) — high implementation + correctness risk in one loop.
7. (Bottleneck enabler) **torch.compile with off-budget warmup** to unlock throughput → enable capacity. Deprioritized: only ~14% gain + multi-minute compile risks the 10-min wall cap.
8. (Capacity, cheaper placement) **2nd ReZero block at layer2/8×8** (depth at the proven stage, ~1.18M params) — cheaper than the 384 widen.

## Combinations

- **#2 + #3 (decoupled WD + stronger aug)**: both throughput-free regularization-regime levers, orthogonal mechanisms (penalty redistribution vs input-space augmentation); stacking them is more likely to clear the >0.1pp bar together than either marginal change alone — though at the cost of single-variable attribution.
- **#1 + #7 (320 widen + torch.compile)**: compile's throughput unlock could let the wider net fit enough epochs to anneal; dominated because compile's 14% is too small to rescue a 1.25× widen and adds wall-cap risk.
- **#4 + #2 (GELU + decoupled WD)**: recipe-alignment + penalty fix; both sub-noise individually, attribution muddied — deferred.

## Candidate Ideas

### 1. Decoupled weight decay (no-decay on BN γ/β + ReZero α)
**Summary**: Split the single SGD param list into two groups — `weight_decay=5e-4` for multi-dim weight matrices (conv 4-D, fc 2-D) and `weight_decay=0.0` for 1-D params (BatchNorm affine γ/β and the ReZero `alpha` scalar). One ~8-line edit at the optimizer construction (`train.py:243-249`); the existing LR-schedule loop already writes LR to every param group so no other change is needed. Hold PEAK_LR=0.4 / wd=5e-4 for clean single-variable attribution. Net is byte-identical to EXP-004 otherwise. (proposals/idea-01.md)

**What it targets**: The diagnosed **regularization ceiling on a saturated, epoch-surplus net**. Decaying BN γ is a spurious penalty (BN already controls activation scale), and decaying the ReZero α actively fights the capacity ramp that delivered EXP-004's +0.13pp (α.grad≈0.0179, so a lr·wd·α term is non-negligible). Removing both redistributes the L2 penalty to where it controls real complexity (conv/fc weights), shifting the fully-annealed optimum where ~96.0 is set.

**Reasoning**: Standard, well-evidenced practice (Bag of Tricks §4; fastai/timm defaults) with isolated effect ~0.1–0.3pp on conv nets. **Zero throughput cost** → expected num_epochs unchanged (~142–150) → cannot lose by under-annealing (the EXP-005/007 failure mode), giving the best downside asymmetry in the candidate set. The α-decoupling is a second, net-specific contributor not in the generic accounting.

**Sources**: proposals/idea-01.md; arXiv:1812.01187 (Bag of Tricks); `knowledge/references/fast-cifar10-recipes.md:27`; EXP-004 analysis (α is live, accuracy-bearing); `03-experiment-learnings.md` (under-anneal Failed Approaches; noise-floor Protocol Finding).

**Estimated Effort**: low (one ~8-line edit, one 300s run + smoke).

**Risk Assessment**: The dominant risk is **sub-noise**: on a net already at 96% with LS 0.2 + Cutout + EMA, the marginal regularization correction may be only a few hundredths, landing in the [95.90, 96.05] null band against a bar that sits at the noise floor. Theoretically sound so unlikely to *hurt* (worst realistic case ≈ baseline, not a regression). Upside is uncertain, not the downside.

### 2. Stronger augmentation — Cutout 8→12 + light RandomErasing
**Summary**: Strengthen train-time augmentation from `Cutout(8)` to `Cutout(12)` (the airbench96 value) plus a light `transforms.RandomErasing(p=0.25, scale=(0.02,0.15), value=0.0)` appended after Cutout, all else byte-identical. Two lines in the `train_tf` pipeline (`train.py:205-213`); RandomErasing is a torchvision built-in (no new dep) operating on the post-Normalize tensor with value=0.0 = mean-fill (matching the existing Cutout-with-mean convention and the std=1 eval space). (proposals/idea-02.md)

**What it targets**: The diagnosed **saturation / epoch surplus** head-on. The net wastes ~100 epochs re-fitting an already-fit train set (overfitting); stronger augmentation makes each epoch harder, slowing *convergence* (consuming the surplus) so the fully-annealed minimum generalizes better. Crucially augmentation runs on the **CPU DataLoader workers** (8 workers, persistent, prefetch×4) in parallel with the GPU → it slows convergence WITHOUT slowing throughput, sidestepping the EXP-005/007 under-anneal trap (which slowed the GPU step).

**Reasoning**: cutout12 is the documented airbench96 value (we under-shoot only on this axis; we already match LS 0.2 and crop pad 4). RandomErasing (Zhong et al. AAAI 2020) is a peer-reviewed label-preserving regularizer. The mechanism is the textbook "convert surplus epochs into accuracy via stronger regularization," and it is matched to the one property that distinguishes it from the failed capacity adds — throughput preservation (expect num_epochs ~142–150).

**Sources**: proposals/idea-02.md; airbench96 (arXiv:2404.00498, cutout12); Zhong et al. AAAI 2020 (Random Erasing); `03-experiment-learnings.md` (epoch surplus, under-anneal Failed Approaches, noise floor).

**Estimated Effort**: low (two transform lines, one run).

**Risk Assessment**: Cutout12 alone is likely sub-noise (airbench's 96.03 vs our 96.00 is +0.03pp and confounded by width/GELU); the **combination** is the bet to clear >0.1pp — honestly marginal-headroom. Secondary risk: if the combined erasing is too strong for ~142 epochs the net mildly under-*fits* (not under-anneals, since throughput holds) → gain cancels; mitigated by light settings and observable via the mid-trajectory (ep25 should stay near ~92%). Worst case ≈ small loss, not a large regression.

### 3. Milder widen layer2 (8×8 stage) 256→320
**Summary**: Widen the proven 8×8 middle stage `layer2` 256→320 — the explicitly pre-registered EXP-007 fallback. Edits exactly two lines (`train.py:150-151`): `conv_bn(128,256)→conv_bn(128,320)`, `GatedResidual(256)→GatedResidual(320)`, and the layer3 stem `conv_bn(256,512)→conv_bn(320,512)`. layer3 output stays 512 → pool/fc untouched. New `num_params == 8,817,203` (+1.03M over baseline, hand-computed; method reproduces EXP-007's 9,997,235 exactly). Hold PEAK_LR=0.4. (proposals/idea-03.md)

**What it targets**: The **capacity/epoch balance** named as the binding constraint by EXP-007. EXP-004 proved 8×8 capacity is still a live lever (+0.13pp); EXP-007's 384 widen under-annealed (94 epochs, still climbing). 320 adds ~half the capacity at ~56% of the 384 widen's incremental compute (channel² scaling of the dominant 8×8 convs) → projected ~115–135 epochs, plausibly enough to anneal.

**Reasoning**: The lowest-regret way to *resolve the EXP-007 ambiguity* (was the loss capacity or epochs?). num_epochs makes even a null informative: ≤110 climbing = under-anneal again (try cheaper capacity); ≥125 flat = capacity-saturated at fixed recipe (abandon capacity scaling).

**Sources**: proposals/idea-03.md; EXP-004 (+0.13pp capacity), EXP-007 analysis (pre-registered 320 fallback, §Next Steps #1); `03-experiment-learnings.md` (under-anneal mechanism, FLOP≠wallclock).

**Estimated Effort**: low (two-token edit on two lines, reuses EXP-007 protocol).

**Risk Assessment**: The author's own honest read is **<50% to clearly clear 96.10**. Two failure modes both land sub-bar: (a) under-anneal recurs if the throughput hit + shared-host contention exceed the estimate (epochs ≤110); (b) saturation — half of EXP-004's +0.13pp ≈ +0.06pp, INSIDE the noise floor, if 142 epochs already saturates the recipe. Unlike ideas 1–2 it carries genuine under-anneal downside (could regress like EXP-007). num_epochs (a host-noisy diagnostic) must be read with the tail shape.

## Review

Cross-model (Codex) adversarial review in `01-idea-review.md`. Scored verdict: Idea 1 (evidence 7/impact 4), **Idea 2 (evidence 6.5/impact 6.5)**, Idea 3 (evidence 6/impact 7). **Pick: Idea 2 — stronger augmentation.** No hard-constraint violations in any idea. Top concerns + resolutions folded into the chosen idea / plan:

1. **"Don't accept sub-noise wins"** (≥96.10 AND clearly above the ~0.1pp floor) → carried verbatim into verification; a 96.03–96.09 single-run result is a NULL.
2. **Idea 2's "epoch surplus = overfitting" is inferred, not measured** (current logs don't show the train/test gap); the concrete failure is that cutout12 + RandomErasing + LS 0.2 UNDER-FITS within the fixed time-anneal → **resolution: pre-register the mid-trajectory check** (ep25/ep50 vs EXP-004's ~92% @ep25; a collapse + still-climbing tail = underfit signal → the plan's fallback is to drop RandomErasing / lower cutout to 10).
3. **Idea 2's "throughput-free" is a hypothesis, not a fact** (RandomErasing is still per-sample worker compute) → **resolution: num_epochs is a first-class diagnostic**; if it falls materially below the EXP-004/006 142–150 band the result is confounded (worker-saturation, not pure regularization).
4. Idea 1's α-decoupling argument is overstated (SGD decay on α ∝ α, not a constant force) and its headroom is the weakest → not chosen, but kept as a low-regret future rider. Idea 3 has the highest ceiling but, after two under-anneal failures, is the worst bet to actually clear ≥96.10 in one run → deferred (still the pre-registered capacity probe if regularization stalls).

## Idea Evaluation

Adopt the reviewer's pick (Idea 2) without override. The verdict aligns with the diagnosis: of the three, Idea 2 is the only one that both (a) targets the named limiter (regularization ceiling on a saturated, epoch-surplus net) and (b) carries no under-anneal downside (throughput preserved on CPU workers), giving the best risk-adjusted chance to clear the +0.1pp bar in a single seed-fixed run. Idea 1 shares the zero-downside property but has clearly the lowest ceiling (reviewer impact 4/10). Idea 3 has the highest ceiling but the highest regression risk and revisits the EXP-005/007 under-anneal family. Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: Stronger augmentation — Cutout 8→12 + light RandomErasing (Idea 2, proposals/idea-02.md)

**Why this idea**:
It is the only finalist that attacks the diagnosed limiter — a **saturated / regularization-bound net with a large (~4×) epoch surplus** — with a mechanism matched to that limiter (stronger input-space regularization converts wasted epochs into a higher annealed generalization ceiling) AND a throughput profile that avoids the recurring failure mode. Augmentation runs on the 8 CPU DataLoader workers in parallel with the GPU step, so unlike the EXP-005/007 capacity adds it does not cut the epoch count and cannot lose by under-annealing. cutout12 is the documented airbench96 value (the single augmentation axis where we under-shoot the reference), and RandomErasing (Zhong et al., AAAI 2020) is a peer-reviewed, label-preserving, torchvision-native regularizer (no new dependency). The combination — not either tweak alone — is the bet to clear the noise floor.

**Hypothesis**:
Replacing `Cutout(8)` with `Cutout(12)` plus a light `RandomErasing(p=0.25, scale=(0.02,0.15), value=0.0)`, all else byte-identical to the EXP-004 recipe, raises `best_test_acc` to **≥96.10%** and clearly above the ~0.1pp noise floor. Concretely: (i) `num_epochs` stays in the ~142–150 band (throughput preserved — the key prediction distinguishing this from under-anneal failures); (ii) the early/mid trajectory sits at or modestly below EXP-004's (ep25 near ~92%, reflecting harder-but-not-broken augmentation), with the fully-annealed low-LR tail closing the gap and finishing higher. Falsified if best <96.10 with normal epochs and a flat tail (augmentation sub-noise / net already optimally regularized at cutout8), or if `best==final` with a still-rising tail and ep25 well below ~92% (under-fit from too-strong aug → back off RandomErasing).
