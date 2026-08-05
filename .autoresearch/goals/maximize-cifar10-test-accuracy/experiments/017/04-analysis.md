# Report EXP-017: Throughput-free BN-affine noise (cheap GhostBN surrogate, layer3-first)
- **Created**: 2026-06-30

## Goal
Maximize CIFAR-10 `best_test_acc` (%) within the fixed 300s training budget, editing only `train.py`. Higher is better. Baseline = **96.38** (EXP-008, commit 07c3760). Improvement bar = ≥96.48 (baseline + 0.1pp) AND clearly above the same-session control beyond the ~0.1–0.2pp noise floor. This experiment tested whether the BN-statistic-noise regularization that gave EXP-016's only positive signal (+0.24pp over its same-session control) clears the bar when injected **throughput-free** at full epochs — the decisive run EXP-016 could not perform because GhostBN's grouping broke the fused BN kernel (~50% slower → under-anneal).

## Idea & Hypothesis
**Chosen idea (idea-01):** wrap the BN in `conv_bn` with a `NoisyBN` that runs the standard fused full-batch BN, then — during training only — perturbs the *normalized* activation per-(sample,channel) with Gaussian noise, a direct elementwise surrogate for GhostBN's per-sub-batch normalization jitter. The mechanism reasoning: normalizing a sample by a noisy sub-batch's (μ̃, σ̃) instead of the full-batch (μ, σ) is algebraically a per-channel multiplicative + additive jitter on the BN output; `NoisyBN` injects that jitter directly while keeping the fused channels_last kernel → ~150 epochs, no under-anneal. This decouples the regularization (which showed signal) from the throughput tax (which capped it).

**Hypothesis:** per-(sample,channel) BN-affine Gaussian noise at the calibrated layer3 ghost-equivalent σ, composed with the existing EMA, lifts `best_test_acc` ≥96.48 over the same-session control. If it ties at healthy epochs, BN/activation-statistic noise is redundant with the existing stack at full epochs → EXP-016's +0.24pp was a low-c0-draw artifact, hardening the backbone-pivot mandate.

## Approach
Single editable file, `train.py`:
- Added `NoisyBN(nn.BatchNorm2d)` (placed before `conv_bn`). Train-only forward runs the fused full-batch BN (`super().forward`, eval-safe, buffers noise-free), then applies the **division-free, β-untouched** perturbation `y' = y + (y−β)·σε_mul + γ·σε_add` with per-(sample,channel) noise of shape `[N,C,1,1]`. Eval path is bit-identical to `nn.BatchNorm2d`; `σ=0` ⇒ exact baseline. `mul`/`add` toggles allow per-component noise.
- Wired into `conv_bn` (10 BN sites), gated by env `BN_NOISE` (σ, default 0) and `BN_NOISE_MIN_CH` (apply only where `num_features ≥` this; 512 = layer3-only, the EXP-016 site).
- **σ calibration (not the brainstorm's 0.1–0.2):** a smoke measured the GhostBN(g=128)-equivalent per-channel stat noise at the 3 layer3 sites over 6 real CIFAR batches, using the corrected multiplicative form `σ_full/σ_ghost−1`: σ_add*≈0.025, σ_mul*≈0.033 → **σ_cal=0.033**. Cells: c0 (σ=0, full-speed control), cA (σ=0.033, 1× ghost-equivalent — PRIMARY), cB (σ=0.083, ~2.5×), cA/cB both layer3-only.
- **Deviation from the "throughput-free" premise:** the M2 probe showed layer3-only noise is **+7.2% slower** (the elementwise ops live in the autograd graph; backward must traverse them — not the RNG draws). mul-only saved only 1.3%, so the faithful mul+add form was kept and the ~140-epoch run (vs c0's 149) accepted as a mild, non-confounding gap (per EXP-014, ±10ep near 150 ≈ 0.02–0.03pp; and the gap works *against* the noise cells, so a real benefit could not be hidden by it).

All M1 smokes passed (σ=0 train+eval ≡ nn.BatchNorm2d at 0.0 diff; noise zero-mean within sampling z<6; per-layer EMA buffer math; gradient finiteness on trainable params only; exactly 3 layer3 sites active at MIN_CH=512).

## Execution
Three sequential same-session `train.py` processes on GPU 1 (`CUDA_VISIBLE_DEVICES=1`, `timeout 600`), each writing its own log + nvidia-smi log. All completed cleanly; no retries, no contention (GPU 1 idle 0–3% throughout; num_epochs 149/140/140 matched the M2 +7% prediction exactly). The two test-harness wrinkles during smoke design (a max-over-samples assertion that tripped on expected sampling noise; the brainstorm's σ being 3–6× too strong) were resolved in planning/calibration before the real runs — the runs themselves were uneventful.

## Results
- **Primary metric**: 96.23 (baseline: 96.38, delta: **−0.15**, −0.16%). Best cell = cB; cA tied the control exactly.
- **Cells**: c0 96.14 (final 96.00 @ep149); cA 96.14 (final 96.09 @ep140); cB 96.23 (final 96.13 @ep140). ep25: c0 92.43 / cA 92.43 / cB 92.23 (cB's mild early dip from stronger noise fully recovers). All fully annealed (best≈final).
- **Observations**:
  - **Decisive null.** The calibrated ghost-equivalent noise (cA, σ=0.033) tied the control **exactly** — 96.14 = 96.14 — at near-full epochs (140 vs 149). The throughput-free reproduction of EXP-016's mechanism shows **no benefit** at the ghost-equivalent magnitude.
  - 2.5× noise (cB, σ=0.083) reached 96.23 = **only +0.09pp** over c0, within the ~0.2pp session noise floor and far below the 96.48 bar — a hint that supra-ghost noise does something tiny, but not a win.
  - **This session's c0 = 96.14, exactly EXP-016's c0 (96.14).** The same-session standard-BN control reproducibly draws ~96.14 on this host, which means **EXP-016's "+0.24pp" was a weak-control-draw artifact**, not a real BN-noise benefit. cB even edged c0 *despite* 9 fewer epochs, confirming the epoch gap did not confound.
- **Analysis**: NoisyBN achieved its intended local effect (calibrated per-(sample,channel) jitter at the proven layer3 site, full-speed kernel, clean anneal, no under-fit) yet moved the metric by nothing at the ghost-equivalent magnitude and only +0.09pp at 2.5×. The intervention worked; the *axis* is the dead end. BN/activation-statistic noise is redundant with the existing regularization stack (Cutout12 + RandomErasing + LS0.2 + EMA) at full epochs. Combined with the exact c0 reproduction, EXP-016's apparent signal is explained as control-draw variance.
- **Key Learning**: A throughput-free, calibrated reproduction of EXP-016's BN-stat noise ties the same-session control exactly (96.14=96.14) — EXP-016's +0.24pp was a low-control draw, and the BN/activation-noise regularization axis is closed at full epochs.

## Verification
- **Conditions**: (a) budget/validity PASS (c0 450.5s/149ep, cA 430.9s/140ep, cB 429.1s/140ep, all valid best_test_acc, all wall < 600s). (b) **FAIL** — best noise cell cB 96.23 < 96.48 bar and only +0.09pp over same-session c0 (96.14), below the >0.1pp clear margin and within the ~0.2pp floor; cA tied c0 exactly. No apparent win → mandatory confirmation re-run not triggered. (c) integrity skipped per protocol after (b) failed (noted clean: only train.py modified, prepare.py byte-unchanged, 1 eval/epoch, seed 42, all smokes passed).
- **Review Notes**: Results confirmed trustworthy. num_epochs 149/140/140 match the independent M2 throughput prediction → equal-condition runs, no contention. The exact c0=96.14 match with EXP-016 is corroborating, not suspicious — it is the expected same-host draw. No scope/process integrity concerns.
- **Verdict**: no-improvement
- **Verdict Basis**: valid result, necessary condition (b) failed — metric did not clear the bar and did not beat the same-session control beyond noise.

## Unexplored Avenues
- **Faithful compile-funded layer3 GhostBN.** NoisyBN is a surrogate: it uses independent-per-sample Gaussian noise, not GhostBN's *group-shared, data-dependent* sub-batch stats. The decisive null here (calibrated ties; 2.5× gives +0.09pp) makes it unlikely the structured version wins materially, but the only fully faithful closer is GhostBN with torch.compile (EXP-014 recipe) recovering the ~50% throughput tax. **Low priority** — the surrogate already spans 1× and 2.5× magnitudes with no signal, so the BN-noise axis is best treated as closed rather than re-probed.
- All input-aug (EXP-006/008/011/015), optimizer (009/010/014), capacity (005/007/014), reg-scalars (012), loss-geometry (013), eval-TTA (006), and BN-stat-noise (016/017) lanes are now saturated at the ~96.3–96.5 ceiling. The unexplored class is **backbone inductive-bias / architectural priors** that change *what features the net can represent* rather than how hard it is regularized.

## Next Steps
- **BlurPool anti-aliased downsampling** (Zhang 2019) replacing the MaxPool/stride downsamples — adds shift-equivariance as an architectural prior; throughput-light, plausibly free at the budget. **Confidence: medium** — a genuinely new axis (inductive bias, not regularization), and the strongest of the EXP-017-brainstorm deferred finalists.
- **Squeeze-Excitation channel attention** in the residual blocks — cheap per-channel recalibration, a different capacity axis than width/depth (both saturated). **Confidence: medium-low** — small param add, must verify it does not cost epochs.
- Treat the BN-noise / GhostBN axis as **closed**; do not re-enter without the compile-funded faithful variant, and only if a backbone pivot also stalls. **Confidence: high** that further BN-noise tuning is wasted effort.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
