# Report EXP-018: BlurPool anti-aliased downsampling (MaxBlurPool / BlurPool-only)
- **Created**: 2026-06-30

## Goal
Maximize CIFAR-10 `best_test_acc` (%) within the fixed 300s training budget, editing only `train.py`. Higher is better. Baseline = **96.38** (EXP-008, commit 07c3760). Bar = ≥96.48 (baseline+0.1pp) AND clearly above the same-session control beyond the ~0.1–0.2pp noise floor. This was the FIRST architectural-inductive-bias experiment, testing whether anti-aliased downsampling (restoring shift-equivariance) lifts the generalization ceiling after capacity/epochs/optimizer/all-regularization axes saturated (12 straight no-improvements).

## Idea & Hypothesis
**Chosen idea (idea-01, Codex review pick 7.5/10):** replace the naive `nn.MaxPool2d(2)` subsampling at layer1/2/3 with **MaxBlurPool** (Zhang 2019, ICML, arXiv:1904.11486) — dense max at stride 1, then a fixed binomial blur (depthwise, stride 2) to subsample. The naive maxpool violates the Nyquist sampling theorem → aliasing → shift-variance, the one structural weakness never touched on this goal. Zhang shows anti-aliased downsampling acts as **effective regularization** that raises clean accuracy. The blur is a fixed buffer (0 trainable params), so a comparison vs MaxPool isolates the inductive-bias change, not capacity.

**Hypothesis:** anti-aliased downsampling reaches best_test_acc ≥96.48 over the same-session control by a clear >0.1pp margin at near-full epochs, replicated. A tie demotes the downsampling inductive bias as ceiling-moving on this small-image, strongly-augmented net.

## Approach
Single editable file, `train.py`. Added env-toggled anti-aliased pooling: `BLUR_KSIZE` (0=baseline `nn.MaxPool2d`, odd 3/5), `BLUR_LAYERS` ("123"), `BLUR_MODE` ("max"=MaxBlurPool dense-max+blur; "blur"=BlurPool-only, anti-aliased strided subsample replacing the max). `BlurPool2d` = fixed binomial depthwise blur (registered buffer, excluded from optimizer/grad) at stride 2 via the conv's built-in zero padding. Wired `make_pool(128/256/512, 1/2/3)` into layer1/2/3; final 4×4 head pool unchanged.

**Key deviation from plan (recorded in 03-execute.md Decisions):** the planned faithful MaxBlurPool ("max" mode) was **throughput-disqualified**. M1 throughput probes found the original reflect-pad + dense-max chain ran at only 0.40× baseline (→~103ep); zero-pad fixed the reflect cost (→0.884×, **132ep**), but the dense `MaxPool2d(stride=1)` at full resolution still pushed max-mode under the hardened `num_epochs ≥ 135` under-anneal gate (ks3 132ep, ks5 127ep). The **BlurPool-only** form (drop the dense max; anti-aliased binomial strided subsample) is genuinely **throughput-free** (ks3 1.028×→153ep, ks5 0.987×→146ep). To respect the under-anneal gate (the #1 failure mode on this goal), the official cells used blur-only mode. All M1 correctness smokes (A baseline parity, B/C 16-8-4 shapes + kernel-sum-1 + buffer-not-param, D finite train backward, F native-fp32 eval + flip + TTA + EMA-buffer invariance) passed for both modes at ks3/ks5.

**Cells (same-session, GPU 1, per-cell nvidia-smi sampler):** c0 (`BLUR_KSIZE=0`, baseline MaxPool, anchor) / cA (`BLUR_KSIZE=3 BLUR_MODE=blur`, PRIMARY, triangle filter) / cB (`BLUR_KSIZE=5 BLUR_MODE=blur`, binomial-4, stronger low-pass).

## Execution
Three sequential cells ran cleanly under `timeout 600`, no retries. GPU 1 was ours throughout (max ~6 GB/cell, no foreign job — gpu_*.log), so the same-session comparison is valid. num_epochs 150/153/146 — the blur cells ran AT or ABOVE c0's epoch count (BlurPool-only is throughput-free; cA +3ep over c0), so this is decisively NOT an under-anneal confound. Wall < 460s all cells; production img/s ~26,700 (above the conservative synthetic probe).

## Results
- **Primary metric**: 96.23 (best blur cell, cA). Baseline 96.38, delta **−0.15**; vs same-session c0 96.31, delta **−0.08pp**.
- **Cells**: c0 96.31 (final 96.29 @ep150); cA (ks3) 96.23 (final 96.12 @ep153); cB (ks5) 96.16 (final 96.10 @ep146). ep25: c0 92.18 / cA 92.41 / cB 91.98.
- **Observations**:
  - **Both anti-aliased cells lost to the same-session MaxPool control** — cA −0.08pp (within the ~0.1–0.2pp noise floor), cB −0.15pp. No cell approached the 96.48 bar.
  - **Monotonic degradation with blur strength** (ks3 −0.08 → ks5 −0.15): stronger low-pass filtering is mildly *harmful*, not helpful — the opposite of what an anti-aliasing benefit would predict. The binomial blur discards high-frequency detail that the classifier uses.
  - **Throughput-free and fully annealed**: cA ran +3 epochs over c0 (153 vs 150) yet still lost, so the null is NOT an epoch/under-anneal artifact (best≈final all cells). ep25 healthy — cA was even slightly *above* c0 early (92.41 vs 92.18), ruling out under-training. The loss is at the annealed ceiling.
- **Analysis**: The intervention achieved its intended local effect (anti-aliased downsampling, throughput-free, well-trained) but moved the metric *negatively*. Two factors explain the null: (1) **no shift-equivariance headroom** — the recipe's `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` augmentation already trains the net to be translation-robust, so the architectural shift-equivariance prior is redundant; and (2) on 32×32 inputs there is little spatial aliasing to correct, unlike Zhang's ImageNet (224×224) gains. There is a confound — blur-only changes both max→avg-style aggregation AND aliased→anti-aliased subsampling — but the **monotonic worsening with stronger blur** argues the low-pass operation itself is the harm (lost detail), so a hidden anti-aliasing benefit being masked by losing max is unlikely; even if MaxBlurPool (preserving max) were run at full epochs, the evidence points to at best a tie. The downsampling inductive bias is not the ceiling limiter here.
- **Key Learning**: Anti-aliased downsampling (throughput-free BlurPool-only) does NOT beat MaxPool on this net — it loses monotonically with blur strength; strong RandomCrop+flip aug already supplies translation invariance and 32×32 has little aliasing headroom, so the shift-equivariance prior is redundant.

## Verification
- **Conditions**: (a0) integrity PASS (clean — summary==per-epoch-max for all cells, 1 summary line each, eval-count==epochs, prepare.py byte-unchanged, only train.py modified, seed 42, num_params 7,784,627 unchanged → not invalid/gamed). (a) budget/validity/≥135ep PASS (150/153/146 ep, all valid, <460s wall). (b) FAIL — best blur cell cA 96.23 < 96.48 bar AND < c0 96.31 (−0.08pp); no apparent win → confirmation re-run not triggered.
- **Review Notes**: Results trustworthy. Throughput-free (blur cells ≥ c0 epochs) removes the under-anneal confound that plagued prior cost-adding changes; full anneal + healthy ep25 rule out under-fit; clean same-session conditions (no contention). The max→avg confound is noted but the monotonic blur-strength degradation makes a masked anti-aliasing benefit implausible.
- **Verdict**: no-improvement
- **Verdict Basis**: valid result, necessary condition (b) failed — metric did not clear the bar and lost to the same-session control.

## Unexplored Avenues
- **Faithful MaxBlurPool (preserve max + anti-alias) at full epochs, compile-funded.** Blur-only confounds max→avg with anti-aliasing. The clean isolation needs MaxBlurPool, which is throughput-bound (132ep) here; the EXP-014 torch.compile recipe could recover the ~12% to run it at ~150ep. **LOW priority** — the monotonic degradation with blur strength (cA→cB) already argues the low-pass itself is mildly harmful, so MaxBlurPool would at best tie; not worth the heavy compile machinery unless a different signal emerges.
- **BlurPool on the stem/whitening or a single large-spatial layer only.** Untested, but same mechanism — low prior given the across-the-board null.
- The downsampling-operator axis joins the saturated list. Remaining genuinely-different architectural levers (from the EXP-018 brainstorm finalists, NOT yet tried): **Squeeze-Excitation channel attention** (idea-02, content-adaptive channel recalibration — a different *functional form*, not spatial) and **AdaptiveConcatPool head** (idea-03, avg⊕max readout — cheap, throughput-free).

## Next Steps
- **Squeeze-Excitation channel attention** (EXP-018 idea-02): add SE blocks (GAP→bottleneck→sigmoid gate) at layer2/3, identity-init the gate inside the ReZero branch; a channel-attention inductive bias orthogonal to the spatial/downsampling axis just closed. **Confidence: low-medium** — a genuinely different mechanism, but SE's gains are ImageNet-scale and may sit in the noise floor on this saturated CIFAR net.
- **AdaptiveConcatPool head** (EXP-018 idea-03): replace MaxPool(4)+fc with avg⊕max concat→fc(1024); throughput-free readout enrichment. **Confidence: low** — cheap to test but likely a sub-noise readout tweak; best as a rider, or a quick standalone before pivoting.
- Treat the **anti-aliasing / downsampling-operator axis as closed**; do not re-run BlurPool variants without a fundamentally different rationale. **Confidence: high** that further blur tuning is wasted. After 13 straight no-improvements (EXP-006→018), the ~96.3–96.5 ceiling is looking increasingly like a genuine architecture+data limit at 300s; if SE and the head also tie, the highest-EV move is a wholesale backbone change (not within-DavidNet tweaks) or accepting the ceiling.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
