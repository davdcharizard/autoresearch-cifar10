# Brainstorm EXP-050
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Goal, metric, direction, constraints, verification live in the goal file.
     Baseline lives in experiment-indices/improve-cifar10-test-accuracy.tsv (96.22, EXP-012, 6c417a4). Bar = 96.32. -->

## Web Search & Literature Review

No new external search — the relevant mechanism is well-established and already partly probed in-project:
- **Small-batch generalization / flat minima** (Keskar et al., ICLR 2017, "On Large-Batch Training…"): large batches converge to *sharp* minima that generalize worse; smaller batches inject more SGD gradient noise → *flatter* minima → better test accuracy at fixed LR/epochs. This is the mechanism this experiment tests.
- **Linear scaling rule / equal-epoch equivalence** (Goyal et al. 2017; Smith et al. 2018): predicts *equal* accuracy across batch sizes at equal epochs when LR is scaled with batch — i.e. NO gain from batch changes per se. This is the competing null hypothesis. The two literatures are in tension; an empirical run resolves which holds for this net.
- In-project anchor — **EXP-025** (`reports/exp-report-025.md`): batch 128→256 found the k=4 net is **COMPUTE-bound** at 256 (dt 8→15-26ms), collapsing updates 61% (35.5k→14k) → −2.38pp. The compute-bound property is decisive for the *downward* direction (see below).

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, 6c417a4). Bar = 96.32. **42 consecutive no-improvements (EXP-013..049)**; 6 lifetime improvements (EXP-000..012 era).

**Every axis tried so far is closed** (goal-learnings): capacity (width k>4 / depth-realloc / fat-head / ResNet-D / BlurPool — all epoch-wall), augmentation (strength/policy/mixing/cooldown/border/occlusion-pattern), LR schedule (peak/shape/floor), regularizer-adds (dropout/GhostBN/SAM/WD — underfit), classifier-head (aggregation/scoring-geometry), feature-routing (multi-scale/deep-supervision), activations (SiLU ×2), weight-averaging (EMA/SWA ×3), optimizer-family (AdamW) + Gradient Centralization (loss-polish), bag-of-tricks incl. **residual scaling / zero-init-γ (EXP-026)**, loss-function (PolyLoss/LS), **pre-activation block ordering (EXP-015, null)**, SE-attention (EXP-008). EXP-049 closed the directive's **"combine near-misses"** route (cooldown+GC regressed to 96.13, below both alone).

**Batch size — tested UPWARD only**: EXP-025 (128→256) regressed hard. The **downward** direction (batch < 128) has NEVER been tested. This is a genuinely untouched axis.

**Two relevant tailwinds for batch-down**:
1. **Compute-bound, not launch-bound** (EXP-025): at batch 256 dt scaled ~linearly with batch. So at batch 64, dt should ~halve (≈4.5-6ms) → roughly **2× the gradient updates** at *similar* total images — the exact INVERSE of the batch-256 update-collapse that caused its regression. Smaller batch here ADDS updates rather than removing them.
2. **Epoch-saturated** (EXP-007/045/046): more epochs beyond ~91 don't help, so the mild image/epoch reduction from a smaller batch (≈70-85 ep) is unlikely to underfit much.

## Candidate Ideas

### 1. Smaller batch size (BATCH_SIZE 128→64) for SGD gradient-noise regularization — single variable
**Summary**: Change `BATCH_SIZE` from 128 to 64 and nothing else — keep `PEAK_LR=0.2`, `WARMUP_FRAC=0.05`, the time-fraction cosine schedule, and all augmentation/optimizer settings identical. At fixed LR, halving the batch *doubles* the relative SGD gradient noise (noise ∝ LR/√B: 0.2/√64 vs 0.2/√128) while keeping the mean update magnitude the same — the canonical Keskar "small batch → flatter minima → better generalization" regime.

**Reasoning**: This is the only genuinely-untested axis left with a distinct, well-documented mechanism that is NOT in any closed family (it is not a capacity add, not an augmentation, not a regularizer module, not a schedule/optimizer/architecture change). The two in-project tailwinds make it clean: EXP-025's compute-bound finding means batch-64 *adds* ~2× updates (not the update-collapse that sank batch-256), and epoch-saturation means the mild image reduction (~70-85 ep) shouldn't underfit. Single-variable → clean attribution. Even a null cleanly closes the downward batch axis (complementing EXP-025's upward closure).

**Sources**: Keskar et al. ICLR 2017; `reports/exp-report-025.md` (compute-bound; launch-bound only at 128); goal-learnings EXP-007/045/046 (epoch-saturation).

**Estimated Effort**: low — a one-line hyperparameter change. Run + verify dt/epochs.

**Risk Assessment**: (a) If a launch-overhead floor keeps dt from fully halving, epochs drop more than expected (~70) → mild underfit. (b) Smith/Goyal equal-epoch equivalence may hold → exact null. (c) batch 64 + LR 0.2 could be slightly noisy early, mitigated by the 5% warmup + BN + label smoothing. Worst case is a clean no-improvement (most likely outcome ~95.9-96.3); no crash risk. dt watched as an early signal.

### 2. LayerScale — learnable per-channel residual-branch scaling (small init)
**Summary**: Multiply each BasicBlock's residual-branch output by a learnable per-channel vector initialized to a small constant (e.g. 1e-1) before the `+= shortcut(x)`. Throughput-free (one fused elementwise multiply per block, ~9 vectors, negligible params).

**Reasoning**: Gives the residual branches a learnable magnitude DOF, which can better-condition signal propagation and sometimes improves generalization at fixed capacity. Throughput-neutral → no epoch-wall confound.

**Sources**: Touvron et al. 2021 (CaiT LayerScale); `reports/exp-report-026.md`.

**Estimated Effort**: low — add a `nn.Parameter` per block and one multiply in `forward`.

**Risk Assessment**: Strong prior it is null — EXP-026 (zero-init residual γ, the closely-related residual-scaling DOF) was "within-noise null, moves loss not top-1" on this shallow net. LayerScale adds an extra DOF but is in the same residual-scaling family, so most likely a repeat null.

### 3. PReLU — learnable-slope activation replacing ReLU
**Summary**: Replace `F.relu` with `nn.PReLU` (learnable negative slope), one parameter per channel. Near-free, tiny param add.

**Reasoning**: A learnable activation can fit a slightly better nonlinearity than fixed ReLU; never tried (only SiLU/Swish were).

**Sources**: He et al. 2015 (PReLU); `reports/exp-report-010.md`, `-028.md` (SiLU null).

**Estimated Effort**: low.

**Risk Assessment**: Activation axis is closed (SiLU null ×2 — "generalization-null here, and smooth activations cost ~1ms/step"). PReLU likely null for the same reason; the per-channel PReLU may also not fully fuse → small dt cost → mild epoch-wall. Low expected value.

## Idea Evaluation

After 42 no-improvements with this exhaustive a map, no remaining lever has strong positive EV; the task under NEVER STOP is to run the best genuinely-fresh experiment, document, and continue.

**Evidence strength / mechanism distinctness**: Candidate 1 is the only option targeting a genuinely-untouched axis (batch-size-downward) with a mechanism (SGD gradient noise → flat minima) that sits in NO closed family. Candidates 2 and 3 both have strong in-project priors that they are null (EXP-026 residual-scaling; EXP-010/028 activations) — they re-probe near-closed families.

**Feasibility / cleanliness**: Candidate 1 is a one-line single-variable change with clean attribution; EXP-025's compute-bound result lets me predict the throughput behavior (dt ~halves, updates ~2×) and the epoch-saturation result bounds the underfit risk. Candidates 2/3 are also low-effort but carry repeat-null priors.

**Risk profile**: All three fail gracefully (no-improvement); none can crash. Candidate 1's only real risk is a larger-than-expected epoch drop, which is measurable and informative.

**Expected impact**: All low. Candidate 1 has the best *information* value (closes/【maps the last untouched axis and resolves the Keskar-vs-Smith tension for this net) and the only non-closed-family mechanism, so it is the clear lead. Candidates 2/3 are alternates for subsequent loops only if 1 is null and nothing better surfaces.

## Chosen Idea
**Selected**: Candidate 1 — Smaller batch size (BATCH_SIZE 128→64), single-variable, LR/warmup/everything else unchanged.

**Why this idea**:
It is the only genuinely-untested axis remaining (batch-size-downward) with a distinct, well-documented generalization mechanism (Keskar gradient-noise → flat minima) that is not in any closed family. EXP-025's compute-bound finding makes it a clean test rather than a repeat of the batch-256 failure: batch-64 ADDS ~2× gradient updates (inverse of 256's update-collapse) at similar total images, and the net's epoch-saturation bounds the underfit risk. One-line change, clean single-variable attribution; even a null result cleanly closes the downward batch axis.

**Hypothesis**:
At batch 64 / LR 0.2 (≈2× relative gradient noise, same mean update), the net is compute-bound so dt ~halves to ≈4.5-6ms giving ~2× updates and ~70-85 epochs (no severe underfit, given epoch-saturation). IF small-batch gradient noise finds a flatter, better-generalizing minimum on this net, best_test_acc ≥ 96.32 (clears the bar). Falsified if it lands within ±0.25pp of baseline (Smith/Goyal equal-epoch equivalence holds → batch noise is neutral here) or below (the image/epoch reduction or excess noise dominates) — either way closing the downward batch-size axis.
