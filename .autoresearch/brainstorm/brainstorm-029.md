# Brainstorm EXP-029
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **SWA — Averaging Weights Leads to Wider Optima (Izmailov et al., arXiv 1803.05407)** (background knowledge; no new fetch)
  Establishes BN running-stat RE-ESTIMATION as a standard, legitimate model-finalization step: whenever the weights or the data distribution the stats were accumulated under differ from eval conditions, recompute the running mean/var with forward passes before evaluating. This is a buffer update of the model itself, not an eval-procedure change.
- **AdaBN — Revisiting Batch Normalization for Practical Domain Adaptation (Li et al., arXiv 1603.04779)** (background knowledge)
  Replacing BN statistics with target-distribution statistics at eval time recovers accuracy under train/test distribution shift with zero weight changes. Our shift is real and self-inflicted: training batches are TA+RE+crop-augmented; the test set is clean. The running stats the evaluator consumes are estimates of the AUGMENTED distribution.
- **knowledge/README.md scan**: cifar10-fast (concat avg+max pooling head pedigree), airbench (GPU-resident pipeline — now known to be metric-irrelevant here, see History), muon-optimizer.md (closed by EXP-028). No new external fetches needed this loop — the strongest candidate is grounded in in-project measured evidence (EXP-025).

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006 recipe). Bar = 96.81. σ calibration (EXP-027): baseline mean ≈96.57, σ ≈0.16 ⇒ candidates need TRUE effects ≥ +0.3; the noise band is ±0.15.
- **22 consecutive misses (007–028).** Closed axes: capacity (uniform width ×3, depth-for-width, allocation, shortcuts), recipe constants (LR heat both sides, warmup, schedule family, WD both sides, momentum both sides heat-compensated, batch + LR-scaling), gradient noise BOTH directions, augmentation pressure both directions + time-varying (EXP-025), init both directions, activations (EXP-026 cost ladder), faster-but-different numerics (EXP-021), optimizer geometry (EXP-028 Muon), EMA smoothing (EXP-011), FixRes clean tail (EXP-025).
- **EXP-028's sharpened lesson**: transit-speed/sample-efficiency gains decay to zero before the plateau; the max-statistic pays ONLY for converged-plateau LEVEL (and length). The next idea must move plateau level directly.
- **NEW (this brainstorm, from re-reading train.py): loader stalls are UNCHARGED — the GPU-resident-pipeline idea is metric-dead.** train.py:216 sets `t0` AFTER the loader yields, so host/loader wait never enters `total_training_time`; the 300s budget counts only charged GPU work (H2D + fwd/bwd/step + sync). Eliminating loader stalls (GPU-resident data, more workers, uint8 transfer) adds ZERO epochs — it only shrinks total wall (493s, already under the 600s cap). Conversely, moving charged work above `t0` would be timing-protocol reward hacking. The throughput axis is closed at BOTH the kernel tier (EXP-021) and the pipeline tier (structurally).
- **Best unexploited measured POSITIVE fragment in the campaign**: EXP-025's one-time **+0.35 boost within ~1 epoch of switching to clean data** — attributed to BN-stat/feature alignment to the clean (test-like) distribution, before clean-data TRAINING ruined it by overfitting. exp-report-025 explicitly names "forward-only BN-stat recalibration (cannot overfit)" as the surviving fragment.
- **Integrity precedent**: Law 3 bars variance harvesting; the EXP-011/016 insight says judge by plateau level. Eval-side compute inflation (TTA ensembling) is the same genus as variance harvesting — it improves the measurement, not the model.

## Candidate Ideas

### 1. Clean-data BN running-stat recalibration before every eval (forward-only, training-invisible)
**Summary**: Before each `evaluator.evaluate(base_model, device)` call, recompute every BN layer's running mean/var on CLEAN train images (test transform: ToTensor+Normalize only): `reset_running_stats()`, set `momentum=None` (exact cumulative average), run K≈16 no-grad forward passes in train() mode on a fixed GPU-resident tensor of K×512 clean train images (preloaded once at startup, ~100MB VRAM), then eval as usual; restore momentum afterward. Training is untouched: train-mode forwards use BATCH stats (running stats are never consumed by training), no weights change (no_grad, no optimizer), no training-budget seconds are charged (recalib happens outside the timed region, like eval itself). The evaluated object remains the plain eager model at unchanged inference cost.

**Reasoning**: The recipe trains on a heavily augmented distribution (RandomCrop pad-4, flip, TrivialAugmentWide, RandomErasing p=0.5) but is tested on clean images — the BN running stats the evaluator consumes estimate the WRONG distribution. EXP-025 measured exactly this alignment being worth **+0.35 within ~1 epoch** of switching distributions (before its failure mode, clean-data TRAINING, kicked in — which forward-only recalibration cannot trigger by construction). AdaBN/SWA establish BN re-estimation as a standard model-finalization step. It is the ONLY candidate class that passes all four structural laws trivially: zero early heat, zero epochs (uncharged), zero training-numerics change, zero gradient-noise change — because training literally cannot see it. It lifts every one of ~139 evals uniformly = a plateau-LEVEL shift, the exact currency the max-statistic rewards.

**Sources**: reports/exp-report-025.md § Results (the +0.35 switch boost + surviving-fragment note); goal-learnings § Failed Approaches (FixRes entry, Insight line); arXiv 1803.05407 §3 (BN re-estimation); arXiv 1603.04779 (AdaBN); train.py:134–145 (augmented train transform), train.py:263 (eval call site).

**Estimated Effort**: low-medium — ~30 lines in train.py (preload clean tensor, recalib helper, call before eval), no new packages (torchvision already provides the clean transform pieces).

**Risk Assessment**: (a) Magnitude risk — the +0.35 of EXP-025 bundled BN alignment WITH feature adaptation; the pure-BN share is unknown, could be +0.1 (sub-bar). (b) Sign risk is bounded: if augmented stats were somehow better for test data, recalibrated evals would read LOWER — visible immediately at ep1–5, no late surprise. (c) Wall cost trivial: 16 forward-only batches ≈ 0.15s/eval, +~20s total (493→~515s, well under 600). (d) Integrity: the model's buffers are updated with train-set data only, evaluated object unchanged — no eval-procedure modification, no extra inference compute, no variance harvesting (uniform mean shift). (e) torch.compile interference: use base_model (eager) for recalib forwards — no recompile triggers.

### 2. Horizontal-flip TTA via an eval wrapper module (REJECTED on integrity before evaluation)
**Summary**: Pass a wrapper module to the evaluator whose forward averages logits over the image and its horizontal flip (optionally under bf16 autocast to pay the 2× inference cost). Worth +0.2–0.5 on CIFAR-10 in the literature.

**Reasoning / why rejected**: Passes the constraint LETTER (only train.py changes; Eval.evaluate() untouched) but fails the adversarial integrity test: it improves the MEASUREMENT (2× inference compute, an implicit 2-view ensemble) rather than the trained model. Same genus as variance harvesting, which this campaign already classified as reward hacking under Law 3. Also nearly busts the wall cap (eval ≈ doubles: +~110s on a 493s run). Recorded so future loops don't re-derive it.

**Sources**: goal file § Hard Constraints; project-insights (max-statistic entry); Law 3 precedent in goal-learnings.

**Estimated Effort**: low. **Risk Assessment**: high probability of `invalid` classification at analysis; cap risk.

### 3. Concat avg+max global pooling head (cifar10-fast pedigree)
**Summary**: Replace `adaptive_avg_pool2d` with concat(avg-pool, max-pool) → fc input 256→512 (fc params 2,570→5,130). Architecture-class change that is nearly free in dt (one extra pooling kernel, ~0.05ms) and arguably free in early heat (max-pooled features carry signal from init; fc trains fast).

**Reasoning**: The one micro-architecture modernization with short-budget CIFAR pedigree (cifar10-fast ResNet-9 uses it) not yet tried; max-pathway preserves high-activation evidence that average pooling dilutes. But no ablation magnitude exists in our references, and the prior for a head-only change is +0.1–0.2 — below the +0.3 effect-size screen on its own.

**Sources**: knowledge/README.md References (cifar10-fast row); train.py:106; EXP-026 protocol (early-dt gate would screen any dt surprise).

**Estimated Effort**: low. **Risk Assessment**: likely noise-band outcome (≤ +0.2); compile graph change forces recompile (fine, warmup absorbs it); fc re-init means a brief early-heat learning cost for the new half of the head.

## Idea Evaluation

**Evidence strength**: Idea 1 is the only candidate backed by an in-project MEASURED positive effect (+0.35, EXP-025) plus two literature anchors (SWA's BN re-estimation, AdaBN); Idea 3 has pedigree but no measured magnitude anywhere; Idea 2 has literature magnitude but is integrity-rejected. **Mechanism clarity**: Idea 1's mechanism is precise — the evaluator consumes running stats estimating the augmented distribution; recalibration makes them estimate the clean distribution the test set is drawn from; EXP-025 demonstrated the misalignment costs several tenths. **Expected impact**: Idea 1: +0.1–0.35 (the BN share of the measured boost); Idea 3: +0.0–0.2. Only Idea 1's range reaches the bar. **Risk profile**: Idea 1 fails gracefully (recalibrated evals simply read in/below the noise band; visible by ep5; cannot crash training since training never reads running stats), wall cost ~+20s. **Feasibility**: both low effort; Idea 1 slightly more code but all patterns exist in-project (EXP-025's second-loader/clean-transform code, GPU-resident tensor like EXP-028's buffers).

**The four laws**: Idea 1 is the FIRST candidate in 23 loops that is structurally invisible to training — zero heat, zero epochs, zero numerics, zero noise — because it acts only on buffers the training pass never consumes. The laws cannot price it; only its raw effect size matters. That alone makes it the correct next probe of the plateau-level currency.

## Chosen Idea
**Selected**: Idea 1 — Clean-data BN running-stat recalibration before every eval

**Why this idea**:
It directly monetizes the campaign's best unexploited measured positive fragment (EXP-025's +0.35 alignment boost) through the only delivery mechanism that cannot trigger that experiment's failure mode (forward-only ⇒ no clean-data overfitting), and it is the first candidate that is provably free in all four structural currencies (heat/epochs/numerics/noise) because training never reads BN running stats. It is a pure plateau-LEVEL intervention — exactly what the max-over-checkpoints metric rewards (EXP-011/016/028 insight) — with trivial wall cost and a fail-graceful signature visible within the first 5 evals.

**Hypothesis**:
Recalibrating BN running stats on 16×512 clean train images (momentum=None cumulative average, forward-only, train-mode, no_grad) immediately before each per-epoch eval will shift the entire eval trajectory upward by the BN-alignment share of EXP-025's +0.35 switch boost. Testable prediction: evals from ep1 onward read above the baseline family at matched epochs, the converged plateau (final-7 median) lands ≥ +0.25 above the baseline mean (≥ 96.8), and best_test_acc ≥ 96.81 — with num_epochs ≈ 139, dt ≈ 22.4ms, and training_seconds = 300.0 unchanged (the change is invisible to the timed loop). If the plateau shift is < +0.1, the conclusion is that EXP-025's boost was feature adaptation rather than BN stats, closing the alignment axis entirely.
