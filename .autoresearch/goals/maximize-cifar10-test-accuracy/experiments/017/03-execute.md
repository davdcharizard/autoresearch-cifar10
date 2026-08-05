# EXP-017: Throughput-free BN-affine noise (cheap GhostBN surrogate, layer3-first)

## Execution

Overall Status & Info:
- **Created**: 2026-06-30
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-017
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented `NoisyBN(nn.BatchNorm2d)` in train.py and wired it into `conv_bn` (10 BN sites), toggled by env `BN_NOISE` (σ, default 0 = exact EXP-008 baseline) and `BN_NOISE_MIN_CH` (512 = layer3-only). Training-only forward runs the fused full-batch BN then perturbs the NORMALIZED activation per-(sample,channel) via the division-free `y' = y + (y−β)·σε_mul + γ·σε_add` (β untouched). All M1 smokes passed: Smoke A (σ=0 train+eval AND σ=0.1 eval ≡ nn.BatchNorm2d, 0.0 diff), Smoke B (noise zero-mean within sampling z<6; mul-perturbation spatially constant 9.5e-6; buffers ≡ nn.BatchNorm2d), Smoke C (per-layer EMA buffers = manual EMA, shape [C]), gradient smoke (finite loss/grads, trainable params only; exactly 3 layer3 sites active with MIN_CH=512). The calibration smoke set the σ values; the M2 probe drove a throughput note (see Decisions).

### Surprises & Discoveries
- **Calibration confirmed the brainstorm's σ=0.1–0.2 was 3–6× too strong.** Measured GhostBN(g=128)-equivalent noise at the 3 layer3 sites (over 6 real CIFAR batches): σ_add*≈0.025, σ_mul*≈0.033 (corrected multiplicative form σ_full/σ_ghost−1). σ_cal = max = **0.033**. So cA=0.033 (1× ghost-equivalent), cB=0.083 (~2.5×).
- **Not perfectly throughput-free.** layer3-only noise = +7.2% slower (25144 vs 27082 img/s, ~138 ep predicted); all-site = +37.4% (the elementwise ops on the large prep/layer1 activations are costly). The cost is the full-activation elementwise ops IN THE AUTOGRAD GRAPH (backward must traverse them), not the RNG draws — so mul-only (`add=False`) saved only ~1.3% (+5.9%, ~140 ep). The premise "throughput-free" holds only approximately at layer3.
- σ_add* < σ_mul* at all 3 sites → the multiplicative (scale) jitter is the dominant ghost-noise component here.

### Decisions
- **Kept the faithful mul+add form** (NoisyBN defaults add=True) rather than the mul-only mitigation: the throughput difference is negligible (7.2% vs 5.9%) and mul+add captures BOTH calibrated components (σ_mul + the smaller σ_add), more faithful to GhostBN. A single σ is applied to both (additive slightly over-injected vs its 0.025 calibration — minor).
- **Accepted the ~138-epoch run (~+7% cost) as a mild, non-confounding gap** rather than torch.compile-funding it. Rationale (EXP-014 evidence): epochs near 150 are worth ≈0.02–0.03pp (154→173 gave +0.03pp), so the ~11-epoch c0-vs-noise gap accounts for ≤~0.02pp — below the 0.1pp bar and 0.2pp noise floor. The gap works AGAINST the noise cells (they run fewer epochs than c0), so a >0.1pp noise-cell win would be a strong positive (as in EXP-016 at 133ep), and a tie cannot be hiding a real benefit the gap erased. Compile-funding (EXP-014 recipe) is heavy machinery for a ~7% recovery; deferred to a faithful-GhostBN follow-up if this wins. Recorded the predicted ~138ep; reject only if a cell drops <~130 (heavy contention).
- σ values: cA=0.033 (calibrated 1×), cB=0.083 (~2.5×), both MIN_CH=512 (layer3, the EXP-016 site); c0 noise=0 (full-speed standard-BN control).

## Experimental Adjustments
- **σ set from calibration, not the brainstorm's 0.1/0.2**: measured layer3 ghost-equivalent σ_cal=0.033 (cA), 2.5×=0.083 (cB). (ref: M1 calibration smoke — σ_add*=0.025, σ_mul*=0.033 over 6 batches)
- **Faithful mul+add kept despite ~+7% cost**: mul-only saved only ~1.3% (cost is graph elementwise ops, not RNG), so no faithfulness sacrifice was warranted. (ref: M2 probe — mul+add 25144 vs mul-only 25485 img/s)

## Run Log

### Run 1 — c0 (control), cA (layer3 σ=0.033), cB (layer3 σ=0.083)

Metadata:
- **Job ID**: (background bash — see TaskOutput)
- **Log file(s)**: experiments/017/run_c0.log, run_cA.log, run_cB.log; gpu_c0/cA/cB.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-30
- **Ended**: 2026-06-30

Description:
- Three sequential same-session cells on GPU 1, each a separate `train.py` process under `timeout 600`. c0 (`BN_NOISE=0`) is the full-speed standard-BN control (~149ep, same-session anchor). cA (`BN_NOISE=0.033 BN_NOISE_MIN_CH=512`) injects the calibrated layer3 ghost-equivalent noise at the EXP-016 site (~140ep) — PRIMARY, the epoch-near-neutral reproduction of EXP-016's +0.24pp signal. cB (`BN_NOISE=0.083 BN_NOISE_MIN_CH=512`) is ~2.5× stronger (~140ep) — maps the noise-strength response. Test: does either noise cell exceed c0 by a clear >0.1pp margin (≥96.48) despite the ~7% epoch gap.

Observations:
- **DECISIVE NULL: the calibrated ghost-equivalent noise (cA, σ=0.033) tied the control EXACTLY — 96.14 vs c0 96.14** — at near-full epochs (140 vs 149). The throughput-free reproduction of EXP-016's mechanism shows NO benefit at the ghost-equivalent magnitude. (source: run_cA.log, run_c0.log)
- **Stronger noise (cB, σ=0.083, 2.5×) reached 96.23 — only +0.09pp over c0**, within the ~0.2pp session noise floor and FAR below the 96.48 bar. A hint that supra-ghost noise does something tiny, but not a win. (source: run_cB.log)
- **This session's c0 = 96.14, EXACTLY EXP-016's c0 (96.14)** → confirms the same-session control reproducibly draws ~96.14 on this host, and **EXP-016's "+0.24pp" was a weak-control-draw artifact**, not a real BN-noise benefit. The mechanism is essentially redundant with the existing stack at full epochs.
- Healthy & not under-fit/over-regularized: ep25 c0 92.43 / cA 92.43 / cB 92.23 (cB's mild early dip from stronger noise recovers); all fully annealed (best≈final: c0 96.14/96.00, cA 96.14/96.09, cB 96.23/96.13).
- num_epochs 149/140/140 matches the M2 +7% prediction exactly → clean equal-condition runs, no contention (GPU 1 idle 0–3% throughout, gpu_*.log). The noise cells' 9-epoch deficit did NOT confound (cB even edged c0 despite fewer epochs; per EXP-014 ±10ep≈0).

Key Metrics:
- c0 best_test_acc: 96.14% (final 96.00 @ep149); 149 ep; 450.5s; 1635 MB (source: run_c0.log)
- cA best_test_acc: 96.14% (final 96.09 @ep140); 140 ep; 430.9s; 1636 MB (source: run_cA.log)
- cB best_test_acc: 96.23% (final 96.13 @ep140); 140 ep; 429.1s; 1636 MB (source: run_cB.log)
- ep25: c0 92.43 / cA 92.43 / cB 92.23 (source: run_*.log)
- calibrated layer3 ghost-equiv σ: σ_add*=0.025, σ_mul*=0.033 → σ_cal=0.033 (M1 calibration smoke)
- M2 throughput: control 27082, layer3-noise 25144 img/s (+7.2%) → predicted/actual ~140ep

## Verification Results

### Conditions Checked
- **(a) Completes within budget, valid best_test_acc, wall < 600s** — PASS. c0 450.5s/149ep, cA 430.9s/140ep, cB 429.1s/140ep; all valid best_test_acc, all < 600s. (source: run_*.log)
- **(b) Best noise cell ≥ 96.48 AND > same-session c0 by a clear (>0.1pp) margin, replicated** — FAIL. Best noise cell = cB 96.23% < 96.48 bar; +0.09pp over c0 (96.14), below the >0.1pp clear margin and within the ~0.2pp noise floor. cA tied c0 exactly (96.14). No apparent win → mandatory confirmation re-run NOT triggered. → no-improvement. Stop; (c) not evaluated per protocol.
- **(c) Integrity** — skipped per protocol after (b). (Noted clean: only train.py modified; prepare.py byte-unchanged; eval count == num_epochs (1/epoch); seed 42; Smokes A/B/C + calibration + gradient all passed.)

### Verdict basis
no-improvement: the calibrated ghost-equivalent BN noise tied the control (cA 96.14 = c0 96.14) at near-full epochs, and 2.5× noise (cB 96.23) only reached +0.09pp (within noise, < 96.48 bar). The throughput-free reproduction of EXP-016's mechanism confirms BN-statistic noise gives no real benefit at full epochs — EXP-016's +0.24 was a low-c0 draw (c0=96.14 in both sessions). Final verdict in 04-analysis.md.

### Informational Metrics
- peak_vram_mb: c0 1635.4 / cA 1635.7 / cB 1635.7 MB (NoisyBN adds no params) (source: run_*.log)
- num_epochs: c0 149 / cA 140 / cB 140 (the ~7% noise overhead, M2-predicted) (source: run_*.log)
- total_seconds: c0 450.5 / cA 430.9 / cB 429.1 (training_seconds 300.0 all) (source: run_*.log)
- num_params: 7,784,627 (unchanged) (source: run_*.log)
- ep25 test_acc: c0 92.43 / cA 92.43 / cB 92.23 (source: run_*.log)
- layer3 ghost(g=128)-equivalent σ (M1 calibration): σ_add*=0.025, σ_mul*=0.033

## Errors & Dead Ends

<!-- Append only. -->

## Human Notes

> (none — autopilot)
