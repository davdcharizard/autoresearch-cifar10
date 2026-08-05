# EXP-018: BlurPool anti-aliased downsampling (MaxBlurPool / BlurPool-only) at layer1/2/3

## Execution

Overall Status & Info:
- **Created**: 2026-06-30
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-018
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Added anti-aliased downsampling to `train.py`, env-toggled by `BLUR_KSIZE` (0=baseline `nn.MaxPool2d`, odd 3/5), `BLUR_LAYERS` ("123"), and `BLUR_MODE` ("max"=MaxBlurPool dense-max+blur; "blur"=BlurPool-only, anti-aliased strided subsample replacing the max). `BlurPool2d` is a fixed binomial depthwise blur (registered buffer, 0 trainable params, excluded from optimizer/grad) applied at stride 2 using the conv's **built-in zero padding**. Wired `make_pool(128/256/512, 1/2/3)` into layer1/2/3 in place of `nn.MaxPool2d(2)`; final 4×4 head pool unchanged. All M1 correctness smokes pass for both modes at ks3/ks5 (Smoke A baseline parity + num_params 7,784,627 unchanged; Smoke B/C exact 16/8/4 sizes + per-channel kernel sum 1.0 + buffer-not-param; Smoke D finite train backward, whiten frozen, blur buffer no grad; Smoke F native-fp32 eval + flip + TTA + contiguous all finite + EMA blur-buffer invariance).

### Surprises & Discoveries
- **The planned reflect-pad MaxBlurPool is severely throughput-bound** (M1 Smoke E). The original `F.pad(reflect)` + dense `MaxPool2d(stride=1)` + depthwise conv chain ran at only **0.40× baseline** in a micro-net probe (full ResNet9: ~0.69× → ~103 ep), failing the ≥135 under-anneal gate — the EXP-016 GhostBN trap. Breakdown probe isolated TWO killers: (1) `F.pad(reflect)` (separate kernel + allocation), and (2) the dense full-resolution `MaxPool2d(2, stride=1)`.
- **Fix 1 — zero-pad in conv** (drop `F.pad(reflect)`): lifted MaxBlurPool ks3 from 0.40→0.70 (micronet) / full-net **0.884 → 132 ep**. Minor border-darkening vs reflect (DC not perfectly preserved at the 1-px border of a 32×32 map) — negligible.
- **Fix 2 — BlurPool-only (drop the dense max)**: anti-aliased strided subsample (blur the conv_bn output, stride-2) is **throughput-FREE** — full-net ks3 **1.028× → 153 ep**, ks5 0.987× → 147 ep. The depthwise binomial blur alone is cheap; the dense max was the cost.
- Full-net throughput (c0 anchor 22,923 img/s ≈ 149 ep): max-mode ks3 L123 132ep / L12 134ep / ks5 127ep (all ≤135 gate); blur-mode ks3 L123 **153ep** / ks5 147ep (both clear).

### Decisions
- **Official cells use BLUR_MODE="blur" (BlurPool-only), NOT the planned MaxBlurPool ("max").** Rationale: the faithful Zhang MaxBlurPool (dense max + blur) costs 12–15% throughput → 127–134 ep, failing the hardened `num_epochs ≥ 135` under-anneal gate (the #1 failure mode on this goal; project-insights High). BlurPool-only is genuinely throughput-free (147–153 ep), so it tests the anti-aliasing inductive bias WITHOUT the under-anneal confound that disqualified GhostBN (EXP-016). This is the gate-respecting choice.
- **Confound noted**: blur-only changes two things vs baseline at once — (a) max→binomial-weighted-average aggregation, and (b) aliased→anti-aliased subsampling. A clean isolation of (b) alone needs MaxBlurPool, which is throughput-bound here. The faithful MaxBlurPool (compile-funded to recover throughput, EXP-014 recipe) is the documented follow-up if blur-only shows signal. A tie is interpreted as "anti-aliased blur-subsampling does not beat max-subsampling at full epochs," with the (a)/(b) entanglement flagged.
- **Cells**: c0 (baseline MaxPool, 149ep anchor) / cA=blur ks3 L123 (PRIMARY, ~153ep) / cB=blur ks5 L123 (stronger low-pass, ~147ep). ks2 dropped (even kernel needs asymmetric padding; ks3/ks5 odd → clean symmetric conv padding). Same-session on GPU 1 with per-cell background nvidia-smi sampling.

## Experimental Adjustments
- **reflect-pad → zero-pad in conv**: removed the dominant throughput cost (`F.pad(reflect)` kernel+alloc); MaxBlurPool ks3 0.40→0.70 (micronet). (ref: M1 breakdown probe /tmp/exp018_bd.py)
- **MaxBlurPool ("max") → BlurPool-only ("blur") for official cells**: max-mode 127–134 ep fails the ≥135 gate; blur-mode 147–153 ep clears it, throughput-free. (ref: M1 Smoke E full-net probe — c0 22,923 img/s, blur ks3 1.028×→153ep, max ks3 0.884×→132ep)

## Run Log

### Run 1 — c0 (baseline), cA (blur ks3 L123), cB (blur ks5 L123)

Metadata:
- **Job ID**: (background bash — see TaskOutput)
- **Log file(s)**: experiments/018/run_c0.log, run_cA.log, run_cB.log; gpu_c0/cA/cB.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-30
- **Ended**: 2026-06-30

Description:
- Three sequential same-session cells on GPU 1, each a separate `train.py` process under `timeout 600`, with a background nvidia-smi sampler per cell (mid-run contention guard). c0 (`BLUR_KSIZE=0`) is the full-speed baseline MaxPool control (~149ep, same-session anchor). cA (`BLUR_KSIZE=3 BLUR_MODE=blur`) is the PRIMARY anti-aliased downsampling cell (BlurPool-only, ks3 triangle filter, ~153ep — throughput-free). cB (`BLUR_KSIZE=5 BLUR_MODE=blur`) is a stronger low-pass (binomial-4, ~147ep) mapping the smoothing-strength response. Test: does either anti-aliased cell exceed c0 by a clear >0.1pp margin (≥96.48) at near-full epochs.

Observations:
- **NO-IMPROVEMENT: both anti-aliased cells LOSE to the same-session control.** cA (ks3) 96.23 = **−0.08pp** vs c0 96.31 (within the ~0.1–0.2pp noise floor); cB (ks5, stronger blur) 96.16 = **−0.15pp**. (source: run_cA.log, run_cB.log, run_c0.log)
- **Monotonic degradation with blur strength** (ks3 −0.08 → ks5 −0.15) → the low-pass filtering itself is mildly HARMFUL (discards high-freq detail), not helpful; no anti-aliasing headroom to exploit. (source: run_cA/cB.log)
- **Throughput-free as predicted**: num_epochs 150/153/146 (blur cells AT or ABOVE c0's epoch count — BlurPool-only is genuinely free; cA even ran +3ep over c0). NOT under-anneal — fully annealed (best≈final: c0 96.31/96.29, cA 96.23/96.12, cB 96.16/96.10). (source: run_*.log)
- **Not under-fit**: ep25 c0 92.18 / cA 92.41 / cB 91.98 — cA is even slightly ABOVE c0 at ep25, ruling out under-training; the loss is at the annealed ceiling, not a convergence artifact. (source: run_*.log)
- **Clean equal conditions**: GPU 1 ours throughout (max mem ~6 GB/cell, no foreign job, gpu_*.log); evals == num_epochs (150/153/146, ≤1/epoch); seed 42; prepare.py byte-unchanged; num_params 7,784,627 unchanged (blur = buffer).

Key Metrics:
- c0 best_test_acc: 96.31% (final 96.29 @ep150); 150 ep; 458.3s wall; 1635.4 MB (source: run_c0.log)
- cA best_test_acc: 96.23% (final 96.12 @ep153); 153 ep; 447.9s wall; 1635.5 MB (source: run_cA.log)
- cB best_test_acc: 96.16% (final 96.10 @ep146); 146 ep; 448.9s wall; 1635.6 MB (source: run_cB.log)
- ep25: c0 92.18 / cA 92.41 / cB 91.98 (source: run_*.log)
- M1 full-net throughput (img/s): c0 22,923 anchor; blur ks3 1.028× (→153ep actual), blur ks5 0.987× (→146ep actual); max-mode ks3 0.884×→132ep (disqualified, see Decisions) (source: /tmp/exp018_smoke2.py probe)

## Verification Results

### Conditions Checked
- **(a0) Metric integrity / anti-gaming (run ALWAYS)** — PASS (clean). Exactly one `best_test_acc:` summary line per cell; summary == max per-epoch test_acc (c0 96.31==96.31, cA 96.23==96.23, cB 96.16==96.16); eval-count == num_epochs (150/153/146); `git diff --quiet -- prepare.py` UNCHANGED; `git status` shows only train.py modified; seed 42; num_params 7,784,627 unchanged. No gaming. → not invalid.
- **(a) Completes within budget, valid metric, wall < 600s, ≥135 epochs** — PASS. All three cells valid best_test_acc, training_seconds 300.0, wall < 460s (no timeout), num_epochs 150/153/146 all ≥135 (the under-anneal gate clears — BlurPool-only throughput-free). (source: run_*.log)
- **(b) Best blur cell ≥96.48 AND > same-session c0 by clear >0.1pp, replicated** — FAIL. Best blur cell = cA 96.23% < 96.48 bar AND < c0 96.31% (−0.08pp, not >c0+0.1pp). cB 96.16 even lower. No apparent win → mandatory confirmation re-run NOT triggered. → no-improvement. Stop; no further conditions.

### Verdict basis
no-improvement: throughput-free anti-aliased downsampling (BlurPool-only) did not beat the same-session MaxPool control — cA (ks3) tied-to-lost at −0.08pp (within noise), cB (ks5) lost −0.15pp, with monotonic degradation as blur strength rose. All cells fully annealed at ≥146 epochs (no under-anneal) and not under-fit (ep25 healthy), so the null is genuine. Final verdict in 04-analysis.md.

### Informational Metrics
- peak_vram_mb: c0 1635.4 / cA 1635.5 / cB 1635.6 MB (blur adds no params) (source: run_*.log)
- num_epochs: c0 150 / cA 153 / cB 146 (blur-only throughput-free; cA +3ep over c0) (source: run_*.log)
- total_seconds: c0 458.3 / cA 447.9 / cB 448.9 (training_seconds 300.0 all) (source: run_*.log)
- num_params: 7,784,627 (unchanged — blur is a buffer) (source: run_*.log)
- ep25 test_acc: c0 92.18 / cA 92.41 / cB 91.98 (source: run_*.log)

## Errors & Dead Ends

<!-- Append only. -->

## Human Notes

> (none — autopilot)
