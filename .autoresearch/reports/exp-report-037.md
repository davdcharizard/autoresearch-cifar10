# Experiment Report: EXP-037 — SE channel attention (r=16, all 9 blocks) with near-identity init

- **Date**: 2026-06-10
- **Verdict**: no-improvement
- **Primary metric**: best_test_acc = **96.37%** (Run 2; Run 1 96.34 — concordant replicates; baseline 96.71, bar 96.81, delta −0.34)
- **Branch**: autoresearch/exp-037 (discarded)
- **Artifacts**: brainstorm/brainstorm-037.md · plans/plan-037.md · logs/exp-log-037.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc %, higher is better) within the fixed 300s charged training budget, modifying only `train.py`. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

## Idea & Hypothesis
**Idea**: After EXP-036 completed the recipe-constant audit (every constant dosed, incumbent always won), the only law-permitted open territory was architecture adding NEW functional capacity while free in early heat, dt, numerics, and noise. SE channel attention (Hu et al., CVPR 2018) was the best-evidenced such mechanism: published in-domain CIFAR-10 ResNet gains +0.5–1.2 at fixed epochs, a converged-LEVEL claim (input-conditioned channel gating). Engineered against each measured law: near-identity init (fc2 zero-weight + bias 2.0 → gate = sigmoid(2) = 0.881 constant at step 0) vs the deferral law; pre-registered early-dt GATE_KILL at ≥26.5ms vs launch-bound pricing; default compile/bf16 and batch/momentum/augmentation untouched.

**Hypothesis**: SE raises the converged plateau LEVEL at measured dt ≤ ~2.5ms extra, predicting best ≥ 96.81. Falsified by (a) GATE_KILL on dt, or (b) clean converged plateau ≤ baseline band — extending the heavy-augmentation absorption law to attention.

## Approach
train.py only (+24/−2 lines): `SEModule` (squeeze = spatial mean; excite = Linear(C→C/16)→ReLU→Linear(C/16→C)→sigmoid; channel-wise mul) inserted after bn2 before the residual add in all 9 BasicBlocks; `_weights_init` taught to skip modules flagged `skip_kaiming` so the global kaiming pass cannot randomize the zero-initialized gates (verified post-construction: fc2 weight absmax 0.0, gate 0.8808). Params 4,319,710 (= baseline + 33,684, hand-calc exact). All hyperparameters unchanged.

## Execution
Two full clean runs (both GATES_CLEAR on poll 1). Run 1 (19:08): SE-dt GATE passed decisively at 24.0/24.9/24.0ms; 129 epochs; best 96.34; rc=0, 473.2s; cold-compile startup 24.4s. Post-hoc fine profile showed 5 windows >29ms → classified contaminated per the plan's pre-registered ≤2 rule → rerun. Run 2 (19:18): 128 epochs; best 96.37; rc=0, 456.8s; warm-cache startup 10.0s — and 9 such windows, all EXACTLY 30.0ms, at an equally clean load. **Root cause: instrument quantization, not contention** — the step line prints pct with 1 decimal (0.1% = 0.3s), so 50-step windows at true ~24.2ms round to 0.4%→24.0ms or 0.5%→30.0ms; the 29ms threshold sat between quantization rungs. Coarse watchdog windows (~400 steps): 24.0–25.5ms, zero exceedances, both runs. Pre-condition intent (uncontaminated) satisfied; no further reruns (the one contamination rerun was spent and the contamination hypothesis affirmatively refuted).

## Results
- **best 96.34 / 96.37 (two draws), test_loss 0.1882/0.1884, 129/128 epochs, dt 24.1/24.2ms (+1.7ms SE cost), params 4,319,710, VRAM 1644/1616MB.**
- **Deficit decomposition lands exactly on zero retained gain**: +1.7ms → −10.5 epochs → expected deficit-only level ≈ 96.57 − 0.15 ≈ 96.42; measured 96.34/96.37 sits at/just-below that within σ. test_loss ~0.188 vs family ~0.185 — basin statistically indistinguishable. SE's gates bought nothing in either currency: the published +0.5–1.2 LEVEL effect did not appear AT ALL, not even partially.
- **The near-identity init pattern worked as engineering**: ep1 35.35/34.82 (family ~38, EXP-018's broken-init signature would be ~20), mild early lag recovered by mid-run — deferral was successfully dodged, so the null is attributable to the mechanism itself, not to a botched warm start.
- **dt law refinement**: 9 SE modules cost +1.7ms total (~0.19ms/module) — an order below the 2.5ms/block full-block cost; small fused pointwise+matmul chains are cheap under default compile. Useful pricing datum: micro-attachments to existing blocks are nearly free; whole blocks are not.
- **Trajectory fit**: 32nd consecutive miss, and the third consecutive experiment (EXP-035 SAM, EXP-036 LS, EXP-037 SE) where a mechanism with solid fixed-epoch literature returned EXACTLY deficit-only under this recipe. The absorption law sharpens: TA+RE at batch 512 with a completed one-cycle anneal saturates not just regularization/flatness but apparently the entire class of "auxiliary capacity/conditioning" benefits measured in lighter-regularization regimes — SENet's CIFAR baselines used crop+flip only. External transfer is now 0-for-14, with all 14 failures explained by five named mechanisms (deferral, max-statistic, dt pricing, noise optimum, augmentation-regime absorption).
- Hypothesis falsified via arm (b): clean converged plateau below the baseline band; the attention axis closes with a two-draw measured datum.

## Verification
First-failure-stop per plan-037, verified on the protocol-clean Run 2 with Run 1 concordant. Pre-condition: fine 50-step profile's slow-window criterion found instrument-invalid at dt≈24ms (pct quantization rungs 24.0/30.0 straddle the 29 threshold — diagnosed, documented in exp-log Experimental Adjustments); evaluated on coarse watchdog windows: 0 exceedances both runs, means 24.1/24.2 ≤ 26 ✓. Integrity: params 4,319,710 ✓, training_seconds 300.0 ✓, eval_lines = num_epochs (128/128, 129/129) ✓. **Condition 1 FAILED on merits: 96.37 < 96.81.** Conditions 2–3 skipped per protocol (incidental: rc=0, 456.8s ≤ 600; evals = epochs). No false-failure risk — two independent clean draws agree to 0.03. Verdict: **no-improvement**.

## Unexplored Avenues
- **Stage-3-only SE** (the pre-registered fallback, never triggered): would cut the dt cost to ~+0.6ms, but the full-SE read shows the GAIN is zero, not the cost too high — a cheaper dose of a zero-effect mechanism is still zero. Closed by this read.
- **ECA / other attention forms (spatial, CBAM)**: same absorption objection now carries a measured SE datum plus EXP-030's max-pool head failure; spatial attention also adds dt. Low prior.
- **Attention WITH reduced augmentation** (trade TA for SE): forbidden direction — the dose-response law (4 points, both sides) says any pressure reduction loses more than SE's literature gain; the certified recipe's augmentation cannot be traded away.

## Next Steps
1. **Treat "auxiliary module with literature gains under lighter regularization" as a screened-out class** (SAM, LS-dose, SE all returned deficit-only); require future candidates to have evidence under HEAVY-augmentation, budget-matched regimes specifically (high confidence in the screen).
2. **The only remaining un-priced structural direction**: changes that REMOVE cost rather than add capability — e.g., a numerics-preserving dt reduction (none identified since EXP-021) or architecture simplification that keeps the function class (low confidence any exists; the EXP-006 conversion law demands numerics equivalence).
3. **σ-aware framing hardens**: with 32 misses, every axis measured, and three consecutive exact-deficit nulls, the recipe's remaining headroom to the bar (~+0.24 above its own mean) is plausibly below ANY single intervention's true effect at this budget; the highest-EV use of further loops is replicate-based variance exploitation only if a mechanism with in-regime evidence emerges (medium confidence).

## Key Learning
SE channel attention — engineered cleanly past the deferral law (near-identity init verified) and priced at only +1.7ms — returned exactly its epoch deficit with a family-identical basin: the heavy-augmentation absorption law extends from regularizers (SAM, LS) to capacity-conditioning modules. Fixed-epoch architecture gains measured on crop+flip baselines do not survive TA+RE + completed-anneal recipes; the in-regime-evidence screen is now the binding filter for any future candidate.
