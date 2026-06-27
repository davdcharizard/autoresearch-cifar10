# Experiment Report EXP-031: Progressive resizing 24→32 (in-step GPU downsample, switch at 50%)

- **Date**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-031.md
- **Plan**: plans/plan-031.md
- **Exp-log**: logs/exp-log-031.md
- **Verdict**: **no-improvement** (96.69 vs bar 96.81; baseline 96.71, Δ −0.02)

## Goal
Maximize CIFAR-10 best_test_acc (%) of a ResNet-20-family model trained within a fixed 300s charged budget, modifying train.py only. Baseline 96.71 @ 1990397 (distribution top; run mean ≈96.57, σ ≈0.16); success bar = 96.81; candidates need TRUE effects ≥ +0.3.

## Idea & Hypothesis
**Idea**: Train the first 50% of the charged budget at 24×24 (in-step, charged GPU bilinear downsample), the second 50% at native 32×32 — numerically the baseline regime. Both shapes pre-compiled in startup warmup (`torch.compile(dynamic=False)`).

**Why it was chosen**: It attacked the epoch count — the one currency with a measured positive conversion law (EXP-006: +25 epochs = +0.48) — and was the campaign's FIRST candidate with regime-MATCHED external evidence: progressive resizing (fastai DAWNBench, MosaicML composer) is natively a wall-clock-budget method, unlike the 0-for-10 fixed-epoch speedrun transfers.

**Hypothesis**: Phase-1 dt ≈14–17ms → ~165–185 total epochs (vs 139); if the low-res quality toll ≤ ~0.2, the extra ~35–46 epoch-equivalents convert at the EXP-006 rate into best ≥ 96.81. Falsifiable: phase-1 dt ≥20ms kills the arithmetic; failure to rejoin the baseline family post-switch signals toll dominating.

## Approach
Four train.py edits on `autoresearch/exp-031` (baseline path byte-identical otherwise): constants `LOW_RES=24`, `RES_SWITCH_FRAC=0.5`; `torch.compile(model, dynamic=False)`; dual-shape startup warmup (3 iters at 32², 3 at 24², no optimizer step); timed-step branch `if progress < RES_SWITCH_FRAC: inputs = F.interpolate(inputs, size=LOW_RES, mode="bilinear")` — charged inside the timed region so the speedup pays for its own downsample.

**Run-2 protocol fixes (uncharged wall side only)**: phase-1 eval thinning to every 3rd epoch (evals there are cosmetic — 24px-train/32px-eval mismatch — and "≤ once/epoch" permits fewer) and 16 loader workers (2×NUM_WORKERS; at 13.5ms/step the GPU outpaces the 8-worker CPU augment pipeline).

## Execution
- **Run 1 — WALL_CAP_KILL at 601s (81% charged, ep 159)**: mechanism perfect (P1 13.3–14.3ms, stall-free switch, P2 22.0–22.7ms, zero contention) but the plan's ~460–490s wall estimate missed that UNCHARGED costs scale with epochs: ~113 phase-1 evals ×1.3s ≈146s + ~120s loader stalls → projected ~680s uncapped. Hypothesis unmeasured (anneal incomplete, best-at-kill 95.24 still climbing). Classified protocol/estimation failure → one adjusted resubmit.
- **Run 2 — clean completion**: rc=0, total 457.0s, startup 17.5s, training 300.0s, **185 epochs / 17,866 steps**, 109 eval lines, VRAM 1613.0MB, params 4,286,026. Per-segment profile: P1 221 windows mean **13.5ms** (0 slow >22), P2 135 windows mean **22.3ms** (0 slow >27; ≤24 numerics guard passed). GPU 0 free at first poll both runs.

## Results
**best_test_acc 96.69 (ep 179); final 96.67; final_test_loss 0.1846. Bar missed by 0.12; −0.02 vs recorded baseline; +0.12 vs baseline mean = within 1σ — no detectable true effect.**

The mechanism delivered everything it promised EXCEPT the conversion:
1. **Throughput arithmetic: fully delivered.** 24px dt = 13.5ms = 0.60× of 22.4ms (near the 0.5625 FLOPs ideal — H20 stays compute-bound at this shape). +46 epochs (185 vs 139), the largest epoch surplus of the campaign (EXP-006 itself only bought +25).
2. **Switch hazards: all bounded as engineered.** No charged compile stall (transition window 17.5ms = blend of phases); eval dipped to 80.49 at the switch epoch then recovered to 86.92 within ONE epoch and climbed monotonically — BN/weight adaptation was fast and cheap, as the self-healing argument predicted. Phase-2 numerics baseline-identical (22.3ms mean, family train-loss curve, identical VRAM/params).
3. **Conversion: zero.** Final-7 median 96.64 ≈ baseline-family plateau (~96.6); best 96.69 ≈ baseline mean +0.12 (noise). The +46-epoch surplus converted at ~0.00/epoch versus EXP-006's +0.019/epoch.

**Why (mechanism)**: The EXP-006 conversion law holds for epochs *of the same training distribution*. Phase-1 epochs here taught 24px statistics — the formative high-LR phase learned features at the wrong object scale, and what phase 2 inherited was roughly equivalent in plateau-value to what the baseline reaches by mid-schedule anyway. By the structural-law lens this is the MAX-STATISTIC law again in a new guise: low-res training is a transit-speed gain (reaches 90% faster, ep ~122 vs baseline's later arrival) whose advantage decays to zero by the converged plateau — the only thing the metric pays for. The regime-matched MosaicML/ImageNet evidence ("quality-neutral at −30% wall") did not transfer downward to CIFAR: at 32px CIFAR images are already information-critical; ImageNet 160px-vs-224px discards redundancy, CIFAR 24px-vs-32px discards signal. External-transfer record: 0-for-11.

**Wall-cap discovery (decisive for future planning)**: the 600s TOTAL cap, not the 300s charged budget, binds high-epoch configs. Each epoch costs ~2.2s charged + ~1.3s uncharged eval; loader sustains only ~20ms/step with 8 workers (stalls uncharged but real wall). Throughput ideas must budget: 300 + startup + epochs×1.3 + stalls ≤ 600 ⇒ epochs ≲ 200 even with stall-free loading — and eval thinning (≤1/epoch is a ceiling, not a floor) plus 2×workers are now validated wall levers (Run 2: 457s at 185 epochs).

## Verification
- Condition 1 (best ≥ 96.81): **FAIL** — 96.69. Pre-condition profile PASS (clean both segments, 185 ≥ 150 epochs, params 4,286,026, training_seconds 300.0). First-failure-stop; conditions 2–3 informationally pass (rc=0, 457.0s ≤ 600; 109 evals ≤ 185 epochs).
- Trustworthiness: high — dt signatures match prediction in both phases, zero slow windows, params/VRAM identical to baseline, eval thinning compliant with the once-per-epoch ceiling, downsample charged in-step (accounting honest). No false-failure risk: the plateau was fully converged (15 evals within ±0.07 of final).
- Verdict basis: clean miss of the bar; effect within noise of baseline mean → **no-improvement**.

## Key Learning
The epoch-conversion law (EXP-006) is conditional on the epochs carrying the SAME training distribution: +46 epochs at 24px converted at zero because low-res learning is a transit-speed gain that the converged-plateau metric never pays for — and CIFAR at 24px loses signal, not redundancy, so even regime-matched (wall-clock) external evidence failed to transfer (0-for-11). Separately: the 600s wall cap is the true binding constraint for any high-epoch idea (~1.3s uncharged eval/epoch + loader stalls), with eval thinning and 2× workers now validated wall levers.

## Unexplored Avenues
- **Interior resolution points (28px; switch fracs 0.3/0.65)**: NOT bracketed by this result — 28px loses less signal and still buys ~+20 epochs. But the zero conversion at 24px despite fast adaptation suggests the toll is distribution-inheritance, not adaptation shock, which 28px only dilutes; expected ≤ +0.1. Low priority.
- **Up-resizing finish (FixRes-style 32→40 final phase)**: inverse direction; pays dt instead of earning it, and EXP-025/029 closed eval-time alignment tricks. Effectively closed.
- **Reverse curriculum (32 early, 24 late)**: violates the deferral law (cheap epochs belong where they're cheap, but the anneal tail is load-bearing at full res — EXP-025); closed by mechanism.
- **Wall levers as enablers**: eval thinning + 16 workers are now proven and reusable for ANY future high-epoch candidate (they bought 144s of wall) — they are protocol tools, not metric ideas.

## Next Steps
1. **Combine near-misses at unchanged distribution**: the only positive-conversion currency left is full-res epochs; nothing remaining cheapens a full-res step without numerics change (EXP-021) — so attack the plateau LEVEL directly: e.g., SWA/weight-averaging over the final plateau WITH augmented-loader BN re-estimation (EXP-029 taught how to do this correctly; the max-statistic rewards plateau level and EMA failed (EXP-011) but SWA's flat-minima mechanism is different — though noise-optimum law (4) is a known headwind). Confidence: low-medium.
2. **Radical: deeper-not-wider at matched dt** (e.g., ResNet-32 at width 3×, channels 48/96/192 — 32-aligned per EXP-005): different capacity SHAPE at similar step cost; allocation position is first-class (EXP-017) and depth at fixed budget is unbracketed in the deferral-law sense. Must pass the early-dt gate ≤23ms. Confidence: low-medium.
3. **Exploit the validated wall levers with the baseline recipe unchanged**: 16 workers alone removes residual baseline loader stalls (~40s) but baseline epochs are dt-bound, not wall-bound — no direct metric path; keep as protocol default only. Confidence: low (as a metric idea).

## Exit Action Results
(no exit actions defined for this goal)
