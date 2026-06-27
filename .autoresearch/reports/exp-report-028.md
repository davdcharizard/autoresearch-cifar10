# Report EXP-028: Muon optimizer for conv weights (airbench-anchored hybrid)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-028.md
- **Plan**: plans/plan-028.md
- **Log**: logs/exp-log-028.md

## Goal

Maximize CIFAR-10 best_test_acc (%) of the ResNet-20-derived recipe within the fixed 300s training budget. Baseline 96.71 @ 1990397 (EXP-006); improvement bar = 96.81 (+0.1pp). EXP-027 calibration: the baseline distribution has mean ≈96.57, σ ≈0.16 — the bar sits ≈+1.5σ above the true mean, so candidates need true effects ≥ +0.3. Question tested: does changing the update GEOMTRY of the conv weights — Muon's orthogonalized momentum, the only optimizer family with a CIFAR-10 speedrun record pedigree — buy enough per-step sample efficiency to beat both the bar and its own step-time tax?

## Idea & Hypothesis

First change of optimizer FAMILY in the campaign (all prior optimizer work was SGD-internal: lr/momentum/WD/schedule, all bracketed and closed). Muon (Jordan): orthogonalize the nesterov momentum of each conv weight via 5 Newton-Schulz iterations in bf16, scale by √max(1, fan_out/fan_in), decoupled WD; fc/BN/biases keep the baseline SGD path. Anchored on airbench's measured CIFAR-10 hyperparameters (lr 0.24, momentum 0.6, convs only) — the only conv-net anchor in the literature. Hypothesis: Muon's benefit class is per-step sample efficiency (airbench's record gains), which is exactly what a fixed-wall-clock budget rewards — IF the gain exceeds the NS-5 eager-matmul step-time tax (estimated +2–4ms on 22.4ms) plus the bar's +0.3 true-effect requirement.

## Approach

train.py only (+42/−4 vs 1990397 on branch autoresearch/exp-028): (1) `MUON_PEAK_LR=0.24`, `MUON_MOMENTUM=0.6`; (2) `zeropower_via_newtonschulz5` (quintic coeffs 3.4445/−4.7750/2.0315, bf16, transpose-if-tall); (3) param split — 19 conv weights (ndim==4) to an inline Muon step with per-param momentum buffers, fc weight keeps SGD+WD, ndim≤1 keeps SGD without WD; (4) Muon block inside the timed region after `optimizer.step()`: nesterov buffer → NS-5 on the (C_out, C_in·k·k) reshape → √(rows/cols) scale → decoupled WD → param update; `muon_lr_now` reuses the time-keyed one-cycle shape (×0.24/0.4). Key decisions: `.reshape` not `.view` (channels_last non-contiguity; copy honestly timed), `o.to(p.dtype)` bf16→fp32 cast. Implementation-time catch: `optimizer.zero_grad()` would NOT clear Muon params' grads (they're outside the optimizer's groups) — silent gradient accumulation; replaced both call sites with `model.zero_grad(set_to_none=True)`.

## Execution

Two runs, one code state. Run 1 (14:22–14:30Z): rc=0, best 96.42, 122 epochs, but post-hoc profile showed 4 of 234 windows >32ms vs the ≤2 contamination limit → discarded per contention protocol and rerun once (epochs matched expected exactly, 122 vs 122.2, so distortion was negligible — but protocol fidelity kept). Run 2 (14:35–14:43Z, byte-identical): rc=0, profile CLEAN (0 of 236 windows >32ms; epochs 123 vs 123.1 expected) → stands as the final run. Side finding: Run 1's spikes did not reproduce → transient foreign contention, not intrinsic Muon-step jitter. The pre-authorized Run-2-at-lr-0.12 trigger (divergence or ep10 <70%) never fired. No retries, no crashes.

## Results

- **Primary metric**: 96.53 (baseline: 96.71, delta: −0.18, −0.19%) — and −0.04 vs the 96.57 baseline MEAN
- **Observations**: Muon's claimed mechanism is REAL and cleanly visible: ep10 85.7 vs baseline family ~78 (+7pp), ep30 92.2, ep60 94.9, ep90 95.9 — consistently ahead at matched epoch count early/mid. But the converged plateau capped at 96.45–96.53 (final-7 median 96.47, tight) with final_test_loss 0.193 vs baseline ~0.185. Cross-run agreement excellent (Run 1 best 96.42 / Run 2 best 96.53 — both inside the baseline noise band). NS-5 measured cost on H20: +2.9ms/step (25.3 vs 22.4, ~13%) → 123 epochs vs 139 (−16). VRAM +18MB (momentum buffers); startup unchanged (NS is eager, no compile cost).
- **Analysis**: Three losses stack against the early gain: (1) the 16-epoch NS tax (~−0.25 by the EXP-006 epochs-curve); (2) the orthogonalized geometry converges to a slightly WORSE basin under this recipe (test_loss 0.193–0.194 in both runs vs ~0.185 — a level shift, not noise); (3) the early sample-efficiency advantage decays to zero as both schedules anneal — it buys arrival time, not plateau height, and MAX-statistic rewards plateau height. Net effect ≈ 0 vs the baseline mean — far from the +0.3 true effect the bar demands. This mirrors EXP-026's lesson from the activation axis: in a fixed-wall-clock regime, per-step quality improvements must beat their own ms-cost ladder, and the baseline's components are each near their measured optimum.
- **Key Learning**: Muon's per-step sample efficiency is real (+7pp at ep10) but buys arrival time, not plateau height: the plateau lands at the baseline mean while NS-5's +2.9ms/step costs 16 epochs and the basin's test_loss is worse. Optimizer-geometry axis closed alongside the SGD-internal axes.

## Verification

- **Conditions**: condition 1 failed (best_test_acc 96.53 < 96.81); conditions 2–3 skipped per first-failure-stop (both would pass: rc=0/447.1s ≤600; eval_lines 123 = epochs)
- **Review Notes**: results confirmed trustworthy — pre-condition contention profile clean on the standing run (0/236 slow windows, epochs exactly at throughput expectation), params 4,286,026 exact, two-run agreement, no variance-spike artifacts (final-7 spread 0.08). The Run 1 discard followed the protocol's letter; its value (96.42) is consistent with Run 2 and changes nothing.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure — valid, clean result that did not meet the bar; no hard-constraint violations (train.py only, GPU 0, 1 eval/epoch, ≤600s, no seed hacking)

## Unexplored Avenues

- **Muon lr/momentum retune (e.g. lr 0.12–0.35 sweep, momentum 0.85)**: the airbench anchor comes from an 8-epoch ResNet-9 regime; ~123 epochs may want different settings. But both clean draws show a basin-quality deficit (test_loss +0.008) that retuning the same geometry is unlikely to flip by +0.3 — low priority.
- **Muon with brief SGD tail**: hand the last ~15% of the budget back to SGD to fix the basin-quality gap while keeping Muon's fast transit. EXP-025 showed mid-run regime handoffs are dangerous (BN/optimizer state shock), and the early advantage already decays to zero before the tail — speculative.
- **NS-3 instead of NS-5 (looser orthogonalization, ~40% cheaper)**: would cut the tax to ~+1.7ms (~9 epochs), but the plateau deficit, not the tax, is the binding loss — the arithmetic still doesn't reach +0.3.

## Next Steps

1. **Re-read in-scope files + papers for untouched axes** (high confidence in necessity, low in any single candidate): with optimizer geometry now closed alongside recipe-space, activations, augmentation pressure, schedule shape, width/depth allocation, and init tricks, the next idea must come from a genuinely unexplored interaction — per the standing directive, think harder and consider combining previous near-misses whose mechanisms are independent.
2. **EMA/weight averaging of the eval model** (medium): a converged-plateau-LEVEL intervention (what MAX-statistic actually rewards) that is nearly free per step — check it is not variance harvesting under Law 3 before planning; it raises the plateau by averaging, not by sampling more draws.
3. **BN-stat recalibration before final evals** (low): forward-only passes are cheap but the effect size measured in EXP-025's fragment was ~+0.1 at best — below the +0.3 screen on its own.

## Exit Action Results
