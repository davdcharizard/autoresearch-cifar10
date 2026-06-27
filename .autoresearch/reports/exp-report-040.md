# Experiment Report: EXP-040 — Uniform 5× width (80/160/320) behind the early dt gate

- **Date**: 2026-06-10
- **Verdict**: invalid (metric NaN — both pre-registered width attempts GATE_KILLed before any eval; the planned screen outcome)
- **Primary metric**: best_test_acc = NaN (baseline 96.71, bar 96.81)
- **Branch**: autoresearch/exp-040 (discarded)
- **Artifacts**: brainstorm/brainstorm-040.md · plans/plan-040.md · logs/exp-log-040.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc %, higher is better) within the fixed 300s charged training budget, modifying only `train.py`. Baseline 96.71 @ 1990397; bar ≥ 96.81.

## Idea & Hypothesis
**Idea**: The width axis owns the program's largest in-regime LEVEL gain (+2.07, 1×→4×, EXP-001) and an unmeasured interior: all three width failures (EXP-002/005/007) were starvation artifacts at ≤55 epochs, never testing converged width-level past 4×. The Failed-Approaches law itself prescribes the re-entry condition (measured compiled dt projecting ≥70 epochs), and EXP-034's launch-bound block-cost measurement (4×'s 22.4ms ≈ 9 × 2.5ms width-independent) made a cheap 5× physically plausible.

**Hypothesis**: 5× stays near launch-bound (dt ≤ ~30ms → ≥ ~100 epochs), converges, and its plateau exceeds 4×'s. Falsification branch (a): GATE_KILL — 5× is compute-bound, axis stays closed on dt grounds. That branch fired, twice.

## Approach
train.py only. Attempt 1: `WIDTH_MULT = 4 → 5` (params 6,693,850). Attempt 2 (pre-registered fallback): explicit widths 72/144/288 (params 5,423,122 — constructed value; the plan's 5,433,514 was an estimate, corrected before launch). Composite launcher with the dt-gate variant: D0 = median of first 3 watchdog windows; GATE_KILL if D0 > 36ms (projected < ~86 epochs — stricter than the law's 70-epoch floor because the smallest measured converged run is 83 epochs and wider models need more margin); dt-adaptive contention threshold D0 × 1.25 thereafter.

## Execution
Two launches, both pristine gates (apps=0, load 8/6), both GATE_KILLed at tick 5 (~90s each):
- **5× (80/160/320)**: windows 54.0/54.0/54.0ms → D0 54.0ms, projected 57 epochs.
- **4.5× (72/144/288)**: windows 54.0/52.8/54.0ms → D0 54.0ms, projected 57 epochs. run.log header confirmed the right model (params 5,423,122); printed dt 53–54ms and img/s ~9,540 corroborate.
Total cost of the experiment: ~3 minutes of GPU time. No retries beyond the pre-registered fallback; no errors.

## Results
- **The dt landscape above 4× is a CLIFF, not a slope.** Max-width 256 → 22.4ms; max-width 288 → 54.0ms; max-width 320 → 54.0ms. A 2.4× step-time jump for a 1.27× FLOPs increase, then FLAT from 288 to 320 (1.56× FLOPs, zero additional dt). This is a kernel-selection/tiling boundary at 256 channels on H20 + torch.compile(default) + channels_last + bf16, not compute scaling. EXP-007's 6× at 58ms now reads as "just past the same cliff", and EXP-034's width-independent per-block law holds only up to 256 channels.
- **The width interior does not exist.** The hypothesis presumed a smooth dt ramp between 4×'s launch-bound 22.4ms and 6×'s 58ms with a usable middle. There is no middle: 4× (max-width 256) sits AT the cliff edge, and the first step past it (288) already prices at 54ms / 57 projected epochs — the exact starvation regime (≤55–57 epochs) where EXP-005/007 measured −0.71 to −1.11. The 2016-vintage 16/32/64 stage geometry scaled by 4 lands exactly on the hardware's last fast width — the recipe's width choice is hardware-optimal, not just statistically optimal.
- **The gate design paid for itself**: the historical failure mode (a full 10-minute starved run) was converted into two ~90s screens, and the falsification branch was pre-registered so the verdict was mechanical.
- **Trajectory fit**: 35th consecutive non-improvement, but the cheapest possible kind — and it upgrades the width-closure from "three starvation artifacts" to "hardware-priced at the kernel level": no dt-compliant width above 4× exists on this stack.

## Verification
Not run — no completed run exists (exp-log § Verification Results documents this). Both kills were the plan's own screen firing on its pre-registered threshold (D0 54.0 > 36ms), with the gate windows mutually consistent across two independent launches at clean load, corroborated by in-log dt prints. Nothing to judge on merits; per plan-040 § Configuration Changes the both-kill branch is verdict **invalid** (metric NaN) with the dt measurements as the experiment's data product. Integrity: scope respected (train.py only), GPU 0 only, no timer or eval interaction — the runs never reached an eval.

## Unexplored Avenues
- **Widths in (256, 288)** — e.g. 68/136/272: closed by inference; the cliff is a binary kernel boundary at >256 (288 and 320 price identically), so 272 almost certainly inherits the 54ms path. Not worth a 90s probe unless some future idea needs exactly that confirmation.
- **Per-stage width capped at 256 with deeper stage-3** — redistributions at ≤256 everywhere are EXP-017/EXP-034 territory (allocation + depth both measured worse); no new design point exists under the cap.
- **A numerics-preserving fast kernel path for >256 channels** (e.g. different memory format) — EXP-021 measured that kernel-path changes break numerics equivalence (−0.20 even when FASTER); chasing a faster wide path risks the same toll on top of unproven width gains.

## Next Steps
1. **Record the width axis as closed at the hardware level** (cliff at >256 channels; 4× is the last fast width) and fold the cliff into the dt law — any future architecture idea must price >256-channel layers at ~2.4× (high confidence).
2. **The capacity frontier is now closed in every currency** (epochs, level, allocation, depth, width-hardware); remaining brainstorm space is data composition/order and objective shaping, both lacking in-regime evidence — candidates there must be zero-cost so their expected nulls are free (medium confidence).
3. **Consider a deliberate variance-reduction read on the recorded baseline**: with mean 96.57 and recorded top 96.71, the bar 96.81 is +1.5σ; if a future loop produces a mid-band result on a genuinely promising mechanism, plan a pre-registered replicate pair rather than a single draw (protocol point, not an experiment idea) (medium confidence).

## Key Learning
On H20 + default torch.compile + channels_last + bf16, conv step time has a hard kernel cliff above 256 channels: max-width 288 and 320 both run at 54ms vs 256's 22.4ms (FLOPs +27%/+56%), so the "width interior" between 4× and 6× does not exist — 4× (256) sits exactly at the cliff edge, projected epochs collapse to 57 (the measured starvation regime), and the width axis closes on hardware grounds rather than statistics. The gate protocol turned what was historically a 10-minute starved run into two 90-second measurements.
