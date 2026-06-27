# Experiment Report: EXP-044 — Within-cliff asymmetric widening (stage widths 64/160/256, dt-gated)

- **Date**: 2026-06-10
- **Verdict**: invalid (pre-registered GATE_KILL branch — metric NaN, no fair mechanism test)
- **Primary metric**: best_test_acc = NaN (baseline 96.71, bar 96.81)
- **Branch**: autoresearch/exp-044 (discarded)
- **Artifacts**: brainstorm/brainstorm-044.md · plans/plan-044.md · logs/exp-log-044.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc %, higher is better) within the fixed 300s charged training budget, modifying only `train.py`. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): baseline mean ≈ 96.57, σ ≈ 0.16.

## Idea & Hypothesis
**Idea**: The last flagged untried single-model gap — asymmetric capacity within the 256-channel kernel cliff. Widen stage 2 from 128 to 160 channels (64/160/256; +18% FLOPs, +507k params; RegNet w1<w2<w3 ordering), the first capacity increase in program history that avoids both the >256 cliff (EXP-040) and the starvation regime (EXP-002/005/007 lived at 40–55 epochs): dense-law projection was 24.8ms → ~125 fully-annealed epochs.

**Hypothesis**: The added stage-2 capacity raises the converged plateau by more than its ~0.20 epoch deficit → best ≥ 96.81. Pre-registered branches: (i) ≥96.81 improvement; (ii) plateau ≈ mean − deficit with family test_loss → capacity level-saturated at 4×; (iii) GATE_KILL > 28ms → 32-aligned-but-not-64-aligned widths misprice; verdict invalid, new kernel-pricing law.

## Approach
train.py only, three edits: `WIDTH_MULT = 4` → `STAGE_WIDTHS = (64, 160, 256)`; `ResNet.__init__` takes a widths tuple; print line updated. Blocks, init, transforms, loader, optimizer, schedule, compile/warmup, timed loop, eval cadence byte-identical to baseline. CPU sanities all passed pre-launch: params exactly 4,793,290 (analytic match), forward shape/finiteness, pad shortcuts (96 channels at BOTH transitions), 2-step train smoke. Launched via the exp042-pattern composite (dual gates → background run → D0-median dt gate at 28ms → watchdog).

## Execution
Single launch; gates clear on poll 1 (apps=0, load=4 — pristine host). **GATE_KILL at watchdog tick 5 (~80s into the run)**: three ~350-step windows measured 32.4/31.7/32.6ms → D0 = 32.4ms > 28ms, projected ~95 epochs. Printed dt agreed (32ms uniform, img/s ~16,050 vs baseline ~23,000) and the host stayed clean through the decision (load 8.5, zero foreign GPU-0 apps) — true kernel cost, not contention. Training was healthy pre-kill (loss falling, evals ep1–10 reached 77.48, no NaN). Per the pre-registered branch: no relaunch, ~2 GPU-minutes total cost. No errors beyond the screen itself.

## Results
- **The dense pricing law does NOT cover non-64-aligned widths.** Prediction: 22.4 + 0.18×13.3 ≈ 24.8ms. Measured: 32.4ms — the +18% FLOPs cost +10.0ms instead of +2.4ms, ~4× the law. Every layer was ≤256 channels, so this is NOT the EXP-040 cliff; it is a second, independent alignment constraint: on H20 + torch.compile(default) + channels_last + bf16, channel counts that are 32-multiples but not 64-multiples fall off the fast-kernel path.
- **Retro-consistency makes this a law, not a one-off**: EXP-005's 80/160/320 collapse (8.6k img/s) was previously attributed to the >256 cliff plus vague "alignment"; EXP-034's 48/96/192 fallback ran 27.4ms where block-count arithmetic predicted less. Both contained non-64-multiples. Three independent observations now triangulate: the H20 fast path wants width ≡ 0 (mod 64).
- **Consequence for the capacity program**: the design space "asymmetric widths within the cliff" collapses from {32-multiples ≤ 256} to permutations of {64, 128, 192, 256} — and 192 is itself suspect (EXP-034's 27.4ms datum contains 192; if 192 also misprices, the space is EMPTY and 64/128/256 is the unique fast configuration). The only surviving candidate is 64/192/256 (+37.5% stage-2, ~+27% FLOPs; dense law says 26.0ms but the law just failed for 160 — a gate-first probe would resolve 192's status in ~90s).
- **At 32.4ms the experiment is unwinnable regardless of statistics**: ~95 epochs is inside the starvation regime where EXP-002/005/007 price deficits at −0.7 to −1.1 — the capacity question cannot be measured at this throughput, which is exactly why the gate pre-registered invalid rather than letting a confounded run produce a fake "capacity fails" datum.
- Trajectory fit: second GATE_KILL in five loops (EXP-040, EXP-042, this) — the ~90s screen has now saved three full loops while charting the hardware envelope (cliff >256; grouped 2.8×; now 64-alignment).

## Verification
Per plan-044, the dt gate is the integrity pre-condition: it failed BY DESIGN on a provably clean host (gates clear at launch, load 8.5 at decision, three consistent windows, printed dt agreeing). Condition 1 (best ≥ 96.81) not evaluable — run killed at ~11% progress; conditions 2–3 skipped. No false-failure risk: contention is excluded by the load/app checks and window consistency; a rerun would re-measure the same kernels. Verdict: **invalid** (metric NaN) exactly as pre-registered in plan-044 Abort Criteria.

## Unexplored Avenues
- **64/192/256 (the only surviving asymmetric-width candidate)**: gate-first probe, ~90s to resolve. Two risks stack: 192 may misprice (EXP-034 datum suggests it), and even at the dense-law 26.0ms it projects ~119 epochs with deficit ~−0.28, needing +0.5 true level — the EXP-009 saturation reading argues capacity is regularization-bound at 4.29M anyway. Low prior, but it is the LAST permutation; running it would close the asymmetric-capacity class with a measurement either way.
- **The statistical question (does capacity at converged epochs pay?) remains unmeasured** — EXP-044 produced a hardware answer, not a statistics answer. But the hardware now constrains the test so tightly (64-multiples only, ≤256) that 64/192/256 is the only remaining instrument.
- A 96/128/256 variant (widen stage 1, both 64-multiples) is in-laws on alignment but costs ~+42% FLOPs at the most expensive resolution → ~29ms even by the dense law, ~107 epochs; dominated by 64/192/256 on every axis.

## Next Steps
1. **Record the 64-alignment law** in project-insights (extends the EXP-040/042 kernel entry) and goal-learnings — channel widths must be ≡ 0 (mod 64) on this stack; dt-gate anything else (high confidence).
2. **Decide the asymmetric-capacity endgame**: either spend ~90s gate-probing 64/192/256 to close the class with data, or close it by arithmetic (saturation evidence + deficit) and return to novel mechanism construction; the gate-probe is cheap enough to justify as the first milestone of whichever loop comes next (medium confidence).
3. **Carry the replicate-pair protocol** for any future promising mid-band read (standing, from exp-report-040/041/042/043) (medium confidence).

## Key Learning
The H20 + compile + channels_last + bf16 stack has a SECOND alignment law independent of the 256-channel cliff: channel widths that are 32-multiples but not 64-multiples fall off the fast-kernel path (160-wide stage 2: 32.4ms measured vs 24.8 dense-law predicted — +10ms for +18% FLOPs, ~4× the law; retro-consistent with EXP-005's 160 and EXP-034's 27.4ms fallback containing 96/192). Within-cliff asymmetric capacity collapses to permutations of {64, 128, 192, 256}, leaving 64/192/256 as the single surviving candidate — and the baseline 64/128/256 increasingly looks like the unique hardware-fast configuration, optimal by construction rather than by luck. Architecture search on fixed hardware is search over the kernel lattice, not over ℝ^widths.
