# Experiment Report: EXP-045 — 64/192/256 gate-first (the last kernel-lattice capacity point)

- **Date**: 2026-06-10
- **Verdict**: invalid (pre-registered GATE_KILL branch — metric NaN, no fair mechanism test)
- **Primary metric**: best_test_acc = NaN (baseline 96.71, bar 96.81)
- **Branch**: autoresearch/exp-045 (discarded)
- **Artifacts**: brainstorm/brainstorm-045.md · plans/plan-045.md · logs/exp-log-045.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc %, higher is better) within the fixed 300s charged training budget, modifying only `train.py`. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context: baseline mean ≈ 96.57, σ ≈ 0.16 (EXP-027).

## Idea & Hypothesis
**Idea**: EXP-044's pre-registered endgame — 64/192/256, the unique surviving within-lattice capacity configuration after the alignment GATE_KILL (all widths 64-multiples, all ≤256). Dual purpose: resolve 192's kernel status (the discriminating test between "64-multiple" and stricter lattice hypotheses) and, if fast, run the first unconfounded measurement of capacity at converged epochs (+40% FLOPs → dense-law 27.7ms → ~112 epochs, deficit ≈ −0.38).

**Hypothesis branches (pre-registered)**: (i) best ≥ 96.81; (ii) gate passes but plateau ≈ mean − deficit → capacity level-saturated; (iii) GATE_KILL > 31ms → 192 misprices, fast lattice = {64,128,256} exactly, verdict invalid.

## Approach
train.py only, identical three-edit mechanics to EXP-044 (`STAGE_WIDTHS = (64, 192, 256)`, widths-tuple `ResNet.__init__`, print line). All four CPU sanities passed pre-launch: params exactly **5,392,714** (analytic match), forward, pad shortcuts (128 at stage-2 transition, 64 at stage-3), train smoke. Launcher: validated exp044 composite with three threshold edits (gate 31ms, contention floor 31ms), diff-confirmed.

## Execution
Single launch; gates clear poll 1 (apps=0, load=17). **GATE_KILL at tick 5 (~80s)**: windows 33.6/33.4/33.4ms → D0 = 33.4ms > 31ms, projected ~93 epochs. Printed dt agreed (34ms uniform, ~15,100 img/s); host clean post-kill (zero GPU-0 apps, load 12.2) — true kernel cost. Training healthy pre-kill (ep10 best 77.47). No relaunch per plan; total cost ~2 GPU-minutes. No errors beyond the screen.

## Results
- **The law is refined, not just confirmed: the fast set is the POWERS OF TWO, not the 64-multiples.** 192 = 3×64 is 64-aligned yet priced +11.0ms over baseline where the dense law predicts +5.3. With EXP-044's 160 datum, the discrimination is complete: fast = {64, 128, 256} (22.4ms); slow = {80, 160, 192} (~33ms) and {288, 320} (54ms). EXP-034's 48/96/192 fallback (27.4ms, all non-powers-of-2 at 12 blocks) fits the law retroactively.
- **Off-lattice dt is FLAT, mirroring the >256 cliff**: 160 at +18% FLOPs → 32.4ms; 192 at +40% FLOPs → 33.4ms. Both sub-256 off-lattice points land on a ~33ms tier nearly independent of FLOPs, exactly as 288/320 share the 54ms tier. The kernel selector appears to fall back to one slow implementation per regime; FLOPs are second-order off the lattice.
- **The asymmetric/within-cliff capacity class is now closed in FULL.** On-lattice capacity-increasing permutations price out before gating: 64/256/256 ≈ +88% FLOPs → ~34ms dense-law (~91 epochs, starved); 128/128/256 ≈ +100% → ~36ms. Off-lattice widths are measured-slow. The baseline 64/128/256 is the unique fast increasing power-of-2 triple with viable epoch arithmetic — hardware-optimal by exhaustion, not by luck.
- **The capacity-at-converged-epochs question dies unmeasured but moot**: there is no instrument left to measure it with on this hardware; every viable test configuration is either slow or starved. Capacity is closed in every currency including instrument-availability.
- Trajectory: third GATE_KILL in six loops (040/042/044/045 — the screen has now charted the full kernel envelope at ~2 GPU-minutes per datum vs ~8 per full run).

## Verification
Per plan-045, the dt gate is the integrity pre-condition and failed BY DESIGN on a provably clean host (dual gates clear, load 12.2 at decision, three consistent ~350-step windows, printed dt agreeing). Condition 1 not evaluable (killed at ~11%); conditions 2–3 skipped. No false-failure risk: contention excluded by app/load checks and window consistency; a rerun would re-measure the same kernels. Verdict: **invalid** (metric NaN), exactly as pre-registered.

## Unexplored Avenues
- **Within-architecture capacity: NONE remain.** The lattice is exhausted; this section intentionally records the absence — future brainstorms should not re-derive width candidates.
- **The flat-tier observation suggests a perverse possibility**: since off-lattice costs are flat (~33ms whether +18% or +40% FLOPs), IF an off-lattice run were ever worth doing, the largest off-lattice config (e.g. 64/192/256 vs 64/160/256) is free relative to the smaller one. But both project ≤95 epochs = starvation; the tier itself is unusable under this budget. Recorded for completeness only.
- **Radical structural recombination** is now the only unexplored frontier (per the autopilot directive's "more radical architectural changes"): candidates must be dt-flat (use only existing fast kernels at existing shapes), deferral-free, and level-targeted — a very tight screen that the next brainstorm must construct against, knowing 0-for-13 external transfer and all in-paradigm classes closed.

## Next Steps
1. **Correct the kernel law in project-insights and goal-learnings** from "≡ 0 mod 64" (EXP-044's hypothesis, now refuted) to "powers of two {64,128,256}; off-lattice = flat slow tiers" — done as part of this loop's distillation (high confidence).
2. **Next brainstorm must open the radical-structural frontier under the full screen stack** (fast-lattice shapes only, deferral-free, plateau-LEVEL mechanism, in-regime evidence) — e.g., evaluate whether ANY remaining train.py-expressible change passes all screens, and if none do, prioritize the highest-information cheap probes (medium confidence).
3. **Protocol carry-over**: replicate pair for promising mid-band reads; D0-median gate for any architecture-shape change, now with the power-of-2 lattice as the prior (high confidence).

## Key Learning
The H20+compile+channels_last+bf16 fast-kernel set is the powers of two: {64, 128, 256}. EXP-044's "64-multiple" hypothesis is refuted — 192 (= 3×64) prices on the same flat ~33ms tier as 160, just as 288/320 share the flat 54ms tier above the cliff. Off the power-of-2 lattice, FLOPs are second-order: the kernel selector falls back to one slow implementation per regime. With on-lattice capacity increases starvation-priced before gating (64/256/256 ≈ 34ms, 128/128/256 ≈ 36ms), the capacity class is closed in every currency INCLUDING instrument availability — the baseline 64/128/256 is the unique hardware-fast increasing triple, optimal by exhaustion. Three ~90s gate probes (044/045 + 042) charted an entire hardware envelope that would have cost three confounded full loops to misread.
