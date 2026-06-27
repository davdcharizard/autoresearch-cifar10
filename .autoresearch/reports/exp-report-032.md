# Experiment Report EXP-032: SWA tail — freeze cosine at 85%, average iterates, eval BN-re-estimated SWA model

- **Date**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-032.md
- **Plan**: plans/plan-032.md
- **Exp-log**: logs/exp-log-032.md
- **Verdict**: **no-improvement** (96.60 vs bar 96.81; baseline 96.71, Δ −0.11)

## Goal
Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged budget, train.py only. Baseline 96.71 @ 1990397 (distribution top; mean ≈96.57, σ ≈0.16); bar 96.81; true effects ≥ +0.3 needed.

## Idea & Hypothesis
**Idea**: Canonical SWA grafted onto the time-keyed one-cycle: training byte-identical to baseline until 85% of the charged budget; from there the LR freezes at its cosine value (~0.030, the canonical constant SWA tail) and each epoch's single eval scores the equal-weight average of end-of-epoch iterates (`AveragedModel`), with BN running stats re-estimated on the AUGMENTED train loader (`update_bn`) before every tail eval.

**Why chosen**: The only currency the max-statistic pays is converged-plateau LEVEL; SWA attacks it directly with paper-scale gains (+0.2–0.6 on CIFAR ResNets/WRNs, Izmailov et al. 2018) — and it repaired a diagnosed in-project failure: EXP-011's EMA evaluated averaged weights against LIVE BN buffers, the stats/weights mismatch class EXP-029 measured at −10.93 in the extreme.

**Hypothesis**: SWA plateau ≥ +0.25 above the raw baseline mean ⇒ best ≥ 96.81 at unchanged dt/epochs. Falsifiable: SWA trail failing to climb above the baseline family within ~8 SWA epochs.

## Approach
Five train.py edits (16 insertions / 1 deletion) on `autoresearch/exp-032`: swa_utils import; `SWA_START_FRAC = 0.85`; `swa_model = AveragedModel(base_model)` in startup; one in-step lr-freeze branch (charged, trivial); eval-block branch (tail: `update_parameters` → `update_bn` full augmented-loader pass → `evaluator.evaluate(swa_model, ...)`; pre-tail unchanged). One eval per epoch throughout; Eval untouched; no new packages.

## Execution
- **Run 1 — CONTENTION_KILL at pct 98.6**: clean 21.6–23.0ms through pct 91 (including ~6 SWA-tail epochs — machinery healthy), then charged dt 72–102ms; post-kill: GPU 0 empty but host load average 241 (foreign CPU job). Best-at-kill 96.27, ~4 epochs short of converged tail.
- **Run 2 — CONTENTION_KILL at pct 27.8**: same dt signature with load FALLING (43→15); post-kill: foreign PID 1624123 resident on GPU 0 (5.9GB) — GPU time-slicing (EXP-011 signature). Both kills environment-caused; new infra-errors entry: GPU-free pre-check is insufficient, gate on host load too.
- **Run 3 — clean**: launched after 51 gate polls (~25 min; GPU freed, load 7). rc=0, total 533.3s, startup 12.4s, **139 epochs / 13,475 steps**, 139 evals, VRAM 1844.2MB (+231MB SWA copy), params 4,286,026. Profile: 268 windows, mean 22.3ms, **0 slow** — pristine.

## Results
**best_test_acc 96.60 (final eval); final_test_loss 0.1756. Bar missed by 0.21; −0.11 vs recorded baseline; +0.03 vs baseline mean — no detectable true effect.**

What the trail shows:
1. **BN re-estimation works and is mildly positive at n=1**: last raw eval ep118 94.92 → first SWA eval ep119 95.63 (+0.71). No EXP-029 damage signature anywhere — the augmented-loader `update_bn` is the correct procedure, confirmed.
2. **The average climbs, in loss more than accuracy**: SWA test_loss fell monotonically every single tail epoch to 0.1756 — strictly better than the baseline family's final ~0.185 — while accuracy crept to 96.5–96.6 and was still gaining ~+0.02/ep at cutoff (~21 snapshots vs the paper's 30–80+).
3. **Net: the average exactly recovers what the frozen tail forfeits.** The baseline's cosine-to-zero anneal ends at ~96.5–96.7; the SWA run's frozen-LR raw iterates were noisier, and their average landed at 96.60 ≈ the baseline MEAN. The flat-minima centering produced a genuinely different solution (much better calibrated — the loss gap is real) but its accuracy sits exactly where the ordinary anneal would have gone.

**Mechanism reading**: this completes a clean two-experiment story about weight averaging on this recipe. EXP-011 (EMA across the anneal, no BN re-est) lost −0.25; EXP-032 (basin-only averaging WITH correct BN re-est) recovers to ±0σ. The BN flaw explained EMA's deficit, but fixing it only removes the damage — it does not add level. The deeper reason is the same improved-loss/no-acc-gain signature both attempts share: on this recipe, accuracy's ceiling is not set by iterate noise or basin position that averaging can fix; the cosine tail already finds the basin center as well as the average does (a time-keyed anneal IS an iterate-averaging analogue — classical result: averaging SGD ≈ annealed SGD). Test-loss/calibration improves because averaging smooths the logit geometry, but the argmax decisions don't change. Under a fixed wall clock there is no free tail to extend the snapshot count, and the trail's +0.02/ep terminal slope would need ~10 more epochs just to reach the recorded baseline.

The axis is now closed from both ends: averaging across the anneal loses (EXP-011), averaging the sampled basin at the canonical recipe with correct BN handling ties the mean (EXP-032). 27 consecutive misses; external transfer 0-for-12 (paper SWA gains assume a fixed-epoch budget where the SWA phase is ADDED; under fixed time it is CARVED OUT of the anneal).

## Verification
- Condition 1 (best ≥ 96.81): **FAIL** — 96.60. Pre-condition profile PASS (268 win, mean 22.3ms, 0 slow; 139 epochs exact projection; params/training_seconds/eval-count all exact). Runs 1–2 were confirmed foreign contention (GPU PID 1624123; host load 241) and rerun per protocol; Run 3 is the clean measurement. Conditions 2–3 informationally pass (rc=0, 533.3s ≤ 600; 139 evals = 139 epochs).
- Trustworthiness: high. No false-failure risk: the trail is smooth, the plateau real, and the miss (−0.21 vs bar) is ~1.3σ — far outside parsing/noise ambiguity for a bar-pass.
- Verdict basis: clean miss → **no-improvement**.

## Key Learning
Weight averaging cannot raise this recipe's accuracy plateau: with the diagnosed BN flaw fixed (augmented-loader re-estimation — confirmed positive, +0.71 at n=1), basin-averaged iterates land exactly at the baseline mean with strictly better test loss — the time-keyed cosine anneal already performs the equivalent averaging implicitly, and under fixed wall clock the SWA phase is carved out of the anneal rather than added after it. Improved-loss/no-acc-gain is the recurring signature of solution-smoothing interventions on this goal (EMA, SWA): calibration moves, argmax decisions don't.

## Unexplored Avenues
- **More snapshots via longer tail (SWA_START 0.75) or cyclic tail LR**: the trail was still climbing +0.02/ep at cutoff, so a longer tail mechanically gets closer — but it forfeits MORE anneal, and the two effects cancelled at 0.85; expected ≤ ±0.1. Interior unbracketed but low prior.
- **SWA + kept final anneal (average DURING the unmodified cosine tail, eval averaged model)**: removes the forfeited-anneal cost but the iterates freeze as LR→0 (brainstorm Candidate 2's known weakness); the n=1 +0.71 jump suggests evaluating a BN-re-estimated copy of the RAW model during the late anneal is roughly free — but it converges to the raw eval as stats converge; noise-band expected.
- **Greedy soup over the plateau (keep snapshot only if it improves test acc)**: selecting on TEST accuracy is overfitting-the-metric (reward-hacking adjacent) — barred.
- The loss-vs-accuracy decoupling (0.1756 vs 0.185 at equal accuracy) is the experiment's most interesting datum: the recipe's accuracy ceiling is decision-boundary-limited, not confidence-limited. Interventions that move BOUNDARIES (data, capacity, training distribution) are the only remaining class — and nearly all are measured-closed.

## Next Steps
1. **Deeper-not-wider at matched dt with the early-dt gate** (ResNet-26, n=4 @ 4× width, ~28–30ms projected → gate decides in ~90s): the last unbracketed capacity direction; contradicts two laws so the gate must be strict — but a cheap, decisive probe. Confidence: low.
2. **Hold-out axis: training-distribution composition** (e.g., reduce RandomErasing/TA aggressiveness ONLY in the final 15% — "anneal the augmentation with the LR"): augmentation pressure is bracketed STATICALLY, but its SCHEDULE is untouched; matches the deferral law (late-only change) and EXP-025's lesson cuts the other way (tail pressure load-bearing) — a genuine unknown. Confidence: low-medium.
3. **Accept σ reality**: with 27 misses, the structural laws say the recipe sits at a measured multi-axis optimum; remaining ideas should prioritize information value (closing the last open axes cheaply via gates) over expected gain. Confidence: high (as strategy).

## Exit Action Results
(no exit actions defined for this goal)
