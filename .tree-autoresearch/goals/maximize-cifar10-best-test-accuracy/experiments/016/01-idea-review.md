# Adversarial Review

## Prioritized feedback

### 1. [Cross-cutting, most important] The metric is a max over ~16+ tail evaluations — variance reduction is partially self-defeating
EXP-011's reported 95.61 decomposes as final-16 mean 95.493 + best-minus-tail premium 0.1169. That premium *is* the max-order-statistic of eval-to-eval noise. Every candidate here reduces that noise:
- A raises ESS 79.2→101.5 ⇒ averaging-noise std ×0.885 ⇒ premium shrinks ~0.013 if the premium were purely averaging noise (it isn't — much is eval-set noise, so this is an upper bound on the loss).
- B raises ESS to 160 ⇒ std ×0.70 ⇒ premium loss up to ~0.035.

So a candidate must raise the **tail mean** by more than its premium loss to raise `best_test_acc`. **Fix (applies to all three):** preregister reporting tail-mean alongside best, and size the expected tail-mean gain against the premium loss before committing the single metric run. Do not claim a win from a change whose only predicted effect is variance reduction.

### 2. [Fatal-adjacent for Candidate A] The bias correction moves the averaged weights far less than "6.30% point mass" implies
The removed mass sits on the copy-in state at t=225 s. The first *retained* sample is only 31 steps later at a tail LR of ≈0.146×peak (cosine at 75%, decaying to 0). The two states are near-identical; the parameter delta from de-biasing is ≈ 6.3% × (boundary state − tail-weighted mean), where the window's total weight drift is already small. Predicted effect is well under the 0.117 noise floor, and under the frozen protocol (seed 42, one run, no rerolls) an A-vs-EXP-011 difference of ±0.05–0.15 is **unidentifiable**. Additionally: the "boundary-to-first-sample decay" clause is **inert** — with zero-init numerator and mass, decaying 0 then normalizing yields exactly the first sample regardless of Δt. Its inclusion signals the kernel arithmetic wasn't traced end-to-end (the rest of it — terminal mass 0.9373 = r¹⁶⁰, ESS 101.9, mean age 21.8 s — I recomputed and it checks out).
**Fix:** if run at all, pair de-biasing with a kernel change large enough to exceed noise (e.g. halve the half-life to ~9 s at fixed de-biasing, isolating age with ESS held roughly comparable), or accept that A is a replication, not an experiment.

### 3. [Candidate A] Lowest information yield: it clears 95.50 by re-deriving a known +0.21
Parenting on EXP-004 makes the local bar 95.50 while EXP-011 already sits at 95.61 with the same package. A formal "improvement" here is near-certain and near-worthless for the global best. **Fix:** parent on EXP-011 (bar 95.71) so the run actually tests the bias-correction hypothesis rather than the EMA hypothesis.

### 4. [Candidate B] The proposal conflates two variables — kernel *shape* and mean *age* — and moves age the wrong way
Uniform gives ESS 160 (1.57× over 101.9, std ×0.80) but mean age 37.4 s vs 21.8 s. You are buying a 20% noise reduction with 15.6 s of extra staleness in a window where LR is still ~15% of peak at entry. **Fix (this is the strongest single refinement available across all three candidates):** use a **mean-age-matched boxcar** — uniform over the final ~44 s (~93 samples) gives mean age ≈22 s, matching EXP-011's bias-corrected kernel, while ESS 93 ≈ 101.5 — no, that's a wash; better, uniform over the final ~60 s gives mean age 30 s and ESS ~128. Pick the width by preregistered rule (match age, maximize ESS subject to age ≤ 25 s) rather than defaulting to the full quarter. This isolates kernel shape and strictly dominates on the axis you can defend.

### 5. [Candidate B] BN buffers: the risk is real but smaller than the SWA literature implies
Izmailov et al. require a BN recalibration pass because averaged weights induce activation statistics no constituent model's running stats describe. You average buffers instead. That's the same shortcut EXP-011 already validated — so it's survivable at short horizons — but mismatch scales with weight dispersion, and uniform-over-75 s has the largest dispersion of any candidate here. **Fix:** preflight-measure the averaged model's train-set BN drift (cheap, accuracy-blind), or shrink the window per item 4, which shrinks dispersion directly.

### 6. [Candidate C] The chosen hyperparameter cancels the claimed mechanism
Lookahead's variance reduction scales with (1−α). At α=0.8, k=5, the effective smoothing horizon is ~6 steps — against an EMA horizon of ~1,600 steps (18.75 s half-life at ~85 steps/s). You cannot simultaneously invoke α=0.8 to dodge displacement contraction *and* claim meaningful trajectory-variance reduction. The residual effect is a 20%-per-5-steps pullback acting as a mild trust region — real, but not the stated mechanism.

### 7. [Candidate C, fatal as specified] It discards the only validated +0.21 and must beat 95.61 from a 95.40 base with a near-null smoother
"No EMA" imposes a ~0.21 handicap before the intervention does anything, with zero diagnostic support for endpoint variance in EXP-004 (final == best, as the proposal itself concedes). Also unaddressed: Lookahead runs from step one, so it perturbs all 25,560 steps and any throughput loss is charged against the 300 s budget; and slow-weights-with-online-BN is an untested eval pairing.
**Fix:** keep EXP-011's EMA as the base (parent on EXP-011) and add Lookahead on top with canonical α=0.5, k=6–10 (the setting Zhang et al. actually feature for CIFAR WRN), so the floor is 95.61 and the intervention has a mechanism proportional to its claim. This turns C from dominated into the highest-ceiling candidate.

### 8. Minor credits (verified, not weaknesses)
- B's odd-cadence argument is sound: period-2 SAM + cadence 31 alternates SAM/non-SAM phase across samples. Keep 31 in any boxcar variant.
- The "~160 samples" figure is internally consistent: 2,449 SAM updates ⇒ ~4,900 tail steps ⇒ 4,900/31 ≈ 158.
- All three preregister an accuracy-blind preflight then exactly one metric run — correct and hard-constraint-compliant. Ensure the preflight explicitly verifies **step count is not reduced**, since state arithmetic is charged.

---

## Scored verdict

**Activation-Anchored Bias-Corrected Full-State EMA**
- Evidence/reasoning: **7/10** — arithmetic is correct and grounded in EXP-011's realized timestamps (I reproduced ESS 101.9 and terminal mass 0.9373), but the mechanism's *magnitude* argument is absent and the boundary-decay clause is inert.
- Potential impact: **3/10** — near-certain to clear 95.50 by replicating a known result; expected delta vs the 95.61 global best is below the 0.117 noise floor and unidentifiable in one run.

**Uniform Full-State Clean-Tail SWA**
- Evidence/reasoning: **5/10** — SWA literature is real but its preconditions (cyclic/constant LR, BN recalibration) are both violated here; the proposal honestly names its own staleness counter-hypothesis but doesn't quantify the age/ESS tradeoff (37.4 s vs 21.8 s for only a 1.57× ESS gain).
- Potential impact: **6/10** — tests the one averaging axis (kernel shape/ESS) never varied in this lineage, at low implementation risk, with a clear refinement path (age-matched boxcar) that can dominate EXP-011 on both age and ESS.

**CIFAR-Grounded High-Alpha Lookahead Without Evaluation EMA**
- Evidence/reasoning: **4/10** — cites a real α=0.8 CIFAR setting, but the choice contradicts the variance-reduction mechanism it claims, and no EXP-004 evidence of endpoint variance is offered.
- Potential impact: **4/10** — highest ceiling in principle (only trajectory-level change, orthogonal to EMA and stackable later), but as specified it forfeits +0.21 and applies a ~6-step smoother in exchange; rises to ~7 if rebased on EXP-011 with α=0.5.

## Pick: **Uniform Full-State Clean-Tail SWA**

It wins because it is the only candidate that both (a) attacks the diagnosed limiter — stable generalization via cheap sparse state arithmetic, the one mechanism this lineage has actually validated — and (b) varies a factor that has never been varied here. A is a replication whose predicted effect is provably below the run-to-run noise it must be measured against; C hands back a validated 0.21 to buy a near-null smoother. B keeps the validated mechanism intact and swaps only the kernel, so its failure is as informative as its success: it directly answers whether age or ESS governs the tail premium, which is the question the next three experiments all depend on.

**Condition on the pick:** adopt item 4 before running. Full-quarter uniform buys a 20% noise reduction for 15.6 s of staleness *and* forfeits up to 0.035 of the max-selection premium — that is a coin flip. A preregistered narrower boxcar (width chosen to hold mean age ≤ 25 s while maximizing ESS) makes B strictly better than EXP-011's kernel on both axes and gives the run a defensible path past 95.61 rather than past 95.50.
