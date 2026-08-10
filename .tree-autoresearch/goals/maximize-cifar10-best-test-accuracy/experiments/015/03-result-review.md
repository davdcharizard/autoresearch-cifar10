# EXP-015 Adversarial Raw-Result Review

## VERDICT: **PASS** (result accepted as valid evidence) — experiment outcome: **no-improvement**

The run is arithmetically self-consistent, integrity-clean, and fully dosed. It does not meet the formal child bar. Nothing in the record justifies an invalid/crash classification.

---

## 1. Independent recomputation

### Preflight timing vector
| Quantity | Recomputed | Emitted | Gate | Status |
|---|---|---|---|---|
| Ratio 1 | 12.674479/12.620640 = 1.0042661 | 1.0042659 | — | ✓ |
| Ratio 2 | 1.0036561 | 1.0036561 | — | ✓ |
| Ratio 3 | 1.0035572 | 1.0035573 | — | ✓ |
| Ratio 4 | 1.0042236 | 1.0042236 | — | ✓ |
| Ratio 5 | 1.0046771 | 1.0046771 | — | ✓ |
| Parent drift | (12.636951−12.620640)/12.631046 = 0.0012913 | .001291 | ≤.03 | ✓ (23× margin) |
| Ratio median | sorted → 1.0042236 | 1.004224 | ≤1.01 | ✓ |
| Ratio max | 1.0046771 | 1.004677 | ≤1.02 | ✓ |
| Ratio MAD | median{0, 4.23e-5, 4.535e-4, 5.675e-4, 6.663e-4} = **0.0004535** | .000452 | ≤.005 | ✓ but see NB-1 |

Projections all clear: 25689.50 ≥ 25300; 131 ≥ 130; 159.33 ≥ 155; 449.79 ≤ 600. Candidate-only peak 610.284 MiB < 1350.

### Step/window partition — exact closure
- SAM-eligible window 4936 + CutMix window 20884 = **25820 = total steps** ✓ (windows disjoint and exhaustive).
- Tail steps = 25820 − 20884 = 4936; SAM 2468/4936 = **0.5000 exactly** ✓ (deterministic every-other-step, as "periodic" implies).
- CutMix 10362/20884 = **0.496169 → .4962** ✓. Against p=0.5 Bernoulli: expected 10442, σ = 72.3, observed −1.11σ. Normal sampling noise ✓.

### Loss-call identities (all exact)
- Ordinary Poly: 25820 − 10362 − 2468 = **12990** ✓
- Poly total: 12990 + 10362 + 2468 = **25820 = steps** ✓ (exactly one descent loss per step)
- Total: 25820 Poly + 2468 CE = **28288** ✓
- Second, independent route: front-window ordinary 20884 − 10362 = 10522; tail ordinary 4936 − 2468 = 2468; 10522 + 2468 = **12990** ✓ — confirms zero CutMix/SAM overlap.

### EMA schedule
- Span 25792 − 20894 = 4898 steps; 159 updates ⇒ 158 intervals × **exactly 31 steps** ✓.
- Odd stride against period-2 SAM ⇒ strict parity alternation ⇒ 80/79 split. Reported ordinary 79 / SAM 80 = 159 ✓.
- 159 ≥ 155 gate ✓; matches projection 159.33.

### Eval accounting
- 25820 = 132×195 + 80 ⇒ 132 full epochs + one partial = **133 epochs**, 133 evals, **≤1 eval/epoch exactly at the bound** ✓.
- CutMix window 20884/195 = **107.09 epochs → 107 live evals**; remaining 26 epochs → **26 EMA evals**; 107+26 = 133 ✓. Swaps = restores = 26, parity 0 ≤ 1 ✓.
- Uncharged overhead 438.8 − 300.0 − 1.0 = 137.8 s / 133 evals = 1.04 s per eval — plausible for CIFAR-10 test on H20 ✓.

### Tail statistics (final-16 EMA)
- Sum = 1523.75; mean = **95.234375** ✓ (exact match)
- min 95.15 ✓, max 95.34 ✓, range **0.19** ✓, final 95.16 ✓
- best − mean = 95.34 − 95.234375 = **0.105625** ✓
- Max of final-16 **equals the global best (95.34)** ⇒ the headline number is a tail value, not an early lucky spike.

### Formal delta
- Child best **95.34** vs parent **95.61** ⇒ **Δ = −0.27**
- Formal bar 95.71 ⇒ **shortfall 0.37**; scientific target 95.69 ⇒ shortfall 0.35
- Tail-mean delta: 95.234375 − 95.493125 = **−0.258750**
- Fairness check on best-of-N inflation: parent best−tailmean = 0.116875, child 0.105625 — nearly identical tail shape, so the deficit is **not** an artifact of differing selection headroom. The gap (0.26–0.27) exceeds the child's entire tail range (0.19).

### Dose/budget
Charged 300.0 (exact); total 438.8 ≤ 600; steps 25820 ≥ 25300; epochs 133 ≥ 130; EMA 159 ≥ 155; parity 0. Actual beat projection on steps (+0.51%) while using less total time — both consistent with "ran faster than preflight predicted," not contradictory. Exit 0, physical GPU0 UUID matched, seed 42, one metric run, no preflight rerun, no retries. Evaluator guarded with 0 test iterations in preflight ⇒ no test-set exposure on the tuning/timing path.

---

## 2. Blockers

**None.** No traceback, CUDA fault, OOM, nonfinite, audit failure, RNG/restore/coverage failure. All dose gates met at full dose. Every emitted aggregate reproduces from the raw counters, with two independent derivations agreeing on the 12990 ordinary-Poly count.

---

## 3. Nonblocking concerns

**NB-1 (arithmetic, cosmetic).** Ratio MAD recomputes to 0.0004535 from the printed ratios; emitted 0.000452 — a 1.5e-6 discrepancy that the displayed 7-decimal precision does not explain. Gate margin is 11×, so no decision changes, but the MAD emitter's median/precision handling is worth a look.

**NB-2 (preflight memory model under-predicts by 2×).** Candidate-only preflight peak 610.284 MiB vs metric peak 1222.4 MiB. The 1350 MiB gate was cleared with 55% headroom in preflight but only **9.4% headroom in the real run**. This gate is currently validated against a measurement that systematically halves the true footprint; a heavier future candidate could pass preflight and OOM.

**NB-3 (loss values are not cross-comparable).** Child final loss .1645 vs parent .1552 and terminal train-loss EMA .009284 are computed under Poly ε=−0.25, which shifts the objective by −0.25(1−p_t) relative to CE. Unless these are CE-evaluated, the loss deltas carry no information about the comparison. **Accuracy is the only comparable metric here** — do not cite the loss gap as corroboration.

**NB-4 (progress is charged-time, not step fraction).** SAM's first step 20886 sits at 80.9% of steps but progress .7500. The schedule is time-based and SAM roughly doubles per-step cost, so the "last 25% of the run" is only **19.1% of optimizer steps** (4936/25820). Any downstream write-up describing SAM as covering 25% of training steps would be wrong.

**NB-5 (tail is declining; the metric is generous, not conservative).** The final-16 EMA sequence falls near-monotonically 95.34 → 95.16. The best was captured early in the EMA window; a longer tail would have scored lower. This strengthens, not weakens, the no-improvement finding.

**NB-6 (single preflight vector).** First-and-only complete vector, correctly not rerun per the no-retry constraint — but ratio MAD across 5 reps is then the sole dispersion evidence for the timing gates. Acceptable; noted for the record.

---

## 4. Is the result trustworthy?

**Yes, as a single-seed measurement of this exact package.** Integrity is unusually well-evidenced: the window partition closes exactly (20884+4936=25820), the loss-call identities close by two independent routes, the EMA stride is an exact integer 31 with the parity split it implies, and eval counts reconcile to the CutMix window boundary at 107/26. Fabricated or mis-instrumented counters would not close this tightly. Charged time is exact, the run is single-shot, and there is no selection across runs or configs.

---

## 5. Classification and key learning

**Classification: no-improvement, metric 95.34.** Valid, fully-dosed, integrity-clean run measuring below the promotion threshold. Not invalid; not a crash. Recorded delta −0.27 vs parent 95.61, formal shortfall 0.37 vs the 95.71 bar.

**Key learning (correctly scoped):** Holding the inherited package fixed — time-aware PreAct WRN-16-4, front-loaded probabilistic CutMix at 0.4962 dose over the first 80.9% of steps, clean-finish periodic SAM at 0.5 dose over the final 19.1% of steps with **CE-only ascent**, and charged-time clean-tail EMA — replacing the descent objective with PolyLoss ε=−0.25 costs −0.27 best / −0.26 tail-mean at dose parity and ~0.4% step-time overhead. **This exact package point is rejected and need not be retried.**

**Overclaims to reject:**

- **"Negative ε hurts."** Not established. ε is confounded with at least three interactions this single run cannot separate: (a) CE-only SAM ascent against a non-CE descent loss creates an ascent/descent objective mismatch that a negative ε *widens*; (b) CutMix soft/mixed targets change the meaning of the (1−p_t) term; (c) clean-tail EMA averages over the last 19.1% of steps where SAM and the modified descent loss both act. The observed deficit could originate in any of these couplings rather than in ε itself.
- **"PolyLoss is worse than CE for this task."** One ε value, one sign, one seed gives no gradient in ε and says nothing about ε>0 or about Poly with an ε-matched SAM ascent. The **PolyLoss family is not ruled out.**
- **"The −0.27 gap is decisive."** Single seed 42. The within-run tail range (0.19) is autocorrelated and does not estimate seed-to-seed variance; typical WRN-16-4 CIFAR-10 seed σ is ~0.10–0.20, placing this at roughly 1.5–2.5σ. Directionally credible and consistent across both best (−0.27) and tail-mean (−0.26), but it is a one-seed result.

**What full dose does buy:** the negative is not attributable to under-training, truncated dose, eval starvation, or timing-gate slack — all gates cleared with margin and the schedule executed as designed. That is what licenses closing this branch without a retry, and nothing more.

## Main-Agent Evidence Corrections

- NB-1 is explained by prompt rounding: the raw ratios retain full precision and emit MAD/median `0.0004515797`; recomputation from seven-decimal prompt values yields the cosmetic `0.0004535` difference.
- `final_test_loss=0.1645` is produced by the frozen evaluator's standard CE and is directly comparable with the parent's CE test loss `0.1552`. Only `final_train_loss_ema=0.009284` is measured under the shifted Poly objective and cannot be compared numerically to parent CE training loss.
