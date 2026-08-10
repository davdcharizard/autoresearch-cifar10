I now have full context. Emitting the review.

---

# Adversarial Review — EXP-008 (CIFAR-10, best_test_acc)

**Baseline:** 93.55% @ `8faf0f3` · improvement bar **≥ 93.65%** (+0.10) · fixed **300 s** counted budget, ≤600 s wall · one H20 · `train.py`-only · no seed hacking · keep 80% boundary, N1/M7, hard-label weak tail, width-2 fixed (`008/01-brainstorm.md:12,15`).

**Finalists:** Width-3 ResNet-20 · Weight Decay 5e-4 · Residual-Block Dropout 0.3.

I verified the load-bearing numbers first. Width-2 run: 27,143 steps, 71 epochs, switch at 80.0%, strong checkpoint **90.08% @ ep56** (loss EMA 0.2283), first weak 92.96%, **best 93.55% @ ep67** (4 epochs before end), final 93.49%/0.2196, last-three slope ≈ +0.005 pts/ep — effectively flat (`007/03-execute.md:59-61,70`). Weak tail = epochs 57–71 = **15 evaluated epochs**. Width-3 probe: 2,412,730 params, 18.983 ms/step, 667.2 MB, ~15,595 calibrated steps (`008/01-brainstorm.md:16`). 15,595/27,143 = 0.575 → the claimed ~42.5% update loss checks out.

---

## Prioritized feedback (most important first)

### 1. [Cross-cutting — the regularization diagnosis is directional, and its evidence is partly unpersisted] Same evidence-gap pattern that EXP-007's review flagged.
The whole EXP-008 field is reweighted by one inline claim: "declining train loss plus flat/rising test loss suggests excess fit late … whether stronger norm control improves the accepted wide model" (`008/01-brainstorm.md:14,17`). But two of the three supporting numbers are **not in the persisted artifacts**: the execute log records only the *final* test loss 0.2196 and the near-switch train EMA 0.2283 — the "train-loss EMA … toward roughly 0.04" and "test loss bottomed near 0.214" (`008/01-brainstorm.md:14`) do not appear in `007/03-execute.md` or `007/04-analysis.md`; the raw log was correctly removed. This is exactly the "premise asserted from an unpersisted trajectory" problem the EXP-007 idea-review raised about the capacity story (`007/01-idea-review.md:17-19`). It does not sink the diagnosis — train≪test with a flat/mildly-rising tail loss is a textbook late-overfit tell — but it means "regularization is the next limiter" is a hypothesis, not a fact.
- **Fix:** Have the plan pre-register the overfit read as a *primary* diagnostic (strong-switch checkpoint vs 90.08%; tail train-EMA; best-vs-final loss gap), so the chosen intervention is judged against whether the overfit signature was real, not only against the scalar metric.

### 2. [Width-3, near-fatal for this slot] It attacks the one limiter this brainstorm just argued *away*, and pays the worst update trade in the goal's history.
The diagnosis explicitly states EXP-007's strong-view gain "removed evidence for immediate under-capacity at width 2" and names norm control as the clean next question (`008/01-brainstorm.md:17`). Width-3 then spends the entire experiment on *more capacity* — internally contradictory with the chosen diagnosis. The marginal economics are also decisively worse than the width-1→2 step that justified this trend: width-1→2 bought 3.98× params for 1.44× step-time; width-2→3 buys only 2.25× params for 1.74× step-time (`008/01-brainstorm.md:41`) — roughly half the per-step-time capacity return. And the fixed-time cost lands precisely where EXP-007 is most fragile: ~15,595 steps ⇒ ~8 weak-tail epochs, but width-2's best landed at **epoch 67 = the 11th weak epoch** (`007/03-execute.md:61`). With eval once per epoch, width-3 plausibly *terminates before the peak forms* — losing both optimization and best-metric sampling in the exact region the metric comes from. This is the double-jeopardy the EXP-007 review named for width-2-vs-1.5 (`007/01-idea-review.md:25-27`), now far more severe.
- **Fix:** Only pursue if the diagnosis is re-cast as "capacity trend, not regularization" — but that contradicts §1's own read. If kept, it needs a pre-registered tail-shape gate (best still rising at final epoch ⇒ under-optimized, expected). Realistically: **defer** until a regularization result confirms/rejects the overfit diagnosis; running width-3 now answers a question the brainstorm didn't ask.

### 3. [Weight Decay 5e-4 — point-value transfer is the real risk, and no sweep is allowed] 5e-4 is a canonical value, not a fitted one for *this* horizon.
The proposal's own shrinkage math shows 5e-4 is not cosmetic: the parameter-only surrogate retains ~73% at 1e-4 vs **~21% at 5e-4** over this schedule (`proposals/idea-04.md:58`). WRN's 5e-4 accompanied ~200 epochs; this run is ~71 (`proposals/idea-04.md:33,42`). The true local optimum could be 2e-4/3e-4, and the protocol forbids sweeping after seeing the result (`proposals/idea-04.md:123`). So a single 5e-4 run risks overshooting into underfit and being scored as "decay doesn't help" when a smaller value would have. **Mitigant that raises this above EXP-007's 2/5 impact:** the intervention now lands on a **4× model** that EXP-007 shows fits the strong views far better (90.08% vs 84.60%) with a low tail train loss — genuine overfit headroom the 0.27M model lacked, so the underfit risk that dominated EXP-007's low-ceiling verdict (`007/01-idea-review.md:37-38`) is materially reduced here.
- **Fix:** Pre-register the underfit tell — if the strong-switch checkpoint drops well below 90.08% or tail train-EMA rises sharply, the lever is over-strong and the follow-up is a *lower* decay, not abandonment. Keep the compute-neutrality gate (≥99% of 27,143 steps) so any step loss flags a confound.

### 4. [Weight Decay 5e-4 — mechanism/limiter gap] The observed excess-fit is a *tail* phenomenon; decay's leverage is concentrated in the *plateau*.
The overfit signal the diagnosis cites is late — test loss creeping 0.214→0.2196 with accuracy flat over the final ~4 weak-tail epochs (`008/01-brainstorm.md:14`; `007/03-execute.md:61`). But idea-04's own analysis shows 5e-4's dominant shrinkage occurs in the 80% high-LR plateau and *automatically weakens* in the low-LR tail (`proposals/idea-04.md:58,61`). So decay attacks fit where the diagnosis didn't locate the problem. The saving link is indirect: lower parameter norm entering the tail still plausibly improves tail generalization — but this is a coupling argument, not a direct hit on the tail overfit.
- **Fix:** State this explicitly in the hypothesis (decay controls plateau norm, which *carries into* the tail) rather than implying it directly suppresses the late excess fit — otherwise a null is hard to interpret.

### 5. [Dropout 0.3 — over-regularization risk echoes EXP-006, and weak-tail behavior is unspecified] Stacking a second every-view regularizer on the N1/M7 plateau is the pattern this goal has already punished.
Placement (after first BN/ReLU, before conv2) is correct WRN-style (`008/01-brainstorm.md:59`) and literature-grounded. Two real gaps: **(a)** p=0.3 on top of RandAugment through the whole plateau is *strictly more* plateau regularization, the same "too aggressive every-view" mechanism that made EXP-006 Cutout regress −0.67 and the CutMix idea get rejected in EXP-007 (`006/04-analysis.md`; `007/01-idea-review.md:21-23`); WRN's 0.3 came with 200 epochs, not 71. **(b)** The brainstorm never says whether dropout is disabled during the **weak hard-label tail** — the clean refinement phase EXP-004 established as load-bearing. If dropout stays on through the tail (eval disables it, but *training* doesn't), it directly fights the clean-refinement mechanism that produces the peak.
- **Fix:** Pre-register p at the low end (0.1–0.15) *or* apply dropout only during the strong plateau and disable it for the weak tail (mirroring the RandAugment→weak switch), and keep the ≥96% step-retention preflight (`008/01-brainstorm.md:63`). Note both choices add unvalidated knobs, which is why this is a weaker single-run bet than the compute-neutral decay.

### 6. [Dropout 0.3 — attribution cost] Lower attributability than weight decay, for similar ceiling.
Dropout inserts an RNG/mask kernel in the timed region (reduces exposure) and perturbs the fixed-seed augmentation/data stream, so a marginal ±0.1 result is less cleanly attributable than a one-literal decay change (`008/01-brainstorm.md:69`). Under a +0.10 = ten-images threshold with no reroll, cleaner attribution is worth real weight.
- **Fix:** None beyond §5; this is an inherent trade that argues for decay-first if the diagnosis is the shared target.

### 7. [Minor — all three are single-run threshold-sensitive] Correctly acknowledged.
Each notes +0.10 = ten test images and forbids rerolls; the wide-model runs are ~71 epochs with a flat tail, so run-to-run noise near 93.5% is non-trivial. No action — noted so it isn't mistaken for an omission.

---

## Scored verdict

| Idea | Evidence & reasoning | Potential impact |
|---|---|---|
| **Weight Decay 5e-4** | **4/5** — one-literal, compute-neutral, canonical value, mechanism matches the diagnosed limiter; docked for point-value transfer risk with no sweep (#3) and the plateau-vs-tail leverage gap (#4). | **3/5** — bounded upside, but genuinely higher than EXP-007's 2/5: now on a 4× model with real overfit headroom, preserving all ~27k precious updates (history repeatedly punishes update loss: EXP-003/005). |
| **Residual-Block Dropout 0.3** | **3/5** — real WRN support and architecture-aware placement, but stacks a second every-view regularizer on the plateau (EXP-006 pattern, #5), unspecified weak-tail behavior, lower attribution (#6). | **3/5** — comparable ceiling to decay if p is right, but p=0.3 likely too strong for 71 epochs; realized upside discounted by over-reg and exposure cost. |
| **Width-3 ResNet-20** | **2.5/5** — internally contradicts this brainstorm's own regularization diagnosis (#2); worst marginal capacity/step-time trade in the goal's history; well-quantified but pointed at the wrong limiter. | **3/5** — highest *raw* ceiling if a capacity trend continued, but heavily discounted: ~8 weak-tail epochs likely terminate before the peak that historically lands at the 11th weak epoch. |

### Winner: **Weight Decay 5e-4**

It is the only finalist that attacks the **diagnosed limiter head-on while preserving every update** — decisive under a fixed-time budget where this goal's history (label smoothing EXP-003, 75% switch EXP-005) repeatedly shows lost exposure erasing gains. It is the most attributable of the three (one literal, same SGD kernel path), and the EXP-007 result flips the calculus that made it a low-ceiling bet before: on the 0.27M model the dominant risk was underfit; on the 1.07M model that fits N1/M7 to a 90.08% strong checkpoint with a low tail train loss, there is genuine overfit headroom for norm control to reclaim. Dropout targets the same limiter but adds an over-regularization risk this goal has already been burned by (EXP-006) plus exposure and attribution costs; width-3 spends the whole run on the capacity axis the diagnosis just set aside and is the most fragile choice under fixed time.

It wins **conditionally on §1**: if the plan-time re-read of the overfit signature is weak (strong-view train loss not actually low, no late test-loss creep), the norm-control premise collapses and none of the three is well-motivated — at which point dropout is *not* a safer substitute (same premise, more risk), and the right move is to re-diagnose rather than escalate regularization.

### Required refinements before planning (in order)
1. **Re-establish the premise.** Pre-register the overfit read (strong-switch checkpoint vs 90.08%, tail train-EMA, best-vs-final loss gap) as a primary diagnostic, since the "≈0.04 train / 0.214 test" evidence is unpersisted (#1).
2. **Frame 5e-4 honestly as plateau-norm control that carries into the tail**, not a direct suppressor of the late excess fit (#4); state the underfit tell and that a null routes to a *lower* decay, not to abandoning the lever (#3).
3. **Keep the compute-neutrality gate** (≥99% of 27,143 steps; assert `weight_decay == 5e-4`, unchanged params/loader/seed) so any exposure change flags a confound, not the intervention.
4. **Hold width-2, N1/M7, 80% boundary, hard-label tail, and 1e-4-except-the-literal fixed** — single-line diff, no dropout/width bundling — so the result is attributable.
5. Explicitly **defer width-3 and dropout** as pre-registered follow-ups keyed to this run's signature: persistent overfit after 5e-4 → dropout or higher decay; underfit → lower decay; clean capacity-starved signature (train loss still high under N1/M7) → reopen the width axis.
