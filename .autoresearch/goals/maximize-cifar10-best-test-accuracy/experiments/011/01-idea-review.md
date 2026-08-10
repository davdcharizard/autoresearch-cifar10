# Adversarial Review — EXP-011 Candidates

All three candidates are single-lever tweaks to the EXP-010 CutMix recipe (94.15% baseline). The acceptance bar is **94.25%** (+0.10 necessary condition, explicitly named in EXP-010 next steps). I read the brainstorm, goal definition, results TSV, learnings, EXP-005/010 analyses, the CutMix paper note, and `train.py` (`CUTMIX_PROBABILITY` at line 23, `cutmix_collate` at 113, phase/loader logic at 273–314).

## Prioritized Critique (most important first)

1. **"Decrease to 0.25" points the lever the wrong way relative to its own evidence — near-fatal reasoning flaw.** The brainstorm justifies it by the "0.35-point strong-checkpoint deficit." But EXP-010 (`010/04-analysis.md` §Results) shows that deficit did *not* cost anything: the **first weak checkpoint was already +0.20 above EXP-007** and the tail finished at its best. That is the signature of regularization that is net-positive and *not* over-dosed. Cutting probability dilutes the exact mechanism that produced +0.60. The idea's own risk assessment concedes this. *Fix:* only run it as the cheap lower arm of a p-bracket {0.25, 0.75}, not as a standalone hypothesis — and state that its prior is negative.

2. **"Stop CutMix at 70%" rests on a premise the data arguably contradicts.** The stated diagnosis is "recovery demand may limit the result" because the tail rose to termination. But a tail still rising *at best* is equally read as "refinement was productive and unfinished," which argues for *more* of the current recipe, not for inserting an earlier hard-label N1/M7 interval. The 70–80% window it creates is **strong-but-hard-label** (RandAugment on integer labels), which is neither the successful weak tail nor validated anywhere. *Fix:* if pursued, gate it on a concrete prediction (e.g., 70% variant must beat EXP-010's 93.16% first weak checkpoint) rather than the vague "recovery demand."

3. **"Stop at 70%" carries the largest confound-and-cost burden for the weakest mechanism.** It adds a *second* loader teardown/rebuild (line 305–308 currently fires once at 80%; a 70% variant doubles it), broadly perturbing the RNG/data stream — the same class of confound EXP-010's analysis already flags as limiting attribution. 70% is not literature-grounded, and EXP-005 (`005/04-analysis.md`) already showed early strong-phase removal hurts. It's the highest-effort, highest-risk, lowest-support candidate. *Fix:* isolate to a single extra transition and keep target-format/worker audits identical to EXP-010.

4. **"Increase to 0.75" has the best-supported mechanism but a real, unaddressed downside risk.** EXP-010's mildly-lower strong checkpoint (89.73 vs 90.08) plus recovered/rising tail genuinely indicates headroom for more regional mixing toward the 94.25% bar. The gap the idea ignores: at 0.75 the strong checkpoint could drop enough that the fixed **~20% hard-label tail cannot recover it** — the tail budget is fixed, the regularization load is not. *Fix:* pre-register a switch-checkpoint floor (analogous to EXP-010's 87.08% underfit marker) so a collapsed strong fit is diagnosed as the failure mode, not confused with noise.

5. **Shared blind spot across all three: single-seed evidence vs a ±0.10 acceptance margin.** The +0.60 at p=0.5 is one fixed-seed realization (both analyses concede "not a precise effect estimate"). A ±0.10 threshold is plausibly inside single-seed CutMix/RNG variance, so any of these could pass or fail on stream luck. *Fix:* whichever is chosen, frame success against a margin comfortably above 0.10 (as EXP-010's 0.50-over-threshold pass did) and treat a bare pass skeptically.

6. **Throughput is not a differentiator (mild positive for the p-tweaks).** EXP-010 measured hard/soft step cost as 10.823/10.829 ms with worker headroom, so raising or lowering probability won't threaten the 300s budget. This removes a risk from both p-tweaks and makes the 70%-stop's extra loader rebuild the only real time cost among the three.

## Scored Verdict

**Increase CutMix Probability to 0.75**
- Evidence/reasoning: **6.5/10** — the only candidate whose lever direction is *supported* by EXP-010's signature (recovered tail + slightly-lower strong fit = under-dosed, not over-dosed); mechanism (more class-bearing regional mixing → localization invariance) traces to the CutMix paper note.
- Impact: **7/10** — directly amplifies the mechanism that already delivered +0.60; highest ceiling for clearing 94.25%, throughput-safe.

**Decrease CutMix Probability to 0.25**
- Evidence/reasoning: **4/10** — cheapest to run, but its rationale is contradicted by the very checkpoints it cites; expected direction is negative.
- Impact: **4/10** — most likely dilutes the winning mechanism; low ceiling.

**Stop CutMix at 70%, Retain N1/M7 to 80%**
- Evidence/reasoning: **4/10** — premise ("recovery-limited") is a contestable reading of a tail that rose to its best; introduces an unvalidated strong-hard-label window.
- Impact: **5/10** — a genuinely distinct lever with some ceiling, but discounted by high implementation/RNG-confound risk and medium effort.

**Pick: Increase CutMix Probability to 0.75.** It is the only candidate whose lever moves in the direction EXP-010's evidence actually points — the mildly-lower strong checkpoint that fully recovered and a tail still climbing at termination are the classic marks of *under*-dosed, net-positive regularization, so more mixing is the head-on attack on the remaining headroom toward 94.25%. It is throughput-safe and a low-effort one-line change (line 23). It beats "0.25" outright (which argues against its own evidence) and beats "70%-stop" on both mechanism strength and risk (no second loader lifecycle, no RNG-stream confound, no unvalidated strong-hard-label phase). Its one real hazard — a strong checkpoint that drops below what the fixed 20% tail can recover — is concrete and cheaply guarded with a pre-registered switch-checkpoint floor; adopt that guard when planning it.
