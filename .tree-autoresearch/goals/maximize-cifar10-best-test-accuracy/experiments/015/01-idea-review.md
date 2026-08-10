# Adversarial Review — Three Finalists

## Independent verification first (all three sets of arithmetic recomputed)

**B's EMA mass/kernel claims — every number checks out.**
- Copy-in anchor terminal mass: 2^(−74.7736/18.75) = 2^(−3.98793) = **0.063025** ✓
- Mean inter-sample gap 74.7736/159 = 0.47028 s → per-sample decay d = 2^(−0.025082) = 0.982766 → newest weight 1−d = **1.7234%** ✓
- Copy-in ESS: Σw² = 0.063025² + (1−d)(1−d³¹⁸)/(1+d) = 0.0039721 + 0.008657 = 0.012629 → 1/Σw² = **79.18** ✓
- Corrected terminal mass 1 − 2^(−74.906/18.75) = **0.937283** ✓; oldest weight (1−d₁)·d¹⁵⁹/m = 0.004881·0.063025/0.937283 = **0.0328%** ✓; newest 0.017234/0.937283 = **1.8387%** ✓; ESS = 0.937283²/0.008657 = **101.48** ✓
- The recurrence is algebraically a normalized ratio estimator (S←dS+(1−d)x, m←dm+(1−d), E=S/m), so "first alpha = 1, copy exact" is exactly true, not approximately. B's estimator claims are sound.

**C's gradient algebra — both identities are exactly right.**
- Hard label: d/dz[−log p_y + ε(1−p_y)] = (1+εp_y)(p−e_y), multiplier ∈ [1,1.25] ✓
- Soft target: d/dz[−Σq log p + ε(1−q·p)] = (p−q) + ε·p⊙((q·p)·1 − q) ✓
- Constituent-decomposition claim verified: λ(1+εp_a)(p−e_a) + (1−λ)(1+εp_b)(p−e_b) = (p−q) + ε[(q·p)p − q⊙p], identical to the soft form, since λp_a+(1−λ)p_b = q·p ✓
- The p=q equilibrium shift is real and I get the same sign: for q=(0.7,0.3), ‖q‖²=0.58, grad_a = ε·0.7(0.58−0.7) = −0.021 (raises z_a), grad_b = +0.021 (lowers z_b). CE contributes zero there, so **Poly-1 installs a new equilibrium biased toward the majority class** — CutMix's soft target is no longer a fixed point.

**A's cadence claim is right but trivial**; gcd(5,31)=1 → ~1/5 of the 160 EMA samples land on a sync step (32) ✓. The consequential arithmetic A did *not* do is below.

---

## Prioritized feedback

### 1. A: unaddressed effective-step halving — the largest un-modeled risk in the set (near-fatal as specified)
`slow ← slow + 0.5(fast − slow)` with fast reset to slow each cycle advances the trajectory by **0.5·D per k steps** where D is the k-step SGD displacement. Over 25,798 steps in a 300 s budget with a tuned annealing schedule, that is an effective-LR/effective-progress halving (momentum retention only partially compensates). The Lookahead paper's CIFAR gains were obtained with inner LR re-tuned for Lookahead, not by dropping k=5/α=0.5 onto a baseline's tuned LR — so "canonical, no new tuning" is self-undermining: the citation does not support the configuration actually proposed. *Refinement:* either compensate inner LR (which introduces the tuned coefficient the proposal claims to avoid), or use α≈0.8 to keep 80% of displacement, or restrict Lookahead to a phase — all of which contradict "all training from step 1."

### 2. A: mechanism is redundant with the object being evaluated (mechanism↔limiter gap)
The sole evaluator is a 160-sample full-state EMA with ESS ≈ 79. Lookahead's mechanism is *variance reduction of the online iterate*. Variance in the iterate has already been averaged away in the evaluated quantity; you cannot harvest it twice. The limiter is stated as a 0.2–0.3 pt **generalization** lift, and Lookahead offers no generalization mechanism the parent lacks (Nesterov + periodic SAM + tail EMA). A's own self-estimate (0.00–+0.15, ranked below a cheaper method) is consistent with this. Combined with #1 the expected sign is arguably negative.

### 3. A: the latency gate is underestimated (concrete gate-breach mechanism)
"One model-sized non-gradient copy" prices HBM traffic, not launches. 2.75 M params ≈ 11 MB → ~11 µs of traffic on H20, but a per-tensor Python loop over ~100–200 parameter tensors × (lerp_ + copy_) = 200–400 kernel launches every 5 steps at ~86 steps/s ≈ 0.24–0.4 ms/step ≈ **2–3.5% latency**, straight through the ≤1.02 gate — and "projected ≥25k steps" leaves only 3.1% headroom off the parent's 25,798. Precedent: width-320 was killed pre-metric at 1.161×. *Refinement:* `torch._foreach_lerp_` + `torch._foreach_copy_` over a pre-built list, or a single flat contiguous slow buffer, inside the existing `no_grad` region; measure launch count, not bytes.

### 4. B: the effect size is structurally capped ~4× below its own success bar (non-fatal but decisive)
Δ(corrected − uncorrected) ≈ 0.0627·(x̄_late − x₁), i.e. ~6.3% of the parameter drift over the final ~53 s. Parent EMA↔live distance is 1.506% and buys the usual ~0.2–0.3 pt; B's displacement is ~0.10–0.15% relative — roughly **1/12 of the EMA-vs-live lever**. Expected gain ≈ +0.02…+0.06, versus formal +0.10 (95.71) and scientific tail +0.197 (95.69 vs 95.493). Airtight math in service of a near-guaranteed null run. *Refinement that raises the ceiling:* the correction is precisely what makes a **longer horizon safe** (no stale-anchor penalty). Pair debiasing with half-life 18.75→~30 s and/or activation at progress 0.65; that moves ESS 79→150+ and mean age materially, converting a rounding error into a real lever. This requires relaxing the "no new tuned coefficient" premise — but the premise is exactly what caps the idea.

### 5. B: the ESS increase works *against* the actual metric (best_test_acc is a max, not a mean)
Parent best 95.61 vs final-16 mean 95.493 → ~+0.12 of max-order-statistic bias across evaluations. Raising ESS 79→101 cuts per-evaluation sd ~11%, shrinking that upward bias by ~0.013 and eating a quarter of B's expected gain. B optimizes the tail mean while the reported metric rewards the max. *Refinement:* state the predicted split (tail mean ↑, max-bias ↓) explicitly in the pre-registration so a flat best_test_acc with a lifted tail mean is not misread as success — and note this is the one place where "improving the estimator" and "improving the metric" diverge.

### 6. B: the stated risk list mis-identifies its own risk
"Correction may follow late SAM noise" is not a real exposure: the newest-sample weight moves only 1.7234% → 1.8387%. The genuine risk is the opposite one B lists second-hand: the 6.3% anchor at age 74.8 s may act as an implicit long-horizon SWA/regularizer whose removal loses flatness. That is testable cheaply in the paired gate by logging EMA↔live distance (parent 1.506%) — a large drop in that distance is the signature of the interpolation benefit being removed.

### 7. C: ε's **sign** opposes C's own diagnosed limiter (prerequisite fix, not fatal)
The multiplier (1+0.25·p_y) ∈ [1,1.25] is maximal exactly where p_y→1 and ≈1 where p_y→0 — anti-focal, upweighting already-solved examples. The stated limiter is residual boundary/hard-example generalization at 95.5%. ε<0 gives multiplier (1−0.25p_y) ∈ [0.75,1], relatively upweighting boundary examples, **and** flips the CutMix equilibrium shift from majority-sharpening to minority-softening (aligning with the validated EXP-002 CutMix rather than opposing it). PolyLoss explicitly permits negative ε and reports task-dependence. The single most valuable refinement to C is to resolve the sign against the limiter rather than inheriting the paper's classification default.

### 8. C: ε=0.25 is anchored to a gradient-norm bound, not to any effect size
Published classification-effective Poly-1 magnitudes are of order 1; 0.25 is ~4× smaller. This is the reward-hacking-inverse risk: chosen for safety of the audit, likely landing inside the sub-0.30 noise band the common evidence warns has already reversed results. *Refinement:* pick |ε| from the desired logit-margin change per step (or from a short accuracy-blind proxy on training loss curvature), then commit; do not defend 0.25 with the 25%-inflation argument, which is a constraint, not a prediction.

### 9. C: both SAM passes are modified, changing the ascent direction of a validated component
EXP-004's periodic SAM was accepted with the parent loss. Replacing the loss on the ascent pass changes ε-ascent geometry (the perturbation now over-weights confident examples), so C ships two entangled changes. *Refinement:* keep the ascent pass on plain CE and apply Poly only to the descent pass — preserves the validated ascent direction, isolates the objective change, and reduces the interaction surface with the 80 SAM-derived EMA samples.

### 10. C: scope the Poly term to clean steps if ε>0 is retained
With front-loaded probabilistic CutMix, the clean-tail region — exactly where the 160 EMA samples are drawn — is predominantly clean. Restricting Poly to non-CutMix steps removes the equilibrium-shift conflict entirely while concentrating the change where the evaluator actually samples. This is strictly dominant over global application if the sign stays positive.

### 11. All three: C's latency claim is the only credible one
256×10 FP32 softmax + gathers ≈ 4 extra tiny launches × 2 passes × 25.8k steps ≈ 0.2–0.3% — 1.002–1.008 is realistic and the ≤1.01 gate is appropriate. Reuse `log_softmax`'s output via `exp` rather than a second softmax to stay at the low end. B's gate exposure is essentially zero. A's is the one at risk (#3).

---

## Scores

| Finalist | Evidence & Reasoning | Potential Impact |
|---|---|---|
| **Canonical Lookahead around Nesterov** | **4/10** — cadence arithmetic correct and the citation is real, but the paper's gains came with re-tuned inner LR, the effective-step halving is un-modeled, and the variance-reduction mechanism is redundant with a 160-sample ESS-79 EMA; the author ranks it below another idea. | **3/10** — self-estimated 0.00–+0.15 against a +0.10/+0.20 bar, with a plausible large *negative* from halved effective progress in a 300 s budget plus real gate exposure. |
| **Activation-anchored bias-corrected EMA** | **8/10** — every mass/ESS/age figure independently reproduced, the recurrence is a genuine ratio estimator with exact first-sample copy, no new coefficient, and both bias and ESS improve simultaneously; only its own risk list is mis-aimed. | **3/10** — displacement is ~6% of the tail drift ≈ 1/12 of the EMA-vs-live lever → ~+0.02…+0.06, and the ESS gain mechanically shrinks the max-order-statistic bias the metric depends on; a near-certain null unless the horizon premise is relaxed. |
| **Bounded soft-target Poly-1** | **5/10** — hard, soft, and constituent-decomposition gradients all verified exactly and the CutMix conflict is honestly derived; but ε's sign contradicts the stated limiter, its magnitude has no effect-size anchor, and PolyLoss offers only task-dependent support. | **6/10** — the only candidate that alters the objective on every one of ~25.8k steps at ~0.3% cost, with a genuine ±0.2 range and the highest ceiling of the three; sign is currently the coin-flip. |

---

## Verdict

**Bounded soft-target Poly-1** wins, but only with prerequisites attached.

Why it beats the others: it is the sole finalist whose ceiling reaches the +0.10 formal / +0.197 tail bar. B is the best-executed *analysis* in the set — I could not break a single one of its numbers — but its own no-new-coefficient premise caps the perturbation at ~1/12 of the EMA-vs-live lever, and the one metric run is too scarce to spend on a change whose honest point estimate is +0.03 and whose variance reduction partly cancels the max-statistic bias the reported metric rides on. A is dominated on both axes: its mechanism duplicates work the EMA already does, its citation does not support its configuration, its effective-step halving is unaddressed and plausibly negative, and it carries the only real gate-breach risk — its own author ranks it second.

C's flaws are parameter-level and fixable inside the idea; B's is structural to the idea's premise. That asymmetry decides it.

**Prerequisite refinements before planning C:**
1. Resolve the sign of ε against the limiter — ε<0 (multiplier 1−|ε|p_y) upweights boundary examples *and* turns the p=q CutMix shift from majority-sharpening into minority-softening; commit to one sign in the pre-registration, accuracy-blind.
2. If ε>0 is retained, scope the Poly term to non-CutMix steps only (removes the verified equilibrium shift; concentrates the change in the clean tail where all 160 EMA samples are drawn).
3. Keep the SAM ascent pass on the parent loss; apply Poly to the descent pass only, so the validated EXP-004 ascent geometry is not co-modified.
4. Anchor |ε| to an intended effect size, not the 25%-gradient-inflation bound; 0.25 risks landing inside the sub-0.30 noise band.
5. Reuse `log_softmax`→`exp` for p rather than a second softmax; verify ≤1.01 by launch count.

If the sign question cannot be settled accuracy-blind, the fallback with the best reasoning-per-run is **B extended** — debiasing plus a lengthened half-life (~30 s) or earlier activation (~0.65), since the correction is precisely what makes the longer horizon safe. That is a different idea than B as written, and should be labeled as such.

`PICK: BOUNDED SOFT-TARGET POLY-1`
## VERDICT: PASS

### 1. Sign alignment — correct as claimed

For hard label $y$, $L = -\log p_y + 0.25\,p_y - 0.25$. Using $\partial_z \text{CE} = p - e_y$ and $\partial_z p_y = p_y(e_y - p)$:

$$\partial_z L = (p - e_y)\,(1 - 0.25\,p_y)$$

So the hard-label gradient is exactly the parent CE gradient rescaled by $1 - 0.25p_y \in [0.75, 1]$. The quoted values check out (0.975 / 0.875 / 0.775 at $p_y = 0.1/0.5/0.9$), and $|\epsilon| = 0.25 < 1$ guarantees the multiplier is strictly positive — no sign inversion and no amplification anywhere on the domain. This is a rescaling only; it does not move the hard-label optimum (gradient still never vanishes for $p_y<1$).

For soft targets the added gradient is $+0.25[(q \odot p) - (q\cdot p)\,p]$, which is **not** parallel to $(p-q)$, so no scalar-multiplier description exists — as the writeup concedes. I verified the direction claim at $p=q$: with $\lambda=0.7$, the component on the majority class is $+0.25\lambda(\lambda-S)=+0.021$ ($S=\sum q_i^2=0.58$) and $-0.021$ on the minority, i.e. descent moves *away* from the majority — equalizing, the reverse of positive $\epsilon$. Magnitude of the resulting fixed-point shift, solving $(u-0.7)+0.1u(1-u)=0$: $u^\star = (11-\sqrt{93})/2 = 0.678$ versus 0.700. So "over-softening" is real in direction but ≈2.2 points at $\lambda=0.7$ — comparable to $\epsilon_{ls}\!\approx\!0.03$ label smoothing. The flagged risk is bounded and benign at this coefficient; the unbounded *ratio* near cancellation is a diagnostic artifact of dividing by a vanishing denominator, not a gradient-magnitude hazard.

### 2. CE-ascent / Poly-descent — coherent, not a fatal mismatch

$\nabla$Poly is a per-example reweighting of $\nabla$CE with weights confined to a 4/3 band, so the batch ascent direction has very high cosine similarity to the true Poly ascent direction; SAM is empirically robust to far cruder ascent approximations (stale/reused directions). More importantly, the perturbation is a *fixed, normalized* $\rho=0.05$ direction that is held bit-identical to the parent, which is what makes this a single-variable delta: RNG replay + BN suppression + identical adversary means the only change versus parent is the descent-pass loss. The tested claim is therefore "Poly-shaped update gradient," not "Poly-1 sharpness-aware minimization" — the writeup must not claim the latter. One asymmetry to note: at perturbed weights $p_y$ is lower by construction, so the multiplier sits nearer 1 and the intervention is *weaker* on SAM steps than on ordinary steps.

### 3. Numerics — adequate

Leaving `F.cross_entropy` untouched preserves the parent's stable fused `log_softmax`+NLL and its autocast fp32 promotion, so the validated path is bit-preserved; the additional `torch.softmax(..., dtype=torch.float32)` promotes only the Poly branch, and its fp32 scalar adds cleanly to the fp32 CE scalar. The extra softmax over a $(B,10)$ tensor plus one autograd node is negligible against a WRN-16-4 step, so the ≤1.01 median / ≤1.02 per-round step-time gate is not endangered by it. Two caveats in concerns below.

### 4. Fixed $|\epsilon|=0.25$ — defensible

Pre-registering one point buys clean falsifiability and avoids data-conditioning the coefficient, and the bounded-multiplier argument gives a genuine a-priori safety guarantee rather than a post-hoc one. The honest cost is interpretive, not procedural: published Poly-1 recommends $\epsilon\approx+1$, so this is a novel-sign probe, and a null result cannot separate "wrong sign" from "magnitude too small." Accept it as a screening decision with the conclusion scoped to $\epsilon=-0.25$.

### Blockers

None.

### Nonblocking concerns

1. **The `multiplier ∈ [0.75, 1]` audit must be scoped to hard-label calls only.** On `cutmix_poly` the gradient is not a multiple of $(p-q)$, so a global assertion would be mathematically false and would trip the decisive evaluator-guarded preflight into a spurious abort. Assert the multiplier identity on `ordinary_poly` / `sam_descent_poly`, and on `cutmix_poly` assert the vector formula against autograd plus a per-example loss-value bound instead.
2. **Define the SAM × CutMix intersection explicitly.** Four counters with no `sam_cutmix` bucket implies disjointness (plausible given front-loaded CutMix and clean-finish SAM), but that is currently implicit. Add `assert sam_ascent_ce == sam_descent_poly` and an assertion that the intersection is empty, so an unintended schedule overlap fails loudly rather than silently taking an undefined path.
3. **Assert the multiplier to tolerance, not exactly.** The identity is realized through autograd over an fp32 softmax with fp16/bf16 logits; use a relative tolerance (~1e-3) or the preflight will fail on rounding.
4. **Effect is confounded with a small tail effective-LR reduction.** The multiplier is ~1 at init and only bites once the model is confident, i.e. precisely in the clean tail where the EMA accumulates — up to 25% gradient attenuation on confident examples there resembles a mild late LR decay. Either outcome sign is explainable by that mechanism; a tail LR-scale control would be the natural follow-up, not a prerequisite.
5. **Power.** The intervention's expected effect (≈2.5–22.5% per-example rescaling; ~0.03-equivalent smoothing on mixed batches) is plausibly at or below single-run noise, so a miss near the 95.69/95.71 thresholds should be reported as "not resolved at this budget," not as refutation.
6. Confirm ascent-pass gradients are zeroed before the descent pass, so the Poly gradient is genuinely the sole optimizer input, and that parameter restoration precedes the single update as stated.

Planning may proceed with this exact fixed package, with concerns 1–2 folded into the audit spec before launch.
