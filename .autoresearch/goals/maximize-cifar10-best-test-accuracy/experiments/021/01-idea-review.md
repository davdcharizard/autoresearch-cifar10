I now have all the context. Here is my adversarial review.

---

# Adversarial Review — EXP-021 Candidates

## Prioritized Feedback (most important first)

### 1. Channels-Last: the accuracy mechanism contradicts the goal's own diagnosis (severe, near-fatal on expected value)
The candidate's entire accuracy payoff runs through **exposure** (+807 updates, ~103k images). But `02-system-understanding.md` § Current Bottleneck states the accuracy limiter is *"generalization under a short strong-view phase,"* not exposure, and explicitly: *"additional exposure has not yet been causally tested at the accepted recipe."* The proposal itself concedes this four times (`idea-channels-last.md` Hypothesis, Risks: *"Exposure may not help… accepted accuracy is generalization-limited"*). So even in the best case — timing gate clears AND 94.25% is hit — the proposal's own decision table admits attribution collapses (*"formal metric clears the goal, but the registered exposure mechanism failed"*). You would be spending the one fixed-seed run to buy a mechanism the diagnosis says is not the binding constraint. *Refinement:* there is no clean fix within scope — this is structural. The only honest framing is that a win here is a trajectory-variance win, not an exposure win, and the diagnosis says the ceiling is elsewhere.

### 2. Channels-Last: the speedup premise is unsupported by its own source and likely to self-veto (severe)
`papers/channels-last-memory-format.md` (the candidate's primary source) states plainly: *"PyTorch's performance guidance presents channels-last primarily together with reduced precision or AMP. None of the sources establishes a speedup for this repository's FP32 ResNet-20 workload."* The workload is the worst case for NHWC FP32: 32×32 spatial, tiny channel counts (stem is 3→32; 3-channel input is the canonical NHWC-unfriendly case), launch-bound small kernels. The 3.09% gate has a high probability of stably missing → **no accuracy run**, exactly the EXP-013 and EXP-016 outcome (invalid/NaN). This goal's history has now shown *twice* that systems/exposure ideas gated on fresh paired timing die before the accuracy run; a third is close to retrying a pattern history shows fails. *Refinement:* if pursued at all, front-load a cheap disposable single-arm channels-last timing probe *before* building the 10-gate correctness battery, so effort isn't sunk into a candidate that dies at the timing gate. But this doesn't fix the low expected value.

### 3. CutMix α=0.5: the mechanism plausibly *hollows out* the useful signal, and direction is unsupported (severe)
The proposal's own math is the counter-argument: a U-shaped Beta(0.5,0.5) moves mass to the endpoints, which *reduces* the mid-λ (≈0.5) events. Those moderate-occlusion events are precisely where CutMix's localization/occlusion benefit lives (`cutmix.md`: "regional-occlusion regularization"). `idea-cutmix-alpha.md` admits this: *"more endpoint events can weaken moderate-size occlusion/localization"* and *"same-class mixing reduces effective intervention."* So the intervention is a plausible-both-ways bet with no evidence for the helpful direction — EXP-010 *won* with the 0.35-point strong deficit intact, so there is no evidence that closing that deficit via α helps rather than dilutes the winning signal. *Refinement:* the distribution gate (≥70% non-trivial rectangles, variance ≥1.25×) proves the *distribution shifted*, not that the shift *helps* — it cannot de-risk the direction. This is genuinely low-ceiling: the proposal concedes *"a scalar refinement around the frontier has limited upside."*

### 4. CutMix α=0.5: runs against recorded strategic steer (moderate)
Both EXP-010 and EXP-011 explicitly de-prioritize further CutMix scalar tuning: EXP-011 Next Steps — *"A new architecture or representation lever on the accepted p=0.5 recipe now has a stronger rationale than more CutMix scalar tuning"*; EXP-010 medium-low — revisit architecture *"if CutMix tuning lacks a mechanism for clearing the new 94.25% threshold."* This is not fatal (α=0.5 ≠ the already-failed p=0.75 point, so it is not an unchanged retry), but it is the weakest-justified of the three against the documented guidance.

### 5. Pool-First Option-A: the core hypothesis leans on an unproven decomposition of EXP-017 (the main weakness of the strongest candidate)
The candidate assumes EXP-017's harm (NLL 0.2024 vs 0.1934, best 94.09 vs 94.15) came from the *learned 1×1 projection + BN*, not the *pooling*. EXP-017's analysis does attribute it to *"the random normalized transition basis"* — supporting the isolation — but this is not proven; the pooling itself could be the diluting agent (box-filtering tiny CIFAR edges), with the learned projection having partially *compensated*. If so, removing the projection makes it worse, not better. *Refinement (concrete and cheap):* pre-register EXP-017's switch (90.20%), first-weak (93.45%), and the NLL-up/best-down signature as the discriminator. If pool-only reproduces improved switch/first-weak but NLL again rises above ~0.19 and best again falls, that *confirms* pooling — not the projection — carries the late-generalization harm, turning even a miss into a clean mechanistic verdict. This is the one candidate where a negative result is still high-value.

### 6. Pool-First Option-A: identity-gradient dilution and branch-spectrum mismatch (moderate, acknowledged)
Average pooling spreads the direct identity gradient from 1.0 (even phase) to 0.25×4, weakening the transition identity path, while the residual branch's stride-2 conv stays phase-sensitive — so the two branches can be spectrally mismatched at the add (`idea-pool-first-option-a.md` Risks list this). This is a real but bounded risk and is correctly gated (autograd 0.25 check, byte-equal non-transition blocks). No fix needed beyond the existing gates; flagging that the expected effect sits *"near the one-seed resolution floor"* — a bare 94.25% pass is weak causal evidence (10 images).

### 7. All three: 19-evaluation parity cap handling (verification note — not a flaw)
Only Channels-Last can add an epoch, and it correctly implements the mandatory cap (`evaluation_count < 18` for weak-tail looks, reserved terminal look, tested against both 69- and 70-epoch schedules → 19 unique looks including terminal). Pool-First and α=0.5 are compute-neutral/negative and cannot manufacture extra looks; both assert ≤19. Constraint satisfied by all three; no violation. Note the cap is a metric-parity guard, not an accuracy mechanism — none of the candidates leans on it for the gain, which is correct.

---

## Scored Verdict

**Deterministic Pool-First Option-A Shortcuts**
- Evidence/reasoning — **4/5**: Cleanest mechanistic argument, tied to a directly-relevant prior result (EXP-017) it isolates; anti-aliasing/ResNet-D grounding is directional not decisive, and the projection-vs-pooling decomposition is assumed, not proven.
- Potential impact — **3.5/5**: Attacks the diagnosed generalization/transition limiter head-on, preserves ~98%+ exposure, informative either way; ceiling is modest (94.25–94.40) and near single-seed resolution.

**CutMix Alpha 0.5 at Fixed Probability**
- Evidence/reasoning — **3/5**: Quantitatively clean single-literal change with strong isolation, but the *direction* of benefit is unsupported and plausibly harmful (hollows out mid-λ occlusion events).
- Potential impact — **2/5**: Self-admitted low ceiling around a validated point; against recorded steer away from CutMix scalar tuning.

**FP32 Full-Model Channels-Last**
- Evidence/reasoning — **2.5/5**: Rigorous, honest, and heavily gated, but both load-bearing premises are weak — its own source denies an FP32 speedup exists, and the exposure→accuracy link is untested and contradicted by the diagnosis.
- Potential impact — **2/5**: Nominal 94.28 ceiling, but the modal outcome is a stable timing-gate miss → no accuracy run (as in EXP-013/016); a win would have muddy attribution.

### Pick: **Deterministic Pool-First Option-A Shortcuts**

It is the only candidate that attacks the *diagnosed* limiter (generalization under short strong views / transition information) with a mechanism the system understanding endorses, while the other two are structurally compromised on the dimension that matters most. Channels-Last routes its entire payoff through exposure — which the goal's diagnosis says is not the binding constraint and which its own cited source says probably won't even materialize as a speedup; its most likely result is a third systems-idea self-veto with no accuracy number. α=0.5 is low-ceiling by its author's admission, points in an unsupported (possibly harmful) direction, and cuts against the recorded guidance to leave CutMix scalar tuning behind. Pool-First is the cleanest scientific move on the board: it isolates the *one* unresolved variable from EXP-017 (deterministic downsampling vs. learned projection/BN), preserves exposure and attribution, and — with the EXP-017 switch/first-weak/NLL signature pre-registered (feedback #5) — yields a decisive mechanistic verdict whether it clears 94.25% or not. That combination of on-target mechanism plus guaranteed information return makes it the strongest bet, not merely the safest.
