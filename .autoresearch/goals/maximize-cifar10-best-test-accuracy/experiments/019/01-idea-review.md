I have thorough grounding across the brainstorm, all three proposals, both papers, the goal definition, system understanding, learnings, results TSV, `train.py`, and the EXP-018 analysis. Here is the review.

---

# Adversarial Review - EXP-019 Candidate Ideas

## Prioritized Feedback (most important first)

### 1. Idea-03 (Weak-Tail EMA) fights its own diagnosis - the tail is monotonic, so there is little noise to smooth and the online endpoint is the thing to keep. **[Fatal to the thesis]**
The whole premise is "damp sub-epoch SGD noise in the weak tail." But the evidence says that noise is nearly absent: EXP-002 measured best-final = 0.01 (`04-results.tsv` line 6; learnings "Dense final-tail evaluation found only a 0.01-point best-versus-final gap"), and EXP-010's best == final == 94.15. EMA of a monotonically-improving, low-noise sequence *lags* the endpoint rather than improving it. Worse, the proposal *forfeits* online late evaluations (idea-03.md:22, "each scheduled evaluation is EMA-only"), and EXP-018 already showed online (94.02%) beat the averaged model (93.85%) on this exact tail (018/04-analysis.md:23). So the design is expected-negative on the one comparison we have direct data for. This is mechanistically the same failure family as EXP-018 (averaging a converging cosine tail), even though the decay policy is technically distinct - the learnings note flags EMA as a "distinct hypothesis," which saves it from being a disqualifying retry, but the headwind is identical.
- **Path to improve:** would require evidence that sub-epoch (intra-epoch) noise exists and is large - e.g., a pre-registered intra-epoch variance probe - *and* a policy that keeps the online endpoint as a fallback (evaluate online at the terminal step, EMA elsewhere) rather than blindly replacing it. Absent that, this idea should be deprioritized.

### 2. Idea-02 (Balanced Mixup/CutMix): alpha=0.2 quietly *reduces* effective regularization and *dilutes* the very "global interpolation" mechanism it sells. **[Material - the core bet needs re-grounding]**
The proposal's headline is "preserve 50% mixed, add a genuinely different global-linearity invariance" (idea-02.md:5-21). But Beta(0.2,0.2) is endpoint-heavy by the proposal's own admission (idea-02.md:19) - most Mixup draws give lambda approximately 0 or lambda approximately 1, i.e. near-clean images with near-hard targets. Two consequences it doesn't reconcile:
- (a) Replacing 25% alpha-1 CutMix (strong regional occlusion) with 25% near-hard Mixup means *total effective regularization drops below EXP-010's*, despite the "50% mixed" framing. Since CutMix at p=0.5 delivered the +0.60 frontier gain over plain width-2 (94.15 vs 93.55; results lines 11,14), halving effective strong-mix regularization risks giving back part of that 0.60 - a concrete regression path toward approximately 93.8-93.9.
- (b) The claimed "whole-image linear interpolation" invariance is barely delivered, because few batches actually sit near lambda=0.5. So the experiment mostly tests "swap a quarter of CutMix events for weak Mixup," a smaller and geometrically-diluted change than advertised.
- **Path to improve:** pre-register the expected effective-lambda mass (fraction of Mixup batches with lambda in [0.3,0.7]) and reframe the hypothesis honestly as a *mild* linearity nudge with *reduced* regional occlusion, OR raise alpha toward the cited 0.4 so Mixup genuinely delivers interpolation - while accepting the 87.08% underfit marker (idea-02.md:98) as the guardrail. The current alpha is caught between "too weak to add invariance" and "chosen to avoid over-regularizing"; make that tradeoff explicit rather than claiming equal mixing.

### 3. Idea-01 (Nesterov): the proposal's own mechanism argument points to an ambiguous-to-negative generalization sign. **[Material - tempers the upside, honestly disclosed]**
idea-01.md:49 concedes that extra current-gradient weight "could also weaken the historical-gradient filtering that the [momentum-generalization] paper associates with generalization." Since the accepted recipe already runs momentum 0.9 and that historical filtering is *the* credited generalization mechanism (papers/momentum-generalization.md:9), Nesterov plausibly trades *toward* faster fit and *away* from the generalization axis that is the diagnosed bottleneck (system-understanding.md:17, "accuracy bottleneck is generalization under a short strong-view phase"). Neither cited paper establishes a CIFAR gain over plain momentum (both are directional; nesterov-nonconvex is a deterministic method on second-order stationarity, not this stochastic recurrence). Combined with 0.10-point single-seed resolution = 10 images (idea-01.md:146), even a "pass" is weak causal evidence.
- **Path to improve:** this is inherently low-ceiling; its value is *cheap disambiguation of the EXP-001 confound*, not a likely frontier move. Frame it as an information/attribution experiment, not an impact bet, and don't over-predict 94.30% off directional theory.

### 4. Idea-02 collator returns a 3-tuple; the training loop and counters must change accordingly - in scope but adds surface. **[Minor]**
`train.py:218` unpacks `for inputs, targets in train_iterator`, and `train.py:221` counts CutMix via `int(targets.ndim == 2)` - which would also fire for Mixup's soft targets. idea-02's `mixed_collate` returns `(inputs, targets, kind)` (idea-02.md:37-49) while the weak loader stays 2-tuple, so the loop needs conditional unpacking plus three-way counters. All within `train.py` scope (constraints allow only `train.py` edits - definition.md:12), but it is real provenance plumbing with RNG-ordering risk under forkserver.
- **Path to improve:** the proposal's bitwise-RNG and proportion gates (idea-02.md:60-69) already cover this; just ensure the `kind`-aware unpack doesn't perturb the `u<0.25` CutMix RNG position it claims to preserve (idea-02.md:13).

### 5. All three share the same weak terminal evidence: single seed, +0.10 gate approximately 10 images. **[Context, not disqualifying]**
This is a protocol constraint (no reroll - definition.md:18), so it applies equally and is not a differentiator. It does mean the *reasoning* and *ceiling* must carry the decision, since a bare numeric pass proves little - which is exactly how I am scoring.

---

## Scored Verdict

**Balanced Mixup/CutMix Geometry (idea-02)**
- Evidence/reasoning - **7/10**: strongest positive analog (CutMix, same augmentation-geometry family, gave the last real +0.60 gain) and Mixup is an established CIFAR method; docked because alpha=0.2 undercuts the stated mechanism and the "equal regularization" framing is misleading (Feedback #2).
- Impact - **8/10**: highest ceiling of the three - lives in the exact lever (strong-phase invariance geometry) that the diagnosis names as the bottleneck, with room for a genuine frontier move if complementarity holds.

**Isolated Nesterov Momentum (idea-01)**
- Evidence/reasoning - **6/10**: cleanest attribution and resolves a real EXP-001 confound, but the cited theory is directional and its own mechanism argument suggests an ambiguous/negative generalization sign (Feedback #3).
- Impact - **4/10**: proposal itself caps the ceiling at a "small online basin shift" (idea-01.md:137); adds no data, capacity, or invariance.

**One-Epoch-Half-Life Weak-Tail EMA (idea-03)**
- Evidence/reasoning - **3/10**: the mechanism targets tail noise the evidence says is approximately 0.01 points, and EXP-018 directly shows online > averaged on this tail, making the online-forfeit expected-negative (Feedback #1).
- Impact - **3/10**: highest effort/risk (BN-buffer approximation, per-step foreach exposure cost) for the least-supported mechanism.

### Pick: **Balanced Mixup/CutMix Geometry (idea-02)**

It wins on merit, not caution. It attacks the diagnosed bottleneck - generalization under the short strong-view phase (system-understanding.md:17) - head-on, in the one family that has actually moved this frontier recently (RandAugment +0.47, CutMix +0.60). It introduces a genuinely different invariance while holding the 50% hard anchor and explicitly avoiding EXP-011's p=0.75 over-regularization failure, giving it both the highest ceiling and a real mechanistic story. Idea-01 is the cleaner, cheaper experiment but is a low-ceiling attribution exercise whose own reasoning hints the generalization sign may be flat-to-negative. Idea-03 is disqualified as the top pick because its core mechanism runs against the established monotonic-tail finding and it forfeits the online endpoint that EXP-018 showed is superior.

**One caveat that should shape execution before idea-02 is run:** address Feedback #2 first - either raise alpha so Mixup actually delivers interpolation (guarded by the 87.08% underfit marker) or pre-register that this is a *mild, reduced-regularization* geometry swap and predict accordingly. Run as-written with alpha=0.2 unexamined, the most likely outcome is a small regression from diluted regularization, not a gain - the idea's ceiling is real but conditional on getting the Mixup strength right.
