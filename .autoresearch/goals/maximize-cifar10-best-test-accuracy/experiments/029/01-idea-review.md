# Adversarial Review — EXP-029 Candidates

## Prioritized Feedback (most important first)

### 1. Channels-last (idea-02): attacks the systems bottleneck, but the *accuracy* limiter is generalization — and the exposure→accuracy link is explicitly unproven
This is the deepest flaw. `02-system-understanding.md` says the accuracy bottleneck is "generalization under a short strong-view phase," and the Headroom Assessment states plainly: *"additional exposure has not yet been causally tested at the accepted recipe."* EXP-013 already delivered +18.91% image throughput (batch 256) and it was never shown to convert to accuracy. Channels-last is a pure exposure play riding on a link the goal's own history has not established. So even the *best case* — a real 3% step speedup — only buys more steps of unknown accuracy value.
- **Compounding uncertainty:** (a) will NHWC even be faster on tiny 32×32 FP32 tensors? The proposal itself concedes official gains are for reduced precision/large tensors and that Option-A slice/pad may trigger fallbacks; (b) even if faster, more steps may not help. Two independent coin-flips multiplied.
- **Attribution is unrecoverable:** the proposal admits different legal cuDNN kernels change reduction order, so a 94.25% result "is the net implementation effect rather than a bitwise extra-exposure counterfactual." You can't even attribute a win to exposure. *Path to improve:* there isn't a clean one within this framing — the systems win doesn't map to the diagnosed limiter. This is the weakest bet, not fatal but low expected value.

### 2. Stochastic depth (idea-03): sits inside a documented failed family and partly leans on the same unproven exposure link
`03-experiment-learnings.md` records **"Identity-oriented residual changes suppress the short strong phase"** (Medium, count 2: EXP-012 preactivation −2.85, EXP-015 zero-gamma −3.25). Batchwise whole-branch bypass is exactly an identity-oriented residual perturbation, stochastically applied. The proposal's own Risk section concedes "even a 10% final drop rate can remove a material fraction of representation depth" with only nine blocks.
- **Transition blocks are the sharp edge:** keeping indices 3 and 6 droppable zeroes newly-padded Option-A channels on drop — and EXP-024 already showed "abrupt late width can create an early class transient," EXP-017/021 retired pool-first transition shortcuts. Dropping a transition branch is a novel way to hit the same transition fragility.
- **Also leans on exposure:** ~5% branch-compute savings → more steps is half the hypothesis, and that half inherits the unproven exposure→accuracy link above.
- **Ceiling is thin:** point prediction 94.25–94.35% barely clears the +0.10 gate while carrying the highest strong-phase-underfit risk. *Path to improve:* the paper's regime is 110–1202 layers; a materially different, defensible version would be the brainstorm's own "stochastic depth + extra depth" combination (add blocks so effective depth is preserved) — but that's explicitly deferred as too broad. As scoped, this is close to retrying a failed mechanism.

### 3. GC (idea-01): the tested operator is *not* the paper's, and BatchNorm may make it partly redundant
GC is the strongest candidate (below), but two honest gaps:
- **Operator ≠ evidence.** The proposal applies `P(g) + λw` (decay after projection), not the paper's `P(g + λw)` invariant-mean formulation — it says so in "Decay and momentum ordering." The ECCV theorem and much of its empirical weight rest on the projected-decay version. So the direct-evidence claim is softer than it reads: the *exact* operator being run has weaker literature backing, and the choice is forced by the local constraint (EXP-008/009 forbid changing decay). This is a genuine mechanism/evidence tension — the thing that made GC work in the paper is partly the thing local history forbids. *Path:* the proposal already isolates and documents this; keep the interpretation gate ("lower switch fit + miss = GC removed useful high-LR drift") — it correctly makes the result falsifiable rather than rescued.
- **BN redundancy — an unstated assumption.** Every one of the 19 convs feeds a BatchNorm. BN is invariant to per-filter weight scaling, so the per-output-filter mean component GC removes is partly already neutralized downstream. GC's regularization may therefore be weaker here than in the paper's setting. *Path:* the preflight already records "removed common-mode norm by stage" — elevate that to a first-class read: if removed-mean norm is a tiny fraction of gradient norm, the mechanism is near-inert and a null result is expected, not a surprise. This should temper the 94.30% point prediction.
- **Residual optimizer-path risk is real but well-contained.** GC is an optimizer-path intervention and those are a High-importance recurring failure (EXP-020/022/028), where even exact first-step and coherent-scale matching didn't prevent collapse because momentum integrates a different trajectory. GC is genuinely narrower (modifies the gradient itself, cannot increase Conv gradient norm, no alternating/pullback state), and the immutable-corpus concentration/update-spike gates are mandatory and appropriate. Not fatal — but the "cannot increase norm" argument is instantaneous, and EXP-028 is the precedent that instantaneous guarantees don't bind the momentum path. The gates, not the norm argument, are what protect this.

### 4. Cross-cutting: only GC's payoff maps to the diagnosed limiter
Both idea-02 and idea-03 route their upside through added exposure; the diagnosis says the limiter is generalization and that exposure has never been shown causal here. GC is the only candidate whose primary mechanism (remove common-mode gradient drift → better generalization) attacks the stated accuracy bottleneck directly, at fixed exposure.

---

## Scored Verdict

**Conv2d-Weight-Only Gradient Centralization (idea-01)**
- *Evidence/reasoning:* **8/10** — direct ECCV conv-only CIFAR precedent + EXP-028's own recommended next probe; docked because the tested `P(g)+λw` variant isn't the paper's theorem and BN may blunt the effect.
- *Impact:* **6/10** — modest but real ceiling; clears +0.10 plausibly (EXP-012/026 already reached within 0.03), attacks the actual limiter at fixed exposure, cleanest attribution.

**FP32 Channels-Last (idea-02)**
- *Evidence/reasoning:* **4/10** — official gains are for reduced precision/large tensors; tiny FP32 32×32 case is explicitly unresolved, and exposure→accuracy is uncausal in this goal's history.
- *Impact:* **4/10** — even a successful speedup yields un-attributable, possibly-null accuracy; double-multiplied uncertainty.

**Conservative Linear Stochastic Depth (idea-03)**
- *Evidence/reasoning:* **4/10** — paper regime (110–1202 layers) transfers poorly to 9 blocks; sits in the documented identity-residual / transition-fragility failure families.
- *Impact:* **5/10** — plausible regularization angle but thin ceiling (barely at gate) against the highest strong-phase-underfit risk.

### Pick: **Conv2d-Weight-Only Gradient Centralization (idea-01)**

It is the only candidate whose mechanism targets the diagnosed accuracy limiter (generalization) rather than routing through the goal's explicitly-unvalidated exposure→accuracy link, and it does so while preserving the entire accepted forward function, ordinary SGD state, data curriculum, and schedule — giving it by far the cleanest attribution. Its residual danger (optimizer-path collapse) is the same family that killed EXP-020/022/028, but this instance is genuinely narrower and is guarded by mandatory immutable-corpus concentration/update-spike gates rather than by the (insufficient) instantaneous norm argument. It carries the strongest evidence base and the most credible mechanism-to-limiter fit; the two must-fix caveats are (1) treat the `P(g)+λw`-vs-`P(g+λw)` gap and (2) the BN-redundancy of the removed mean as first-class interpretation reads, so a small or null result is diagnosed honestly rather than rescued. Channels-last and stochastic depth both bet on exposure the diagnosis says is unproven, and stochastic depth additionally re-enters a failed perturbation family — so neither outranks GC on merit.
