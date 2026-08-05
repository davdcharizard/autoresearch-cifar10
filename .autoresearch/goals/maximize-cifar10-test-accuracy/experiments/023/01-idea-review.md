**Prioritized Feedback**

1. **A’s biggest flaw is cost optimism.** The EMA weights are maintained for free, but EMA logits are not free: a no-grad teacher forward is still GPU time. Given the under-anneal history in [03-experiment-learnings.md](.autoresearch/goals/maximize-cifar10-test-accuracy/03-experiment-learnings.md), all-step KD is dangerous. Make KD tail-only and require a smoke showing `num_steps >= 12610`; otherwise it repeats the SAM/capacity failure mode.

2. **A should fix the KL direction.** The brainstorm writes `KL(student || teacher)`, but standard KD should use detached teacher probabilities as the target: `KL(teacher || student)` / cross-entropy from teacher soft labels. Reverse KL is a real implementation-risk, not a detail.

3. **A is the only candidate with a genuinely new signal.** Static LS 0.2 was already tuned, and input augmentation is saturated across occlusion/mixing/transform. EMA KD adds input-dependent class-similarity targets, which the brainstorm’s Web Search section correctly distinguishes from label smoothing. But test it with reduced LS as a planned arm; LS0.2 + KD may over-soften.

4. **B is safe but probably small.** PolyLoss is throughput-free, so it avoids the project’s #1 failure mode. But it mostly reshapes CE confidence weighting on a clean, balanced dataset where LS0.2 already beat LS0.1 and scalar regularization tied in EXP-012. It is a good cheap probe, not the strongest bet to clear 96.48.

5. **B needs a precise LS interaction plan.** `p_t` is ambiguous under label smoothing. Decide whether PolyLoss uses hard-label `p_y` plus existing LS CE, replaces LS CE, or sweeps LS with epsilon. Otherwise a “PolyLoss win/loss” will be hard to interpret.

6. **C is the most redundant.** Weak-strong consistency is mostly another augmentation-invariance regularizer, while strong aug is already central and further aug mechanisms tied in EXP-011/015. With no unlabeled data, pseudo-targeting weak predictions can also fight true labels when the weak view is wrong.

7. **C’s cost story is still bad enough.** Depending on implementation it may be one extra no-grad forward rather than “two forwards beyond baseline,” but it still spends GPU step budget. If gated late enough to keep steps, its useful training exposure shrinks; if not gated, it risks the known under-anneal trap.

8. **Multiple-comparison reward hacking is the main constraint risk.** Sweeping `alpha/T/gate`, `epsilon`, or consistency weights against `best_test_acc` can easily pick a noise draw near the 0.1pp floor. Pre-register small grids, compare same-session controls, and confirm any apparent winner. Do not evaluate raw+EMA separately per epoch; the goal allows at most one validation per epoch.

**Scored Verdict**

- **A. EMA self-distillation**
  - Evidence/reasoning: **8/10**. Best aligned with the newly diagnosed loss/target bottleneck and supported by EMA/mean-teacher/KD rationale, but cost and KL-direction details must be tightened.
  - Potential impact: **8/10**. It has the only credible >0.1pp upside, provided tail gating preserves the anneal gate.

- **B. PolyLoss / loss reshaping**
  - Evidence/reasoning: **6/10**. Clean and cheap, but the mechanism overlaps partly with already-tested target/loss scalar tuning.
  - Potential impact: **5/10**. Could clear the bar if lucky, but likely sub-noise on this saturated LS0.2 + strong-aug recipe.

- **C. Weak-strong consistency**
  - Evidence/reasoning: **4/10**. The semi-supervised evidence does not transfer cleanly to fully supervised CIFAR with strong labels and already-saturated augmentation.
  - Potential impact: **4/10**. Costly enough to threaten annealing and redundant enough that the upside is weak.

**Pick: A, EMA self-distillation.** It is the highest-EV bet because it attacks the untouched learning-signal axis with a genuinely different target source, not just another scalar loss tweak or augmentation-consistency regularizer. Run it only as a constrained tail-gated KD experiment with correct forward-KL teacher targets, a `num_steps >= 12610` smoke gate, and a reduced-LS arm to avoid double-softening.
