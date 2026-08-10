# Adversarial Review — EXP-003 Candidate Ideas

All three finalists respect the hard constraints (train.py-only, no new deps, seed 42 fixed, `Eval.evaluate()` untouched, ≤1 eval/epoch, single predeclared run). None is a fatal violation, and none is an unchanged retry of a failed approach (EXP-001's failure was schedule-side; all three preserve the EXP-002 schedule). So the decision is on merit, not disqualification.

## Prioritized concrete feedback

**1. Width test attacks capacity, but the diagnosis names generalization as the limiter — this is a mechanism/limiter mismatch (most important).**
`01-brainstorm.md:24` and `03-experiment-learnings.md` frame the remaining gap as "generalization under hard-label cross-entropy," with capacity called only "an orthogonal possibility." The width proposal (`idea-03.md:88-90`) itself concedes the likely failure mode is "better training loss but unchanged test accuracy" — i.e., exactly what you expect when generalization, not capacity, bounds the metric. So the highest-ceiling idea is aimed at the least-supported limiter, and has a real path to *regressing* best_test_acc (more capacity → lower train loss → worse overfit) on top of the throughput hit. *Refinement:* this idea would be much stronger paired with an isolated regularizer, but that violates its own de-bundling premise; as a standalone first test it is the weakest-justified of the three.

**2. Width's throughput projection is speculative and the schedule is not retuned for it.**
`idea-03.md:74-76` admits the compute-bound bound is ~17,200 steps / 44 epochs but *hopes* for 23k–30k steps by assuming the tiny model is launch-bound. That is an unverified assumption on which the entire viability rests. Even the optimistic case discards 25–40% of the ~38,629 validated updates, and LR 0.1 is kept fixed (`idea-03.md:63-64`) — correct for attribution, but it means a wider, under-optimized model on fewer updates. If it lands <60 epochs the run is "throughput-inconclusive" (`idea-03.md:151`) — a likely non-result, not an improvement. *Refinement:* measure step time in a short dry-run before committing a full 300s run; if >13 ms/step, drop to width 20.

**3. Mixup at α=0.2 over ~95 epochs risks the undertraining failure the proposal itself flags — the horizon is short for mixup.**
`idea-02.md:171-177` and the mixup literature: canonical CIFAR mixup gains come from ~200-epoch recipes. Here the fixed budget yields only ~37.5k–38.5k steps (`idea-02.md:158`). Mixup slows finite-horizon fitting precisely when best_test_acc typically peaks in the low-LR tail. *Refinement:* the single strongest change would be to disable mixup during the 20% refinement tail so the endpoint fits clean targets — the proposal defers this to a follow-up (`idea-02.md:107-111`), but it is the mechanism most likely to convert a sub-threshold tail into a passing one, and is worth folding in given best_test_acc is usually a tail checkpoint.

**4. Mixup keeps BN statistics fitted on interpolated images while eval uses clean images (`idea-02.md:181-183`).** Conservative α limits this but does not remove the train/eval distribution shift; combined with the tail-on policy above, it directly threatens the evaluated checkpoint. *Refinement:* same as #3 (mixup-off tail lets BN running stats re-settle on clean-like inputs).

**5. Label smoothing's supporting evidence is the weakest link in its own argument, and its risk is a top-1/calibration divergence.**
The "train loss 0.06 vs test loss 0.2843" gap (`idea-01.md:16`, `:76`) is explicitly non-comparable (augmented train batches vs clean test) — the proposal honestly says so (`idea-01.md:20`, `:35`). So the diagnosis is suggestive, not evidence. More importantly, the NeurIPS paper (`label-smoothing.md:7-8`) shows LS most reliably improves *calibration/NLL*, which the proposal flags as a genuine failure mode (`idea-01.md:91`): it can lower test loss without moving top-1. Under a metric that is strictly best_test_acc with a tight +0.10 threshold, that is the central risk. *Refinement:* none available within one run (can't sweep ε); but the near-zero cost means the downside is bounded — it will not perturb the validated 38,600-step horizon or the schedule, so a null result is cheap and clean.

**6. Minor — all three lean on a +0.10–0.47 range against a +0.10 threshold that sits inside single-run variance.** Every proposal correctly forbids reroll (good, per constraint `01-definition.md:18`), but the expected gains for LS especially cluster near the threshold. This is inherent to the goal, not a flaw to fix, but it means the *ceiling* matters: an idea whose median outcome is ~+0.1 is a coin-flip pass.

## Scored verdict

**Modest Label Smoothing (ε=0.05)**
- Evidence/reasoning — **8/10**: cleanest de-bundled test, preserves the validated schedule and step count exactly; weakened only by an admittedly non-comparable loss diagnosis and a real top-1-vs-calibration risk.
- Potential impact — **5/10**: modest ceiling (~+0.1 to +0.27), near-zero chance of large regression; upside capped because it adds no data or capacity.

**Conservative Mixup (α=0.2)**
- Evidence/reasoning — **6.5/10**: solid literature and a correct, sync-free implementation, but attacks the same target as LS with a materially higher undertraining/BN-shift risk over a short horizon, and the always-on tail policy works against the evaluated checkpoint.
- Potential impact — **6.5/10**: higher ceiling than LS (input-manifold regularization), but a wider outcome distribution with a real negative tail under 95 epochs.

**1.5×-Width ResNet-20 (24/48/96)**
- Evidence/reasoning — **4.5/10**: WRN evidence is only directional and from far longer recipes; the mechanism targets capacity while the local diagnosis names generalization, and the throughput projection is an unverified assumption.
- Potential impact — **6/10**: genuinely high ceiling *if* capacity were the limiter, but symmetric downside — self-admitted overfit mode plus 25–40% fewer updates make regression as plausable as improvement.

## Winner: **Modest Label Smoothing (ε=0.05)**

It is the only finalist that attacks the *diagnosed* limiter (hard-label generalization) with a mechanism that leaves the validated EXP-002 horizon, schedule, throughput, and BN statistics fully intact — so any delta is cleanly attributable and the downside is tightly bounded. Mixup shares its target but adds convergence, BN-shift, and tail-fit risks that the short 300 s / ~95-epoch budget amplifies; the width test aims at the least-supported limiter (capacity) and carries the largest regression risk from lost updates plus its own admitted overfitting mode. Label smoothing wins not because it is the safe choice but because it has the tightest evidence-to-risk alignment with the diagnosis; its lower ceiling is the one real cost.

If the team wants to trade certainty for ceiling, **Mixup with the refinement in feedback #3 (mixup-off during the 20% low-LR tail)** is the second pick and the more compelling exploratory bet than width — it raises the ceiling while removing mixup's biggest fixed-budget liability. Width should be deferred until a regularizer has been shown to help, since the diagnosis does not support capacity as the current bottleneck.
