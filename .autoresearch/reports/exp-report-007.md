# Report EXP-007: torch.compile (reduce-overhead) to buy more epochs
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md
- **Plan**: plans/plan-007.md
- **Log**: logs/exp-log-007.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s training budget on one H20, editing only `train.py`.
Higher is better. Baseline **96.00%** (EXP-003); success bar ≥ **96.10%** (baseline + 0.1pp).

## Idea & Hypothesis
Chosen idea: wrap the training forward with `torch.compile` to cut per-step kernel-launch overhead (the k=4 net
is launch-bound on H20, ~45% util), fitting more epochs of the *identical* 96.00% recipe into 300s. Selected
because the project's strongest established lever is "more epochs of the proven recipe" (EXP-003: 54→77 epochs →
+0.58pp) and compile leaves optimization dynamics untouched → cleanest possible attribution. Hypothesis: more
epochs (target ≳90 vs 77) lift `best_test_acc` past 96.10 (expected ~96.2–96.6%) at unchanged params/VRAM.

## Approach
Two edits to `train.py` only (no new dep — `torch.compile` is core torch 2.9.1): (1) added
`compiled_model = torch.compile(model, mode="reduce-overhead")` after the model is on device; (2) routed the
training forward through `compiled_model(inputs)` (inside the existing bf16 autocast). Eval left UNCHANGED on the
eager `model` (shared weights via `._orig_mod`) to avoid eval-time recompiles on the differing eval batch shapes.
Mode chosen by a planning-phase smoke test: default compile was net-negative (1.03× for ~13.6s compile cost);
`reduce-overhead` (CUDA graphs) gave 8.1ms vs 9.4ms eager. Compile cost charged honestly to the budget (first
in-loop step). Everything else byte-for-byte fixed (k=4, Cutout, PEAK_LR 0.2, WD 1e-4, label smoothing, batch
128, bf16, channels_last, Nesterov, cosine-to-0, seed 42).

## Execution
One run, no retries, no errors, no recompiles/graph-breaks. Clean startup (params 4,299,866 unchanged).
Steady-state **dt = 8ms/step, ~15,400 img/s** vs the prior ~10–11ms / ~11,600 — a ~31% full-loop throughput gain
(larger than the 1.16× isolated-forward smoke test, because the compiled path also trims the real loop's eager
overhead). Loss decreased normally; eval climbed to a stable plateau (95.81–95.92 over ep 84–89). Completed at
300.0s training / 402.9s total, **89 epochs / 34,523 steps**, peak VRAM 453.8 MB.

## Results
- **Primary metric**: **95.92%** (baseline 96.00%, delta **−0.08 pp**, −0.08%) — below the +0.1 bar and the
  baseline (within the ~0.2pp noise band).
- **Observations**: The *mechanism* fully succeeded — compile bought **+12 epochs (+28% steps)** net of its
  one-time cost, exactly the throughput lever intended. But the **extra epochs did not raise accuracy**:
  final_test_loss 0.2081 ≈ EXP-003's 0.204 (no overfitting, no gain), and the eval plateaued in the high-95.8s/
  low-95.9s from ~ep 80 on.
- **Analysis**: This decouples two things EXP-003 had conflated. EXP-003's +0.58pp came from fixing
  **undertraining** (54→77 epochs of a freshly-regularized model). EXP-007 shows that past ~77 epochs the k=4 +
  Cutout model is **converged** — 77→89 epochs adds gradient steps along the same cosine schedule but lands at
  the same ~96.0% ceiling. So "more epochs / more throughput" is now a **saturated** lever, joining width,
  regularization, and weight-averaging. The −0.08pp is noise, not a real regression. Net: four levers
  (capacity, regularization, weight-averaging, training-length) are all at their local ceiling for this
  architecture at this budget; 96.0% is a genuine plateau.
- **Key Learning**: `torch.compile(reduce-overhead)` cleanly buys ~+30% throughput / +12 epochs here, but the
  k=4+Cutout model is epoch-saturated past ~77 — extra training no longer helps. The freed throughput is only
  valuable if spent on a *different* axis (e.g. more capacity, which compile now makes affordable).

## Verification
- **Conditions**: Cond 1 (clean completion in budget) PASS; Cond 2 (best_test_acc ≥ 96.10) **FAIL** (95.92);
  Cond 3 (no constraint violations) skipped — aborted after Cond 2 per protocol.
- **Review Notes**: Results trustworthy — clean single run, frozen eval, seed 42, exactly one eval/epoch,
  `num_params` unchanged (4,299,866) and compile is execution-only (no math/eval change) → no reward-hacking
  surface. Throughput gain independently visible in the per-step dt. The −0.08pp is within noise.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, trustworthy run; primary metric below the +0.1 bar (verification condition 2 failed).

## Unexplored Avenues
- **Compile + more capacity (the real payoff)**: compile's ~30% speedup makes previously-too-slow models
  affordable. EXP-004 found k=6 compute-bound at ~22ms/step → only 35 epochs (underfit). With reduce-overhead
  that could drop toward ~15ms → potentially ~50+ epochs, possibly enough to stop underfitting. **k=5 (or
  compiled k=6)** is now the most promising untried direction — it re-opens the *capacity* axis that the 300s
  budget had closed. This is the strongest lead.
- **Compile + larger batch + LR scaling**: stack the two throughput levers (compile 8ms + batch-256 utilization)
  for even more steps — but epochs are saturated, so only worthwhile in service of capacity, not alone.
- **A fundamentally different architecture at k=4** (SE, better stem/activation, deeper-thinner): changes the
  accuracy-per-epoch ceiling rather than the epoch count. SE's launch-kernel cost is now partly hidden by compile
  (CUDA graphs), reducing the throughput risk that deprioritized it earlier.

## Next Steps
1. **Compiled higher-capacity model (k=5, or k=6 with compile)** — *medium-high confidence*; compile re-opens the
   capacity axis (EXP-004's k=6 was compute-bound at 22ms; compile should cut that materially). Capacity was the
   single biggest historical lever (+2.84pp at k=4); more capacity that now *fits enough epochs* is the best shot.
2. **Compiled k=4 + SE blocks** — *medium confidence*; a different accuracy-per-epoch axis, with compile/CUDA-
   graphs absorbing much of SE's extra-kernel throughput cost that made it risky before.
3. **Accept 96.0% as a strong plateau and pursue accuracy-per-epoch (architecture) over more-training** —
   *strategic note*; four levers (width≤k4, regularization, weight-averaging, training-length) are saturated.

## Exit Action Results
- None defined for this goal — skipped.
