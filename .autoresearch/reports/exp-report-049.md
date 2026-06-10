# Report EXP-049: Augmentation cooldown (EXP-034) + Gradient Centralization (EXP-031) combined
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-049.md
- **Plan**: plans/plan-049.md
- **Log**: logs/exp-log-049.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the fixed 300s budget on a single H20. Baseline = **96.22%** (EXP-012, 6c417a4); bar = baseline + 0.1 = **96.32%**. After 41 consecutive no-improvements with every single accuracy axis mapped and closed, this experiment executes the standing directive's "combine previous near-misses": it tests whether the two best throughput-neutral near-misses, applied together, add to clear the +0.1 bar.

## Idea & Hypothesis
Chosen idea: re-apply BOTH proven throughput-neutral near-misses unchanged and simultaneously — the EXP-034 augmentation cooldown (@0.10: disable TrivialAugment + Cutout for the final 10% of the budget so the low-LR tail fine-tunes on the clean test-aligned distribution; the only ≥baseline result at 96.26) and the EXP-031 compiled+hoisted Gradient Centralization (centralize the 23 conv/fc weight grads to per-output-unit zero-mean each step; the best loss-improver at loss 0.1894, top-1 96.14). Reasoning: the levers act on orthogonal axes (input-distribution alignment × gradient-space regularization). **Synergy hypothesis**: GC converges the model to a lower-loss / better-conditioned state whose top-1 advantage is *masked* by the aug-train↔clean-test mismatch; the clean cooldown tail removes that mismatch exactly when GC's better-conditioned weights can fine-tune to clean-distribution boundaries — letting GC's confirmed loss advantage finally surface as top-1. Hypothesis: throughput-neutral (dt~8ms, ~91 ep), and IF the two sub-noise levers add, `best_test_acc ≥ 96.32`; falsified if it lands within ±0.25pp of 96.26 without clearing the bar.

## Approach
Two independent, individually-proven edit sets to `train.py`, with no code interaction:
- **Gradient Centralization (EXP-031, 3 edits)**: module-level `_gradient_centralize(grads)` (out-of-place per-output-unit mean-subtract over fan-in dims) wrapped as `_gc_compiled = torch.compile(...)` in DEFAULT mode (zero_grad reallocates grads each step → CUDA-graph static addresses invalid; separate from the model's reduce-overhead compile); hoisted `gc_params = [p for p in model.parameters() if p.ndim > 1]` (23 targets) once; call site between `loss.backward()` and `optimizer.step()` computing centralized grads and reassigning `p.grad`.
- **Augmentation cooldown (EXP-034, 4 edits)**: `COOLDOWN_FRAC = 0.10`; `train_tf_clean` (full pipeline minus TrivialAugment); `aug_cooled` flag with an epoch-boundary `train_set.transform` swap once `total_training_time/TIME_BUDGET_S ≥ 0.90` (with a marker print); Cutout gated behind `if not aug_cooled`.
Tail LR left untouched (frozen near-zero, NOT reheated — per the EXP-035 caution that reheating regresses). GC runs every step including the clean cooldown phase. Smoke test confirmed: params 4,299,866 (unchanged), 23 GC targets, COOLDOWN_FRAC 0.1, GC per-output mean ≈ 3e-8 (zeroed), clean transform free of TrivialAugment, diff = `train.py` only.

## Execution
One clean run on idle GPU 1, 402.3s wall, exit 0, no retries, no NaN/traceback. dt steady 8ms (647×8ms, 55×9ms) — throughput-neutral, NO CUDA-graph break (the GC grad reassignment did not perturb the model's reduce-overhead forward graph, as EXP-031 established). The cooldown marker fired once at `>>> aug cooldown ON at ep 82 frac 0.90`. 91 epochs (= baseline ~91, no epoch confound). Early convergence was healthy (ep1 54.58%, ep2 66.64% — if anything faster than baseline's ~45.7% ep1, consistent with GC's known early-training benefit).

## Results
- **Primary metric**: best_test_acc **96.13%** (baseline 96.22, delta **−0.09pp**, −0.09%) @ ep86 — below the 96.32 bar by 0.19pp. final_test_acc 96.07% @ ep91; final_test_loss 0.1983.
- **Observations**:
  - **Throughput-neutral, clean fair test**: dt 8ms, 91 ep = baseline — no epoch or throughput confound. This is an uncontaminated test of the combination.
  - **The cooldown mechanism worked**: pre-cooldown base 95.77 (ep80) → post-cooldown peak 96.13 (ep86), a healthy +0.36 climb over ~6 clean epochs (comparable to EXP-034's tail climb).
  - **But the combination REGRESSED vs both components alone**: 96.13 < baseline 96.22 (−0.09) AND < EXP-034 cooldown-alone 96.26 (−0.13).
  - **GC's loss benefit washed out**: final_test_loss 0.1983 ≈ baseline 0.195, NOT EXP-031's standalone 0.1894. And the pre-cooldown base (95.77 @ep80) was LOWER than EXP-034's (96.05 @ep83) — GC did not raise the augmented-phase base.
- **Analysis**: The synergy hypothesis is answered cleanly NEGATIVELY. The two sub-noise levers did not add — they slightly anti-combined. The mechanism: GC's standalone loss improvement (0.1894) did NOT persist in the presence of the cooldown (loss 0.1983 ≈ baseline), so there was no better-conditioned low-loss state for the clean tail to "cash in"; and GC failed to lift the pre-cooldown base, so the (healthy) cooldown climb started from a lower point and only reached 96.13. This is consistent with the firmly-established **polish-vs-top1 wall**: GC is a loss-polish lever whose effect is real only on loss and only in isolation; layering it under the cooldown removed even that, leaving a result governed by run-to-run base jitter (here a low draw). The "combine near-misses" route — combining two individually sub-noise levers — does not break the plateau; it lands within the same ±0.25pp noise band, on the low side this run.
- **Key Learning**: Combining the two best near-misses (EXP-034 cooldown + EXP-031 GC) regressed to 96.13 (below both alone) — GC's loss benefit washed out and it didn't raise the pre-cooldown base; two individually sub-noise levers do not add.

## Verification
- **Conditions**: Condition 1 (`best_test_acc ≥ 96.32`) FAILED at 96.13; conditions 2 (clean run within budget — 402.3s < 600, params unchanged, no crash) and 3 (no hard-constraint violations — `train.py` only, eval untouched, once/epoch, no new deps, seed 42, deterministic, cooldown fired once) both PASSED.
- **Review Notes**: Results trustworthy and notably clean — throughput-neutral (8ms) at matched epochs (91 = baseline), cooldown verified to fire once at frac 0.90, GC verified to zero per-output-unit grad means. No epoch/throughput/scope confound; no integrity concerns. The −0.09pp is a genuine matched-budget combination result.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid clean throughput-neutral run; primary necessary condition failed (−0.09pp vs baseline, −0.19pp vs bar).

## Unexplored Avenues
- **Cooldown + a loss-improver that DOES persist under the cooldown**: GC's loss gain vanished in combination; a different polish lever (e.g. PolyLoss EXP-041, which improved loss/calibration via the loss function itself rather than the gradient) might persist through the clean tail since it is not a gradient-space op. But this re-treads two closed axes again and is also near-certainly sub-noise — low value.
- **Cooldown + LS-off in the clean tail** (brainstorm-049 candidate 2): a same-mechanism-reinforcing pairing (clean data + hard targets), never tried; but carries the EXP-035 tail-sensitivity regression risk and LS-down was null full-run. Low-medium value.
- The "combine two sub-noise near-misses" strategy is now evidenced as a plateau itself (this experiment): the components don't add, and base jitter dominates. Further cooldown combinations are low value.

## Next Steps
- **CORRECTION — pre-activation is NOT a fresh axis**: brainstorm-049's candidate 3 (pre-activation BN→ReLU→Conv blocks) was already tried and found null in **EXP-015** ("Pre-activation (true-WRN) block ordering … no gain"). Do NOT re-run it. (This catch is why goal-learnings must be re-read each loop.)
- **Combine-near-misses route is now closed too** (this experiment): two sub-noise levers don't add. The only directive-endorsed remaining branch is a genuinely NEW mechanism, not recombination or known structural variants.
- **Dig for an untried mechanism** (low confidence of gain, but the NEVER-STOP mandate): candidates not yet tested individually — a learnable activation (PReLU, near-free negative slope), or a stem/early-downsample redesign. Each is a single fresh sub-lever, low-confidence but information-positive. The next brainstorm should enumerate what mechanism class has genuinely never been touched, cross-checked against goal-learnings to avoid re-treads (as pre-activation would have been).
- **Document-the-ceiling confirmation run** (high information, ~zero gain): the honest fallback — a clean baseline replication characterizing the ±0.25pp band — remains available if the next brainstorm finds no genuinely-new mechanism. Continue per NEVER STOP rather than stopping.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
