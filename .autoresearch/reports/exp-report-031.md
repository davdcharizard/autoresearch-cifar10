# Report EXP-031: Throughput-optimized Gradient Centralization (compiled + hoisted)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-031.md
- **Plan**: plans/plan-031.md
- **Log**: logs/exp-log-031.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) by editing only `train.py` within the fixed 300s training-compute budget on a single H20. Baseline = **96.22** (EXP-012, commit 6c417a4); success bar = **96.32** (baseline + 0.1pp). This experiment tested whether removing EXP-030's throughput confound (a fair, throughput-neutral run of Gradient Centralization) lifts accuracy to/past the bar.

## Idea & Hypothesis
**Chosen idea**: Re-run EXP-030's Gradient Centralization (centralize every `ndim>1` weight grad to per-output-unit zero-mean over fan-in, between backward and step) with the execution optimized: a `torch.compile`d out-of-place centralization over a pre-hoisted list of the 23 conv/fc weight grads, to eliminate the ~1ms/step eager host overhead that cost EXP-030 ~3 epochs.

**Why selected**: EXP-030 was the project's strongest near-miss — GC TIED baseline (96.21) AND improved loss (0.1934<0.195) DESPITE a 3-epoch handicap (88 vs 91 ep) caused entirely by the eager per-param GC loop's host overhead. The GC math is FLOP-trivial; the lost epochs were a pure execution artifact. `torch.compile` (proven +30% on this launch-bound net, EXP-007) fuses the 23 mean+subtract pairs into one compiled call. Removing the confound was the most promising remaining shot at the bar.

**Hypothesis**: the compile+hoist fix restores ~91 epochs / dt~8ms and lifts `best_test_acc` above EXP-030's 96.21 — toward or past 96.32. Falsifiable two ways: (1) epochs stay ~88 → fix didn't take; (2) epochs recover to ~91 but accuracy stays within ±0.2pp of 96.22 → GC's benefit is genuinely below the noise floor.

## Approach
Three edits to `train.py`, identical GC math to EXP-030, only the execution changed:
1. Module-level `_gradient_centralize(grads)` returns each weight grad centralized **out-of-place** (`g - g.mean(dim=fan_in, keepdim=True)`), wrapped as `_gc_compiled = torch.compile(_gradient_centralize)` (DEFAULT mode — `zero_grad(set_to_none=True)` reallocates grads each step, so reduce-overhead/CUDA-graph static addresses would be invalid).
2. Hoist `gc_params = [p for p in model.parameters() if p.ndim > 1]` once before the training loop (the 23 GC targets; skips iterating the 45 BN/bias params and rebuilding reduction dims each step).
3. Call site between `loss.backward()` and `optimizer.step()`: `centralized = _gc_compiled([p.grad for p in gc_params])` then reassign `p.grad = cg`. Out-of-place + reassign sidesteps the in-place-mutation-under-compile clone-writeback ambiguity (where Inductor could no-op GC by writing to a clone).
No config/param change. Smoke test verified: params 4,299,866; 23 targets / 45 excluded; max |per-output-unit mean| after GC = 7.6e-9; compiled matches eager to 2.98e-8; compiled GC ~17% faster than eager (30-iter micro-bench).

## Execution
An initial launch was killed externally by a user `/exit` after printing only the header (no training); relaunched cleanly. One full run, exit 0, no NaN/Traceback, 91 evals for 91 epochs, total_seconds 405.1 (<600). Trajectory tracked baseline throughout. No retries, no adjustments.

## Results
- **Primary metric**: best_test_acc **96.14%** (baseline 96.22, delta **−0.08pp**, −0.08%) — below baseline and below the 96.32 bar.
- **Observations**: **The throughput fix worked perfectly** — num_epochs **91** (= baseline; vs EXP-030's 88), dt **8ms steady** (641/703 sampled lines at 8ms). num_steps 35161, peak_vram 469.8 MB. final_test_loss **0.1894 — BETTER than baseline 0.195 AND better than EXP-030's 0.1934**.
- **Analysis**: This is the clean, throughput-neutral fair test of GC that EXP-030 could not deliver, and it is **decisive**. At full throughput GC lands at 96.14 (within the ±0.2pp noise floor of baseline 96.22 and of EXP-030's 96.21) while improving loss. The hypothesis's mechanism (compile restores epochs) was CONFIRMED; the hoped-for accuracy gain was FALSIFIED via falsification-path (2). The 3 lost epochs in EXP-030 were NOT masking a top-1 gain — GC simply does not move top-1 on this net. GC consistently lowers test loss (0.1894, project-competitive) but leaves accuracy flat → it is another **convergence-POLISH lever**, joining the polish-vs-top1 cluster (EMA/SWA EXP-006/019/020, Bag-of-Tricks EXP-026, LS-down EXP-023): all improve loss/calibration/flatness, none move top-1 once the recipe is at its capacity ceiling. EXP-030's 96.21 was simply the noise-favorable tail of the same null. The optimizer/gradient-dynamics axis, entered via GC, yields the same outcome as the polish axis.
- **Key Learning**: A throughput-neutral fair test of GC (91 ep/8ms, fix confirmed) shows GC improves test LOSS (0.1894<0.195) but NOT top-1 (96.14, within noise) — GC is a convergence-polish lever, resolving the EXP-030 near-miss as a loss-only effect.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) **FAILED** (96.14 < 96.32, also below baseline). Cond 2 (clean completion <600s) and Cond 3 (no constraint violations: train.py-only, params 4,299,866 unchanged, ≤1 eval/epoch, no new deps, seed unchanged) both PASS (informational).
- **Review Notes**: Results trustworthy. best_test_acc parsed from the summary block; throughput-neutrality VERIFIED (91 ep / 8ms = baseline), so the comparison is clean and fair — this is the experiment's central success (the fix worked) even though the metric did not improve. No integrity concern: pure torch + torch.compile, train.py only.
- **Verdict**: **no-improvement**
- **Verdict Basis**: Cond 1 verification condition failed (96.14 < 96.32); all hard constraints satisfied → no-improvement, not invalid.

## Unexplored Avenues
- **Other optimizer/gradient-dynamics members** (the axis is now characterized via GC but not exhausted): AGC (adaptive gradient clipping, Brock 2021), gradient-noise injection, LARS/LAMB layer-wise normalization. CAUTION: GC just showed the axis behaves like the polish cluster (loss-not-top1) on this net — these are now LOWER confidence than before EXP-030/031, since the most-documented member (GC) gave a top-1 null. Each also carries the same per-param-loop throughput cost (would need the same compile/hoist treatment).
- **GC as a loss/calibration lever** (not for this top-1 goal): GC reliably produces project-competitive loss (0.1894); if the goal ever shifts to test-loss or ECE, GC is a validated free win. Out of scope for the current accuracy goal.
- **GC + a top-1-moving change** (compose, not solo): GC's landscape regularization might compound with a genuine capacity/generalization change, but no such change has been found in 31 experiments — speculative.

## Next Steps
1. **Close the GC sub-axis; treat the optimizer/gradient-dynamics axis as polish-like (HIGH)**: GC is the best-documented member and gave a clean top-1 null at a fair test — record GC as a polish lever and DE-prioritize the optimizer axis for top-1 (it behaves like EMA/SWA/Bag-of-Tricks). Only revisit for a loss/calibration goal.
2. **AGC as the one remaining well-documented optimizer-class probe (LOW-MEDIUM)**: if continuing to map the axis, AGC is the next member, but expectation is now a loss-not-top1 null given GC; budget it as an axis-closer, not a likely gain. Would need compile/hoist treatment to stay throughput-neutral.
3. **Accept the 96.22 plateau as the firm k=4/300s ceiling (HIGH confidence)**: with ~23 axes closed (scalar knobs, aug family, regularizer-adding, compute-adding/epoch-wall, weight-averaging/polish, downsampling, activation, LR-schedule, and now optimizer/gradient-dynamics via GC), the recipe is generalization-bound at fixed capacity. Per NEVER-STOP, continue probing — but expect axis-closing nulls; the highest-value remaining moves are genuinely novel mechanisms, not more members of mapped axes. Reference project-insights polish-vs-top1 (now +GC) and the epoch-wall insights.

## Exit Action Results
<!-- No exit actions defined in the goal file. -->
- N/A — goal file defines no exit actions.
