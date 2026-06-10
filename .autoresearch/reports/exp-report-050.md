# Report EXP-050: Smaller batch size (128→64) for SGD gradient-noise regularization
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-050.md
- **Plan**: plans/plan-050.md
- **Log**: logs/exp-log-050.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the fixed 300s budget on a single H20. Baseline = **96.22%** (EXP-012, 6c417a4); bar = baseline + 0.1 = **96.32%**. With every previously-tried axis closed (42 no-improvements through EXP-049), this experiment probed the one genuinely-untested axis: batch size DOWNWARD.

## Idea & Hypothesis
Chosen idea: halve `BATCH_SIZE` 128→64, single-variable (LR 0.2, warmup 0.05, schedule, optimizer, augmentation, seed all unchanged). At fixed LR, halving the batch doubles the relative SGD gradient noise (∝ LR/√B) while keeping the mean update magnitude constant — the canonical Keskar (2017) "small batch → flatter minima → better generalization" regime. Reasoning: this is the only remaining axis with a distinct mechanism NOT in any closed family, and EXP-025's compute-bound finding predicted batch-64 would ADD updates (inverse of the batch-256 update-collapse) at similar total images, with epoch-saturation (EXP-007/045/046) bounding underfit risk. Hypothesis: dt ~halves to ≈4.5-6ms → ~2× updates / ~70-85 epochs; IF small-batch noise finds a better-generalizing minimum, best_test_acc ≥ 96.32; else within ±0.25pp (closing the downward batch axis).

## Approach
A single-line change to `train.py`: `BATCH_SIZE = 128` → `64`. LR/warmup deliberately NOT linear-scaled (the goal is more noise at the same mean step — the flat-minima test — which requires holding LR fixed; linear scaling would preserve dynamics and defeat the test). Smoke test: AST clean, `git diff` = the one `BATCH_SIZE` line, 781 batches/epoch (50000//64, drop_last=True), params unchanged 4,299,866.

## Execution
One clean run on idle GPU 1, 378.3s wall, exit 0, no retries, no NaN/divergence. Stable throughout despite the higher relative noise (warmup + BN + label smoothing held the early steps).

## Results
- **Primary metric**: best_test_acc **95.36%** (baseline 96.22, delta **−0.86pp**, −0.89%) @ ep66 (= final epoch) — 0.96pp below the 96.32 bar.
- **Observations**:
  - **dt only PARTIALLY halved (5-6ms, mostly 6ms) — a launch-overhead floor**, not the ~4ms full compute-scaling the EXP-025-based premise predicted. img/s fell ~26% (≈11,500 vs ~15,600) → total images dropped → only **66 epochs** (vs baseline 91). Updates rose to 51,121 (~1.4× baseline ~35.5k, short of the hoped 2×).
  - **Slower early convergence** from the 2× relative gradient noise at fixed LR 0.2: ep1 28.20% vs baseline ~45.7%.
  - **final_test_loss 0.2138 > baseline 0.195** (worse on loss), and test_acc was still monotonically CLIMBING at the final epoch (ep62 95.32 → ep66 95.36) — the model is **under-resolved / under-trained**, not overfit and not benefiting from a flatter minimum.
  - peak_vram 342.5 MB (lower than baseline ~454, as expected for the smaller batch).
- **Analysis**: The hypothesis is answered NEGATIVELY, and the mechanism is clear: the Keskar flat-minima benefit did not materialize because two costs dominated. (1) The net has a per-step launch-overhead FLOOR (~6ms) below batch 128, so halving the batch did NOT halve dt — img/s dropped, total images fell, and epochs collapsed to 66. (2) The 2× relative gradient noise at unchanged LR 0.2 slowed convergence (ep1 28% vs 46%). Together these left the model under-trained (loss 0.2138 > 0.195, still climbing at the final epoch). This is the MIRROR IMAGE of EXP-025 (batch 256 → compute-bound update-collapse → −2.38pp): there the larger batch cut updates; here the smaller batch cuts images-per-budget AND slows convergence. Both directions away from 128 regress. The result resolves the Keskar-vs-Smith tension for this net in the negative (no flat-minima gain observed) — though strictly the test is confounded by the epoch drop (epochs were not held equal), the practical conclusion stands: batch 64 does not help under this compute-gated budget.
- **Key Learning**: Halving the batch to 64 at fixed LR regressed −0.86pp (95.36): a launch-overhead floor capped dt at 6ms → only 66 epochs, and 2× gradient noise slowed convergence → under-resolved (loss 0.214); the batch-size axis is now closed in both directions, with 128 optimal.

## Verification
- **Conditions**: Condition 1 (`best_test_acc ≥ 96.32`) FAILED at 95.36; conditions 2 (clean run within budget — 378.3s < 600, params unchanged, no crash) and 3 (no hard-constraint violations — `train.py` 1-line diff, eval untouched, once/epoch, no new deps, seed 42, no seed hacking) both PASSED.
- **Review Notes**: Results trustworthy — clean uncontended run, single-line diff, deterministic. The only caveat is that this is not a perfectly isolated "gradient-noise" test (epochs dropped from 91→66 due to the launch floor), but that confound is itself part of the finding: smaller batch is throughput-penalized below 128 on this net. No integrity concerns.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid clean run; primary necessary condition failed (−0.86pp vs baseline, −0.96pp vs bar).

## Unexplored Avenues
- **Batch 96 (a milder reduction)**: would lose fewer images/epochs than 64 while adding some noise — but the launch floor still applies (dt won't drop proportionally) and the mechanism showed no upside even at 64, so near-certainly a smaller regression, not a gain. Not worth a loop.
- **Batch 64 WITH more epochs by cutting other compute**: there is no spare compute to cut (the recipe is already minimal); infeasible within budget.
- The batch-size axis is best considered CLOSED in both directions (128 is the throughput/quality optimum: launch-bound floor below it, compute-bound update-collapse above it). Gradient-noise regularization via batch size is not a lever here.

## Next Steps
- **The batch-size axis is now closed both directions** (EXP-025 up, EXP-050 down); 128 is optimal. Cross off the last "untouched axis" I had identified.
- **Genuinely-new mechanism search is nearly exhausted** (high-confidence assessment): after 43 no-improvements, the remaining untried single levers are all in or adjacent to closed families — LayerScale (brainstorm-050 cand 2, but EXP-026 residual-scaling was null), PReLU (cand 3, but activations closed EXP-010/028). Low confidence either helps; both are clean throughput-neutral tests worth running once each under NEVER STOP before declaring the mechanism space empty.
- **Document-the-ceiling confirmation run** (high information, ~zero expected gain): a clean baseline replication characterizing the ±0.25pp band remains the honest fallback once the last fresh sub-levers are tried. Continue per NEVER STOP — keep running the best-available fresh experiment each loop rather than stopping.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
