# Report EXP-018: Late Arithmetic SWA with In-Budget BN Recalibration
- **Created**: 2026-08-06

## Goal

Raise CIFAR-10 `best_test_acc` above the 94.15% frontier at `7c1e7d8` while modifying only `train.py` under the fixed seed-42, one-H20, 300-second training and evaluator protocol. Formal improvement required at least 94.25%, and this experiment additionally required the final SWA model itself to clear that bar and the pre-install online best.

## Idea & Hypothesis

Uniformly average weak-tail parameter states from completed epoch endpoints in `[86%,98%)`, install the mean once, and spend the final 2% of the same counted budget cumulatively recalibrating BatchNorm on hard weak data. The hypothesis predicted at least seven nondegenerate snapshots, at least 97% of accepted optimizer exposure, final SWA accuracy of at least 94.25%, and improved late calibration versus EXP-010's 0.1934 NLL.

## Approach

`train.py` gained importable helpers for detached FP32 arithmetic averaging, charged synchronized snapshot/install transactions, ordered parameter installation, and cumulative BN refresh with momentum restoration. Snapshotting began at 86%, SGD stopped at 98%, and the active persistent weak loader supplied no-grad refresh batches until the 300-second counter expired. The terminal SWA evaluation replaced the terminal online evaluation; one provenance line recorded install step, online best, snapshot spread/cost, refresh counters, and final SWA metrics. Mandatory external Claude idea, plan, revised-plan, implementation, and corrected-source reviews all succeeded without fallback.

## Execution

Arithmetic, real refresh, and five-fresh-process timing gates passed. The arithmetic mean matched FP64 reference within `1.19e-7`; 780 real refresh batches proved iterator recreation, cumulative counters, momentum restoration, and state isolation. Timing projected eight snapshots, 26,412 steps, 18 evaluations, and 325.0 seconds total. One fixed-seed production run exited zero without retry in 332.4 seconds total.

## Results

- **Primary metric**: 94.02% (baseline: 94.15%, delta: -0.13 points, -0.14% relative)
- **Observations**: The online path was already weaker at the 80% switch (88.09% versus 89.73%), although first weak recovered to 93.21% versus 93.16%. Online best reached 94.02%. Eight snapshots covered 87.14-97.30% with median consecutive normalized RMS `7.26e-3` and first-last `2.67e-2`, ruling out a frozen/no-op average. After 1,624 cumulative BN batches, final SWA accuracy was 93.85%, 0.17 below its own online best, and NLL was 0.2037 versus EXP-010's 0.1934. Exposure remained healthy at 26,453 steps (98.35% of EXP-010).
- **Analysis**: Cross-run baseline comparison is partly exposed to ordinary fixed-time CUDA trajectory variance because the online path diverged before SWA began. The within-run comparison is nevertheless causal and decisive for the intervention: the averaged parameters had substantial spread, BN statistics were recomputed over more than four dataset passes, and the resulting SWA model was worse than the online checkpoint it replaced. Uniform averaging across this monotonically refining annealed tail pulls the solution toward earlier, less useful states rather than finding a better calibrated center. This rejects the exact 86-98% arithmetic window and cumulative weak-BN policy; it does not test EMA or textbook constant/cyclic-LR SWA.
- **Key Learning**: Late SWA had ample parameter spread but trailed its online checkpoint and worsened NLL; this annealed uniform window is harmful.

## Verification

- **Conditions**: primary accuracy and final-SWA gates failed; every process-integrity condition passed.
- **Review Notes**: Results are trustworthy. The run had 10 finite summary fields, one H20, only `train.py` tracked, 300.0 counted seconds, 18 unique evaluations, one 80% switch, 49.79% CutMix, eight stopped workers, eight valid snapshots, aligned BN counters, and `install_step == num_steps == 26,453`.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=94.02%` missed 94.25%, while final SWA `93.85%` also missed the threshold and trailed pre-SWA online best by 0.17; no hard constraint failed.

## Unexplored Avenues

- A short-timescale EMA could weight recent tail states more heavily and avoid the backward bias of a uniform mean, but its decay is a new tunable mechanism and cannot be inferred from this one window.
- Textbook SWA with constant or cyclic LR could create basin exploration rather than average a converging cosine path, but it changes the accepted schedule and must justify lost terminal refinement.
- Averaging only a narrower final subset may reduce lag, but the prior idea reviews found near-terminal states too correlated; shifting this exact window is not a justified retry.

## Next Steps

- **High confidence**: retire uniform endpoint averaging on the accepted cosine tail and preserve online terminal weights.
- **Medium confidence**: test isolated Nesterov as the remaining clean optimizer confound only if review accepts its modest evidence and 1.9x first-update semantics.
- **Medium confidence**: seek an identity-preserving mechanism that targets calibration/generalization without another transition or residual-suppression change.

