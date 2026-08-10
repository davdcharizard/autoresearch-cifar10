# Report EXP-030: Raise the Weak-Tail Start LR to 0.02
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the moving baseline of 94.15% at `7c1e7d8`. A valid improvement required at least 94.25% under the fixed seed-42, one-H20, 300-counted-second, `train.py`-only protocol.

## Idea & Hypothesis

Raise only `ANNEAL_START_LR` from 0.01 to 0.02 while preserving the complete accepted 80% LR-0.1 N1/M7 plus p=0.5 alpha-1 CutMix phase and the same `1e-4` cosine endpoint. The idea won adversarial review because it probed an untuned zero-overhead refinement amplitude without moving any protected data/target boundary or evaluator checkpoint. The hypothesis predicted that nearly twice the integrated weak-tail LR would accelerate useful hard-label adaptation and reach at least 94.25%; overshoot and increased effective coupled-decay displacement were preregistered risks.

## Approach

Changed exactly one tracked line in `train.py`: `ANNEAL_START_LR = 0.01 -> 0.02`. Compile, Ruff, formatting, pre-commit, whitespace, scope, and AST/source-consumer checks passed. A no-test CPU controller verified that the accepted and candidate schedules were identical at LR 0.1 through 80%, candidate LR began at 0.02 strictly above the boundary, the tail was monotone, and both ended at `1e-4`. Independently seeded model/RNG hashes matched, parameter count was 1,073,962, momentum buffers remained equal, and a copied-state first weak update measured exactly 2.000000 aggregate displacement. The latter was treated only as arithmetic/scope sanity, not accuracy evidence.

## Execution

One fixed-seed run executed on the sole idle 97,871-MiB H20 and exited zero without retry. The 80.0% transition occurred once, stopped all eight workers, and reported 10,652 CutMix batches among 21,408 strong batches (49.757%). The first weak steps used the registered 0.0200 LR. The run completed 26,758 updates in 300.0 counted and 331.8 total seconds, with 19 unique evaluation epochs, 598.7 MiB peak VRAM, and a complete summary. No code, infrastructure, lifecycle, target, or numeric failure occurred.

## Results

- **Primary metric**: 93.90% (baseline: 94.15%, delta: -0.25 percentage points, -0.27% relative)
- **Observations**: The unchanged-program strong phase was noisy: 83.26%, 86.89%, 86.37%, and 81.38% at the four early checkpoints, then 88.56% at the switch, 1.17 points below EXP010's historical 89.73%. With no corroborating scope or environment anomaly, this is trajectory context rather than an integrity failure. The first weak checkpoint recovered to 93.08%, only 0.08 below EXP010's 93.16%. The candidate then reached 93.87% at epoch 62, peaked at 93.90% at epoch 66, and regressed to 93.79% final. Final NLL was 0.2083 versus EXP010's 0.1934. Its final printed weak train-loss EMA was 0.0264 versus EXP010's 0.0459, while exposure retained 99.48% of EXP010.
- **Analysis**: The intended local effect—larger weak-tail optimization motion—executed, but it did not improve generalization. The near-matched first weak checkpoint shows that LR 0.02 rapidly converted the weaker switch state, yet the later trajectory remained below EXP010, produced worse NLL, and developed a 0.11-point best-final regression. Much lower weak training loss alongside worse test loss/top-1 is consistent with over-refinement or overshoot: the extra tail motion fit the weak hard-label objective more aggressively without improving invariant test performance. Because the single-scalar instruction graph, exposure, evaluator count, CutMix rate, lifecycle, memory, and timing all matched the accepted regime, the miss is not explained by systems overhead. Historical-run CUDA nondeterminism prevents attributing the full 0.25-point delta precisely, but the exact 0.02 operating point provides no positive evidence and is rejected without adjacent-LR tuning.
- **Key Learning**: Doubling the weak-tail start LR lowered training loss but worsened NLL and peak accuracy; the accepted 0.01 quench better preserves refinement generalization.

## Verification

- **Conditions**: Completion, numeric summary, fixed timing, hardware, scope, parameter count, phase/target provenance, lifecycle, CutMix fraction, exposure, and evaluator cadence passed. Primary accuracy failed: 93.90% <94.25%.
- **Review Notes**: Results are trustworthy as a valid single-run protocol result. Exactly one idle H20 and one seed-42 completion were used; only the reviewed scalar changed; the process exited zero; counted/total time was 300.0/331.8 seconds; all 19 evaluator epochs were unique; and no reroll occurred. As preregistered, the benchmark does not force deterministic CUDA, so the effect size is not a reproducibility claim.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid run finished 0.25 points below the moving baseline and 0.35 below the required gate.

## Unexplored Avenues

- A lower tail start such as 0.005 would test stronger quenching, but the accepted 0.01 already finishes at its best with better NLL, so scalar interpolation now has weak expected value and risks parameter chasing.
- A schedule that retains 0.01 but changes only the tail curve shape could redistribute motion without doubling the boundary shock. It remains distinct, but requires a mechanism stronger than post hoc repair of this miss.
- Tail regularization rather than LR amplitude could reduce the observed weak-objective overfit, but any added per-step operator must first satisfy the strict exposure gate established by EXP029.

## Next Steps

- **High confidence**: restore and preserve `ANNEAL_START_LR=0.01`; retire the exact 0.02 point and avoid adjacent scalar tuning without new evidence.
- **Medium confidence**: prioritize a zero-overhead mechanism that changes useful representation or data geometry while preserving the full accepted curriculum, rather than increasing terminal fit pressure.
- **Medium-low confidence**: revisit a pure systems lever only if it demonstrates meaningful convolution/backward speedup with evaluation opportunities held comparable; exposure-to-accuracy remains unproven.

## Exit Action Results

- None defined.
