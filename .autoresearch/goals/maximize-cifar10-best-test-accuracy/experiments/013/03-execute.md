# EXP-013: Batch-256 Linear Scaling

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-013
- **Commit**: (pending — no commit for failed feasibility)
- **Outcome**: failed — paired throughput feasibility veto

## Implementation Notes

### Summary

Implemented the reviewed batch-256 candidate as four training constant changes plus the mandatory Claude-review measurement control that fixes evaluation to 19 elapsed-progress checkpoints. Functional checks proved model/RNG identity with EXP-010, finite hard and probability-target LR-0.2 steps, exact parameters, scaled LR boundaries, and 19 unique simulated evaluation epochs. The first H20 feasibility gate then failed, so no production loader audit or accuracy run was allowed.

### Surprises & Discoveries

The earlier serial probe projected a 28.44% image-exposure gain, but the more rigorous five fresh-process paired design measured only 18.91%. The result was highly stable (control/candidate CV 0.474%/0.197%), so the discrepancy reflects control timing methodology/state rather than transient H20 contention. Fresh paired measurement was necessary and worked as intended.

### Decisions

No threshold was relaxed. The plan explicitly required at least 20% more image exposure because batch 256 sacrifices roughly 40% of optimizer updates; 18.91% was insufficient even though p95 throughput and memory passed. The experiment stops before accuracy, with no batch-512, warmup, fusion, precision, LR, or evaluation fallback. Mandatory Claude idea and plan reviews both completed successfully; no fallback reviewer was used.

## Experimental Adjustments

- **Fixed 19 evaluation opportunities before timing**: Mandatory Claude plan review found that shorter batch-256 epochs would otherwise create extra chances to maximize `best_test_acc`; fixed elapsed checkpoints matched EXP-010's observation count. (ref: `02-plan-review.md`)
- **Stopped at paired feasibility**: Stable fresh pairs measured 1.18914x image throughput and 4.094M projected slots, below the pre-registered 1.20x / 4.131M gates. (ref: `00-paired-timing.md`)

## Run Log

### Run 1 — Feasibility Only

Metadata:
- **Job ID**: N/A (fresh local subprocess pairs)
- **Log file(s)**: `00-paired-timing.md`
- **WandB**: N/A
- **Status**: failed feasibility
- **Started**: 2026-08-06 05:29 UTC
- **Ended**: 2026-08-06 05:32 UTC

Description:
- Five alternating fresh-process H20 pairs tested the candidate's only claimed systems benefit before any accuracy run. Every arm used fresh model/optimizer state, 100 warmups, and 500 synchronized hard/soft steps. The candidate required at least 20% more image throughput and 4.131M ratio-projected slots to justify its optimizer-update loss.

Observations:
- Functional gate passed: identical model state/RNG, 1,073,962 parameters, finite LR-0.2 hard/soft steps, scaled LR endpoints, and 19 unique evaluation epochs. (source: `00-functional-check.py` execution output)
- Paired gate failed: step ratio 1.68189, image throughput 1.18914x, and 4,093,952 projected slots. Trial CV and memory passed, ruling out instability or OOM. (source: `00-paired-timing.md`)
- No full training output or primary metric exists because the reviewed abort criterion correctly prevented launch.

Key Metrics:
- control/candidate median mean step: 10.8437 / 18.2380 ms (source: `00-paired-timing.md`)
- image-throughput ratio: 1.18914x; gate 1.20x (source: `00-paired-timing.md`)
- projected candidate steps/image slots: 15,992 / 4,093,952 (source: `00-paired-timing.md`)
- control/candidate CV: 0.474% / 0.197%; peak allocation 598.7 / 1,120.2 MB (source: `00-paired-timing.md`)

## Verification Results

### Conditions Checked

- **Primary accuracy**: not run — candidate failed a mandatory pre-run mechanism gate.
- **Paired throughput**: fail — 1.18914x <1.20x and 4,093,952 <4,131,000 slots.
- **Scope/functionality**: pass before veto — only `train.py`, expected model identity, finite target paths, and fixed evaluator count.
- **Remaining loader, wall, lifecycle, and full-run conditions**: skipped after the paired feasibility failure.

### Informational Metrics

- Fresh paired timing values are recorded inline above; no accuracy or full-run summary metrics exist.

## Errors & Dead Ends

### 2026-08-06 — Batch-256 throughput below mechanism floor
- Error: `image_throughput_ratio=1.18914 < 1.20; projected_images=4,093,952 < 4,131,000`
- Root cause: batch-256 step cost is 1.68189x control, slightly beyond the throughput knee required to compensate for lost updates.
- Source: `00-paired-timing.md`
- Do NOT retry: do not relax the gate, rerun stable timing, or add a second performance mechanism to rescue EXP-013.

## Human Notes

> Autopilot execution; no intervention requested.
