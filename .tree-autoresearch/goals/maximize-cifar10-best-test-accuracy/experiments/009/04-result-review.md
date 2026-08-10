# Claude Adversarial Result Review: EXP-009

## Stop Decision

EXP-009 stopped at a preregistered preflight gate before training. Candidate/parent median latency had to be at most 1.075 and projected steps at least 26,000; measured values were 1.20715 and about 23,154. Both conditions independently failed. The package was rejected on cost, not quality.

## Integrity

- No `run.log`, evaluation, or accuracy artifact exists.
- `train.py` is the only tracked change.
- Live-gate and implementation-integrity checks passed.
- The latency/exposure gate was written before implementation and benchmarking, so the stop is not post-hoc.

## Safe Learning

- The fixed four-block FP32 SE package adds about 20.7% median step latency, roughly 2.8 times the allowed 7.5% overhead.
- That cost projects about 23.2k steps, materially below the 26k exposure floor.
- Nothing was learned about whether SE helps or hurts CIFAR-10 accuracy; the hypothesis remains untested.

## Verdict

Claude characterized this as a preflight reject rather than a genuine crash. The tree schema has no preflight-reject category and explicitly maps no produced result to `crash/NaN`; the node uses that mechanical encoding while the report preserves the distinction.

## Next Direction

A future SE-family design would need less placement, a narrower bottleneck, or a fused/cheaper gate path, then must pass the unchanged parent-relative cost gate before accuracy. Alternatively, move to a representation mechanism without multiple FP32 launch points.
