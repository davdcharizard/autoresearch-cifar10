# Report EXP-031: Channels_last + LR cooldown 1e-4
- **Created**: 2026-05-29

## Results
- **Primary metric**: 93.04% (baseline: 96.39%, delta: -3.35%)
- **Key observation**: Only 35 epochs — previous channels_last runs got 59-64. Likely GPU contention/thermal throttling, not a valid experiment result.
- **Verdict**: no-improvement (anomalous infrastructure issue)

## Exit Action Results
