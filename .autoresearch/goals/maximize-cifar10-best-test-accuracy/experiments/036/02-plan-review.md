# Adversarial Plan Review — EXP-036

## Prioritized Concerns

1. EXP010's 89.73/93.16/0.1934 anchors are written too compactly and could be mistaken for EXP036 outputs; every observed value must be parsed from `run.log` before comparison.
2. The 94.27 point prediction is only two images above the formal gate; a one-seed max-over-checkpoints pass/miss is noise-limited regardless of extensive preflight.
3. Loader wait occurs before `t0`, so reflection cannot directly reduce counted optimizer exposure. The binding feasibility risk is total wall time reaching 600 seconds, not the 99% step floor.
4. A single candidate-only concentration step can be a benign data-shift transient; accepted controls do not see reflected inputs. The data-only safety rule needs persistence/repetition, not optimizer-path strictness.
5. A parent-process paired corpus is a bounded proxy and cannot reproduce the exact eight-worker production stream; a clean pass must not be overstated.
6. The proposed 31-minute maximum preflight is disproportionate to a two-keyword, low-margin candidate; reduce corpus and loader trials while preserving the decisive semantics/wall checks.

## Resolution

- Clarified that EXP010 values are comparison anchors only and all EXP036 metrics are parsed from the completed log.
- Made the single-seed/noise limitation explicit in the verdict language; no confirmation reroll is allowed.
- Removed the actual step floor as a validity condition and made loader demand margin plus projected/actual total wall the binding systems gates; steps remain informational consistency evidence.
- Defined candidate-only concentration failure prospectively as two consecutive or at least three total >95% steps, while retaining finite-state and whole-model excursion bounds.
- Explicitly scoped the paired corpus as a semantics/safety proxy that cannot certify production worker trajectories.
- Reduced the paired corpus to 32 strong+16 weak batches and loader timing to three fresh pairs/two post-warmup epochs per pipeline.
