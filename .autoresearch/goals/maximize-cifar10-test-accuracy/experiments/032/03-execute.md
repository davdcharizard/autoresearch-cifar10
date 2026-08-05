# EXP-032: Reflection-Padded Random Crops

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-032
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: Failed pre-score qualification - loader stability gate

## Implementation Notes

### Summary

Changed the existing four-pixel `RandomCrop` padding mode from constant zero to reflection and nothing else in production. The ignored verifier separately proves pre-RandAugment pixel confinement with a NumPy oracle, explicitly replays crop/flip/RandAugment decisions, traces those decisions per sample in forkserver workers, and measures real-loader wall stability.

### Surprises & Discoveries

Reflection remains active after RandAugment disables, so accepted/candidate hard-tail images are intentionally not pixel-identical. Plan review also established that main-process sampler RNG and target order cannot stand in for worker-local crop, flip, and private RandAugment decisions.

### Decisions

The pixel oracle stops before RandAugment and flips its padding mask with the image. Worker traces decode RandAugment op/sign independently from the private pre-state. Loader latency is used only for wall feasibility because iterator waiting occurs before the scored `t0`; counted exposure remains anchored to the source-identical GPU body.

## Experimental Adjustments

- **Separated pixel and decision oracles**: Pixel confinement is checked before RandAugment against NumPy reflection, while active full transforms are checked through decision traces. (ref: `02-plan-review.md` concerns 1-3)
- **Defined loader projections**: Explicit differential and absolute wall formulas replace an invalid loader-derived counted-exposure ratio. (ref: `02-plan-review.md` concern 4)
- **Precommitted low-exposure verdict**: Primary success remains valid; a low-exposure miss closes exact reflection but not the wider geometry family. (ref: `02-plan-review.md` concern 6)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - score not launched
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: Not submitted
- **Started**: N/A
- **Ended**: 2026-07-27T00:01:07Z

Description:
- This will be the sole fixed-seed score only if reflection padding preserves the complete crop/flip/RandAugment decision stream and stable worker delivery. The intervention tests whether removing artificial zero crop borders improves boundary quality while preserving the accepted GPU learner and counted exposure. The primary threshold is 94.42%; final accuracy and loss are corroboration only.

Observations:
- Static audit passed: exactly one production argument changed, compilation succeeds, local CIFAR-10 and one idle H20 are available, and no scored log exists. (source: setup commands)
- Semantic preflight passed across 49,920 active and 4,096 inactive worker traces. All sampler indices, worker ids, crop/flip decisions, activity flags, decoded RandAugment decisions, private-state hashes, targets, and terminal main RNG states matched. (source: semantic preflight)
- Reflection affected only padding-derived pixels before RandAugment across all 162 crop/flip cases; the measured padded-window contact rate was 98.8020%, and 21 private RandAugment decisions matched. (source: semantic preflight)
- Real-loader timing failed the preregistered stability gate. Candidate active epochs had 11.20% CV, including a 4.0061-second outlier, while the accepted active arm had 0.97% CV. Per plan, the stable qualification miss was not rerun and the score was skipped. (source: loader timing preflight)

Key Metrics:
- **Accepted active loader**: median 2.8304 s; CV 0.9735%; raw `[2.8176, 2.8290, 2.8313, 2.8917, 2.8766, 2.8295]` s.
- **Candidate active loader**: median 3.0941 s; CV 11.1988%; raw `[4.0061, 2.9626, 3.2606, 3.1525, 2.9708, 3.0357]` s.
- **Accepted inactive loader**: median 2.7877 s; CV 0.2078%.
- **Candidate inactive loader**: median 2.8135 s; CV 1.2367%.
- **Weighted loader medians**: accepted 2.8155 s; candidate 2.9959 s.
- **Projected wall**: 369.34 s differential; 444.41 s absolute; both below 500 s.
- **Projected counted exposure**: unchanged 133.00736 passes; 133.2205 projected complete epochs.
- **Scored runs**: 0; `run.log` was never created.

## Verification Results

### Conditions Checked

- Static source/environment audit: **PASS**.
- Independent construction/model/optimizer/RNG audit: **PASS**; 987,098 parameters.
- Pixel, decision-stream, worker-trace, and cutoff semantics: **PASS**.
- Loader batch shape/count and finite data: **PASS**.
- Loader wall projections below 500 seconds: **PASS**.
- Every loader timing CV <=5%: **FAIL**; candidate active CV was 11.1988%.
- Sole fixed-seed score: **SKIPPED by preregistered abort criterion**.

### Informational Metrics

- Active trace samples: 49,920; inactive trace samples: 4,096.
- Exhaustive pre-RandAugment crop/flip cases: 162.
- Padding-contact rate: 0.9880200.
- Independently decoded RandAugment decisions: 21.

## Errors & Dead Ends

- Reflection padding produced unstable active-phase worker delivery in the balanced real-loader preflight. The candidate's 11.20% active CV exceeded the 5% bound despite acceptable wall projections. This is a stable feasibility miss under the plan, not an infrastructure retry condition; do not rerun timing, launch the score, or rescue the same experiment with another padding variant.

## Human Notes

> Autopilot local-only execution; no user intervention requested.
