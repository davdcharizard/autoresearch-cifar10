# EXP-024: Depth-Preserving Final-Stage Widening to 160

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-024
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Added an optional final-stage channel override to the existing generic `ResNet` constructor and set production to 160 final channels while retaining width-2 stem/layer1/layer2 channels and all three blocks per stage. The candidate is therefore a postactivation 32/64/160 ResNet-20 with unchanged Option-A slicing/padding, global average pooling, training recipe, evaluator, and logging. Static checks prove nine blocks, 19 convolutions, stage outputs 32/64/160, pads 32/96, `Linear(160,10)`, FP32 state, and exactly 1,507,818 parameters.

### Surprises & Discoveries

The accepted constructor already centralizes all stage construction, so the isolated architecture required one optional argument rather than any `BasicBlock` or transition rewrite. Ruff reformatted the multiline constructor and production instantiation; no semantic adjustment was needed.

### Decisions

- Keep `WIDTH_MULTIPLIER = 2` as the accepted early-stage contract and add `FINAL_STAGE_CHANNELS = 160` as an explicit candidate-only override.
- Let the wider graph consume its natural seed-42 initialization stream; do not realign later tensors or data RNG, because the scored intervention is the ordinary net effect of this architecture.
- Build safety/timing controls explicitly as `ResNet(3,10,2)` and candidates as `ResNet(3,10,2,160)`, independent of the mutated production constant.

## Experimental Adjustments

- **No post-veto rescue**: The candidate-only concentration gate fired once at step 2; the plan required no such event, so timing and production were skipped without changing the threshold, width, seed, or corpus. (ref: `preflight-report.json`)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A — production not launched
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/024/preflight-report.json`
- **WandB**: N/A
- **Status**: failed before production
- **Started**: 2026-08-06 15:33 UTC
- **Ended**: 2026-08-06 15:34 UTC

Description:
- One seed-42 local H20 run of the 32/64/160 FP32 ResNet-20 on the accepted N1/M7+CutMix strong phase and hard weak tail. An exact-corpus numerical safety probe and five-pair H20 timing gate must pass before production. Formal success requires `best_test_acc >=94.25%`; switch, first-weak, NLL, and exposure diagnose whether the asymmetric final stage preserves optimization and improves semantic capacity.

Observations:

- Static implementation and scope passed: only `train.py` differs; the candidate is FP32 ResNet-20 with stages 32/64/160, nine blocks, 19 convolutions, Option-A pads 32/96, and exactly 1,507,818 parameters. (source: static command output; `preflight-report.json` facts)
- The controller persisted three 100-batch production-path buckets in one 472,747,293-byte corpus with SHA-256 `d4294f5adb2e58e0847366231458b21901c6f01f270d4cd1c9eae14a05b64565`; fresh explicit control and candidate arms each replayed 100 strong-hard and 100 strong-soft batches from seed 42. (source: `preflight-report.json`)
- At step 2, the candidate placed 126/128 predictions in one class (98.4375%) while the matched control placed 78/128 (60.9375%). Candidate/control losses were 3.8348/5.0240 on that batch. This met the predeclared candidate-only concentration veto. (source: `preflight-report.json` class-share/loss arrays)
- Both arms nevertheless remained finite through all 200 replay steps and reached BN counters of 200. Terminal loss EMA was 2.08604 control versus 1.96966 candidate (ratio 0.94421), but lower loss cannot override the safety veto. (source: `preflight-report.json`)
- Timing and the scored seed-42 production run were not launched; no `run.log` was created. (source: execution status)

Key Metrics:

- safety status: failed; candidate-only concentration events: 1 at step 2 (source: `preflight-report.json`)
- step-2 class share: candidate 98.4375%, control 60.9375% (source: `preflight-report.json`)
- terminal loss EMA: candidate 1.96966, control 2.08604, ratio 0.94421 (source: `preflight-report.json`)
- corpus: 300 batches, SHA-256 `d4294f5adb2e58e0847366231458b21901c6f01f270d4cd1c9eae14a05b64565` (source: `preflight-report.json`)
- best_test_acc: unavailable; production not launched (source: execution status)

## Verification Results

### Conditions Checked

- **Primary metric improvement — skipped**: no scored production result was produced after the mandatory safety veto.
- **Completion/numeric summary — skipped**: production was not launched.
- **Fixed budget and <10-minute runtime — skipped**: production was not launched.

### Informational Metrics

## Errors & Dead Ends

### 2026-08-06 — Candidate-only class concentration in asymmetric final-stage width preflight
- Error: `candidate-only concentration at steps [2]`
- Root cause: The 32/64/160 graph produced 98.4375% one-class predictions at matched step 2 while the accepted 32/64/128 control was at 60.9375%; the net cause may include the 64-to-160 Option-A geometry, wider stage initialization, classifier, or coupled early optimization.
- Source: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/024/preflight-report.json`
- Do NOT retry: Do not relax the concentration gate, rerun the corpus, or rescue EXP-024 with width 144/192, learned transitions, another seed, or optimizer changes.

## Human Notes

> Autopilot session; no human intervention requested.
