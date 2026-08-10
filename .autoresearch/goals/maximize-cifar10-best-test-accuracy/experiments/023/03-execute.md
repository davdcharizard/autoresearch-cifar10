# EXP-023: FP32 Width-3 ResNet-14 Depth-Width Rebalance

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-023
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Changed only `NUM_BLOCKS` from 3 to 2 and `WIDTH_MULTIPLIER` from 2 to 3 in tracked `train.py`, plus the descriptive ResNet comment. Existing generic construction now produces a postactivation FP32 ResNet-14 with stages 48/96/192, six residual blocks, 13 convolutions, two unchanged Option-A transitions, a 192-to-10 classifier, and exactly 1,540,474 parameters. All optimizer, data, CutMix, schedule, timer, worker, seed, and evaluator code remains unchanged.

### Surprises & Discoveries

No implementation surprise: the accepted model is already parameterized cleanly by depth and width, so the reviewed architecture requires no model-logic change. Static hooks confirmed stage outputs `(N,48,32,32)`, `(N,96,16,16)`, and `(N,192,8,8)`.

### Decisions

- Keep architecture construction's natural seed-42 RNG consumption; do not realign shared weights or data RNG, because the scored intervention is the normal net architecture effect.
- Controllers must instantiate accepted control explicitly as `ResNet(3,10,2)` and candidate as `ResNet(2,10,3)` rather than importing mutated constants.
- Cross-architecture 200-step loss/gradient/update ratios are diagnostics only. Only finite state and repeated candidate-only concentration are safety vetoes before the load-bearing timing gate.

## Experimental Adjustments

- **Cross-architecture safety refinement**: Demoted loss/gradient magnitude ratios from vetoes to diagnostics because unequal depth/width/parameter count make them non-equivalent. (ref: `02-plan-review.md` concern 2)

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 2441776 (supervisor PID 2441775)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 14:59 UTC
- **Ended**: 2026-08-06 15:06 UTC

Description:
- One seed-42 local H20 run of FP32 width-3 ResNet-14 on the accepted N1/M7+CutMix and hard-tail recipe. Safety and five-pair timing must pass before production. Expected success requires at least 20,000 updates and `best_test_acc >=94.25%`; switch, first-weak, NLL, and exposure will diagnose any miss.

Observations:

- Static scope/structure passed: only `train.py` differs; candidate has 1,540,474 parameters, 13 convolutions, and stage outputs 48/96/192 at 32/16/8 resolution.
- Immutable-corpus safety passed on 200 batches (100 hard, 100 CutMix), SHA-256 `bb51acc71b9f351762914a832c13c2524678874ee8d066141ea5bb9ada43c4fd`, with no candidate-only concentration; candidate/control terminal loss-EMA ratio was 0.820794 (diagnostic). (source: `preflight-report.json`)
- Five-pair H20 timing passed: weighted ratio 1.162780, max pair 1.167083, projected 23,132 steps, candidate peak 491.59 MiB, control/candidate CV 0.191%/0.260%, and projected total 384.53 seconds. (source: `timing-report.json`)
- Production log initialized correctly with CUDA, ResNet-14, 1,540,474 parameters, 300-second budget, and early finite loss 1.9982 at step 100. (source: `run.log` initial output)
- Production completed once with no error: 300.0 counted seconds, 329.2 total, 23,465 steps, 62 epochs, and 491.6 MiB peak allocation. (source: `run.log` final summary)
- Strong-phase checkpoints rose 86.00% -> 87.24% -> 88.68% -> 89.28%, then switched at 88.77% (above the 87.08% underfit marker but 0.96 below EXP-010). The first weak checkpoint reached 93.34%, above EXP-010's 93.16%. (source: `run.log` evaluation lines and switch line)
- The tail peaked at 94.00%/0.1930 NLL on epoch 53, then ended 93.89%/0.1979. The 94.00% best missed the 94.25% requirement and trailed the 94.15% baseline by 0.15 point. (source: `run.log` epoch 53 and final summary)
- Protocol integrity held: one 80.0% switch, eight stopped workers, 9,392/18,749 CutMix batches (50.093%), 18 unique evaluations, expected parameter count, and no retry. (source: `run.log`)

Key Metrics:

- preflight candidate/control loss-EMA ratio: 0.820794 (source: `preflight-report.json`)
- timing weighted ratio: 1.162780; projected steps: 23,132 (source: `timing-report.json`)
- best_test_acc: 94.00% (baseline 94.15%, threshold 94.25%; source: `run.log` final summary)
- final_test_acc: 93.89%; final_test_loss: 0.1979 (source: `run.log` final summary)
- training_seconds: 300.0; total_seconds: 329.2; startup_seconds: 1.2 (source: `run.log` final summary)
- num_steps: 23,465; num_epochs: 62; num_params: 1,540,474; peak_vram_mb: 491.6 (source: `run.log` final summary)
- switch accuracy: 88.77%; first weak accuracy: 93.34%; peak NLL: 0.1930 at 94.00%; final/best gap: -0.11 point (source: `run.log` trajectory)

## Verification Results

### Conditions Checked

- **Primary metric improvement — failed**: `best_test_acc=94.00%`, below baseline 94.15% and required threshold 94.25%. (source: `run.log` final summary; baseline query)
- **Completion/numeric summary — skipped**: verification aborted immediately after the primary necessary-condition failure, per protocol.
- **Fixed budget and <10-minute runtime — skipped**: verification aborted immediately after the primary necessary-condition failure, per protocol.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> Autopilot session; no human intervention requested.
