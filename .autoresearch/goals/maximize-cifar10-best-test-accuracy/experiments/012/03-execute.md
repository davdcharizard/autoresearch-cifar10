# EXP-012: Canonical Full Preactivation

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-012
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed — valid run, no improvement

## Implementation Notes

### Summary

Replaced the accepted postactivation residual network with the reviewed canonical full-preactivation package while leaving the complete EXP-010 training recipe byte-identical. Each block now performs BN-ReLU-Conv-BN-ReLU-Conv-add, ordinary shortcuts transport raw residual sums, the first and two transition shortcuts transport preactivated values, and the network ends with BN-ReLU before pooling. Shared randomized-module construction order, parameter count, optimizer membership, and all non-model mechanics are preserved.

### Surprises & Discoveries

The paired H20 benchmark measured the preactivation candidate slightly faster in training (0.99378x control) and 16 MB lower in peak allocation despite moving transition normalization before stride-2 convolution. This resolves the external review's memory concern empirically and confirms that equal operator/activation totals translate to compute-neutral behavior on this H20.

### Decisions

No implementation-time deviation was needed. The explicit `preactivate_shortcut` flag is true only for the first network block and is automatically true for the two shape-changing blocks; this makes the paper's boundary exception visible while raw identity remains exact for all six ordinary blocks. The mandatory Claude reviews both completed successfully; no fallback reviewer was used.

## Experimental Adjustments

- **Retired SE before execution**: mandatory paired timing measured 1.23324x training cost and only 21,810 projected steps, so the external idea review's conditional preactivation choice was used without consuming a full run. (ref: `00-se-timing.md`, `01-idea-review.md`)

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 2279275 (`timeout` supervisor 2279274)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 04:43:22 UTC
- **Ended**: 2026-08-06 04:49:22 UTC

Description:
- One fixed-seed local run of the reviewed canonical full-preactivation width-2 ResNet-20 on the unchanged EXP-010 p=0.5 CutMix recipe. The candidate must retain the fixed 300-second counted budget and finish within 600 seconds total. The primary hypothesis is `best_test_acc >=94.25%`; trajectory and exposure diagnostics cannot trigger tuning or a rerun.

Observations:
- Structural gate passed: bitwise shared Conv/Linear tensors and CPU RNG, exact shortcut semantics, 1,073,962 parameters, complete optimizer membership, and finite nonzero hard/soft gradients. (source: `00-preactivation-check.py` execution output)
- Timing gate passed: training ratio 0.99378, projected 27,066 steps, control/candidate CV 0.501%/0.292%, inference ratio 1.00925, and projected total 330.73 seconds. (source: `00-preactivation-timing.md`)
- Run launched on one idle H20 and reached step 50 with finite loss 2.0377 at about 11 ms/step. (source: `run.log` initial progress line)
- The strong phase ended at 86.88%, 0.20 points below the 87.08 compounded-underfit marker, with 10,787/21,665 mixed batches (49.79%); all eight workers stopped at the 80.0% transition. (source: `run.log` L14-L15)
- The first weak checkpoint recovered immediately to 93.48%; the trajectory peaked and finished at 94.22%, missing the 94.25 acceptance gate by 0.03 points. (source: `run.log` L17-L45)

Key Metrics:
- best_test_acc: 94.22% at epoch 70, +0.07 versus baseline and -0.03 versus gate (source: `run.log` L43-L45)
- final_test_acc / final_test_loss: 94.22% / 0.1974 (source: `run.log` L46-L47)
- training / total / startup seconds: 300.0 / 332.2 / 1.0 (source: `run.log` L48-L50)
- peak_vram_mb: 582.7 MB (source: `run.log` L51)
- epochs / steps / parameters: 70 / 27,029 / 1,073,962 (source: `run.log` L52-L54)
- first weak checkpoint: 93.48% at epoch 57; switch checkpoint: 86.88% at epoch 56 (source: `run.log` L14-L17)

## Verification Results

### Conditions Checked

- **Run completion and summary**: pass — exit 0 with all ten numeric summary fields. (source: `run.log` L45-L54)
- **Fixed time and wall limit**: pass — 300.0 counted seconds and 332.2 total, below 600. (source: `run.log` L48-L49)
- **Protocol and scope**: pass — one tracked file (`train.py`), 1,073,962 parameters, one 80.0% switch, eight workers stopped, 49.79% mixed strong batches, and no duplicate evaluation epoch. (source: git diff; `run.log` L15, L54)
- **Exposure**: pass — 27,029 steps, above the 26,091 projection floor and 131 more than EXP-010. (source: `run.log` L53)
- **Primary improvement gate**: **fail** — 94.22% is below required 94.25%, despite exceeding the 94.15 baseline by 0.07. This is a valid no-improvement and cannot be rerun. (source: `run.log` L45)

### Informational Metrics

- final_test_acc: 94.22%; final_test_loss: 0.1974; peak_vram_mb: 582.7; num_epochs: 70; num_steps: 27,029. (source: `run.log` L46-L53)
- Strong switch / first weak: 86.88% / 93.48%; final equals best and the last five checkpoints end 94.06, 94.16, 93.98, 94.15, 94.22. (source: `run.log` L14-L17, L37-L45)

## Errors & Dead Ends

None.

## Human Notes

> Autopilot execution; no intervention requested.
