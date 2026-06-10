# EXP-005: Isolated Cutout

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md
- **Plan**: plans/plan-005.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-005
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary
Implemented isolated cutout on top of the EXP-002 FP32 throughput baseline. `train.py` now defines cutout constants and appends a single `transforms.RandomErasing` operation to the training transform, while preserving model architecture, optimizer, LR milestones, batch size, FP32 compile, channels-last, seed, and once-per-epoch evaluation.

### Surprises & Discoveries
No implementation surprises. `RandomErasing` is available in the project `uv` environment and static checks pass.

### Decisions
The cutout operation is applied after normalization with `value=0.0`, so the masked square is zero in model input space. `CUTOUT_AREA` is derived from `CUTOUT_SIZE=16`, giving an exact 16x16 square through `scale=(0.25, 0.25)` and `ratio=(1.0, 1.0)`.

## Experimental Adjustments
- **Relaunched on GPU 1**: The initial launch used `CUDA_VISIBLE_DEVICES=0`, but `nvidia-smi` showed GPU 0 had other activity from another run while GPU 1 was free. Run 1 was terminated before using it as a measurement, and the real measured run is relaunched with `CUDA_VISIBLE_DEVICES=1`. (ref: Run 1 observation)

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 4121535
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: aborted
- **Started**: 2026-06-08 11:48
- **Ended**: 2026-06-08 11:49

Description:
- Run the cutout-only FP32 throughput ResNet-20 recipe locally on one GPU with output redirected to `run.log`. The experiment isolates cutout from EXP-000's failed label-smoothing, Nesterov, and slow-cosine bundle. The expected result is improved generalization without a material loss of optimizer steps, reaching at least `92.05%` `best_test_acc`.

Observations:
- Aborted before final measurement because another training run was active on physical GPU 0 while GPU 1 was free. The partial log reached only early training and is not used for metrics. (source: `nvidia-smi` check; partial run.log before removal)

Key Metrics:

### Run 2

Metadata:
- **Job ID**: local PID 4124968
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 11:50
- **Ended**: 2026-06-08 11:57 UTC

Description:
- Relaunch the same EXP-005 cutout-only recipe on physical GPU 1, which was free at the pre-run occupancy check. This is the real measurement run for EXP-005. The expected result remains improved generalization from isolated cutout without material throughput loss, with success requiring `best_test_acc >= 92.05%`.

Observations:
- Startup is clean under `CUDA_VISIBLE_DEVICES=1`: log reports `Device: cuda`, `ResNet-20 | params: 269,722`, `Time budget: 300s`, and `Batches per epoch: 390`; `nvidia-smi` shows GPU 1 in use and GPU 0 idle at launch. (source: run.log startup lines; `nvidia-smi` launch check)
- Early cutout training is running without transform, CUDA, or compile errors. Evaluation reached 78.99% by epoch 13; early accuracy is lower than the non-cutout baseline but consistent with stronger augmentation. (source: run.log lines 6-30)
- Mid-run accuracy remains far behind the baseline trajectory despite high throughput, reaching only 84.75% by epoch 34. This suggests 16x16 cutout is a strong regularizer for the fixed-budget recipe. (source: run.log lines 72-80)
- After the first LR drop at step 32000, accuracy recovered into the low 91s but plateaued below the baseline. The best epoch was 110 at 91.72%, followed by flat/slightly lower evaluations through epoch 119. (source: run.log lines 204-244)

Key Metrics:
- best_test_acc: 91.72%
- final_test_acc: 91.34%
- final_test_loss: 0.2666
- training_seconds: 300.0
- total_seconds: 407.8
- startup_seconds: 2.5
- peak_vram_mb: 379.0
- num_epochs: 119
- num_steps: 46238
- num_params: 269,722

## Verification Results

### Conditions Checked
- Baseline and threshold: PASS. Experiment index reports baseline `91.95`; with the +0.10 point margin, EXP-005 requires `best_test_acc >= 92.05`. (source: `exp-index.sh baseline`; goal file)
- Completion and numeric metric: PASS. `run.log` reports a numeric `best_test_acc: 91.72%`. (source: run.log line 244)
- Primary metric condition: FAIL. `91.72% < 92.05%`, so EXP-005 is classified as no-improvement/research failure under the tightened threshold. (source: run.log line 244)
- Scope review: skipped — aborted verification after primary metric failure per execution protocol.
- Validation cadence review: skipped — aborted verification after primary metric failure per execution protocol.

### Informational Metrics
- final_test_acc: 91.34% (source: run.log line 245)
- final_test_loss: 0.2666 (source: run.log line 246)
- training_seconds: 300.0 (source: run.log line 247)
- total_seconds: 407.8 (source: run.log line 248)
- startup_seconds: 2.5 (source: run.log line 249)
- peak_vram_mb: 379.0 (source: run.log line 250)
- num_epochs: 119 (source: run.log line 251)
- num_steps: 46238 (source: run.log line 252)
- num_params: 269,722 (source: run.log line 253)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
