# EXP-066: CutMix Probability 0.75

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-066.md
- **Plan**: plans/plan-066.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-066
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Created `autoresearch/exp-066` from `autoresearch/dev` and implemented the planned one-scalar bracket by changing only `CUTMIX_PROB` in `train.py` from `0.5` to `0.75`. The existing CutMix implementation, alpha, endpoint label smoothing, architecture, optimizer, LR schedule, transform stack, compile/channels-last path, and evaluation cadence were left unchanged.

### Surprises & Discoveries

No implementation surprises. The EXP-064 CutMix path already exposes `CUTMIX_PROB` as a single top-level constant, so the bracket required no training-loop or loss-function edits.

### Decisions

Kept the experiment to a one-line hyperparameter change to preserve clean attribution against EXP-064/065. Preflight confirmed the tracked diff is limited to `train.py`; `python3 -m py_compile train.py` and `uv run ruff check train.py` both passed.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 51941; shell PID 3669768, uv PID 3669769, main Python PID 3669772
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 20:02
- **Ended**: 2026-06-09 20:08

Description:
- Run the one-scalar `CUTMIX_PROB=0.75` bracket locally on a single available GPU using the fixed CIFAR-10 training harness. The expected useful signal is whether stronger regional mixing can raise the peak checkpoint from the 94.11% baseline to at least the 94.21% noise-guard threshold. Output is captured to `run.log`, which will be used for startup markers, LR milestone checks, final metrics, and error diagnosis.

Observations:
- Startup markers confirmed CUDA execution, ResNet-20 with 822,790 parameters, `CutMix alpha: 1.0, prob: 0.75, label smoothing: 0.05`, 300s budget, and 390 batches per epoch. GPU0 was selected and showed expected early memory use. (source: run.log L1-L5; `nvidia-smi` at launch)
- Pre-drop training remained stable, with best accuracy reaching 88.61% by epoch 42 and no traceback/OOM/non-finite markers. The first LR drop occurred at step 21000 in epoch 54 with LR 0.0100; post-drop evals climbed to 93.08% by epoch 58. (source: run.log L89-L123)
- Post-drop training peaked at 93.74% by epoch 66 and plateaued below the 94.21% threshold until a late epoch-89 jump reached 94.11%. This matched the baseline but did not exceed the +0.10pp noise guard. (source: run.log L137-L183)
- Run exited cleanly with final summary metrics; final `best_test_acc` remained 94.11%, so this is a valid no-improvement result. (source: run.log L193-L202)

Key Metrics:
- `best_test_acc`: 94.11%
- `final_test_acc`: 93.47%
- `final_test_loss`: 0.2780
- `training_seconds`: 300.0
- `total_seconds`: 388.7
- `startup_seconds`: 2.0
- `peak_vram_mb`: 661.9
- `num_epochs`: 93
- `num_steps`: 35,953
- `num_params`: 822,790
- Verdict for execution: valid no-improvement because 94.11% equals the current baseline but is below the 94.21% improvement threshold.

## Verification Results

### Conditions Checked
- Baseline check: `exp-index.sh baseline` reported `baseline=94.11`, `baseline_commit=1119ff8`, `total_experiments=67`, `improvements=10`; pass.
- Scoped diff check: `git diff --name-only` listed only `train.py`; pass.
- Compile check: `python3 -m py_compile train.py` exited 0; pass.
- Style check: `uv run ruff check train.py` reported `All checks passed!`; pass.
- Execution summary check: `run.log` contains numeric final metrics including `best_test_acc: 94.11%`; pass.
- Model-depth check: `run.log` reports `ResNet-20 | params: 822,790`; pass.
- CutMix settings check: `run.log` reports `CutMix alpha: 1.0, prob: 0.75, label smoothing: 0.05`; pass.
- Batch-geometry check: `run.log` reports `Batches per epoch: 390`; pass.
- LR-drop check: `run.log` contains `step 21000 ep 54 ... lr: 0.0100`; pass.
- Error scan: `rg -n "Traceback|RuntimeError|CUDA out of memory|nan|inf" run.log` returned no matches; pass.
- Classification check: valid run but `94.11% < 94.21%`; classified as no-improvement.

### Informational Metrics
- The run completed 35,953 steps and 93 epochs, fewer than EXP-064/065 because late-epoch batch times slowed, but still reached the required first LR drop.
- Parameter count stayed at 822,790 and peak VRAM stayed low at 661.9 MB, so the probability bracket did not change model size or material memory footprint.
- Final accuracy was 93.47%, above EXP-064's final 93.02% but below EXP-065's final 93.76%; the primary best checkpoint only tied the baseline.

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention during implementation.
