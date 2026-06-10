# EXP-074: CutMix Endpoint Hard Labels

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-074.md
- **Plan**: plans/plan-074.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-074
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed — no-improvement

## Implementation Notes

### Summary

EXP-074 implements the approved CutMix endpoint hard-label probe. The branch `autoresearch/exp-074` was created from `autoresearch/dev`, and `train.py` was the only tracked file changed. The implementation adds `CLEAN_LABEL_SMOOTHING = 0.05`, changes `CUTMIX_LABEL_SMOOTHING` to `0.0`, keeps the non-CutMix loss on the clean smoothing constant, and leaves both CutMix endpoint losses wired to the CutMix endpoint constant.

### Surprises & Discoveries

No implementation surprises. The current training loop already separated CutMix and clean loss branches, so the change was limited to constants, startup logging, and replacing the hard-coded clean smoothing literal with the new clean constant.

### Decisions

Used separate named constants rather than a bare `0.05` in the clean branch so the log and diff make clear that EXP-074 is not a global label-smoothing deviation. The startup line prints `endpoint label smoothing: 0.0` and `clean label smoothing: 0.05` for verification.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 3556 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-09 21:52
- **Ended**: 2026-06-09 21:59 UTC

Description:
- This run tests whether CutMix's area-weighted label mixture plus endpoint label smoothing over-softens mixed-batch supervision. It keeps clean batches at label smoothing 0.05, changes only CutMix endpoint losses to label smoothing 0.0, and preserves architecture, optimizer, schedule, transforms, batch size, compile/channels-last, seed, and evaluation. The active baseline is 94.11%, so the required improvement threshold is 94.21%.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found GPU0 idle and GPU1 occupied by another checkout; Run 1 launched on GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed the expected scope and smoothing split: `Device: cuda`, `CutMix alpha: 1.0, prob: 0.5, endpoint label smoothing: 0.0, clean label smoothing: 0.05`, and unchanged `ResNet-20 | params: 822,790` (source: run.log L1-L4).
- Early output reached epoch 2 with no error signatures; `test_acc` improved from 43.27% to 59.97% (source: run.log L7-L9).
- First LR drop was reached at step 21000 with `lr: 0.0100`; post-drop convergence climbed from 91.37% at epoch 54 to a current best of 93.81% by epoch 68 with no error signatures (source: run.log L112-L141).
- Late training set a best of 94.17% at epoch 91, then the final epochs ended at 93.19% without clearing the 94.21% noise-guard threshold (source: run.log L187-L226).

Key Metrics:
- `best_test_acc`: 94.17%
- `final_test_acc`: 93.19%
- `final_test_loss`: 0.2488
- `training_seconds`: 300.0
- `total_seconds`: 398.2
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 105
- `num_steps`: 40577
- `num_params`: 822,790
- Verdict: no-improvement. Baseline is 94.11%, but EXP-074 did not reach the required 94.21% improvement threshold.

## Verification Results

### Conditions Checked
- Scope: `git diff --name-only` listed only `train.py`; passed.
- Syntax: `python3 -m py_compile train.py` exited 0; passed.
- Style: `uv run ruff check train.py` reported `All checks passed!`; passed.
- Loss-smoothing implementation: `git diff train.py` shows clean smoothing preserved as `CLEAN_LABEL_SMOOTHING = 0.05`, CutMix endpoint smoothing changed to `CUTMIX_LABEL_SMOOTHING = 0.0`, and non-CutMix loss uses the clean constant; passed.
- Startup log markers: `run.log` L1-L5 confirmed `Device: cuda`, unchanged parameter count, `CutMix alpha: 1.0`, `prob: 0.5`, `endpoint label smoothing: 0.0`, `clean label smoothing: 0.05`, 300s budget, and 390 batches/epoch; passed.
- Final metric: `run.log` L217-L226 reported numeric final metrics including `best_test_acc: 94.17%`; passed.
- Improvement threshold: 94.17% is below the required 94.21% threshold, so the experiment is no-improvement despite being +0.06pp above the 94.11% baseline.

### Informational Metrics
- final_test_acc: 93.19%
- final_test_loss: 0.2488
- training_seconds: 300.0
- total_seconds: 398.2
- startup_seconds: 1.9
- peak_vram_mb: 660.4
- num_epochs: 105
- num_steps: 40577
- num_params: 822,790

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention during implementation.
