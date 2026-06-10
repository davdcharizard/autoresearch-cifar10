# EXP-075: Fan-Out Conv Init Plus CutMix Endpoint Hard Labels

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-075.md
- **Plan**: plans/plan-075.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-075
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

EXP-075 implements the approved coupled near-miss test. The branch `autoresearch/exp-075` was created from `autoresearch/dev`, and `train.py` was the only tracked file changed. The implementation adds a Conv2d-specific fan-out ReLU Kaiming initialization branch, keeps Linear initialization on the existing default Kaiming normal behavior, adds separate clean and CutMix endpoint label-smoothing constants, sets CutMix endpoint smoothing to 0.0, and keeps clean batches at smoothing 0.05.

### Surprises & Discoveries

No implementation surprises. The prior EXP-072 and EXP-074 code paths map cleanly onto the current `train.py`: `_weights_init` already centralizes initialization, and the training loop already separates CutMix and clean loss branches.

### Decisions

Kept Linear initialization unchanged rather than adding classifier-specific calibration, because the plan tests the exact EXP-072 Conv2d fan-out mechanism plus the EXP-074 endpoint-hardening mechanism. Added explicit startup markers for both the initialization variant and smoothing split so the run log can distinguish this coupled experiment from either isolated prior experiment.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 84885 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 22:06 UTC
- **Ended**: 2026-06-09 22:14 UTC

Description:
- This run tests whether the two strongest recent near-misses compose: EXP-072's Conv2d fan-out initialization and EXP-074's hard CutMix endpoint labels. It preserves the successful static CutMix exposure, clean-batch label smoothing 0.05, architecture, optimizer, schedule, transforms, batch size, compile/channels-last, seed, and evaluation harness. The active baseline is 94.11%, so the required improvement threshold is 94.21%.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found both GPU0 and GPU1 idle; Run 1 selected GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed the expected coupled variant: `Device: cuda`, unchanged `ResNet-20 | params: 822,790`, `Conv init: kaiming fan_out relu for Conv2d; default kaiming for Linear`, and `CutMix alpha: 1.0, prob: 0.5, endpoint label smoothing: 0.0, clean label smoothing: 0.05` (source: run.log L1-L6).
- Early output reached epoch 10 with no error signatures; `test_acc` improved from 48.18% to a current best of 78.70% (source: run.log L8-L26).
- First LR drop was observed at step 21000 with `lr: 0.0100`; post-drop convergence rose from 91.42% at epoch 54 to 93.92% by epoch 81, with no error signatures in the monitored log patterns (source: run.log L113-L168).
- Run completed cleanly with `best_test_acc=93.92%`, below the 94.21% improvement threshold; verdict is `no-improvement` (source: run.log L218-L227).

Key Metrics:
- best_test_acc: 93.92%
- final_test_acc: 93.26%
- final_test_loss: 0.2506
- training_seconds: 300.0
- total_seconds: 398.1
- startup_seconds: 1.9
- peak_vram_mb: 660.4
- num_epochs: 105
- num_steps: 40676
- num_params: 822,790
- Verdict: no-improvement; did not meet `best_test_acc >= 94.21%`.

## Verification Results

### Conditions Checked
- Code-scope constraint: PASS. `git diff --name-only` listed only `train.py`.
- Syntax: PASS. `python3 -m py_compile train.py` exited 0.
- Style: PASS. `uv run ruff check train.py` reported `All checks passed!`.
- Implementation and startup markers: PASS. `git diff train.py` shows only the planned Conv2d fan-out init and smoothing split; `run.log` prints the Conv init marker and `CutMix alpha: 1.0, prob: 0.5, endpoint label smoothing: 0.0, clean label smoothing: 0.05`.
- Primary metric availability: PASS. `run.log` contains numeric `best_test_acc: 93.92%` and `peak_vram_mb: 660.4` (source: run.log L218-L224).
- Hard constraints: PASS. No files outside `train.py` were modified; startup confirms unchanged parameter count 822,790 and unchanged CutMix alpha/prob. The diff leaves seed, architecture widths, optimizer, LR schedule, normalization, validation cadence, dependencies, and time-budget behavior unchanged.
- Improvement threshold: FAIL for improvement classification. Baseline is 94.11%; active noise guard requires at least 94.21%, and EXP-075 reached 93.92%, so it is `no-improvement`.

### Informational Metrics
- final_test_acc: 93.26%
- final_test_loss: 0.2506
- training_seconds: 300.0
- total_seconds: 398.1
- startup_seconds: 1.9
- peak_vram_mb: 660.4
- num_epochs: 105
- num_steps: 40676
- num_params: 822,790

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention during implementation.
