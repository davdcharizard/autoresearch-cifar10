# EXP-010: Back-loaded 1-2-3 stage depth

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-010
- **Base Node**: 002
- **Commit**: 160b62f
- **Outcome**: failed

## Implementation Notes

### Summary

The implementation changes only the static six-item `block_specs` in `train.py`, moving the second 64-channel block to a third 256-channel block, plus the human-readable model/config labels. The training loop, CutMix implementation and private generators, optimizer, time schedules, timer boundaries, evaluator, validation cadence, seed, and summary remain unchanged. Static/source, CPU FP32, and physical-GPU-0 BF16 invariants passed before latency measurement.

### Surprises & Discoveries

Seed 42 makes repeated candidate construction bitwise deterministic, but shared parent/candidate stem and first-block weights are not all bitwise equal: 10 of 14 shared-prefix state tensors matched. PyTorch layer constructors consume shape-dependent RNG before the model-wide Kaiming `apply`, so later architecture shapes change the RNG state from which even earlier modules are reinitialized. This is normal fixed-seed architecture behavior and means attribution remains package-level.

### Decisions

Claude's plan review led to a minimal static block-list change rather than retaining a general benchmark-only stage allocator in production. The parent is loaded from exact commit `a36dc09` under a unique temporary module name, and the binding preflight permits at most one rerun only if parent round medians show >7.5% drift; the first valid measurement is decisive.

## Experimental Adjustments

- **Corrected disposable invariant expectation**: Replaced a mistaken cross-architecture shared-prefix bitwise assertion with fixed-seed candidate self-determinism plus an explicit equality count, consistent with the reviewed plan's expected-divergence language. No production code or experiment setting changed. (ref: preflight invariant harness before Run 1)
- **Added repository root to disposable GPU smoke `sys.path`**: The first `/tmp` smoke stopped at `ModuleNotFoundError: train`; adding the reviewed cwd/import path allowed the same unchanged smoke to run. (ref: Errors & Dead Ends)

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 1277209; training PID 1277218; exec session 52085
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log
- **WandB**: N/A
- **Status**: failed (research result below parent)
- **Started**: 2026-08-06 03:22:04 UTC
- **Ended**: 2026-08-06 03:29:56 UTC

Description:
- Run the fixed 1-2-3 stage allocation once from EXP-002 on physical GPU 0 after it passes the accuracy-blind parent-relative latency gate. The intervention preserves six residual blocks and exact Conv/Linear MACs while moving capacity to 8x8 semantic features. Formal success requires at least 95.33% best test accuracy; the stronger architecture hypothesis requires at least 95.53% and 26,500 steps.

Observations:
- CPU/source checks passed: parent/candidate block orders were exact; parameter counts were 2,748,890/3,855,578; both had 392,612,352 MACs/image; all six candidate shapes and gradients passed. Candidate seed-42 construction was bitwise self-deterministic; 10/14 shared-prefix tensors matched across architectures. (source: `/tmp/exp010_invariants.py` output, 2026-08-05)
- Physical GPU 0 is NVIDIA H20 with 97,871 MiB. The BF16/channels-last batch-256 smoke passed with loss 2.381805, finite nonzero gradients, and 477.37 MiB peak allocation. (source: `/tmp/exp010_gpu_smoke.py` output, 2026-08-05)
- The first paired latency measurement was valid: parent round-median drift was 2.15%, below the 7.5% contamination threshold, so no remeasurement was permitted. Candidate median/p90 training latency was 9.241963/9.410895 ms versus parent 10.000283/10.200921 ms; ratios 0.924170/0.922553 passed the 1.05/1.08 gates. Candidate evaluation median was also faster at 2.478465 versus 2.599955 ms. (source: `/tmp/exp010_latency.json`, 2026-08-05)
- The valid preflight projects 30,243 steps, 155.1 epochs, and 471.5 seconds total, all passing the >=26,500-step and <600-second gates. Joint benchmark peak allocation was 622.55 MiB and losses/gradients stayed finite. (source: `/tmp/exp010_latency.json`, 2026-08-05)
- The full-run log wrote the correct device, 1-2-3 configuration, 3,855,578 parameters, 300-second budget, and 195 batches/epoch immediately after launch. (source: `run.log` L1-L5)
- Early accuracy progressed from 54.56% at epoch 1 to 82.42% at epoch 10, 88.48% by epoch 25, 90.80% by epoch 50, 92.10% by epoch 75, and 93.37% by epoch 100. There is no retained EXP-002 early curve, so this cannot isolate early-feature starvation. (source: `run.log` L7-L205)
- The clean tail reached 95.03% at epoch 142 and 95.04% at epoch 145, then stayed in 94.85-95.04% through epoch 157. The last step had smooth loss 0.0007, LR 0.0020, and effective drop path 0.000. (source: `run.log` L289-L319)
- The process exited 0 with no error/nonfinite match, 157 evaluations for 157 epochs, and a complete summary. Nevertheless, best accuracy 95.04% was 0.19 points below parent EXP-002 and 0.29 below the formal 95.33% threshold, so the experiment is a research failure with no retry. (source: `run.log` L319-L331)

Key Metrics:
- preflight median latency ratio: 0.924170 (candidate 9.241963 ms / parent 10.000283 ms)
- preflight p90 latency ratio: 0.922553 (candidate 9.410895 ms / parent 10.200921 ms)
- projected optimizer steps: 30,243; projected total time: 471.5 seconds
- `best_test_acc`: 95.04% (parent 95.23%, delta -0.19 points; source: `run.log` L322)
- `final_test_acc`: 95.04%; `final_test_loss`: 0.2131 (source: `run.log` L323-L324)
- `training_seconds`: 300.0; `total_seconds`: 455.7; `startup_seconds`: 1.1 (source: `run.log` L325-L327)
- `peak_vram_mb`: 1,193.7; `num_epochs`: 157; `num_steps`: 30,558; `num_params`: 3,855,578 (source: `run.log` L328-L331)
- CutMix exposure: 11,165/22,510 = 0.4960 (source: `run.log` L320)

## Verification Results

### Conditions Checked

- **Parent-relative accuracy - FAIL**: parent EXP-002 is 95.23%, requiring at least 95.33%; candidate best was 95.04%, a -0.19-point parent delta and -0.29-point threshold miss. (source: `tree.sh show ... 002`; `run.log` L322)
- **Clean completion/budget/summary - skipped as a formal second check after the failed primary condition**. Run-1 evidence nevertheless confirms exit 0, 300.0 charged seconds, 455.7 total seconds, 157/157 evaluations, and a complete summary; this supports `no-improvement` rather than `crash`.

### Informational Metrics

- Skipped by verification protocol because the primary necessary condition failed. Observed run values are retained under Run 1 Key Metrics, not treated as passing verification outputs.

## Errors & Dead Ends

### 2026-08-05 - Disposable GPU smoke import path
- Error: `ModuleNotFoundError: No module named 'train'`
- Root cause: Running a script located in `/tmp` placed `/tmp`, not the repository root, on `sys.path` despite the shell cwd.
- Source: `/tmp/exp010_gpu_smoke.py` first invocation before Run 1
- Do NOT retry: Do not run external harness files without explicitly adding the repository root to `sys.path`.

### 2026-08-06 - Back-loaded allocation missed parent accuracy
- Error: `best_test_acc 95.04% < parent 95.23% < required 95.33%`
- Root cause: The fixed 1-2-3 package produced no detectable improvement despite higher throughput/exposure. Whether early-feature removal, redistributed drop-path dose, or the changed architecture RNG realization caused the miss is unresolved.
- Source: `run.log` L289-L331
- Do NOT retry: Do not rerun or tune the same 1-2-3 allocation from this metric result; treat the full allocation-plus-drop-dose package as a failed leaf.

## Human Notes

> User requires physical GPU 0 and Claude-only adversarial reviews. Both idea and plan reviews completed successfully with Claude; no fallback reviewer was used.
