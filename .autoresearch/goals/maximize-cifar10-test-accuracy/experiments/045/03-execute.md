# EXP-045: ResNet-D Projection Shortcuts

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-045
- **Commit**: (pending - committed on loop success)
- **PR**: N/A (offline local run)
- **Outcome**: failed - valid normal-exposure metric miss

## Implementation Notes

### Summary

Changed only `PreActBlock` in `train.py`: stride-2 projection shortcuts now average each non-overlapping 2x2 cell before the existing pointwise projection, whose stride becomes one. The main branch, layer1 direct projection, raw identity shortcuts, all parameter construction order, and every training component remain unchanged. Added an ignored evaluator-blocked preflight for source/state/topology, independent phase forward/backward, controls, optimizer update, complete-step timing, and exposure gates.

### Surprises & Discoveries

The accepted model's stated 52 tensors are its parameter tensors; its complete state dict has 97 entries once BatchNorm parameters and buffers are counted. The preflight records and checks both quantities explicitly. No production-code structural surprise required a design change.

### Decisions

The average pool is registered after the existing projection so parameter construction and RNG consumption remain exact. Identity shortcuts retain raw `x`; only projection shortcuts consume the preactivated tensor. Timing constructs only one GPU arm per window and restores common CPU state and RNG for every window.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 1437163 (launcher 1437156)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-27 07:34:58 UTC
- **Ended**: 2026-07-27 07:40:51 UTC

Description:
- One fixed-seed local H20 score of the exact two-transition ResNet-D shortcut treatment will run only after semantic and timing gates pass. It tests whether aggregating all four shortcut phases improves CIFAR-10 phase stability while retaining at least 127 passes. The valid score will not be rerun regardless of outcome.

Observations:
- Static source/resource audit: branch `autoresearch/maximize-cifar10-test-accuracy-045` at accepted `a7c42dc`; only `train.py` differs; frozen files are exact; one idle NVIDIA H20; model has 1,003,482 parameters across 52 parameter tensors and 97 state entries (source: preflight audit stdout, 2026-07-27).
- Semantic gate passed after two preflight-only repairs. Independent FP64/FP32 phase forward errors were `1.11e-16`/`5.96e-8`, gradient errors were zero, transition shortcut RMS ratios were `1.000786`/`1.000806`, main-to-shortcut RMS ratios were `0.693843`/`0.881363`, controls were exact, and Nesterov update error was `1.49e-8` (source: semantic preflight stdout, 2026-07-27).
- Timing gate passed all 16 retained windows: arm CVs `0.0010-0.0041`, regime-ratio CVs `0.0037-0.0044`, retentions `[0.982457, 0.980673, 0.978287, 0.984691]`, median projected exposure `127.901854` passes, and candidate peak `622.165 MiB` (source: timing preflight stdout, 2026-07-27).
- Sole score launched with the exact command and produced `Device: cuda`, 1,003,482 parameters, a 300-second budget, and 195 batches per epoch at startup (source: `run.log` L1-L4).
- The sole score exited zero with one finite summary. Mixup stopped once at step 16,177/195.0s and RandAugment exhausted at step 16,185/195.1s; 26 evaluations followed the every-fifth cadence through final epoch 130, and no error signature appeared (source: `run.log` L1-L71).
- Best accuracy reached `94.11%` at epoch 125 and ended `94.06%`/`0.2512`; normal exposure makes this an attributable architectural miss rather than timing or infrastructure failure (source: `run.log` L58-L71).

Key Metrics:
- `best_test_acc`: `94.11%`, `0.37` below baseline and `0.47` below threshold (source: `run.log` L62).
- `final_test_acc` / `final_test_loss`: `94.06%` / `0.2512` versus accepted `94.45%` / `0.2456` (source: `run.log` L63-L64; accepted EXP036 report).
- Exposure: `25,215` steps = `129.1008` passes across 130 epochs (source: `run.log` L69-L70).
- Counted/wall/startup: `300.0/340.5/1.1s`; peak VRAM `1,108.4MiB`; params `1,003,482` (source: `run.log` L65-L71).

## Verification Results

### Conditions Checked

- **Completion/resource contract - PASS**: exit 0; one H20; one finite summary; 300.0s counted/340.5s wall; correct transitions; 26 unique evaluations; 1,003,482 params; 129.1008 passes; no errors (source: `run.log` L1-L71 and local audit).
- **Primary metric improvement - FAIL**: best `94.11%` is below baseline `94.48%` and required `94.58%` (source: `run.log` L62).
- **Hypothesis support - FAIL**: exposure cleared 127 passes but accuracy did not clear 94.58%; the exact two-transition treatment is rejected.
- **Corroboration - skipped after metric failure**: final `94.06%` and loss `0.2512` are recorded but not alternate criteria.

### Informational Metrics

- Skipped under fail-fast verification after primary failure; raw values remain above.

## Errors & Dead Ends

### 2026-07-27 - Accepted-arm capture compatibility
- Error: `AttributeError: 'PreActBlock' object has no attribute 'shortcut_pool'` during semantic preflight.
- Root cause: The observational capture helper accessed the candidate-only module attribute directly on the accepted control.
- Source: semantic preflight stdout before any counted run, after both independent phase oracles passed.
- Do NOT retry: Do not assume candidate-only state-free module attributes exist on the accepted source; use `getattr(..., None)` and observational hooks.

### 2026-07-27 - Topology probe mutated BatchNorm controls
- Error: block-local main output equality failed after topology probing.
- Root cause: The shape-only full-model topology probe used default training mode and updated candidate BatchNorm running buffers before the accepted/candidate control.
- Source: second semantic preflight before any counted run; independent phase forward/gradient oracles remained exact.
- Do NOT retry: All shape-only probes preceding state controls must run in evaluation mode or use disposable models.

## Human Notes

> Autopilot requested; user asleep. Offline/local only, no GitHub or network.
