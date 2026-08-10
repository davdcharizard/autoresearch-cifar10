# EXP-017: Full-Run Eligible-Weight Gradient Centralization

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-017
- **Base Node**: 002
- **Commit**: 4c22c9c
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented reference-ordered GC only in `train.py`. The code inventories the exact 16 convolution and one classifier weight tensors, materializes the parent's coupled L2 term on all 44 parameter gradients, centralizes the 17 eligible regularized directions, then uses the existing PyTorch Nesterov SGD state machine with internal decay disabled. Sparse charged FP64 audits, exact path/dose reconciliation, excluded-element accounting, final optimizer-state finiteness, and additive final-16 evaluation context were added without changing evaluator calls or metric selection.

### Surprises & Discoveries

The official repository's CIFAR `SGD_GC` implementation adds coupled weight decay before centralization and momentum, whereas the initial brainstorm variant centralized only the raw data gradient. The reviewed plan was corrected before implementation. The frozen model has 44 parameter tensors: 17 eligible tensors containing 2,745,264 elements and 27 excluded BN-affine/bias tensors containing 3,626 elements.

The deterministic full-model GPU-0 smoke passed with FP32 gradients under BF16 autocast, FP64 decomposition error `9.856645908319e-10`, and maximum post-GC row-mean residual `1.103789726287e-09` (source: `/tmp/exp017_gc_smoke.py` stdout, 2026-08-06 12:55 UTC).

### Decisions

To match the official ordering without replacing PyTorch's optimizer, coupled L2 is applied once via `torch._foreach_add_` before GC and the optimizer's internal `weight_decay` is set to zero. Deterministic two-step checks established that excluded parameter and momentum-buffer updates match parent SGD, while eligible updates match an explicit `decay -> centralize -> momentum -> Nesterov` reference.

No implementation fallback is permitted if heterogeneous broadcast foreach dispatch is slow; profiler evidence is informational and the preregistered paired timing gate remains decisive.

## Experimental Adjustments

- **Added repository path to the separate deterministic smoke harness**: `/tmp` is the script directory and the current repository was not automatically importable; inserting the fixed repository root repaired verification step 3 without changing experiment code. This smoke repair is outside verification step 4's explicitly single-repair decisive-preflight ledger. (ref: 2026-08-06 smoke import error below)
- **Corrected the 1,024-step audit-sample assertion from two to three**: the fixed rule audits one-based steps 1, 512, and 1,024. The first preflight stopped at this structural assertion before paired timing and emitted no numeric vector, so the plan permits this single recorded harness repair. (ref: `/tmp/exp017_preflight.log` L1-L10 before rerun)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: not launched — preflight failed before metric access
- **Started**: N/A
- **Ended**: 2026-08-06 12:59 UTC

Description:
- One fixed-seed, physical-GPU-0 CIFAR-10 metric run will test reference-ordered full-run Gradient Centralization against EXP-002. It will launch only after the single accuracy-blind structural and paired-latency preflight passes. The expected valid result is a complete 300-second charged run with unchanged evaluation semantics and exactly one GC call per optimizer step.

Observations:

- The metric process was not launched. Importing parent/candidate modules constructed their inherited `Eval` objects and test-loader objects, after which both evaluator globals were replaced by guards before any trace. Neither attempt iterated a test loader, called `evaluate`, or computed accuracy before stopping in the 1,024-step training-only trace (source: `/tmp/exp017_preflight.log` L1-L10 and preflight control flow).
- The permitted repaired preflight stopped on allocation stability before paired timing and therefore emitted no complete numeric vector. Static inspection identified the harness-only cause: `production_trace` appended 1,024 device-resident loss scalars to `losses`, making `final_allocation` exceed the step-32 baseline independently of model/optimizer state (source: `/tmp/exp017_preflight.py` `production_trace`; `/tmp/exp017_preflight.log` L5-L9).

Key Metrics:

- GPU identity: physical GPU 0, NVIDIA H20, 97,871 MiB, UUID `GPU-b1bc897d-2183-dad2-8302-8800bc02a633`; one visible CUDA device under `CUDA_VISIBLE_DEVICES=0` (source: preflight setup commands, 2026-08-06 12:54 UTC).
- Metric launches: `0`; test accuracy values observed: `0`; complete preflight numeric vectors: `0` (source: execution protocol and `/tmp/exp017_preflight.log`).
- Charged metric-training time consumed: `0 s`; preflight GPU work was accuracy-blind feasibility work outside the frozen 300-second metric budget (source: no `run.log` and no metric-command launch).

## Verification Results

### Conditions Checked

- Parent/scope/syntax: passed before preflight; parent EXP-002 metric is 95.23, only tracked `train.py` changed, `git diff --check` and `py_compile` passed.
- GPU identity: passed; physical GPU 0 and single-device visibility matched the frozen H20 identity.
- Deterministic math/integration smoke: passed after one separate import-path repair; decomposition error `9.856645908319e-10`, residual `1.103789726287e-09`, exact inventory, RNG neutrality, excluded-update parity, and reference Nesterov ordering passed.
- Accuracy-blind paired preflight: failed operationally after its one permitted repair; no complete timing vector was produced.
- Exactly one metric launch: skipped because the decisive preflight did not pass.
- Primary metric condition: skipped because no metric run was authorized.

### Informational Metrics

- No metric-run informational values exist; `best_test_acc` is `NaN` for this failed leaf.

## Errors & Dead Ends

### 2026-08-06 — Temporary smoke harness could not import repository module
- Error: `ModuleNotFoundError: No module named 'train'`
- Root cause: Python initialized `sys.path` from `/tmp`, the temporary script's directory, rather than adding the current repository root.
- Source: `/tmp/exp017_gc_smoke.py` initial invocation, before CUDA/model construction
- Do NOT retry: Do not launch repository-importing `/tmp` harnesses without inserting the fixed repository root into `sys.path`.

### 2026-08-06 — Preflight audit sample-count assertion was off by one
- Error: `AssertionError: audit["samples"] == 2` after the 1,024-step trace
- Root cause: The harness counted periodic steps 512 and 1,024 but omitted the separately mandated first-step audit.
- Source: `/tmp/exp017_preflight.log` L1-L10 before rerun
- Do NOT retry: For cadence-512 one-based auditing, use `1 + floor((steps - 1) / 512)`, which equals three at 1,024 steps.

### 2026-08-06 — Repaired preflight retained device loss scalars during allocation check
- Error: `AssertionError: final_allocation == baseline_allocation` after the repaired 1,024-step trace
- Root cause: The temporary harness appended each detached GPU loss scalar to a 1,024-element list before checking allocation stability; this retained device storage and invalidated the check independently of candidate state.
- Source: `/tmp/exp017_preflight.log` L1-L10 after rerun; `/tmp/exp017_preflight.py` `production_trace`
- Do NOT retry: Do not retain per-step device tensors in allocation-stability harnesses; reduce finiteness into a device scalar or check/release values before taking the baseline. The preregistered single repair was already consumed, so EXP017 receives no further preflight or metric attempt.

## Human Notes

> Autopilot execution; no human intervention during this phase.
