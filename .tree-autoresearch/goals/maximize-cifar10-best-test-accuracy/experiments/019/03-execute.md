# EXP-019: Corrected Reference-Ordered Gradient Centralization

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-019
- **Base Node**: 002
- **Commit**: a103071
- **Outcome**: failed

## Implementation Notes

### Summary

Reapplied EXP017's already-proven scientific mechanism to the clean EXP002 branch, modifying only `train.py`. The implementation inventories all 44 parameter tensors, externally adds coupled `1e-4` L2 to every gradient, centralizes exactly 16 convolution and one classifier regularized directions per output row, and then calls unchanged PyTorch momentum/Nesterov with optimizer-internal decay disabled. GC runs on all steps inside charged time. Sparse cadence-512 FP64 fixed device scalars record total/conv/classifier regularized and removed energy, centralized energy, residual, decomposition, and finiteness; final output adds exact path/inventory/final-16 context while preserving inherited evaluation and summary keys.

### Surprises & Discoveries

Claude's plan review incorrectly suspected that importing `train.py` would launch training; the existing `if __name__ == "__main__": main()` guard already prevents that, although module import does construct the frozen evaluator/test loader before it is replaced by an accuracy guard. The installed Torch 2.9.1 has no `_foreach_mean`, so the 17 heterogeneous reductions remain explicit while their broadcast subtraction is fused through `_foreach_sub_`. The first deterministic smoke exposed a temporary-harness logic mistake: comparing excluded bias trajectories over two loss-derived steps was invalid because GC-altered eligible weights changed the second raw data gradient. Injecting identical raw gradients per step corrected the test without changing production code or the decisive-preflight repair ledger.

### Decisions

Production logic intentionally matches EXP017 rather than introducing an implementation fallback: one foreach coupled-decay add, 17 means, one foreach subtraction, unchanged SGD state. Audit tensors are all allocated before training and only scalar values persist across steps; transient row means are explicitly deleted. Final integrity is printed before a hard failure, so any accounting or numerical defect remains visible in `run.log`. The decisive preflight is separately preregistered and has not yet consumed its one permitted pre-vector repair.

## Experimental Adjustments

- **Corrected smoke excluded-parity inputs**: supplied identical raw gradients on both checked steps, separating effective-decay parity from intended eligible-trajectory divergence. This was a smoke-only repair before decisive preflight and did not alter `train.py`. (ref: initial `/tmp/exp019_gc_smoke.py` assertion, 2026-08-06 UTC)
- **Consumed the sole decisive-preflight repair on reference arithmetic**: replaced an out-of-place `raw + decay*parameter` expectation with clone plus in-place `add_`/`sub_`, matching production's FP32 operation order exactly. The first attempt stopped before timing, allocation, JSON, or metrics; production code and gates are unchanged. (ref: `/tmp/exp019_preflight.log` initial traceback at line 239)

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 2565815 (uv PID 2565816; unified exec session 39130)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 15:31:19 UTC
- **Ended**: 2026-08-06 15:38:49 UTC

Description:
- One fixed-seed physical-GPU-0 run will test official-order full-run Gradient Centralization against EXP002's 95.23% parent after the corrected accuracy-blind preflight passes. The mechanism preserves the inherited WRN, CutMix/drop-path streams, learning-rate path, Nesterov state machine, 300-second charged budget, and one evaluation per epoch. Formal improvement requires `best_test_acc >=95.33%`; a complete valid miss is definitive negative evidence for this exact composition.

Observations:
- Scope, Ruff, syntax, and diff checks passed with only `train.py` modified. The corrected deterministic GPU smoke emitted `SMOKE_PASS torch=2.9.1+cu128 tensors=44 eligible=17 excluded=27 elements=2745264 rows=2266 decomposition=2.059e-10 residual=1.666e-09 truncated_calls=7`. (source: `/tmp/exp019_gc_smoke.py` invocation, 2026-08-06 UTC)
- After the single permitted pre-vector reference-arithmetic repair, the decisive preflight passed: median candidate/parent ratio 1.007687, maximum 1.014301, MAD/median 0.006098, parent drift 0.004559, projected 27,736 steps / 143 epochs, and informational total 465.948 seconds. The 1,056-step trace made 1,056 GC calls and three audits; final live allocation grew 655,360 bytes (below 1 MiB), reserved allocation stayed 734,003,200 bytes, and evaluator calls were zero. (source: `/tmp/exp019_preflight.log` L1)
- The sole metric log began writing immediately and confirmed CUDA, unchanged WRN/CutMix settings, official regularized-direction GC with audit cadence 512, optimizer-internal decay zero, and exact 17/27 tensor plus 2,745,264/3,626 element reconciliation. (source: `run.log` L1-L6, checked 2026-08-06 15:31:28 UTC)
- The only metric process exited 0 with no traceback, CUDA/OOM, assertion, NaN, Inf, or GC-integrity failure. It completed 300.0 charged seconds, 27,976 steps, and all 144 epoch evaluations in 449.8 total seconds. (source: unified exec session 39130 exit 0; `run.log` L295-L312)
- GC reconciled exactly: 27,976 calls split into 10,261 CutMix, 10,416 early-clean, and 7,299 late-clean steps, with 55 correct one-based audits. Decomposition error was `4.0005e-09`, maximum row-mean residual `3.5183e-09`, and final/audited nonfinite counts were zero. (source: `run.log` L296-L300)
- GC removed 6.4162 of 29.1016 audited regularized squared-energy units: energy fraction 0.22047 and norm ratio 0.46955. Convolution norm ratio was 0.41889 and classifier norm ratio 0.93213, so the intervention was large rather than redundant. The final-16 plateau was 94.966875% mean, 94.82-95.07% range, and 95.02% final. (source: `run.log` L297-L301)

Key Metrics:
- `best_test_acc`: 95.07%, -0.16 points versus EXP002 and -0.26 below the 95.33% formal threshold (source: `run.log` L303; parent `tree.sh show 002`)
- `final_test_acc`: 95.02%; `final_test_loss`: 0.2046 (source: `run.log` L304-L305)
- `training_seconds`: 300.0; `total_seconds`: 449.8; `startup_seconds`: 1.3 (source: `run.log` L306-L308)
- `num_steps`: 27,976 across 144 epochs; `num_params`: 2,748,890; `peak_vram_mb`: 1,180.0 MiB (source: `run.log` L309-L312)
- CutMix exposure: 10,261/20,677 = 0.4963 (source: `run.log` L295)
- GC mechanism: 27,976 calls, 55 audits, removed energy fraction 0.22047 / norm ratio 0.46955, conv norm ratio 0.41889, classifier norm ratio 0.93213, decomposition error `4.0005e-09`, residual `3.5183e-09`, zero nonfinite counts (source: `run.log` L296-L300)
- Stability: final-16 mean 94.966875%, range 94.82-95.07%, final 95.02%, best premium 0.103125 points (source: `run.log` L301)

## Verification Results

### Conditions Checked

- **Execution integrity and frozen budget - PASS**: the sole run exited 0, charged training was 300.0 seconds, total runtime 449.8 seconds, all 144 epochs were evaluated once, inherited summary/model/configuration remained intact, only `train.py` changed, and all GC inventory/path/audit/numerical/finiteness conditions passed. (source: unified exec session 39130; `run.log` L294-L312; `git diff --name-only a36dc09`)
- **Parent-relative primary metric - FAIL**: parent EXP002 is 95.23%, so the necessary threshold is 95.33%; EXP019 reached 95.07%, a -0.16-point delta versus parent. This complete valid result definitively rejects the exact official-order full-run GC composition. (source: `tree.sh show ... 002`; `run.log` L303)
- **Global-frontier context - skipped after necessary-condition failure**: 95.53, 95.61, and 95.71 were not reached. (source: `run.log` L303)

### Informational Metrics

Not promoted as verification outputs because the primary necessary condition failed. Exact summary, stability, exposure, and mechanism values are preserved under Run 1 Key Metrics above for analysis.

## Errors & Dead Ends

### 2026-08-06 — Smoke compared excluded trajectories after eligible divergence
- Error: `AssertionError: torch.equal(candidate_model.bias, parent_model.bias)` on smoke step 2
- Root cause: the first GC update intentionally changed eligible weights, so the second loss-derived raw bias gradient was no longer shared; the test confounded data-gradient divergence with excluded decay/update parity.
- Source: initial `/tmp/exp019_gc_smoke.py` line 173 invocation, before decisive preflight
- Do NOT retry: compare excluded updates under explicitly identical raw gradients when eligible trajectories have already diverged.

### 2026-08-06 — Preflight reference used a different FP32 addition form
- Error: `AssertionError: torch.equal(candidate_parameter.grad, expected)` before the long trace
- Root cause: the harness expectation used out-of-place multiply/add while production used in-place foreach `add_`; mathematically equivalent FP32 operation forms need not be bitwise equal.
- Source: `/tmp/exp019_preflight.log` initial traceback at line 239; no JSON vector emitted
- Do NOT retry: bitwise reference checks must clone raw gradients and reproduce production's in-place add/sub ordering exactly.

No production code or infrastructure errors occurred. The metric outcome is a research failure: official-order full-run GC materially removed eligible regularized-gradient energy but reduced `best_test_acc` by 0.16 points versus EXP002; it must not be metric-retried, narrowed, or tuned within EXP019.

## Human Notes

> Autopilot session; no user intervention during implementation or feasibility checks.
