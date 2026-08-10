# EXP-018: Direct Canonical Lookahead on EXP002

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-018
- **Base Node**: 002
- **Commit**: d88242f
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented fixed `k=5`, `alpha=0.5` parameter-only Lookahead in `train.py`. All 44 trainable FP32 parameters receive one detached slow copy; every fifth completed inherited SGD/Nesterov update interpolates slow toward fast and copies slow back while retaining momentum and excluding all BatchNorm buffers. Sparse cadence-128-sync fixed device scalars audit normalized slow-fast distance and interpolation displacement inside charged time. Evaluation always uses slow parameters: synchronized boundaries evaluate live state, while any unsynchronized boundary uses a preallocated fast snapshot and exception-safe exact restoration. Final output adds exact cadence/path/evaluation reconciliation and final-16 stability context without changing inherited summary keys.

### Surprises & Discoveries

The reviewed plan initially relied on the inherited 195-batch epoch being divisible by five. Claude's adversarial plan review correctly identified that this could silently expose fast weights if loader length or an evaluation boundary changed. The implementation therefore tests `step % LOOKAHEAD_K` at every evaluation and swaps on every unsynchronized boundary, not merely the expected final partial epoch. The pinned Torch 2.9.1 build provides the required heterogeneous `torch._foreach_lerp_` and `torch._foreach_copy_` operations. The deterministic GPU smoke passed with 44 tensors, 2,748,890 elements, exact step-5 synchronization, retained momentum, RNG-neutral state copies, BN-buffer exclusion, and BF16/channels-last gradients.

### Decisions

`completed_step` is defined explicitly as the inherited pre-increment `step + 1`; sync `i` must occur at completed step `5*i`, making off-by-one cadence failures abort-grade. Audit distance is sampled at sync 1 and every 128th sync to retain a mechanism signal without placing 44 reductions on every fifth charged update. A second preallocated parameter-sized restore bank is used only to make unsynchronized evaluation exception-safe and bitwise auditable; it never belongs to the optimizer or model state. The preflight imports the same production Lookahead helpers instead of reimplementing their numerical path.

## Experimental Adjustments

- **Generalized slow-weight evaluation to every unsynchronized boundary**: removes the reviewed silent-fast-evaluation path while preserving the chosen slow-weight semantics. (ref: `02-plan-review.md` concern 1)
- **Pinned post-update cadence and projection formulas**: `completed_step=step+1`, first sync 5, exact last-sync reconciliation, and explicit dose/total-runtime equations remove reviewed indexing and feasibility ambiguity. (ref: `02-plan-review.md` concerns 2-4)

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 2493272 (uv PID 2493273; unified exec session 72585)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 14:52:11 UTC
- **Ended**: 2026-08-06 14:59:42 UTC

Description:
- One fixed-seed local GPU-0 run will test canonical parameter-only Lookahead against EXP002's 95.23% parent after the preregistered accuracy-blind preflight passes. The intervention retains the inherited model, CutMix/drop-path stream, SGD/Nesterov state, learning-rate schedule, 300-second charged budget, and single evaluation per epoch. Formal improvement requires a complete integrity-valid run with `best_test_acc >=95.33%`; achieved dose, slow-fast distance, and the final-16 plateau are interpretation context only.

Observations:
- Syntax, Ruff, and diff checks passed with only `train.py` modified. The deterministic physical-GPU-0 smoke emitted `SMOKE_PASS torch=2.9.1+cu128 tensors=44 elements=2748890 first_sync=5 last_sync=10 full_loss=2.538086`. (source: `/tmp/exp018_lookahead_smoke.py` invocation, 2026-08-06 UTC)
- The first complete decisive preflight passed without repair or metric access: median candidate/parent ratio 1.005970, maximum 1.016019, MAD/median 0.009195, parent drift 0.007977, projected 27,784 steps and 143 epochs, and projected total 465.948 seconds. The 1,056-step trace produced 211 exact syncs and two audit samples; allocation was unchanged at 125,040,640 bytes and evaluator calls were zero. (source: `/tmp/exp018_preflight.log` L1)
- The sole metric log began writing immediately and confirmed CUDA, the unchanged 2,748,890-parameter WRN/CutMix configuration, fixed Lookahead `k=5`, `alpha=0.5`, cadence-128 audits, and the exact 44-tensor/2,748,890-element slow inventory. (source: `run.log` L1-L6, checked 2026-08-06 14:52:17 UTC)
- The only metric process exited 0 with no traceback, CUDA/OOM, assertion, NaN, or Inf match. It completed 300.0 charged seconds and all 146 epoch evaluations in 450.1 total seconds. (source: unified exec session 72585 exit 0; `run.log` L298-L314)
- Lookahead was active and exactly reconciled: 5,668 of 5,668 expected syncs from step 5 through 28,340, split 2,030 early-CutMix, 2,154 early-clean, and 1,484 late-clean. All 145 synchronized evaluations plus the one unsynchronized final evaluation were classified; the final fast restore was exact. (source: `run.log` L300-L302)
- The final-16 slow-weight plateau was 94.7219% mean, 94.61-94.82% range, 94.69% final, with a 0.0981-point best premium. Canonical feedback therefore produced a stable but materially lower solution than EXP002, rather than a noisy isolated miss. (source: `run.log` L303-L307)

Key Metrics:
- `best_test_acc`: 94.82%, -0.41 points versus EXP002 and -0.51 below the 95.33% formal threshold (source: `run.log` L305; parent `tree.sh show 002`)
- `final_test_acc`: 94.69%; `final_test_loss`: 0.2246 (source: `run.log` L306-L307)
- `training_seconds`: 300.0; `total_seconds`: 450.1; `startup_seconds`: 1.0 (source: `run.log` L308-L310)
- `num_steps`: 28,341 across 146 epochs; `num_params`: 2,748,890 (source: `run.log` L312-L314)
- `peak_vram_mb`: 1,202.1 MiB; CutMix exposure 10,380/20,920 = 0.4962 (source: `run.log` L299, L311)
- Lookahead mechanism: 5,668 exact syncs, 45 sparse audits, normalized pre-sync distance mean 0.01327042 / max 0.02369419, cumulative audited displacement 18.36242593, final normalized gap 0.00001917, zero audit nonfinite tensors, and zero restore failures (source: `run.log` L300-L302)

## Verification Results

### Conditions Checked

- **Execution integrity and frozen budget - PASS**: the sole run exited 0, charged training was 300.0 seconds, total runtime 450.1 seconds, all 146 epochs were evaluated once, all inherited summary keys were present, model size stayed 2,748,890, scope remained only `train.py`, and all Lookahead cadence/path/evaluation/finiteness/restoration assertions passed. (source: unified exec session 72585; `run.log` L298-L314; `git diff --name-only a36dc09`)
- **Parent-relative primary metric - FAIL**: parent EXP002 is 95.23%, so the necessary threshold is 95.33%; EXP018 reached only 94.82%, a -0.41-point delta versus parent. This is a valid research failure and verification stops here. (source: `tree.sh show ... 002`; `run.log` L305)
- **Global-frontier context - skipped after necessary-condition failure**: 95.53, 95.61, and 95.71 context thresholds were not reached in any case. (source: `run.log` L305)

### Informational Metrics

Not promoted as verification outputs because the primary necessary condition failed. Exact run values and mechanism diagnostics are preserved under Run 1 Key Metrics above for analysis.

## Errors & Dead Ends

No infrastructure or code errors occurred. The completed metric result is a research failure: canonical direct Lookahead on EXP002 reduced `best_test_acc` from 95.23% to 94.82% despite higher step exposure and exact mechanism integrity; it must not be metric-retried or tuned within EXP018.

## Human Notes

> Autopilot session; no user intervention during implementation or feasibility checks.
