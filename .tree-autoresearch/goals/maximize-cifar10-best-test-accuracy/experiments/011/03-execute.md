# EXP-011: Cadence-31 charged-time EMA

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-011
- **Base Node**: 004
- **Commit**: d68f73a
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented a full-state charged-time EMA in `train.py` without changing the EXP-004 online model, data, optimizer, CutMix, or SAM path. The new state machine initializes at the first cadence-31 post-optimizer sample after 75% charged progress, derives decay from step-entry charged-time intervals and the 18.75-second half-life, averages floating parameters and persistent buffers, copies integer buffers, and accumulates consecutive parameter distances before the existing charged synchronization. Evaluation remains exactly once per epoch and routes to the live model before EMA activation or an exception-safe EMA swap afterward; the terminal summary includes state, dose, decay, evaluation, restoration, RNG, distance, and BatchNorm audits.

### Surprises & Discoveries

- The first CPU swap smoke exposed that `torch._foreach_copy_` into leaf parameters must run under an explicit no-grad context even though the shadow update method already had one. This was fixed by decorating the evaluation swap/restore method itself.
- The frozen evaluator calls `model.eval()` and may create a DataLoader iterator, so RNG invariance is checked separately around the EMA swap and restore rather than across evaluator work.

### Decisions

- Runtime RNG assertions bracket every charged EMA update and separately bracket evaluation swap and restore operations. This makes the no-RNG claim executable without assuming the frozen evaluator is RNG-neutral.
- Full-state restoration is independently verified through a fresh `state_dict(keep_vars=True)` enumeration, while optimizer parameter and momentum-buffer identities are captured and compared around every EMA evaluation.
- Nonfinite checks that require scalar synchronization are deferred to evaluation or the post-budget audit; consecutive distance work itself stays asynchronous and charged.
- Claude's pre-run adversarial audit found no blocking issue. Its sole recommended code adjustment preserves the complete terminal summary on a final-audit failure before raising a nonzero exit, improving invalid-run evidence without changing a successful path.

## Experimental Adjustments

- **Decorated EMA evaluation with `torch.no_grad()`**: Required for legal in-place swap/restore copies into leaf parameters; no experimental configuration or online training semantics changed. (ref: Run 1 failure and error entry)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local CPU smoke)
- **Log file(s)**: terminal capture; harness `/tmp/exp011_cpu_smoke.py`
- **WandB**: N/A
- **Status**: failed
- **Started**: 2026-08-06 04:11 UTC
- **Ended**: 2026-08-06 04:12 UTC

Description:
- Deterministic CPU smoke for full-state inventory, first-copy and half-life arithmetic, integer-buffer copying, parity, RNG preservation, optimizer exclusion, exact successful and exceptional evaluation restoration, and fixed-seed initialization parity with parent commit `1a8d0de`. This gate queries no accuracy and is run before GPU feasibility work. The first attempt was expected to validate the newly integrated swap path.

Observations:
- EMA construction, updates, and arithmetic reached the evaluation stage, where the first swap failed before invoking the fake evaluator because leaf parameters were modified outside no-grad. The `finally` restore encountered the same error. (source: terminal traceback, `/tmp/exp011_cpu_smoke.py` line 121; candidate `train.py` original evaluation copy sites)

Key Metrics:
- evaluator calls before failure: 0 (source: terminal traceback)

### Run 2

Metadata:
- **Job ID**: N/A (local CPU smoke retry)
- **Log file(s)**: terminal capture; harness `/tmp/exp011_cpu_smoke.py`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 04:14 UTC
- **Ended**: 2026-08-06 04:14 UTC

Description:
- Repeated the identical deterministic CPU smoke after adding the missing no-grad boundary. The retry covers exact time-decay arithmetic under two cadence partitions, full persistent-state inventory, integer-copy behavior, parity, shadow exclusion, successful and exceptional swap restoration, optimizer momentum preservation, RNG parity, and fixed-seed parent/candidate initialization. No accuracy is queried.

Observations:
- All assertions passed, including two exact live-state restoration checks (one after a deliberately raised evaluator exception) and fixed-seed equality across all 83 full-WRN state keys. (source: terminal output `CPU_SMOKE_OK updates=2 restore_checks=2 state_keys=7 full_state_keys=83`)

Key Metrics:
- state-machine updates: 2; restoration checks: 2; tiny state keys: 7; full-WRN state keys: 83 (source: Run 2 terminal output)

### Run 3

Metadata:
- **Job ID**: local GPU preflight
- **Log file(s)**: `/tmp/exp011_preflight.log`
- **WandB**: N/A
- **Status**: failed
- **Started**: 2026-08-06 04:24 UTC
- **Ended**: 2026-08-06 04:25 UTC

Description:
- Accuracy-blind full-WRN GPU-0 correctness and latency preflight using paired parent/candidate BF16 channels-last steps. It included fixed-seed online-state comparisons, ordinary and production-faithful SAM paths, five alternating latency rounds, cadence sampling, synthetic evaluation restoration, and resource projections. The run was designed to stop before emitting gate metrics if any integrity assertion failed.

Observations:
- The run reached the post-training cadence audit and failed because the benchmark's SAM-only timing sequence used nonconsecutive even step IDs, several of which were divisible by 31. Those artificial due steps created a SAM sample skew that the consecutive production schedule cannot create. (source: `/tmp/exp011_preflight.log` traceback at harness line 275)

Key Metrics:
- metric queried: none; gate metrics emitted: none (source: `/tmp/exp011_preflight.log`)

### Run 4

Metadata:
- **Job ID**: local GPU preflight retry
- **Log file(s)**: `/tmp/exp011_preflight.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 04:27 UTC
- **Ended**: 2026-08-06 04:28 UTC

Description:
- Repeated the full accuracy-blind GPU-0 preflight after reserving cadence accounting for an explicit production-parity sequence. The run compared parent and candidate across five alternating BF16/channels-last latency rounds with 200 ordinary and 100 production-faithful SAM steps per arm, then exercised 30 full-WRN cadence samples and a 40-batch synthetic evaluation swap. All online state, optimizer momentum, SAM replay/restoration, BN, RNG, and evaluation restoration assertions remained active.

Observations:
- Physical GPU 0 was the only visible device and reported `NVIDIA H20`, 102,245,335,040 bytes / 97,871 MiB. All integrity checks passed; candidate and parent online model state plus momentum buffers remained bitwise equal after the paired workload. (source: pre-command GPU checks and `/tmp/exp011_preflight.log` terminal `GPU_PREFLIGHT_OK`)
- Parent round drift was 0.006916, below the 0.075 discard threshold, so the first valid measurement is decisive. The candidate latency ratio was 0.999591 and cleared the <=1.02 gate. (source: `/tmp/exp011_preflight.log` `PREFLIGHT_JSON`)
- EMA produced 30 samples split 15 ordinary / 15 SAM, 29 finite nonzero consecutive distances, one exact EMA evaluation restoration, and zero restore, coverage, nonfinite, or RNG failures. (source: `/tmp/exp011_preflight.log` `ema_audit_lines`)

Key Metrics:
- parent weighted median: 11.380019 ms; candidate weighted median: 11.375366 ms; ratio: 0.999591 (source: `/tmp/exp011_preflight.log` `PREFLIGHT_JSON`)
- weighted latency formula: `0.875 * ordinary_median + 0.125 * SAM_median`, matching the 75% ordinary prefix plus the 50/50 ordinary/SAM tail (source: `/tmp/exp011_gpu_preflight.py` round aggregation)
- parent weighted round medians: 11.380019, 11.336501, 11.355050, 11.414900, 11.411166 ms; drift: 0.006916 (source: `/tmp/exp011_preflight.log` `PREFLIGHT_JSON`)
- candidate weighted round medians: 11.375366, 11.334659, 11.353448, 11.394762, 11.408320 ms (source: `/tmp/exp011_preflight.log` `PREFLIGHT_JSON`)
- calibrated projected steps: 25,570; projected epochs: 132; projected total runtime: 458.208 s (source: `/tmp/exp011_preflight.log` `PREFLIGHT_JSON`)
- peak allocated VRAM: 1,277.040 MiB, below the 1.30 GiB gate (source: `/tmp/exp011_preflight.log` `PREFLIGHT_JSON`)
- synthetic 40-batch evaluation forward: 6.001425 s; measured swap/restore/check overhead: 0.006876 s (source: `/tmp/exp011_preflight.log` `PREFLIGHT_JSON`)
- EMA cadence: 30 updates, 15 ordinary, 15 SAM, first/last step 31/930; consecutive L2 min/mean/max 0.01068191/0.43175113/12.02877045 (source: `/tmp/exp011_preflight.log` `PREFLIGHT_JSON`)
- final preflight EMA/live parameter L2: 10.72766781; relative: 0.1362609416; BN mean/variance L2: 32.74259034/57.00185850; BN variance ratio min/mean/max: 0.20207265/0.93249675/1.92811203 (source: `/tmp/exp011_preflight.log` `PREFLIGHT_JSON`)

### Run 5

Metadata:
- **Job ID**: local PID 1421480
- **Log file(s)**: repository-root `run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 04:43 UTC
- **Ended**: 2026-08-06 04:50 UTC

Description:
- The single preregistered fixed-seed accuracy run, launched with `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`. Physical GPU 0 was reconfirmed immediately before launch as the sole visible NVIDIA H20 with 97,871 MiB. The run uses the exact EXP-004 online model/CutMix/SAM package plus cadence-31 charged-time EMA; intermediate accuracy cannot trigger pruning or retry.

Observations:
- Startup completed and printed CUDA device, 2,748,890 parameters, the full fixed configuration including `ema_half_life_s=18.75`, the 300-second budget, and 195 batches per epoch. GPU process activity was present. (source: `run.log` startup lines; local process/GPU checks at 2026-08-06 04:43 UTC)
- The process exited 0 after a complete 300.0-second charged run. It produced 133 evaluations for 133 epochs, split 106 live and 27 EMA, with exactly one source per epoch and 27 successful EMA swaps/restores. No traceback, CUDA/OOM, timeout, NaN/Inf, audit-failure, or runtime-error signature was present. (source: `run.log` lines 1-289; evaluation/error scans)
- The fixed parent mechanisms remained aligned: CutMix applied 10,345/20,857 eligible batches (0.4960), SAM applied 2,471/4,941 (0.5001), and SAM began at step 20,858 and progress 0.7500. Compared with EXP-004's 10,253/20,662 CutMix and 2,449/4,898 SAM counts, the small dose increase follows the greater realized throughput. (source: `run.log` lines 272-273; EXP-004 `03-execute.md`)
- EMA began at the first due cadence step 20,863 and progress 0.7504, ended at step 25,792/progress 0.9997, and accumulated 160 samples split exactly 80 ordinary / 80 SAM over 74.7736 seconds. All 159 consecutive distances were finite and nonzero; restore, coverage, nonfinite, and RNG failure counts were zero. (source: `run.log` lines 274-278)
- The EMA-only tail reached a best 95.61% at epoch 123, with adjacent tail evaluations at 95.53% and 95.54%; final EMA accuracy was 95.46%. Epochs 118-133 ranged 95.44-95.61 with mean 95.493125. The formal best is not a lone spike, but live-tail performance was intentionally not measured, limiting causal attribution. (source: `run.log` epoch-evaluation records 118-133 and summary lines 280-282)
- Claude's post-run adversarial review independently recomputed cadence divisibility, time-decay arithmetic, eligibility counts, evaluation routing, inventory, dose, timing, preflight reconciliation, and line offsets. It found no blocking concern and classified the run as a trustworthy improvement, while rejecting a claim that EMA alone caused the full delta. (source: `03-result-review.md`)

Key Metrics:
- `best_test_acc`: 95.61%, +0.21 points versus EXP-004 at 95.40%; passes the formal 95.50% threshold but not the stronger 95.70% target (source: `run.log` line 280; `tree.sh show ... 004`)
- `final_test_acc`: 95.46%; `final_test_loss`: 0.1552 (source: `run.log` lines 281-282)
- `training_seconds`: 300.0; `total_seconds`: 447.9; `startup_seconds`: 1.1 (source: `run.log` lines 283-285)
- `peak_vram_mb`: 1,222.4; `num_epochs`: 133; `num_steps`: 25,798; `num_params`: 2,748,890 (source: `run.log` lines 286-289)
- CutMix: 10,345/20,857 = 0.4960; SAM: 2,471/4,941 = 0.5001, first step/progress 20,858/0.7500 (source: `run.log` lines 272-273)
- EMA: 160 updates, first/last step 20,863/25,792, first/last progress 0.7504/0.9997, first/last charged time 225.1324/299.9060 s, span 74.7736 s, oldest coefficient 0.063025, parity 80/80 (source: `run.log` line 274)
- EMA interval min/mean/max: 0.457694/0.470274/0.534709 s; decay min/mean/max: 0.980427055/0.982765296/0.983222391 (source: `run.log` line 275)
- evaluation routing: 106 live + 27 EMA = 133; swaps/restoration checks: 27/27; restore/coverage/nonfinite/RNG failures: 0/0/0/0 (source: `run.log` line 276)
- consecutive parameter L2 min/mean/max: 0.03019896/0.36667874/1.30348098; final EMA/live parameter L2: 0.77567250, relative 0.0150632216 (source: `run.log` line 277)
- final EMA/live BN mean/variance L2: 0.30974478/0.33713258; variance ratio min/mean/max: 0.96376991/1.04175885/1.21857619 (source: `run.log` line 277)
- EMA state inventory: 44 parameter tensors / 2,748,890 elements, 26 floating buffers / 3,616 elements, 13 integer buffers / 13 elements (source: `run.log` line 278)

## Verification Results

### Conditions Checked

- **Primary accuracy improvement**: passed. `best_test_acc=95.61%` is 0.21 points above parent EXP-004 at 95.40% and exceeds the required 95.50% threshold. The preregistered stronger 95.70% target was not reached. (source: `run.log` line 280; `tree.sh show ... 004`)
- **Successful bounded execution**: passed. Exit 0, `training_seconds=300.0` in `[299.5,301.0]`, `total_seconds=447.9 < 600`, complete summary, physical GPU 0, and 133 evaluations for 133 epochs. (source: `run.log` lines 1-289; pre-launch GPU check)
- **Fixed mechanism dose**: passed. `num_steps=25,798 >= 25,200`, EMA updates `160 >= 145`, parity difference 0, and all 159 consecutive distances were finite and strictly nonzero. (source: `run.log` lines 274, 277, 288)
- **Online-parent integrity**: passed. CutMix/SAM boundary and ratios match the frozen package, parameters remained 2,748,890, evaluation sources sum exactly to epochs, and restoration/coverage/nonfinite/RNG failures were all zero. (source: `run.log` lines 272-289; parent EXP-004 execution record)
- **Tracked scope**: passed. `git diff --name-only 1a8d0de` returned only `train.py`; `python -m py_compile train.py` and `git diff --check` passed. (source: post-run static checks)
- **Adversarial result integrity**: passed. Claude independently classified 95.61 as a trustworthy formal improvement over 95.40 with no blocking concern or invalidity/dose trigger. (source: `03-result-review.md`)

### Informational Metrics

- `best_test_acc=95.61%`, `final_test_acc=95.46%`, `final_test_loss=0.1552`
- `training_seconds=300.0`, `total_seconds=447.9`, `startup_seconds=1.1`
- `peak_vram_mb=1222.4`, `num_epochs=133`, `num_steps=25798`, `num_params=2748890`
- CutMix `10345/20857=0.4960`; SAM `2471/4941=0.5001`; EMA `160` updates with `80/80` parity
- EMA-tail epochs 118-133: range 95.44-95.61, mean 95.493125; final 95.46

## Errors & Dead Ends

### 2026-08-06 - EMA evaluation copy lacked no-grad
- Error: `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.`
- Root cause: `ChargedTimeEMA.evaluate` copied EMA and restore tensors into trainable parameters without a method-level `torch.no_grad()` context.
- Source: Run 1 terminal traceback at candidate `train.py` evaluation swap and restore copy sites.
- Do NOT retry: do not perform model-state swaps through in-place leaf-parameter operations unless the entire swap/evaluate/restore method is no-grad.

### 2026-08-06 - Preflight used nonproduction cadence IDs
- Error: `AssertionError` on `abs(candidate_ema.ordinary_samples - candidate_ema.sam_samples) <= 1`.
- Root cause: the latency harness's SAM-only block used selected even step IDs rather than a consecutive production sequence; incidental cadence-31 hits were therefore all SAM samples and biased the parity audit.
- Source: Run 3, `/tmp/exp011_preflight.log` traceback at harness line 275.
- Do NOT retry: do not mix selected parity-only benchmark step IDs with production cadence accounting; reserve EMA samples for an explicit consecutive cadence sequence.

## Human Notes

> The user requires Claude to be the sole adversarial reviewer. Claude performed the idea, plan, implementation, and result reviews without fallback.
