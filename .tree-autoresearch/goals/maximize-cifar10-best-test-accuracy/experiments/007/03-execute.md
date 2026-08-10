# EXP-007: Literature-Scale ASAM in the Validated Clean Tail

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-007
- **Base Node**: 004
- **Commit**: `428cefd`
- **Outcome**: failed - valid run missed the formal metric threshold

## Implementation Notes

### Summary

Replaced only EXP-004's late Euclidean SAM perturbation helper with named-parameter p=2 ASAM using fixed `rho=0.5` and `eta=0.01`. The helper snapshots all 44 parameters, derives 30 non-bias scales from complete snapshots, keeps 14 bias scales at one, constructs `rho*s^2*g/||s*g||` with foreach operations, and restores through the inherited exception-safe two-pass path. First-pulse production diagnostics reduce actual epsilon buffers to adaptive radius, normalized maximum, Euclidean norm, maximum scale, and conv/BatchNorm/classifier/bias energy shares. Parent data, architecture, CutMix, global RNG/drop path, BatchNorm handling, optimizer, schedule, evaluator, and summary remain unchanged.

### Surprises & Discoveries

- A representative full-WRN GPU pulse was nondegenerate (`radius=0.500000`, Euclidean norm `0.390258`) but its epsilon energy concentrated in biases (53.94%) and BatchNorm weights (30.82%), with 15.13% in convolution weights and 0.11% in `fc.weight`. This validates Claude's request for group diagnostics; the fixed literature package remains unchanged.
- ASAM hot-path overhead was smaller than expected: median pulse latency was 20.0858 ms versus 20.0628 ms for actual-parent SAM, only 1.0011x.
- Two initial FP64 harness failures were harness mistakes: the bias discriminator incorrectly expected unit-scale bias to differ from the one-scale formula, then the expectation computed `D` in FP64 although the plan requires FP32. Neither required an implementation change.

### Decisions

- Actual epsilon-derived geometry is checked once on the first production pulse and materialized immediately. Later pulses run the identical foreach path and check finite positive `D`, avoiding repeated normalization buffers while retaining a discriminating production audit.
- Bias scales are initialized once and stay at one; only non-bias scale buffers are refreshed from snapshots per pulse.
- Group shares are attribution diagnostics only. Their concentration cannot abort, tune, or trigger a rerun.

## Experimental Adjustments

- None. The implementation uses the Claude-reviewed fixed ASAM package and cadence without scalar or group changes.

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 588335 (timeout wrapper; execution session 52831)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 15:41:10 UTC
- **Ended**: 2026-08-05 15:48:54 UTC

Description:
- One fixed-seed physical-GPU-0 run replacing only EXP-004's sparse late SAM geometry with literature-scale p=2 ASAM. Full CutMix, independent images, period-two second-pass cadence, and evaluator behavior remain fixed. Formal success begins at 95.50%; the preregistered mechanism-sized target is 95.70% with at least 25,000 steps. Intermediate accuracy cannot alter execution.

Observations:
- Static checks passed: `py_compile`, Ruff, and `git diff --check` exited 0; only tracked `train.py` differs from parent `1a8d0de`.
- FP64 geometry, one-scale discrimination, invalid-gradient rejection, partial-add restoration, RNG isolation, exact 44/30/14 tensor inventory, and bitwise parent initialization checks passed.
- GPU-0 actual-parent CutMix/clean paths matched bitwise for outputs, gradients, optimizer, BatchNorm, and RNG. Scheduled ASAM passed actual radius, RNG replay, one BatchNorm update, exact restore, and one optimizer update.
- Latency gate passed: parent median/p90 20.0628/20.2317 ms; ASAM 20.0858/20.1514 ms; weighted projected horizon approximately 25,557 steps. Candidate peak in the isolated smoke was 608.9 MiB versus parent 587.9 MiB.
- Prelaunch GPU 0 was H20 97,871 MiB with 94,117 MiB free and 0% utilization. One unrelated process held 3,384 MiB; no process from this repository remained after benchmarks.
- Run 1 initialized successfully on CUDA with 2,748,890 parameters, the exact ASAM rho/eta/start/period config, a 300-second budget, and 195 batches per epoch (source: `run.log` startup lines).
- First production ASAM geometry passed at step 20,636/progress 0.7501: actual radius 0.500000, normalized maximum 0.061053, Euclidean norm 0.450053 within the 0.990419 upper bound, maximum scale 1.976884, and both group-share sets summed to one (source: `run.log` activation output and line 273).
- Run 1 exited 0 with no traceback, OOM, nonfinite value, geometry violation, restoration failure, or overlap. All discriminating and structural protocol gates passed before accuracy was read (source: `run.log` lines 271-284 and protocol parser output).
- The single primary metric read returned `best_test_acc=95.34%`, below the formal 95.50% gate and 95.70% mechanism target. The valid result is final with no retry (source: `run.log` line 275).
- The printed CutMix last progress rounded a structurally strict `<0.75` value to `0.7500` at four decimals. Verification accepted the structural cutoff with this documented precision limitation; the discriminating ASAM gates were unaffected.

Key Metrics:
- Primary/final accuracy: 95.34% / 95.18%; final loss 0.1550 (source: `run.log` lines 275-277).
- Runtime: 300.0 charged seconds, 463.6 total seconds, 1.1 startup seconds; peak VRAM 1,213.3 MiB (source: `run.log` lines 278-281).
- Work: 132 epochs/evaluations, 25,575 optimizer steps, 2,748,890 parameters (source: `run.log` lines 270, 282-284; eval count checked programmatically as 132).
- CutMix: 10,237/20,634, ratio 0.4961, last progress printed 0.7500; ASAM: 2,470/4,941, ratio 0.4999, first step 20,636/progress 0.7501 (source: `run.log` lines 271-272).
- ASAM geometry: denominator min/mean/max 0.005270/0.071997/0.332359; first radius 0.500000; normalized max 0.061053; Euclidean norm 0.450053; max scale 1.976884; zero failures (source: `run.log` line 273).
- First-pulse denominator shares conv/BN/fc/bias: 0.141254/0.239253/0.011648/0.607845; epsilon shares: 0.137905/0.097499/0.014345/0.750251 (source: `run.log` line 273).

## Verification Results

### Conditions Checked

- Prelaunch scope, static, geometry, parent parity, GPU integration, and latency/exposure gates: pass; detailed values are in Run 1 observations.
- Full-run discriminating ASAM gates: pass. Actual epsilon radius, normalized maximum, Euclidean norm, group sums, coverage, finite denominator, and all failure counters met preregistered requirements (source: `run.log` line 273).
- Full-run structural protocol: pass with documented four-decimal CutMix boundary rounding. Exit 0, 300.0 charged seconds, 463.6 total seconds, 25,575 steps, 132/132 evaluations, exact parameter count, and unique summary all passed (source: `run.log` lines 271-284).
- Necessary metric improvement: fail. Parent 95.40% plus 0.10 requires 95.50%; observed 95.34%, delta -0.06. The separate 95.70 mechanism-sized target also failed (source: tree node 004 and `run.log` line 275).

### Informational Metrics

- Final test accuracy/loss: 95.18% / 0.1550 versus parent 95.40% / 0.1654.
- Steps: 25,575 versus parent 25,560; total runtime 463.6s versus 457.3s; peak VRAM 1,213.3 MiB versus 1,190.5 MiB.

## Errors & Dead Ends

- Three corrected verification-parser assertions were harness-only mistakes and did not alter experiment code or configuration: unit-scale bias was initially expected to distinguish the one-scale formula, FP64 toy `D` was compared to the required FP32 reduction, and `nonfinite_failures=0` matched an over-broad error regex.

## Human Notes

> User requires physical GPU 0 and Claude adversarial review with no fallback. Claude completed both idea and plan reviews for EXP-007.
