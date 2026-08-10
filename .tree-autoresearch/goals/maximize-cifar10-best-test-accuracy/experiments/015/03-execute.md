# EXP-015: Confidence-attenuating soft-target Poly-1

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-015
- **Base Node**: 011
- **Commit**: `9e1be61`
- **Outcome**: failed

## Implementation Notes

### Summary

Added a fixed `epsilon=-0.25` hard/soft-target Poly-1 helper while preserving the parent's fused CE. Ordinary and CutMix optimizer paths use Poly-1; scheduled SAM retains parent CE for its base perturbation pass and uses hard Poly-1 only for the perturbed optimizer-gradient pass. A small loss-audit object owns counters at the actual loss methods, with terminal reconciliation, and evaluation progress plus terminal debiased train loss are reported outside charged work.

### Surprises & Discoveries

The first analytic smoke showed that an unconditional FP32 softmax prevents the planned FP64 closed-form check from reaching `1e-10` tolerance even though displayed values match. The helper now preserves FP64/FP32 inputs and promotes only BF16/FP16 production logits to FP32; production semantics remain exactly the planned FP32 probability path.

### Decisions

Kept all heavy gradient and probability diagnostics in transient harnesses. Production performs only the required softmax/gathers and integer counter increments, avoiding diagnostic kernels or synchronizations that could consume fixed-time exposure.

## Experimental Adjustments

- **Preserve FP64 only for analytic inputs**: promote BF16/FP16 logits to FP32 in production but do not downcast FP64 test fixtures. This resolved a pre-GPU formula-test precision mismatch without changing the fixed loss. (ref: initial CPU formula smoke)

## Run Log

### Run 1

Metadata:
- **Job ID**: local process (unified session 99535)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`; `/tmp/exp015_*` checks
- **WandB**: N/A
- **Status**: completed metric; verification failed
- **Started**: 2026-08-06 11:02:33 UTC
- **Ended**: 2026-08-06 11:09:55 UTC

Description:
- Run the fixed negative Poly-1 package only after exact formula/state audits and the first complete accuracy-blind paired GPU-0 gate pass. The sole metric process will use seed 42 and the inherited 300-second charged budget. No coefficient, phase, SAM-policy, timing, or metric retry is permitted.

Observations:
- CPU formula checkpoint passed exact hard/soft FP64 gradients, loss nonnegativity, counter identities, unchanged 2,748,890 parameters, compile, diff check, and `train.py`-only tracked scope.
- Physical-GPU-0 integration passed at 610.284 MiB candidate-only peak with finite batch-256 gradients, CE ascent 2.458984, Poly descent 2.359073, exact SAM restore, one BN update, and 30 EMA samples split 15/15. EMA success and injected-exception paths restored state/modes exactly with two checks and zero failures.
- The single complete gate passed without rerun: parent drift 0.001291, ratio dispersion 0.000452, median ratio 1.004224, maximum 1.004677, projected 25,689.50 steps/131 epochs/159.33 EMA samples, and projected total 449.79 seconds. No evaluator or test loader was used.
- The sole metric process exited 0 on physical GPU 0 with a complete summary after 300.0 charged and 438.8 total seconds. All 133 epochs had exactly one evaluation, loss counters reconciled, and SAM/EMA audits had zero restoration, coverage, nonfinite, or RNG failures (source: `run.log` lines 272-292).
- Formal accuracy failed: best 95.34% is 0.27 points below parent 95.61 and 0.37 below the required 95.71. Final accuracy/loss were 95.16%/0.1645 (source: `run.log` lines 282-284).
- Final-16 EMA values were `95.31,95.31,95.34,95.30,95.30,95.32,95.27,95.23,95.23,95.20,95.18,95.15,95.15,95.15,95.15,95.16`, mean 95.234375, range 95.15-95.34, final 95.16, over progress 0.857828-1.000000. The best-minus-tail premium was 0.105625 points.
- Claude Opus independently returned PASS, recomputed every gate/counter/cadence/tail statistic, and confirmed a trustworthy `no-improvement` classification at metric 95.34 (source: `03-result-review.md`).

Key Metrics:
- parent weighted rounds: 12.620640, 12.631604, 12.636951, 12.631046, 12.624291 ms (source: `/tmp/exp015_preflight.log` `PREFLIGHT_JSON`)
- candidate weighted rounds: 12.674479, 12.677786, 12.681904, 12.684394, 12.683336 ms (source: `/tmp/exp015_preflight.log` `PREFLIGHT_JSON`)
- paired ratios: 1.0042659, 1.0036561, 1.0035573, 1.0042236, 1.0046771 (source: `/tmp/exp015_preflight.log` `PREFLIGHT_JSON`)
- candidate-only peak 610.284 MiB; informational joint-process peak 705.664 MiB (source: `/tmp/exp015_gpu_smoke.log`; `/tmp/exp015_preflight.log`)
- trace step-200 parent/candidate loss 1.078941/0.994572 and gradient L2 1.155214/1.539534, report-only (source: `/tmp/exp015_preflight.log`)
- metric summary: best/final/loss `95.34%/95.16%/0.1645`; charged/total/startup `300.0/438.8/1.0s`; peak `1222.4 MiB`; epochs/steps/params `133/25820/2748890`; terminal train loss EMA `0.009284` (source: `run.log` lines 282-292)
- mechanism dose: CutMix `10362/20884`; SAM `2468/4936` from progress 0.7500; EMA 159 samples split 79/80, 107 live + 26 EMA evals, 26 exact restores (source: `run.log` lines 272-278)
- loss calls: ordinary Poly 12,990; CutMix Poly 10,362; SAM ascent CE 2,468; SAM descent Poly 2,468; 25,820 Poly and 28,288 total calls (source: `run.log` lines 279-280)

## Verification Results

### Conditions Checked

- **Physical GPU, source scope, run integrity**: PASS - physical/visible GPU 0 UUID matched NVIDIA H20; exit 0; only `train.py` changed; charged 300.0s, total 438.8s, complete summary and clean audits.
- **Primary metric**: FAIL - 95.34% `<95.71%`; verification stopped at the first failed necessary result condition.
- **Stable mechanism support**: informationally FAIL - full dose passed (25,820 steps, 159 EMA samples, parity 79/80), but final-16 mean 95.234375 `<95.69` and also below parent 95.493125.

### Informational Metrics

- Negative Poly-1 preserved full optimizer/EMA exposure but reduced the stable EMA tail by 0.258750 points versus the historical parent and increased final loss from parent 0.1552 to 0.1645.
- The terminal training-loss EMA is under the shifted Poly objective and is not numerically comparable to parent CE; final test loss remains comparable because evaluation CE is frozen.

## Errors & Dead Ends

### 2026-08-06 - analytic helper downcast limited FP64 tolerance
- Error: hard-gradient difference exceeded `rtol=1e-10` despite matching at displayed precision.
- Root cause: the helper forced its probability path to FP32 for every input dtype, including FP64 analytic fixtures.
- Source: initial CPU formula smoke before any GPU command or numeric gate.
- Do NOT retry: preserve FP64 inputs for analytic tests while promoting only BF16/FP16 production logits to FP32.

## Human Notes

> Autopilot session; no execution-phase intervention.
