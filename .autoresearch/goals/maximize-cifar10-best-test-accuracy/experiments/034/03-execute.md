# EXP-034: Conv2d-Only Kaiming Fan-Out Initialization

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-034
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed — immutable-corpus class-concentration veto; production not authorized

## Implementation Notes

### Summary

Split the existing initializer by module type: Conv2d now uses explicit Kaiming normal fan-out/ReLU, while Linear retains the accepted literal default Kaiming call. No production graph, optimizer, data, schedule, evaluator, timer, logging, or runtime line changed.

### Surprises & Discoveries

The implementation is exactly a three-line initializer split. Its scientific effect is nevertheless concentrated: only the stem and two widening Conv tensors change scale, while their following BatchNorm layers approximately cancel forward scale but not relative SGD updates.

### Decisions

No production diagnostics or timing campaign were added because initialization precedes the training timer and leaves recurring operations identical. Exact tensor/RNG proofs and byte-identical trajectory instrumentation remain isolated in the ignored preflight controller.

## Experimental Adjustments

- **Restore the temporary accepted initializer as a staticmethod**: the first controller attempt restored a raw function descriptor and caused candidate construction to receive an extra bound argument. Re-wrapping the unchanged saved function fixes only the ignored comparison harness. (ref: Errors & Dead Ends — controller staticmethod restoration)

## Run Log

### Run 1

Metadata:
- **Job ID**: local preflight; no production PID
- **Log file(s)**: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/034/preflight.log`; production `run.log` only if authorized
- **WandB**: N/A
- **Status**: failed preflight; production skipped
- **Started**: 2026-08-06 20:55 UTC
- **Ended**: 2026-08-06 20:57 UTC

Description:
- The candidate first undergoes exact constructor/RNG/tensor-scale checks and a copied 200-strong/64-weak trajectory safety screen. Only a complete pass authorizes one seed-42 production run. The expected benefit is improved gradient transport at zero recurring cost; the main risk is amplified relative stem updates despite near-invariant initial BN-normalized features.

Observations:
- Construction passed exactly: 19 Conv/19 BN/one Linear, 1,073,962 parameters, identical CPU/CUDA RNG states and every unaffected tensor, with only `conv1`, `layer2.0.conv1`, and `layer3.0.conv1` rescaled by 0.306186/0.707107/0.707107 at maximum error `1.49e-8`. (source: `preflight-report.json` `construction`)
- Initial train-mode function passed on real hard/soft batches. Relative logit L2 was 0.000437/0.000406 and loss ratio 0.999899/0.999944; changed pre-BN RMS matched analytic scales and post-BN RMS ratios were 0.99920-0.999998. (source: `preflight-report.json` `initial_function`)
- Corpus integrity passed: strong/weak file hashes matched registered EXP022/028 values; 200 strong batches contained 94 hard/106 soft targets and 64 weak batches were hard. Tensor hashes were `4242043f...ad40` and `df97b02a...eae`. (source: `preflight-report.json` `corpus`)
- Trajectory vetoed production: the candidate reached 99.22-100% one-class predictions at steps 9, 14, 15, 16, 24, and 25 while control shares were 65.63-89.84%. Max candidate/control logit/gradient/update ratios were 3.0972/2.3976/1.9522; maximum stem relative update was 13.9878%. (source: `preflight-report.json` `trajectory`)
- Lower loss did not waive the veto: candidate/control terminal strong/weak EMA ratios were 0.97649/0.97591, all BN counters reached 264, and state remained finite. (source: `preflight-report.json` `trajectory`)
- Two post-veto control/control repeats under the production-default backend were not bitwise identical, but had no one-sided concentration against each other and neither exceeded 95% after the shared ordinary step-1/2 transient. This supports treating the candidate's six later collapses as intervention-specific while preserving the reviewer-requested noise caveat. (source: `control-repeat-a.json`, `control-repeat-b.json`)
- The seed-42 production run and evaluator were not executed after the safety veto. No `run.log` was created.

Key Metrics:
- best_test_acc: unavailable — production correctly blocked before evaluation.
- exact changed-tensor norm ratios: 0.30618623/0.70710677/0.70710677. (source: `preflight-report.json` `construction`)
- max trajectory logit/gradient/update ratios: 3.0972x/2.3976x/1.9522x; concentration steps: 9,14,15,16,24,25. (source: `preflight-report.json` `trajectory`)
- minimum candidate/control BN running variance: 0.019817/0.056332; terminal strong/weak EMA ratios: 0.97649/0.97591. (source: `preflight-report.json` `trajectory`)

## Verification Results

### Conditions Checked

- **Baseline/source — pass**: moving baseline 94.15% at `7c1e7d8`; tracked scope is only `train.py`, with user-owned `data/` preserved.
- **Static/exact construction — pass**: code quality, graph, tensor, scale, RNG, Linear, BN, bias, buffer, and parameter checks passed.
- **Initial function — pass**: real hard/soft post-BN activation, logit, loss, finiteness, variance, and counter checks passed.
- **Corpus integrity — pass**: registered strong/weak file hashes, schemas, target ranks, tensor hashes, and post-replay immutability passed.
- **Trajectory safety — fail**: candidate-only >95% class concentration occurred at six registered steps; the pre-production abort criterion is met.
- **Production/evaluator/verdict — skipped**: aborted immediately after the trajectory failure.

### Informational Metrics

- No scored training metrics; this is a preflight veto rather than an accuracy result.
- `train.py` SHA-256: `f95d1fdee720191ae199b9d28a94b25224f2cf2d2fb2ba6921ae8053bba4e3c5`.
- Preflight report SHA-256: `7699baab7ef9395bf1b68fe568392b8bb6103e9ee0ea035dfff6922d22e38815`.

## Errors & Dead Ends

### 2026-08-06 — Controller staticmethod restoration
- Error: `TypeError: ResNet._weights_init() takes 1 positional argument but 2 were given` before construction evidence.
- Root cause: the ignored controller temporarily replaced the static initializer, then restored the saved raw function without the `staticmethod` descriptor.
- Source: `preflight.log` initial attempt, `make_model`/candidate construction traceback.
- Do NOT retry: restore temporary static methods with `staticmethod(original)`; do not change production `train.py` or experimental semantics.

### 2026-08-06 — Fan-out immutable-corpus concentration veto
- Error: `candidate-only concentration at [9, 14, 15, 16, 24, 25]` with candidate shares 0.9922-1.0000.
- Root cause: BN nearly cancels the initial forward rescaling, but the 0.306x stem and 0.707x transition parameter scales amplify relative SGD geometry; the candidate diverged into repeated early one-class states despite lower loss EMA.
- Source: `preflight-report.json` SHA-256 `7699baab7ef9395bf1b68fe568392b8bb6103e9ee0ea035dfff6922d22e38815`.
- Do NOT retry: do not exclude the stem, select transitions only, interpolate scales, compensate LR/decay, relax concentration gates, reroll the corpus/seed, or run production.

## Human Notes

> Autopilot; no human intervention.
