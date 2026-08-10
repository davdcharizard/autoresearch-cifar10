# EXP-037: Mean-Centered Stem Convolution

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-037
- **Commit**: (pending — committed on loop success)
- **Outcome**: invalid — preregistered mechanism-survival gate failed before production

## Implementation Notes

### Summary

Added one Conv2d subclass that subtracts each output filter's coefficient mean in forward and used it only for the image-facing stem. Stored parameters, initialization, module inventory, residual convolutions, BN, data, optimizer, schedule, timer, and evaluator remain accepted.

### Surprises & Discoveries

The implementation needs no custom autograd or parameter mutation: ordinary subtraction makes the stem data gradient the intended mean-zero projection automatically.

### Decisions

Kept raw stored weights and coupled decay untouched so only the forward representation is projected; no variance scaling, cache, epsilon, all-layer expansion, or null-space optimizer rewrite was introduced.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local preflight; no production PID yet
- **Log file(s)**: ignored EXP037 preflight/timing logs; root `run.log` only if authorized
- **WandB**: N/A
- **Status**: completed; invalid preflight
- **Started**: 2026-08-06 22:02 UTC
- **Ended**: 2026-08-06 22:04 UTC

Description:
- Construction/oracle and control-qualified post-BN mechanism-survival checks precede immutable-corpus safety and paired timing. Only complete passes authorize one seed-42 production run. The expected mechanism is removal of stem DC response with a non-expansive projection; the main risk is immediate BatchNorm redundancy.

Observations:

- Construction matched the accepted model exactly: 19 convolutions, 19 BatchNorm layers, one linear layer, 1,073,962 parameters, and only `conv1` used the projected subclass.
- The FP64 oracle was exact for outputs and input gradients. Effective per-filter means were at numerical zero (`6.17e-18`), projected norm did not exceed raw norm, and raw-weight gradient means were numerical zero (`3.95e-16`).
- At initialization, the candidate produced substantial pooled/logit divergence while exact accepted controls had zero divergence, so the projection was initially functional.
- After 64 registered strong batches, accepted control/control divergence was already large. Candidate maximum survival was `1.0518` versus a `0.6530` control floor on the hard view, and `1.1741` versus `0.7927` on CutMix: only about `1.61x` and `1.48x`, below the preregistered `5x` separation requirement.
- Candidate losses and logit scales remained finite and within ratio bounds, and no candidate-only concentration veto fired. The sole failures were the two step-64 mechanism-null gates.

Key Metrics:

- Hard step64: pooled relative L2 `1.0518`, logit relative L2 `0.8081`, loss ratio `0.9465`; control floor `0.6530`.
- CutMix step64: pooled relative L2 `1.1741`, logit relative L2 `1.0155`, loss ratio `0.9665`; control floor `0.7927`.
- Controller SHA256 `79bac4d86fa5e8d50038368ffdc49ade9642a183323717491e878cf0fa0542cc`; tracked `train.py` SHA256 `5e51ebad46262bc4ccdc2a065ccb834dcdc682a5ba1fca42aedf66e93bc9fdac`.

## Verification Results

### Conditions Checked

- PASS: tracked scope, compile, Ruff, format, pre-commit, GPU availability, exact construction/state/RNG, module inventory, FP64 projection oracle, finite loss/logit ratios, and concentration checks.
- FAIL: the hard and CutMix step-64 candidate divergences did not exceed `5x` their matching accepted control/control divergence.
- NOT RUN by plan: full 264-batch safety replay, seven-pair timing, and the seed-42 production run.

### Informational Metrics

- Initial hard candidate: post-BN RMS relative `7.82e-05`, pooled relative L2 `0.2498`, logit relative L2 `0.1988`.
- Initial CutMix candidate: post-BN RMS relative `7.02e-05`, pooled relative L2 `0.2129`, logit relative L2 `0.1793`.

## Errors & Dead Ends

- The controller exited nonzero after durably writing `mechanism-report.json`, as intended for a failed gate. The PyTorch scalar-conversion warning in the oracle is diagnostic-only and did not affect exact comparisons.
- The large accepted control/control divergence after 64 GPU updates makes a bare absolute divergence misleading. Under the reviewed protocol, mean-centered stem lacks enough control-relative separation to justify a production trial.

## Human Notes

> Autopilot; no human intervention.
