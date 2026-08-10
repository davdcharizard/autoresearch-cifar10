# EXP-038: Output-RMS-Matched Cosine Classifier

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-038
- **Commit**: (pending — committed on loop success)
- **Outcome**: invalid — classifier row-dispersion safety gate failed before long replay/timing/production

## Implementation Notes

### Summary

Added one frozen scale constant and replaced only the final affine invocation with L2-normalized pooled features and classifier rows followed by bias-free `F.linear`. Model and Linear construction, all stored tensors, pooling, data, optimizer, schedule, timer, evaluator, and summary remain accepted. The originally proposed evaluation-cadence control was removed during plan review to preserve exact baseline comparability.

### Surprises & Discoveries

The implementation needs no new module or state: functional normalization reuses the existing Linear weight, and omitting the bias from `F.linear` leaves it stored in the unchanged optimizer group with `grad is None`. Actual production-path checks must prove SGD skips that parameter, including coupled decay.

### Decisions

Kept the full-precision scale literal `22.786916732788086` and epsilon `1e-6` fixed. No temperature rounding, bias deletion, head-specific LR/decay, phase switch, or evaluator edit was introduced. Following adversarial review, slow inverse-norm drift is tested by a 10,240-step observed replay rather than a short linear projection.

## Experimental Adjustments

- **Preserved accepted evaluation cadence**: Removed the brainstorm's fixed-19-look idea because changing cadence would confound a non-speed classifier experiment against the existing baseline. (ref: `02-plan-review.md` concern 1)
- **Extended observed drift replay**: Replaced a 5,120-step linear full-horizon projection with 8,192 strong plus 2,048 weak observed updates and consecutive-window gates. (ref: `02-plan-review.md` concern 5)

## Run Log

### Run 1

Metadata:
- **Job ID**: local preflight; production PID pending
- **Log file(s)**: ignored EXP038 controller logs/JSON; root `run.log` only if all gates authorize production
- **WandB**: N/A
- **Status**: completed; invalid preflight
- **Started**: 2026-08-06 22:30 UTC
- **Ended**: 2026-08-06 22:38 UTC

Description:
- Implement and qualify the frozen output-RMS-matched cosine classifier against exact construction/formula/calibration checks, immutable short and long trajectory gates, and paired H20 timing. Only complete passes authorize one seed-42 production run. The expected mechanism is bounded angular classification without radial feature/weight shortcuts; the principal risk is inverse-norm optimizer amplification under unchanged global SGD and decay.

Observations:

- Tracked scope and static checks passed. Construction matched 19 Conv/19 BN/one Linear, 1,073,962 parameters, named state, and CPU/CUDA RNG exactly. FP64 random/tiny/zero output and VJP errors were at numerical precision; actual logit bounds and epsilon behavior were finite.
- Default cuDNN reproduced the frozen calibration within about 9 ppm: accepted/candidate CutMix logit RMS was `2.760024/2.760048`; the derived scale was `22.786719` versus frozen `22.786917`. Hard-view RMS ratio was `0.998027`. Real production SGD left the unused bias `grad is None` and bitwise unchanged.
- The short 200-strong/64-weak replay passed its global gates: exact264 BN increments, finite state, no candidate-only concentration, no update spikes, and candidate/control maximum ratios `0.6226` logits, `0.9894` gradients, `0.9079` updates, `0.9811` strong loss EMA, and `0.9587` weak loss EMA.
- The candidate nevertheless crossed the long-stage lifetime row max/min bound (`<=3`) at step4, peaked at `4.5122` on step15, and remained above3 through step264 (`261/264` looks). At the peak, all row norms were finite (`1.9618` minimum), class share was only `43.75%`, and loss was `4.5053`; this is angular-head row specialization rather than class collapse.
- Because the planned long replay evaluates the same lifetime bound from initialization, it was logically unable to pass. Long replay, timing, and seed42 production were not run; no scale/epsilon/LR/decay/bias rescue was attempted.

Key Metrics:

- Construction controller SHA256 `077c81512bb815f220241f16babe7d44c999e0785054450a3d4dea76ce9e4481`; trajectory controller SHA256 `10b52531a58c249ea060c63bb4e62fe3b4d033931476b1f5b226ae0a923cf63e`; tracked `train.py` SHA256 `7b3d2f01bbbc6b6a337415c89f2d07bdf4a91aa205d5c1d060acdf59cc96edfb`.
- Candidate short terminal loss EMA: strong `1.9878`, weak `1.6603`; candidate maximum whole-update fraction `0.02184`; candidate-specific concentration steps: none.
- Candidate row max/min: first failure `3.1314` at step4, maximum `4.5122` at step15, terminal `3.1028` at step264.

## Verification Results

### Conditions Checked

- PASS: corpus hashes/schemas, tracked scope/static checks, exact construction/RNG/inventory, FP64 formula/VJPs, finite zero-vector behavior, logit bound, CutMix/hard RMS calibration, real SGD bias immutability, and all registered 264-step global/concentration/update gates.
- FAIL: the preregistered lifetime classifier row max/min ceiling `<=3` was exceeded at 261 of 264 observed candidate steps, starting at step4 and peaking at `4.5122`.
- SKIPPED by plan: 10,240-step survival replay, seven-pair timing, and seed42 production.

### Informational Metrics

- Initial calibration: accepted RMS `2.760024`, candidate RMS `2.760048`, accepted/candidate CE `5.9342/6.0142`; hard-view RMS ratio `0.998027`.
- Short global ratios versus two accepted controls: logits `0.6226`, gradients `0.9894`, updates `0.9079`, strong/weak loss EMA `0.9811/0.9587`.

## Errors & Dead Ends

### 2026-08-06 — Construction capture hook replaced layer output
- Error: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1 and 128x10)`
- Root cause: the controller's `register_forward_hook` lambda returned the captured pooled tensor via `setdefault`; PyTorch interpreted the non-`None` return as a replacement for `layer3` output.
- Source: ignored `experiments/038/construction.log` traceback in first preflight attempt.
- Do NOT retry: never return a tensor from diagnostic forward hooks; assign inside a named function with implicit `None` return.

### 2026-08-06 — Calibration tolerance and zero-vector oracle were falsely strict
- Error: `RuntimeError: calibration scale 22.78671926261512`; zero-vector expected gradients serialized nonfinite errors.
- Root cause: default cuDNN changed the derived scale by about 9 ppm from the frozen provenance run, while the manual oracle used an undefined scalar-square-root derivative at zero.
- Source: ignored `construction-report.json` from the second preflight attempt; candidate RMS still matched accepted within 9 ppm and all actual gradients were finite.
- Do NOT retry: do not demand bitwise CUDA calibration or use scalar square-root norms at zero; use a predeclared 50-ppm scale tolerance and `torch.linalg.vector_norm` with explicit finite-error assertions.

### 2026-08-06 — Cosine classifier row dispersion exceeded lifetime bound
- Error: candidate classifier row max/min exceeded `3` at step4 and 261/264 replay looks, peaking at `4.5122`.
- Root cause: output-RMS parity and bounded logits did not preserve row-wise optimizer geometry; normalized class rows specialized at substantially different raw norms under shared LR/momentum/decay.
- Source: ignored `trajectory-short-report.json`, candidate metrics steps4-264.
- Do NOT retry: do not relax the row-dispersion bound or tune scale/epsilon/LR/decay after observing the trace; any future angular head needs a distinct intrinsic row-norm-control mechanism.

## Human Notes

> Autopilot; no human intervention.
