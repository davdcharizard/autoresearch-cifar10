# EXP-033: Conservative Small-Area Random Erasing

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-033
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed — preregistered trajectory safety veto; production not authorized

## Implementation Notes

### Summary

Added a top-level, forkserver-picklable wrapper around torchvision RandomErasing and inserted one p=0.25, 2-10%-area, ratio 0.3-3.3 mean-fill erasing operation after the accepted strong ToTensor transform and before Normalize. The weak/evaluation policies, CutMix collator contract, model, optimizer, schedule, evaluator, and production logging remain unchanged.

### Surprises & Discoveries

The existing strong collator already returns the same two-field hard/soft target contract needed by the training loop. Keeping all erasing provenance in disposable preflight controllers avoids both a variable-length batch contract and production-time tensor scanning. Torchvision uses rounded rather than floored integer mask dimensions: dense enumeration encloses legal achieved areas at 1.5625-10.9375%, so the reviewed fidelity bound was corrected to 1.0-11.0% without changing the requested 2-10% policy.

### Decisions

Retained per-image `torch.random.fork_rng(devices=[])` as the reviewed attribution control so erasing does not advance accepted crop/flip/RandAugment draws. This is intentionally load-bearing: production is blocked if paired timing shows more than 1% weighted overhead.

## Experimental Adjustments

- **Clone captured worker RNG states before replay**: `torch.set_rng_state` first segfaulted on a pinned shared-memory view and then rejected a non-contiguous slice. Contiguous parent-owned clones resolved the controller issue without changing the corpus stream. (ref: Errors & Dead Ends — pinned RNG state)
- **Correct achieved-area fidelity bound to 1.0-11.0%**: torchvision rounds both rectangle dimensions, and a legal request can realize 112/1024=10.9375%. The intervention remains exactly scale=(0.02,0.10). (ref: exact-corpus geometry)

## Run Log

### Run 1

Metadata:
- **Job ID**: local preflight; no production PID
- **Log file(s)**: `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/033/preflight-*.log`; production `run.log` only if authorized
- **WandB**: N/A
- **Status**: failed preflight; production skipped
- **Started**: 2026-08-06 20:22 UTC
- **Ended**: 2026-08-06 20:31 UTC

Description:
- The candidate will first undergo semantic, exact-corpus trajectory, live-worker, and paired full-path timing gates. Only a complete pass authorizes the one seed-42 production run. Expected policy exposure is 23.5-26.5% erased strong examples and 1.1-1.9% unconditional pixels, with at least 99% paired exposure retention.

Observations:
- Static semantics passed: exact mean fill/normalized zero, bitwise outside-mask preservation, saved-state reproduction, unchanged downstream CPU/CUDA RNG, pickling, target contract, and 1,073,962 parameters. (source: `preflight.log` stage `semantic`)
- Exact corpus passed with SHA-256 `eff90f5701a303152c9ee44082713fbf63ce31e2561159a108c941a421cecb3d`: 200 strong batches selected from the natural first 205 to register exactly 100 hard/100 CutMix decisions, plus 64 weak batches; all eight workers stopped. (source: `preflight.log` stage `corpus`)
- Geometry passed: 6,384/25,600 erased examples (24.9375%), 100% placement, achieved area 1.7578-10.9375%, conditional mean 5.9499%, unconditional mean 1.4838%, and maximum final post-CutMix effective area 19.7266%. (source: `preflight-report.json` `geometry`)
- Trajectory vetoed production: candidate-only >95% class concentration occurred at steps 4, 20, 21, and 22. Candidate/control maxima were loss 1.0525x, logit RMS 8.9304x, gradient norm 4.4059x, and update norm 3.0674x; strong/weak terminal loss EMA ratios 0.9590/0.9269 cannot override the geometry veto. (source: `preflight-report.json` `trajectory`)
- Live delivery was otherwise ample at 159.07 batches/s and 89.08% of control, with 24.9814% erased examples, 49.54% CutMix, all workers covered/stopped, and weak rebuild 2.797s/no weak zero-fill pixels. Its p95 wait was 45.014ms versus the absolute 1.5ms gate (control also burst to 38.830ms), an independent live-gate failure. (source: `preflight-report.json` `lifecycle`)
- Fresh paired full-step timing and the seed-42 production run were not executed after the trajectory veto. No `run.log` was created.

Key Metrics:
- best_test_acc: unavailable — production correctly blocked before evaluation.
- exact corpus erased fraction: 24.9375%; unconditional area: 1.4838%; effective maximum: 19.7266%. (source: `preflight-report.json` `geometry`)
- trajectory max logit/gradient/update ratios: 8.9304x/4.4059x/3.0674x; concentration steps: 4,20,21,22. (source: `preflight-report.json` `trajectory`)
- live candidate/control throughput: 159.07/178.57 batches/s = 89.08%; candidate wait median/p95: 0.072/45.014ms. (source: `preflight-report.json` `lifecycle`)

## Verification Results

### Conditions Checked

- **Baseline/source — pass**: moving baseline 94.15% at `7c1e7d8`; branch scope is only tracked `train.py`, with user-owned `data/` preserved.
- **Static/semantic — pass**: compile, Ruff, format, pre-commit, diff, RNG, policy, pickling, target, and parameter checks passed.
- **Exact corpus — pass**: every geometry/provenance bound passed on the hashed 200-strong/64-weak corpus.
- **Trajectory safety — fail**: candidate-only concentration and per-step logit/gradient/update ratios exceeded registered limits. This immediately blocks production.
- **Worker/lifecycle — fail (informational after trajectory veto)**: throughput, policy proportions, coverage, shutdown, and weak rebuild passed; absolute p95 wait exceeded 1.5ms.
- **Paired timing — skipped**: aborted after trajectory failure.
- **Production/evaluator/verdict — skipped**: aborted after trajectory failure.

### Informational Metrics

- No scored training metrics; the candidate is an invalid preflight veto rather than a no-improvement accuracy result.
- `train.py` SHA-256: `72a8b59420a2b1f06aad18f85a3493386d791a08f6af064a29aa991d66f380cd`.
- Preflight report SHA-256: `278efad81e7fb2f562a4ca5757e75fb9d51540a4c6f20ea703e8b55b2bc6f76d`.

## Errors & Dead Ends

### 2026-08-06 — Pinned worker RNG state replay fault
- Error: `exit 139 in torch.set_rng_state`, then `RuntimeError: Invalid mt19937 state` on a pinned/shared or sliced loader state.
- Root cause: the controller passed a pinned shared-memory state view, then a row view, directly to CPUGeneratorImpl.
- Source: `preflight.log` initial controller attempts; Python faulthandler pointed to `erase_batch`.
- Do NOT retry: clone the batched RNG tensor into contiguous parent-owned memory and clone each state row before `set_rng_state`.

### 2026-08-06 — Random Erasing exact-corpus trajectory veto
- Error: `candidate-only concentration [4, 20, 21, 22]`; max logit/gradient/update ratios `8.9304/4.4059/3.0674` exceeded 1.5.
- Root cause: even 1.48% unconditional mean-fill deletion changed the high-LR early trajectory enough to amplify class geometry and optimizer displacement on the registered corpus.
- Source: `preflight-report.json` `trajectory`; report SHA-256 `278efad81e7fb2f562a4ca5757e75fb9d51540a4c6f20ea703e8b55b2bc6f76d`.
- Do NOT retry: do not tune p/scale/ratio/fill, reroll corpus/seed, or proceed to timing/production for this exact policy.

## Human Notes

> Autopilot; no human intervention.
