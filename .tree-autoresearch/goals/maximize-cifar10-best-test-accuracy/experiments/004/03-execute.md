# EXP-004: Clean-Finish Periodic SAM

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-004
- **Base Node**: 002
- **Commit**: 1a8d0de
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the reviewed period-two SAM intervention entirely in `train.py`. The existing EXP-002 forward/loss path remains the first pass; eligible even one-based steps in the final clean quarter add a second perturbed pass, then restore exact FP32 parameter snapshots before the single Nesterov update. CUDA RNG replay preserves stochastic-depth masks and future RNG state, while temporary BatchNorm tracking suppression ensures only the first pass updates running buffers. Mechanism counters and first-application metadata make cadence auditable.

### Surprises & Discoveries

- Claude's plan review identified that sharing one autocast context could reuse cached BF16 weight casts and silently hide the perturbation. The implementation uses separate autocast contexts, and both tiny-model and real-WRN smokes confirmed distinct first/second losses.
- `torch._foreach_copy_` and `torch._foreach_add_` support the model's FP32 parameter lists, allowing preallocated snapshots and perturbations without per-pulse tensor allocation.
- The actual WRN BF16 smoke preserved CUDA RNG parity and incremented every BatchNorm `num_batches_tracked` buffer exactly once across two forwards.

### Decisions

- Use the exact predicate `progress >= 0.75 and (step + 1) % 2 == 0`; counters increment only after a successful SAM optimizer update.
- Abort only for a nonfinite or exactly zero global norm. `SAM_EPS=1e-12` is solely a denominator guard.
- Use stage-aware restoration: perturbation helper failures restore internally, while second-pass failures restore BatchNorm flags and exact snapshots in `finally` before propagating.

## Experimental Adjustments

- **Period four -> period two before implementation**: Claude's idea review found that three intervening Nesterov updates could dilute the sharpness-aware component; the stronger clean-tail dose remains within the charged-time budget. (ref: `01-idea-review.md`)
- **Overhead-inclusive exposure bounds**: Claude's plan review added fixed RNG/norm/snapshot costs, changing the expected range to 25,000-25,800 steps and 2,000-2,400 SAM pulses, with hard mechanism floors of 24,000 and 1,800. (ref: `02-plan-review.md`)

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 137642
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 12:02:15 UTC
- **Ended**: 2026-08-05 12:11:28 UTC

Description:
- One fixed-seed run of the EXP-002 WRN/CutMix parent with rho-0.05 SAM on every second step after 75% charged progress. All SAM work is inside the existing 300-second training timer and the process is restricted to physical GPU 0 with a 600-second outer timeout. The expected mechanism dose is 2,000-2,400 pulses while retaining 25,000-25,800 optimizer steps. Improvement requires `best_test_acc >= 95.33%`; no intermediate metric can trigger a retry or adjustment.

Observations:
- Pre-run tiny-model smoke: gradient norm 0.467865, perturbation norm 0.050000, first loss 1.176025, perturbed second loss 1.198730; exact restore, CUDA RNG parity, one BatchNorm update, and one momentum update passed.
- Pre-run WRN BF16/channels-last smoke: gradient norm 4.746192, first loss 2.307617, perturbed second loss 2.544922; exact restore, CUDA RNG parity, and one BatchNorm update passed.
- Startup reported CUDA, 2,748,890 parameters, `sam_rho=0.05`, `sam_start=0.75`, and `sam_period=2`; the first 150 steps showed `sam: 0/0` as required before the clean-tail boundary. (source: `run.log` startup/progress output)
- SAM activated only after the clean-tail boundary. The final audit was `applied=2449`, `eligible=4898`, `ratio=0.5000`, `first_step=20664`, `first_progress=0.7500`; CutMix stopped at 10,253 of 20,662 eligible batches with ratio 0.4962. (source: `run.log` lines 270-271)
- The run completed exit 0 with no traceback, NaN/Inf, CUDA, OOM, timeout, overlap, or invalid-gradient signature. It retained 25,560 optimizer steps and one evaluation per each of 132 epochs. (source: `run.log` lines 1-282)
- Late accuracy reached 95.25% at epoch 125 after an earlier 95.36% best, then climbed to a final and best 95.40% at epoch 132; success did not rely on an isolated early checkpoint. (source: `run.log` lines 253-269)
- Claude's post-run adversarial integrity review approved the result as a trustworthy improvement after checking log freshness, exact cadence arithmetic, timing, evaluation count, scope, seed, RNG replay, BatchNorm handling, parameter restoration, and live BF16 perturbation.

Key Metrics:
- `best_test_acc`: 95.40% (source: `run.log` line 273)
- `final_test_acc`: 95.40% (source: `run.log` line 274)
- `final_test_loss`: 0.1654 (source: `run.log` line 275)
- `training_seconds`: 300.0 (source: `run.log` line 276)
- `total_seconds`: 457.3 (source: `run.log` line 277)
- `startup_seconds`: 1.2 (source: `run.log` line 278)
- `peak_vram_mb`: 1190.5 (source: `run.log` line 279)
- `num_epochs`: 132 (source: `run.log` line 280)
- `num_steps`: 25560 (source: `run.log` line 281)
- `num_params`: 2748890 (source: `run.log` line 282)
- SAM exposure: 2449/4898 = 0.5000, first step 20664 at progress 0.7500 (source: `run.log` line 271)

## Verification Results

### Conditions Checked

- **Primary accuracy improvement**: passed. `best_test_acc=95.40%` is 0.17 points above the 95.23% parent and exceeds the required 95.33% threshold. (source: `run.log` line 273; parent: `tree.sh show ... 002`)
- **Successful bounded execution**: passed. Exit code was 0; `training_seconds=300.0`, `total_seconds=457.3`, complete summary present, physical GPU 0 exposed, and exactly 132 evaluations occurred for 132 epochs. (source: `run.log` lines 1-282)
- **Mechanism and scope integrity**: passed. First SAM progress was 0.7500, exposure was exactly 0.5000 with 2,449 pulses, total steps 25,560 exceeded the 24,000 floor, parameters remained 2,748,890, and only `train.py` changed. (source: `run.log` lines 270-282; `git status --short`)
- **Adversarial integrity review**: passed. Claude classified EXP-004 as a protocol-clean improvement at 95.40%, with no stale output, evaluator change, timing violation, seed hacking, mechanism underdose, or two-pass correctness flaw.

### Informational Metrics

- `final_test_acc=95.40%`, `final_test_loss=0.1654`
- `training_seconds=300.0`, `total_seconds=457.3`, `startup_seconds=1.2`
- `peak_vram_mb=1190.5`, `num_epochs=132`, `num_steps=25560`, `num_params=2748890`
- CutMix exposure `10253/20662=0.4962`; SAM exposure `2449/4898=0.5000`

## Errors & Dead Ends

## Human Notes

> The user requires Claude to be the sole adversarial reviewer; both idea and plan reviews used Claude without fallback.
