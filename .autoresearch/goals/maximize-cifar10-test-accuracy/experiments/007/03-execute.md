# EXP-007: Widen layer2 (8×8 stage) 256→384

## Execution

Overall Status & Info:
- **Created**: 2026-06-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-007
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (valid run; best_test_acc 95.85% below the 96.10 bar → no-improvement, under-annealing)

## Implementation Notes

### Summary
Implemented the EXP-007 capacity-via-width change on the EXP-004 base (96.00%). Two edits in `ResNet9.__init__` (`train.py:150-151`): `layer2` `conv_bn(128,256)→conv_bn(128,384)` and `GatedResidual(256)→GatedResidual(384)`; `layer3` stem `conv_bn(256,512)→conv_bn(384,512)` to match layer2's new 384-channel output. layer3 output stays 512, so `pool`/`fc=Linear(512,10)` are untouched. All HPs (PEAK_LR=0.4), schedule, EMA, whitening, augmentation, loop, and `forward` are byte-identical to EXP-004. Milestone 1 passed: `py_compile` clean; `git diff ae31206` confined to exactly the two width lines; `PEAK_LR` still 0.4; in-process smoke confirmed train+eval forward `[8,10]` finite, internal shapes flow (layer1 128@16×16 → layer2 **384@8×8** → layer3 512@4×4), `fc` stays Linear(512,10), **num_params == 9,997,235 exactly** (the hand-computed value; +2,212,608 over EXP-004's 7,784,627), and the `layer2[2]` GatedResidual is identity-init (α=0, `allclose(blk(h),h)`).

### Surprises & Discoveries
- None. The width change rippled exactly as planned (only layer2 out + layer3 in); no other shape depends on the 256 middle width. The plan-review's exact param prediction (9,997,235) matched the smoke to the digit, confirming nothing else changed shape.

### Decisions
- **Gate index is `layer2[2]`** (Sequential = conv_bn, MaxPool2d, GatedResidual) — the plan-review caught an earlier `[3]` typo; verified `[2]` in the smoke.
- **Held PEAK_LR=0.4** for single-variable attribution, accepting the documented caveat that the widened main-path convs are not identity-preserving (so an LR/optimization mismatch is a possible — but not assumed — failure mode, distinguished in analysis via the trajectory).

## Experimental Adjustments

<!-- none -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (local, background) — exit code captured to run_exit.txt
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-28
- **Ended**: 2026-06-28 (exit 0)

Description:
- Official EXP-007 run on GPU 1. Widened layer2 256→384, all else byte-identical to the EXP-004 recipe. Tested whether the added width lifts `best_test_acc` from 96.00% to ≥96.10% with capacity outrunning the throughput-driven epoch loss.

Observations:
- **NO-IMPROVEMENT: best_test_acc 95.85%** (baseline 96.00%, **−0.15pp**), below the 96.10 bar and below baseline. Clean run, exit 0, no divergence. (source: run.log:196)
- **Decisive cause = severe UNDER-ANNEALING from throughput loss, exactly the predicted failure mode.** The widen fit only **94 epochs / 9069 steps** (vs EXP-004's 142–150) — far below the plan's ≤110 under-annealing cutoff. The wider net (9.997M params, +28% over 7.784M) plus shared-host contention (img/s oscillated 20k peaks ↔ 6–8k dips, vs EXP-004's steady ~26k) cut the step count by a third. (source: run.log:203-204, step lines)
- **The net was STILL CLIMBING when the budget ran out — not capacity-saturated.** Tail: ep90 95.67 → ep92 95.79 → ep93 95.81 → ep94 **95.85** (best==final==95.85, monotonically rising in the last epochs). The low-LR tail — where most accuracy lands (EXP-001 finding) — was cut short; the model never finished annealing. So added capacity is likely useful, but 94 epochs is too few to realize it. (source: run.log eval trace ep83-94)
- peak VRAM 1666.8 MB (~equal to EXP-004's 1635 despite +28% params — 8×8/16×16 activations are small); whitening 0.08s off-budget; total wall 439.9s < 600s. (source: run.log:199-202)

Key Metrics:
- best_test_acc: 95.85% @ ep94 (source: run.log:196; max per-epoch best=95.85 == summary)
- final_test_acc: 95.85% @ ep94 | final_test_loss: 0.3325 (still rising at budget end) (source: run.log:197,198)
- training_seconds: 300.0 | total_seconds: 439.9 (source: run.log:199,200)
- **num_epochs: 94 | num_steps: 9069** (vs EXP-004 142–150 → −33% epochs; the headline diagnostic) (source: run.log:203,204)
- peak_vram_mb: 1666.8 | num_params: 9,997,235 (source: run.log:202,205)

## Verification Results

### Conditions Checked

1. **Clean run within wall guard** — PASS. `RUN_EXIT=0` (not 124); one `^best_test_acc:` line; `total_seconds 439.9` < 600. (source: run_exit.txt, run.log:196,200)
2. **Full training budget + scope/integrity intact** — PASS. `training_seconds 300.0` ≥ 295; `prepare.py` byte-unchanged vs ae31206; tracked diff = only `train.py`; untracked = only the whitelisted `run_exit.txt` (no stray `.py`); diff confined to the two `__init__` width lines; `PEAK_LR`=0.4 unchanged; `num_params 9,997,235` == hand-computed/smoke value (widen took effect, nothing else changed shape). (source: git, run.log:205)
3. **Improvement ≥ +0.1pp** — **FAIL → no-improvement**. `best_test_acc 95.85%` < bar 96.10% (−0.15pp vs the 96.00 baseline). Verification stops here. Metric genuine: max per-epoch best 95.85 == summary, from `Eval.evaluate`, one eval/epoch, seeds unchanged. (source: run.log)

**Necessary condition 3 failed → verdict: no-improvement** (valid run — clean, in-scope, metric genuine — but below the bar, due to under-annealing).

### Informational Metrics
- **num_epochs / num_steps: 94 / 9069** (source: run.log:203,204) — ≤110 under-annealing cutoff TRIPPED. Per the plan's pre-registered diagnostic, this points to the **256→320 fallback** (milder widen to recover epochs), NOT capacity-saturation (the net was still climbing at ep94).
- img/s: ~20k peak but 6–8k dips (source: run.log step lines) — vs EXP-004's steady ~26k; the wider net + shared-host contention (GPU 0 busy) drove the epoch shortfall.
- peak_vram_mb: 1666.8 (source: run.log:202) — negligible rise over EXP-004 (1635); VRAM is not the constraint, throughput is.
- num_params: 9,997,235 (source: run.log:205) — +2,212,608 over EXP-004; the capacity was added but couldn't be annealed in 94 epochs.
- total_seconds: 439.9 (source: run.log:200) — within the ~430–490s estimate.

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
