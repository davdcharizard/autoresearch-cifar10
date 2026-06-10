# EXP-058: Shallower-but-wider ResNet-14 (6 blocks, k=5) — dt-reducing capacity quadrant

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-058.md
- **Plan**: plans/plan-058.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-058
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Two-constant edit per plan: `NUM_BLOCKS` 3→2 (ResNet-20 9-block → ResNet-14 6-block) and `WIDTH_MULT` 4→5 (stages {64,128,256}→{80,160,320}). The `ResNet`/`BasicBlock` classes already parameterize both, so no structural code change. Everything else (CPU AugMix-50%, GPU Cutout, SGD+Nesterov, time-fraction cosine LR, bf16, channels_last, torch.compile reduce-overhead, batch 128, seed 42) is unchanged. Smoke verified: AST OK, scope = train.py only, ResNet-14 builds with 4,290,874 params (iso-param with baseline 4,299,866, −0.2%), forward (8,3,32,32)→(8,10) finite.

### Surprises & Discoveries
ResNet-14 k=5 lands at 4,290,874 params — within 0.2% of the k=4 ResNet-20 baseline (4,299,866). This makes EXP-058 an essentially exact iso-param depth↔width reallocation, the clean mirror of EXP-044 (deeper-narrower iso-param ResNet-32 k=3, 4.17M). The capacity variable is held; only the depth/width split (and the resulting dt) changes.

### Decisions
Chose k=5 (not k=6) specifically because it is iso-param with the baseline — isolating the depth↔width split from a capacity change, and keeping the wide-conv dt premium bounded (k=6 would be 6.17M params and a much steeper memory-bandwidth wall). The hypothesis is about the dt/accuracy frontier at fixed capacity, so iso-param is the correct control.

## Experimental Adjustments

- **Early gate tripped (dt 12ms → ~64 ep, epoch wall) but let run to completion rather than abort**: at step 50-350 dt is flat 12ms (50% over the 8ms baseline) → projected epochs (300/0.012)/390 ≈ 64, below the plan's ~73 abort threshold. This is the hypothesis's predicted NEGATIVE branch — the wide-conv (k=5, 320-ch) memory-bandwidth wall dominates even at 6 blocks. Notably dt=12ms EXACTLY matches EXP-044's deeper-narrower ResNet-32 (also 12ms): the wall is wide-conv memory bandwidth, NOT block count, so reducing blocks did NOT buy dt headroom — the k=5 320-ch convs cost what they cost. DECISION: let the run finish (it self-limits at 300s Σdt, ~7 min wall) instead of aborting, because the completed best_test_acc at the epoch-walled config is a directly comparable data point to EXP-044 (which ran to completion, 92.58 @ 60 ep) and completes the symmetric iso-param depth↔width picture. Trivial extra cost; stronger conclusion. (source: run.log step 50-350, dt 12ms; ep1 test_acc 37.44% vs baseline ~45.7% — early underfit signal.)

## Run Log

### Run 1 (ResNet-14 k=5)

Metadata:
- **Job ID**: background bash ID bizechkmg (local), GPU 0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- ResNet-14 (6 blocks) k=5, iso-param (4,290,874) with baseline. Tests the untested dt-reducing quadrant: fewer blocks → lower launch-bound dt, reinvested into width. Launched on idle GPU 0 (both GPUs 0 MiB/0% pre-launch). Expected: if dt ≤ ~8.5ms (epochs ≥ ~80) the capacity holds and best_test_acc could clear 96.55; if dt > ~10.5ms (epochs < ~73) the wide-conv memory wall dominates → epoch-wall regression. Bar = 96.55.

Observations:
- **Early gate** (@ step 350, ep1): dt flat 12ms (architectural, matches EXP-044's wide-conv wall) → projected ~64 ep (epoch wall, < 73 threshold). No contention (steady dt, img/s ~10,500, GPU 0 launched idle). Loss descending 2.58→1.90, no NaN. ep1 test_acc 37.44% (< baseline ~45.7% — early underfit signal consistent with fewer epochs). Hypothesis NEGATIVE branch confirmed; letting it finish for the completed number.

Key Metrics:
- **best_test_acc: 95.24%** (baseline 96.45, bar 96.55 → **−1.21pp, no-improvement**)
- final_test_loss: 0.2234 (≫ baseline 0.195 / EXP-054 0.1968 — clear UNDERFIT, the epoch-wall signature)
- num_epochs: 61, num_steps: 23622 (epoch wall confirmed: 61 ≪ ~80 saturation floor, vs baseline ~91)
- total_seconds: 418.7 (< 600 ✓), training_seconds 300.0, peak_vram_mb: 444.9, num_params: 4,290,874 ✓
- dt dist: 442×12ms, 28×13ms, 1×14ms, 1×17ms (flat 12ms = +50% over 8ms baseline; architectural, identical to EXP-044's wide-conv wall)
- Trajectory: best 95.24 @ep~57, still mildly climbing/plateauing at the truncated 61-ep budget (under-resolved).

## Verification Results

### Conditions Checked

- **Necessary condition 1 — best_test_acc >= 96.55**: actual **95.24** → **FAIL**. Verdict = no-improvement. (source: run.log `best_test_acc: 95.24%`)
- (Stop at first failed necessary condition. For completeness:) Condition 2 — clean completion: total_seconds 418.7 < 600 ✓, num_params 4,290,874 (intended ResNet-14 k=5) ✓, NaN/traceback 0 ✓. Condition 3 — scope: `git diff --name-only` = train.py only ✓; prepare.py/eval untouched ✓; evaluate() once/epoch (loop unchanged) ✓; no new deps ✓; seed 42 unchanged ✓; ran on uncontended GPU 0 (steady 12ms, launched idle) ✓.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> {none — autopilot}
