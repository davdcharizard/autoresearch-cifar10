# EXP-044: Depth↔width reallocation — deeper-narrower iso-param ResNet (ResNet-32, k=3)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-044.md
- **Plan**: plans/plan-044.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-044
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the plan's single code change in `train.py`: `NUM_BLOCKS` 3→5 and `WIDTH_MULT` 4→3 (L19-20). This reallocates the ~4.3M-param budget from width into depth — from 3 blocks/stage @ {64,128,256} (ResNet-20, k=4) to 5 blocks/stage @ {48,96,192} (ResNet-32, k=3). All other code (recipe, optimizer, schedule, augmentation, torch.compile, seed) is byte-identical → clean single-variable depth-vs-width test. Pre-launch param/FLOP math (replicating the exact baseline 4,299,866) predicts 4,166,970 params (96.9%) and 605M FLOPs (97.8%) — both slightly below baseline, favorable for dt. GPU 0 was busy (71% util, another user's job); GPU 1 idle → running on `CUDA_VISIBLE_DEVICES=1` to avoid contention-confounded dt (infra-errors).

### Surprises & Discoveries
- (to be filled if anything unexpected during run)

### Decisions
- Chose the uniform deeper-narrower config (5 blocks/stage, k=3) over asymmetric per-stage block counts: it is the closest to BOTH iso-param (96.9%) and iso-FLOP (97.8%) among candidates evaluated, is a canonical ResNet-32 depth, and needs only a two-constant change (no `_make_layer` restructuring), keeping the test maximally clean.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID recorded at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (380.4s total wall)

Description:
- Running the deeper-narrower ResNet-32 (k=3, {48,96,192}, ~4.17M params) under the byte-identical 300s recipe on idle GPU 1. Testing whether depth's superior per-param generalization on CIFAR (He 2016) breaks the width-saturated 96.22 plateau. Expect ResNet-32 banner with params 4,166,970. The load-bearing question is throughput: 15 sequential blocks (vs 9) at ≈iso-FLOP — if dt stays ≈8ms and epochs ≥77, it is a fair depth test; if epochs < 77, it is dt-confounded (underfit).

Observations:
- Banner confirmed `ResNet-32 | params: 4,166,970` (= predicted 96.9% of baseline) (source: run.log L2).
- **dt rock-steady at 12ms** from step 50 → step 23200, flat (460/464 sampled steps = 12ms, 3×13ms, 1×14ms) — NOT contention (contention spikes; this is a flat architectural floor). GPU verified idle before launch (GPU 1, 0%) and after run (both GPUs 0%). (source: `tr '\r' '\n' < run.log | grep dt`)
- dt 8→12ms (+50%) despite ≈iso-FLOP (97.8%) — WORSE than EXP-038's fat-head realloc (+31%). 15 sequential conv+BN blocks (vs 9) at lower per-layer arithmetic intensity → memory-bandwidth/launch bound (more kernels, each doing less work). Confirms iso-FLOP ≠ iso-dt (EXP-015/038) and that depth compounds it.
- **Only 60 epochs / 23,243 steps** (vs baseline ~91 / ~35k) → below the ~77-epoch saturation point → severe UNDERFIT: final_test_loss 0.2905 ≫ baseline 0.195.
- Slower per-EPOCH convergence too: ep1 20.7%, ep2 30.2%, ep3 32.4% (vs baseline ~55% by ep1) — deeper net harder to optimize early + narrower layers have less capacity for the strong TA+Cutout aug. (source: run.log eval lines)
- No errors/NaN/OOM. peak_vram 569MB (clean). startup 2.1s. (source: run.log summary block)

Key Metrics:
- best_test_acc: 92.58% @ ep45-ish (best) (source: run.log summary block)
- final_test_acc: 92.55% @ ep60; final_test_loss: 0.2905 (source: run.log summary)
- num_epochs: 60; num_steps: 23,243; training_seconds: 300.0; total_seconds: 380.4; peak_vram_mb: 569.4; num_params: 4,166,970

## Verification Results

### Conditions Checked
1. **Run completes cleanly within budget** — PASS. Summary block printed; total_seconds 380.4 < 600; training_seconds 300.0. No crash/NaN/OOM. (source: run.log summary block)
2. **Fairness gate (dt / epochs)** — FAILED. Steady dt 12ms ≠ ~8ms; num_epochs 60 < 77 saturation point. Result is dt-CONFOUNDED (severe underfit), not a clean depth test. Recorded for analysis.
3. **Primary necessary condition (`best_test_acc ≥ 96.32`)** — FAIL. best_test_acc 92.58 < 96.32 (−3.64pp vs baseline 96.22). → no-improvement.
4. **No hard-constraint violations** — PASS. `git diff --name-only` = `train.py` only; prepare.py/eval untouched; evaluate() once/epoch (loop unchanged); no new deps; seed 42 unchanged.

Verdict: **no-improvement** (primary condition fails; dt-confounded underfit). Run completed cleanly → Outcome = completed.

### Informational Metrics
- peak_vram_mb: 569.4 (slightly above baseline 491 — more BN buffers/activations from 15 blocks despite narrower width).
- num_epochs / num_steps: 60 / 23,243 (vs ~91 / ~35k baseline) — the dt-driven epoch deficit.
- dt distribution: steady 12ms (the fairness-gate failure).

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
