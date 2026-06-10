# EXP-059: GPU faithful AugMix at the proven p=0.5 coverage (W=3)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-059.md
- **Plan**: plans/plan-059.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-059
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Added `_aug_chain(x)` and `gpu_augmix(x, width=3)` (verbatim from plan-057) after `cutout_batch`. In the train loop, before Cutout, apply gpu_augmix to a random ~50% subset (`idx = randperm(bsz)[:bsz//2]; inputs[idx] = gpu_augmix(inputs[idx], width=3)`) — computing the augmentation ONLY on the subset (not full-batch-then-mask) so dt stays ~9.5ms. Removed the CPU `RandomApply([AugMix()], p=0.5)`; CPU pipeline → crop+flip+ToTensor+Normalize. Smoke: AST OK, scope train.py only, subset-apply (128,3,32,32) finite float32 channels_last, aug-on-subset ~1.78ms standalone, num_params 4,299,866 unchanged.

### Surprises & Discoveries
None — the gpu_augmix primitives port cleanly from EXP-057; the only new piece is the subset index-apply. aug-on-subset (~1.78ms standalone) is the same order as EXP-057's full-batch ~1.79ms because the standalone timing is dominated by launch overhead, not the per-sample grid_sample; the in-loop dt premium is the real test (gated).

### Decisions
Compute the augmentation on the ~50% subset (`gpu_augmix(inputs[idx])`) rather than on the full batch then masking — this is the dt-critical choice: EXP-057's W=3 on the FULL batch cost 11ms (epoch wall); running W=3 on only ~64 samples should cost ~half that aug-portion → dt ~9.5ms (~81 ep). Used in-place advanced-index assignment `inputs[idx] = ...` (eager, outside torch.compile, like cutout_batch); the compiled forward receives the full channels_last batch.

## Experimental Adjustments

- **Early gate: dt ~11ms (not ~9.5ms) → ~69 ep; let W=3 run finish instead of the planned W=2 fallback**: at step 5700/ep15/~85s wall, dt is steady 10-11ms → projected epochs (300/0.011)/390 ≈ 69, below the plan's ~75 W=2-fallback threshold. ROOT CAUSE: the subset trick (compute aug on only ~64 samples) saved almost no dt because `grid_sample`/the affine chain are LAUNCH-bound, not throughput-bound — halving the sample count barely halves the launch cost (smoke foreshadowed this: subset 1.78ms ≈ EXP-057 full-batch 1.79ms). So GPU W=3 costs ~11ms regardless of coverage. **KEY REALIZATION**: CPU AugMix is FREE w.r.t. the Σdt/epoch budget (parallel dataloader workers) — EXP-054 ran W=3@50%@**91 ep**=96.45. The GPU path puts the same augmentation IN the timed loop, costing epochs → only ~69 ep for the identical config. The GPU delivery is fundamentally epoch-DISADVANTAGED vs the free CPU path and cannot match it. DEVIATION from plan (W=2 fallback): W=2 would have the SAME launch-bound epoch disadvantage AND less chain diversity than the proven W=3 — it cannot rescue the comparison. Letting the W=3 run finish (69 ep) directly tests the proven CPU config under the GPU epoch cost — the cleaner, more informative data point. No contention (wall/Σdt 1.33×, GPU 0 solo). (source: run.log step 5700; ps etimes 85s; smoke timing.)

## Run Log

### Run 1 (GPU AugMix W=3, 50% coverage)

Metadata:
- **Job ID**: background bash ID blo4yd0jj (local), GPU 0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- GPU faithful AugMix (W=3 Dirichlet + Beta clean-mix) on a random ~50% subset of each batch — the proven-optimal coverage (EXP-054), via the validated GPU path with continuous-affine chains. Removed CPU AugMix. Launched on idle GPU 0 (both 0 MiB/0% pre-launch). Bar = 96.55.

Observations:
- **Early gate** (@ step 5700, ep15, ~85s): dt steady 10-11ms (subset trick didn't save dt — grid_sample launch-bound) → projected ~69 ep (below ~75 floor). wall/Σdt 1.33× (no contention, GPU 0 solo). Loss descending 0.89, no NaN, ep14 test_acc 85.67%. Letting it finish — see Experimental Adjustments for the epoch-disadvantage realization.

Key Metrics:
- **best_test_acc: 95.57%** (baseline 96.45, bar 96.55 → **−0.88pp, no-improvement**)
- final_test_loss: 0.2133 (≫ EXP-054's 0.1968 — underfit; ≈ EXP-057's 0.2115)
- num_epochs: 71, num_steps: 27620 (epoch-disadvantaged: 71 vs CPU EXP-054's 91 at the SAME W3@50% config)
- total_seconds: 379.2 (< 600 ✓), peak_vram_mb: 452.9, num_params: 4,299,866 ✓
- dt dist: 113×10ms, 430×11ms, 6×12ms (steady ~11ms; subset trick did not cut the launch-bound aug cost)
- **Decisive cross-comparison**: GPU 50% (95.57 @71ep) ≈ GPU 100% (EXP-057 95.64 @75ep) — coverage barely changed the GPU result; both ~0.85pp below the CPU 50% baseline (96.45 @91ep). The entire gap is the ~20 lost epochs from putting aug in the timed loop, NOT coverage or policy.

## Verification Results

### Conditions Checked

- **Necessary condition 1 — best_test_acc >= 96.55**: actual **95.57** → **FAIL**. Verdict = no-improvement. (source: run.log `best_test_acc: 95.57%`)
- (Stop at first failed necessary condition. For completeness:) Condition 2 — clean completion: total_seconds 379.2 < 600 ✓, num_params 4,299,866 ✓, NaN/traceback 0 ✓. Condition 3 — scope: `git diff --name-only` = train.py only ✓; prepare.py/eval untouched ✓; evaluate() once/epoch ✓; no new deps (affine_grid/grid_sample core torch) ✓; seed 42 unchanged ✓; ran on uncontended GPU 0 (steady 11ms, wall/Σdt 1.33×) ✓.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> {none — autopilot}
