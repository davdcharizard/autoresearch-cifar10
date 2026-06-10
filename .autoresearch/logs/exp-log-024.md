# EXP-024: BlurPool / anti-aliased downsampling (Zhang 2019)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md
- **Plan**: plans/plan-024.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-024
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (run clean; verification Cond 1 failed → no-improvement, compute-confounded; verdict rendered in analyze)

## Implementation Notes

### Summary
Added a parameter-free `BlurPool2d` module (fixed 3×3 binomial depthwise kernel as a registered buffer; `F.conv2d(..., stride, padding=1, groups=channels)`) and rewired `BasicBlock` for the two downsample sites (Milestone 1): `conv1` is now stride-1, a `BlurPool2d(out, stride=2)` does the stride-2 subsample after `relu(bn1(conv1(x)))` and before `conv2`, and the projection shortcut anti-aliases its input (`BlurPool2d(in, stride=2)`) before a stride-1 1×1 conv. Non-downsample blocks (stride==1) are unchanged. Verified: ruff + AST clean, `git diff --name-only` = train.py only, **params = 4,299,866 (UNCHANGED)** and a forward smoke test gives correct (B,10) output — spatial alignment holds (32→16→8→4 matches the strided-conv output).

### Surprises & Discoveries
The bare-`python` smoke test failed on `import torchvision` (env only has it under `uv`); re-running under `uv run python` confirmed params + forward shape. No code issue.

### Decisions
Split the shortcut into three explicit cases (stride!=1 → BlurPool+1×1; stride==1 & in!=out → plain 1×1; else Identity) to keep correctness general; in this net only the first and third occur. Used DC-preserving normalized binomial filter (sums to 1) so the blur doesn't rescale activations.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — background bash task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08

Description:
- Runs the EXP-012 recipe with anti-aliased downsampling (BlurPool) at the two stride-2 sites, testing whether restoring shift-invariance reduces the generalization gap and lifts best_test_acc above 96.32. KEY confound to watch: moving conv1 to stride-1 ~4×'s its FLOPs at the two heaviest convs → may cut epochs below the baseline 91 (k=4 is launch-bound so it MAY be absorbed). Expected: small top-1 gain if epochs hold; or regression/confound if the added compute craters epochs (EXP-015 / capacity pattern).

Observations:
- Clean run: params 4,299,866 (unchanged — blur kernels are buffers), clean compile (no graph break on the depthwise fixed-kernel conv), no traceback, no NaN (source: run.log L2, summary).
- best_test_acc 95.66% vs baseline 96.22 = **−0.56pp**, below the 96.32 bar. **CONFOUND MATERIALIZED: num_epochs dropped 91→77** (dt rose 8→9-10ms ~+15-25%). BlurPool's ~4× FLOPs at the two heaviest downsample convs were only partly absorbed by the launch-bound headroom and cost ~15% of epochs. final_test_loss ROSE 0.195→0.2085 (same LS, comparable) — consistent with under-training from fewer epochs (source: run.log summary + early dt samples).
- Per the mandatory confound-attribution rule (plan §6): num_epochs 77 < 85 → result is COMPUTE-CONFOUNDED, not a clean test of anti-aliasing's merit. The regression cannot be cleanly attributed to anti-aliasing (the 14 fewer epochs alone plausibly explain most/all of it, mirroring EXP-015 and capacity EXP-004/009).

Key Metrics:
- best_test_acc: 95.66% (source: run.log `best_test_acc:` line)
- num_epochs: **77** (vs baseline 91 — KEY confound diagnostic); num_steps: 29767
- final_test_loss: 0.2085 (vs baseline 0.195, ROSE — under-training); final_test_acc: 95.57%; total_seconds: 398.1; peak_vram_mb: 577.9; num_params: 4,299,866 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAILED**. best_test_acc = 95.66% < 96.32 (−0.56pp vs baseline 96.22). (source: run.log `best_test_acc: 95.66%`)
- **Cond 2 — clean completion within budget**: skipped — aborted after Cond 1 failed. (Would pass: total_seconds=398.1 < 600, Traceback=0, metrics present.)
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 1 failed. (Would pass: git diff = train.py only, num_params=4,299,866, eval-count=77 == num_epochs=77, nn ops are core torch / no new deps, seed 42 intact.)

Verdict basis: first necessary condition failed → no-improvement. **Result is compute-confounded** (num_epochs 77 vs baseline 91) — the regression is NOT a clean test of anti-aliasing.

### Informational Metrics

- Not collected (only when all conditions pass). For the record: num_epochs=77 (vs baseline 91 — confound), num_steps=29767, final_test_loss=0.2085 (vs baseline 0.195, ROSE → under-training), peak_vram_mb=577.9.

## Errors & Dead Ends

<!-- none -->

## Human Notes

> (none — autopilot)
