# EXP-057: Full-coverage faithful GPU AugMix (multi-chain + Beta clean-mix)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-057.md
- **Plan**: plans/plan-057.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-057
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Extended the EXP-056 GPU-aug primitives into a faithful AugMix. Added `_aug_chain(x)` (one independently-augmented version: per-sample affine rotate/shear/scale + photometric brightness/contrast) and `gpu_augmix(x, width=3)` (mix 3 chains with per-image Dirichlet(1) weights via normalized Exp(1), then Beta(1,1)=Uniform convex-mix with the ORIGINAL clean image). Wired `inputs = gpu_augmix(inputs)` into the train loop before `cutout_batch`; removed the CPU `RandomApply([AugMix()])` (CPU → crop+flip+ToTensor+Normalize). Reproduces the EXP-054 winner (w3 AugMix) at 100% coverage (was wall-forced to 50% on CPU). Smoke: AST OK, scope=train.py only, gpu_augmix ~1.79ms/batch, output (128,3,32,32) finite float32 channels_last.

### Surprises & Discoveries
Clean-mix boundedness directly verified in the smoke test: mean|augmix−clean| = 0.41 vs mean|single_chain−clean| = 0.82 — the Dirichlet chain-averaging + Beta clean-mix HALVES the per-image distortion vs a raw chain. This is precisely the shift-bounding property EXP-056's harsh stack lacked (which over-distorted → 95.39); strong a-priori reason to expect a much lower test loss this time.

### Decisions
Each chain = one affine+photometric pass (effective depth ~2) with independent random params, rather than AugMix's variable-depth (1-3) random-op sequences — captures the multi-chain-diversity + clean-mix essence at lower implementation risk while keeping the op pool (rotate/shear/scale/brightness/contrast) GPU-correct on normalized data. W=3 matches the torchvision AugMix default and the EXP-054 winner. Magnitudes kept gentle (EXP-053: magnitude interior-optimal; mixing softens further).

## Experimental Adjustments

- **Run 1 (W=3) ABORTED on the epoch-wall gate → Run 2 relaunched W=2**: W=3 (3 chains, 3 grid_samples) cost ~3ms over the 8ms baseline → steady dt 11.0ms → projected ~65 epochs, below the ~76 under-training floor (wall/Σdt 1.31× — NOT contention, genuinely the GPU-aug compute). Per the plan contingency, dropped to width=2 (2 chains, ~2ms → dt ~10ms → ~77 ep). The in-loop grid_sample cost (~1ms/chain) exceeded the ~0.5ms/chain the standalone smoke implied (channels_last + loop memory pressure). (ref: Run 1 window: steady dt 11.0ms, proj 65 ep)

## Run Log

### Run 1 (ABORTED — epoch-wall gate, W=3 too costly)

Metadata:
- **Job ID**: background bash ID bvszka92o (local) — STOPPED
- **Log file(s)**: run.log (overwritten by Run 2)
- **Status**: aborted (epoch-wall gate)
- **Started/Ended**: 2026-06-09 (aborted at ~step 4500)

Observations:
- W=3 steady dt 11.0ms/step (window +21s wall / +1450 steps / +16s Σdt), wall/Σdt 1.31× (no contention, GPU 1 solo), projected ~65 epochs < 76 floor → aborted per plan. Loss descending normally, no NaN. (source: run.log Run 1; ps etimes)

### Run 2 (W=2)

Metadata:
- **Job ID**: background bash ID bg96yhlp4 (local). [Note: a first W=2 launch attempt (bixa5005x) failed exit 144 — its `pkill -f train.py` self-matched the launching shell AND killed the orphaned Run-1 python that had survived TaskStop; relaunched cleanly without pkill. No contention; GPU 1 verified idle before relaunch.]
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (exit 0)

Description:
- gpu_augmix(width=2): full-coverage AugMix with 2 Dirichlet-mixed chains + Beta clean-mix, on idle GPU 1. Same shift-bounding fix, lighter compute to protect the epoch budget. Expect dt ~10ms, ~77 ep, wall ~430s. Early gate: dt>11ms → drop photometric; wall/Σdt≫2.5 → contention. Bar = 96.55.

Observations:
- **Early gate PASSED** (@ step 7900, ep21, etimes ~113s): dt steady 10ms → projected epochs (300/0.010)/390 = 76.9 ≥ 76 floor. rem=215s ⇒ Σdt≈85s ⇒ wall/Σdt ≈ 1.33× (no contention; matches Run 1's 1.31×). Loss descending 0.88, no NaN. test_acc trajectory normal: ep19 86.88% → ep20 87.57%. GPU 1 solo (GPU 0 = unrelated v2.9.5-gpt-5-5 run, no cross-contention). (source: run.log L~7900; ps etimes)

Key Metrics:
- **best_test_acc: 95.64%** (baseline 96.45, bar 96.55 → **−0.81pp, no-improvement**)
- final_test_loss: 0.2115 (vs EXP-054 0.1968; vs EXP-056 0.224 — clean-mix lowers loss vs the harsh stack but still over-regularized vs the 50% subset)
- num_epochs: 75, num_steps: 29004 (W=2 dt held the epoch budget as gated — 75 ≈ projected 77)
- total_seconds: 382.6 (< 600 ✓), peak_vram_mb: 452.9, num_params: 4,299,866 ✓
- dt dist: 540×10ms, 34×11ms, 4×12ms, 1×14ms, 1×26ms (steady ~10ms, no contention spikes)
- Trajectory: smooth, best plateaued ~95.6 from ep72 (95.63→95.64) — converged, not under-trained.

## Verification Results

### Conditions Checked

- **Necessary condition 1 — best_test_acc >= 96.55**: actual **95.64** → **FAIL**. Verdict = no-improvement. (source: run.log `best_test_acc: 95.64%`)
- (Per protocol, stop at first failed necessary condition. For completeness:) Necessary condition 2 — clean completion: total_seconds 382.6 < 600 ✓, num_params 4,299,866 ✓, NaN/traceback count 0 ✓. Necessary condition 3 — scope: `git diff --name-only` = train.py only ✓; affine_grid/grid_sample core torch (no new dep) ✓; seed 42 unchanged ✓; ran on uncontended GPU 1 (wall/Σdt 1.33×, fair dt-budget) ✓.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> {none — autopilot}
