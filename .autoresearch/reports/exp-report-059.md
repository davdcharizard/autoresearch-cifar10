# Report EXP-059: GPU faithful AugMix at the proven p=0.5 coverage (W=3)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-059.md
- **Plan**: plans/plan-059.md
- **Log**: logs/exp-log-059.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within a fixed 300s GPU-time (Σdt) budget on a single H20, editing only train.py. Baseline = **96.45** (EXP-054); bar = **96.55** (+0.1pp).

## Idea & Hypothesis
EXP-056/057 validated the GPU-aug infra but ran at 100% coverage (both regressed). ~50% coverage is the PROVEN interior optimum (EXP-054 CPU AugMix p=0.5 = 96.45). This loop ran the one untried same-family variant: faithful GPU AugMix (W=3 Dirichlet + Beta clean-mix) on a random ~50% subset, delivering continuous-magnitude affine chains at the proven coverage. Hypothesis: matching the proven coverage via the GPU path (potentially richer continuous chains) clears 96.55.

## Approach
Added `_aug_chain`/`gpu_augmix` (from EXP-057) after `cutout_batch`; in the train loop, before Cutout, applied gpu_augmix(W=3) to a random ~50% subset (`idx = randperm(bsz)[:bsz//2]; inputs[idx] = gpu_augmix(inputs[idx])`) — computing aug only on the subset to try to keep dt low. Removed CPU AugMix; CPU → crop+flip+ToTensor+Normalize. num_params unchanged (4,299,866).

## Execution
Single run on idle GPU 0 (blo4yd0jj). Early gate (step 5700/ep15): dt steady ~11ms (NOT the hoped ~9.5ms) → projected ~69 ep, below the ~75 W=2-fallback threshold. Root cause: `grid_sample`/the affine chain are LAUNCH-bound, so computing W=3 aug on 64 samples costs nearly as much as on 128 — the subset trick saved almost no dt. Per the new understanding (below), I let the W=3 run finish (deviating from the planned W=2 fallback) because W=2 has the same epoch disadvantage and W=3 directly tests the proven config. Clean, uncontended (wall/Σdt 1.33×), exit 0: 71 ep, 379.2s.

## Results
- **Primary metric**: **95.57%** (baseline 96.45, delta **−0.88pp**, −0.91%)
- **Observations**: dt ~11ms, final_test_loss 0.2133 (underfit, ≫ EXP-054's 0.1968). **Decisive cross-comparison: GPU W3@50% (95.57 @71ep) ≈ GPU 100% W2 (EXP-057 95.64 @75ep)** — coverage (50 vs 100%) barely changed the GPU result; both land ~0.85pp below the CPU 50% baseline (96.45 @91ep).
- **Analysis**: Hypothesis REJECTED, with a clean mechanistic explanation. The CPU AugMix that achieved 96.45 is **FREE w.r.t. the Σdt/epoch budget** — it runs in the parallel dataloader workers, off the timed training step, so EXP-054 got W3@50% AND 91 epochs. The GPU path puts the identical augmentation INSIDE the timed loop, costing ~3ms/step → only ~71 epochs for the same W3@50% config. The entire 0.88pp gap is the ~20 lost epochs (underfit), not coverage, policy, or representation. This is confirmed three ways: (1) GPU 50% ≈ GPU 100% (coverage irrelevant on the GPU path), (2) both ≪ CPU 50%, (3) loss 0.213 ≫ 0.197 (underfit signature). The subset trick failed to help because the affine/grid_sample ops are launch-bound (halving samples ≠ halving cost).
- **Key Learning**: Moving augmentation from the CPU dataloader (free w.r.t. the Σdt budget) onto the GPU (inside the timed loop) is strictly epoch-DISADVANTAGED — it cannot match the same augmentation delivered free on CPU; the GPU-aug path is closed for any aug that CPU workers can deliver within the 600s wall.

## Verification
- **Conditions**: Necessary condition 1 (`best_test_acc >= 96.55`) FAILED (95.57). Conditions 2 (379.2s<600, params 4,299,866, 0 NaN) and 3 (scope train.py only, no new deps, seed 42, uncontended) hold.
- **Review Notes**: Trustworthy — steady 11ms (no contention), converged-but-underfit trajectory, fair dt-budget. Regression is real.
- **Verdict**: **no-improvement**
- **Verdict Basis**: valid run, necessary condition 1 failed (epoch-disadvantaged underfit); no hard-constraint violation.

## Unexplored Avenues
- **GPU-aug is only worth it for augmentation CPU CANNOT deliver within the 600s wall.** EXP-052 showed uniform CPU AugMix w3 needs ~792s (wall-infeasible) — that is the ONLY regime where the GPU path's epoch cost could be worth paying (deliver an aug richer than CPU's wall-limited 50%). But EXP-057 (GPU 100%) and EXP-059 (GPU 50%) both show the epoch cost sinks it regardless. The GPU-aug avenue is effectively closed.
- **CPU-side richer aug within the wall**: the only way to add augmentation diversity for FREE (epoch-wise) is on CPU within the 600s wall — but that frontier is mapped (EXP-052/054/055: w3/p=0.5 is the wall-feasible optimum). No headroom.
- The augmentation lever — across CPU (wall-limited) and GPU (epoch-limited) delivery — is now fully exhausted from both sides.

## Next Steps
- **Augmentation is now exhausted from BOTH delivery paths** (high confidence): CPU (free epochs, wall-limited to w3/p=0.5=96.45) and GPU (unlimited coverage, epoch-limited, ~95.6 regardless). Do NOT propose further GPU-aug or CPU-aug variants.
- **The plateau is mapped across every standard lever** (augmentation, capacity ×4 directions, optimizer/schedule/normalization/head/residual/batch/dropout/activation, throughput→epochs). 96.45 appears to be at/near the achievable ceiling for this k=4 ResNet-20 at 300s.
- **Remaining genuine long-shots (low confidence)**: (a) a throughput win that lowers the 8ms GPU-step floor WITHOUT a numerics penalty would buy epochs to spend on richer CPU aug — but EXP-045/046 found reduce-overhead is already accuracy-optimal and max-autotune carries a penalty; this needs a NEW dt reducer (e.g. a hand-fused kernel or a cheaper-but-accuracy-neutral conv path), unclear one exists. (b) Per NEVER-STOP, continue principled long-shots accepting most will be no-improvement on this deeply-mapped plateau. (c) A scalar-knob never isolated (WARMUP_FRAC) as a cheap clean probe — low ceiling.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
- (none — no exit actions defined for this goal)
