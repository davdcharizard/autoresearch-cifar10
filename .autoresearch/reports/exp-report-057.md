# Report EXP-057: Full-coverage faithful GPU AugMix (multi-chain + Beta clean-mix)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-057.md
- **Plan**: plans/plan-057.md
- **Log**: logs/exp-log-057.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within a fixed 300s GPU-time (Σdt) budget on a single H20, editing only train.py. Baseline = **96.45** (EXP-054, commit 86161d9); improvement bar = **96.55** (+0.1pp).

## Idea & Hypothesis
Chosen idea: replace EXP-054's CPU `RandomApply([AugMix() w3], p=0.5)` (50%-subset AugMix, wall-forced) with a faithful **GPU-side AugMix at full coverage** — `gpu_augmix(x)`: build W independently-augmented chains (per-sample affine rotate/shear/scale + photometric brightness/contrast via `grid_sample`), convex-mix them with per-image Dirichlet(1) weights, then Beta(1,1)=Uniform convex-mix the blend with the ORIGINAL clean image. This restores AugMix's two defining shift-bounding properties (multi-chain diversity + clean-mix) that EXP-056's harsh single-stack omitted, now affordable on the idle GPU. Hypothesis: full-coverage exposure of ALL images to the proven gentle diverse augmentation (vs EXP-054's 50%) clears 96.55; the clean-mix bounds the per-image shift so it should not over-regularize like EXP-056.

## Approach
- Added `_aug_chain(x)` (one affine+photometric pass, gentle magnitudes: rot ±12°, shear ±0.1, scale [0.9,1.1], brightness ±0.1, contrast [0.85,1.15]) and `gpu_augmix(x, width)` (Dirichlet-mixed chains + Beta clean-mix, returns channels_last) after `cutout_batch`. Wired `inputs = gpu_augmix(inputs, width=2)` into the train loop before Cutout. Removed the CPU AugMix line (CPU → crop+flip+ToTensor+Normalize).
- **Deviation from plan**: launched W=3 (plan default) but the Milestone-2 gate tripped (steady dt 11.0ms → projected ~65 epochs < 76 floor; in-loop grid_sample cost ~1ms/chain, higher than the standalone smoke's ~0.5ms). Dropped to **W=2** per the plan's contingency. No model/optimizer/schedule/seed/batch/compile changes; num_params unchanged (4,299,866).

## Execution
- **Run 1 (W=3)**: aborted on the epoch-wall gate (dt 11.0ms, ~65 ep, wall/Σdt 1.31× — genuine compute, not contention).
- A first W=2 relaunch (bixa5005x) failed exit 144: its `pkill -f train.py` self-matched the launching shell and also killed an orphaned Run-1 python that survived TaskStop. Relaunched cleanly without pkill after verifying GPU 1 idle.
- **Run 2 (W=2, bg96yhlp4)**: clean on GPU 1. Early gate PASSED (dt 10ms → 76.9 ep projected; wall/Σdt 1.33×, no contention; test_acc trajectory normal). Ran to completion, exit 0. 75 epochs, 29004 steps, total 382.6s, peak VRAM 452.9MB, 0 NaN/traceback.

## Results
- **Primary metric**: **95.64%** (baseline 96.45, delta **−0.81pp**, −0.84%)
- **Observations**: final_test_loss 0.2115 — between EXP-054's 0.1968 (50% subset) and EXP-056's 0.224 (harsh full-coverage stack). The clean-mix demonstrably softens the distortion (EXP-057 −0.81 vs EXP-056 −1.06), but full coverage still lands well below the 50%-subset baseline. Convergence was smooth and complete (best plateaued at ~95.6 from ep72; not under-trained), so the deficit is a regularization-strength effect, not an epoch-budget artifact.
- **Analysis**: Hypothesis REJECTED. Exposing 100% of images to AugMix — even faithful, shift-bounded AugMix — over-regularizes the train signal at this 300s/~75-epoch budget. The EXP-054 50%-subset coverage is a genuine optimum, not merely the best CPU could afford. The "coverage>50% is the untried beneficial direction" premise was wrong: EXP-055 showed coverage<50% hurts, and EXP-057 now shows coverage>50% also hurts — 50% is a true interior peak.
- **Key Learning**: Full-coverage AugMix over-regularizes at the 300s budget even with the shift-bounding clean-mix; EXP-054's 50% coverage is a real interior optimum, not a wall artifact.

## Verification
- **Conditions**: Necessary condition 1 (`best_test_acc >= 96.55`) FAILED (95.64). Conditions 2 (clean completion: 382.6s<600, params correct, 0 NaN) and 3 (scope: train.py only, no new deps, seed 42, uncontended GPU) both hold.
- **Review Notes**: Results confirmed trustworthy — steady 10ms dt (no contention spikes), smooth converged trajectory, fair dt-budget, single editable file. The regression is real, not a measurement artifact.
- **Verdict**: **no-improvement**
- **Verdict Basis**: valid run, necessary condition 1 failed (metric below bar by a wide margin); no hard-constraint violation.

## Unexplored Avenues
- **GPU AugMix on a stochastic ~50% subset (brainstorm candidate 3)**: faithful clean-mixed GPU AugMix applied to only ~50% of each batch — replicates the EXP-054 winner's coverage but with full-strength GPU chains. By construction sits near 96.45; could match-or-slightly-beat if GPU chains are richer than CPU's. Lower ceiling but the only remaining same-idea variant not yet tested.
- **Coverage between 50% and 100%** (e.g. p=0.6–0.7): EXP-057 (100%) and EXP-054 (50%) bracket the peak; a single intermediate point could confirm whether the optimum is exactly ~50% or slightly higher. Low EV given both endpoints favor ~50%.
- The augmentation-coverage/diversity lever is now mapped on BOTH sides of 50% — this idea family is effectively exhausted for clearing +0.1pp. Future loops should pivot OFF the augmentation axis.

## Next Steps
- **Pivot away from augmentation** (high confidence it's exhausted): the CPU lever was mapped at its frontier (EXP-052/053/054/055) and the GPU unlock has now been mapped on coverage (EXP-056 harsh, EXP-057 faithful-full). All roads lead back to the 96.45 50%-subset peak.
- **CRITICAL — do NOT propose EMA/SWA, LR-schedule, capacity, normalization, label-mixing, Cutout strength/pattern, or downsampling**: every one of these is a recorded closed axis in goal-learnings (EMA/SWA is a count≥3 recurring High-Importance failure — it approaches cosine-to-0 from below but never beats it on top-1). The problem is heavily mapped.
- **Activation function swap (ReLU → SiLU / Mish / GELU)** — architecture-internal, throughput-cheap, frequently +0.1–0.3pp on CIFAR ResNets; appears NOT yet tried (verify against the TSV in brainstorm) (medium confidence).
- **Stochastic depth (DropPath on residual branches)** — a regularizer with a DIFFERENT mechanism than input augmentation (drops whole blocks, not pixels), cheap, may help a regularization-saturated-on-aug net; verify untried (medium-low confidence).
- **GPU AugMix on 50% subset** — low-risk hedge to confirm the GPU path can at least match 96.45; low ceiling for beating the bar (low confidence).

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
- (none — no exit actions defined for this goal)
