# Report EXP-011: Mixup (mild α=0.2) GPU-vectorized, stacked on Cutout + compile enabler
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md
- **Plan**: plans/plan-011.md
- **Log**: logs/exp-log-011.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s budget on one H20, editing only `train.py`. Higher is
better. Baseline **96.00%** (EXP-003); success bar ≥ **96.10%**.

## Idea & Hypothesis
Chosen idea: add mild Mixup (α=0.2, per-batch λ) GPU-vectorized and stacked on the existing Cutout, with the
validated `torch.compile(reduce-overhead)` enabler for epoch budget. Rationale: data augmentation is the project's
only proven non-capacity lever (Cutout ~+1.1pp), and the "regularization saturated" finding (EXP-005) was about
*weight decay*, not augmentation — so the augmentation axis was under-explored (only Cutout tried). Mixup is the
canonical *complementary* mechanism (interpolation vs Cutout's occlusion). Hypothesis: improved generalization
lifting `best_test_acc` to ~96.1–96.4%.

## Approach
`train.py`-only edits: added `MIXUP_ALPHA=0.2`; in the training loop after the Cutout line, per-batch Mixup
(λ~Beta(0.2,0.2), batch permute, input lerp, permuted targets) with the loss as the Mixup convex combination of two
label-smoothed cross-entropies; added the compile enabler with the training forward routed through it; eval kept
eager. Mixup mixing + loss live in the eager loop (only the forward is compiled), so the per-step varying λ poses no
CUDA-graph risk. Mixup is parameter-free → num_params unchanged (a built-in sanity check). Seed-consistent (Beta→CPU
gen, randperm→CUDA gen, both seeded).

## Execution
One run, no retries/errors, clean compile, exit 0. Steady-state **dt = 9ms/step (~14,877 img/s)** ≈ compiled-k4 —
Mixup's per-step cost (Beta scalar + permute + lerp + 1 extra CE) is negligible. Fit **88 epochs** — a fair,
fully-converged test (not epoch-starved). Train loss read higher/noisier (expected from mixed targets), no NaN.
Completed 398.3s, peak VRAM 453.7 MB, params 4,299,866 (unchanged).

## Results
- **Primary metric**: **95.86%** (baseline 96.00, delta **−0.14 pp**, −0.15%) — below the +0.1 bar and baseline.
- **Observations**: best 95.86 ≈ compiled-k4 95.92 (EXP-007) within the ~0.2pp noise band → mild Mixup added no
  accuracy. **final_test_loss 0.2898** is markedly higher than baseline 0.204 / compiled-k4 0.208 — this is the
  expected Mixup artifact (training on interpolated/soft targets yields less-confident predictions → higher CE on
  one-hot test labels), NOT a classification regression, since accuracy is flat.
- **Analysis**: A fair, converged negative (88 epochs, dt 9ms). At α=0.2, Mixup neither helped nor hurt accuracy.
  Mechanistically this is consistent with the established picture: the k=4 model is already well-regularized by
  Cutout + label smoothing + WD (overfit gap is small; loss was 0.204), so adding a second augmentation regularizer
  yields diminishing returns rather than a new gain — unlike the original Cutout add, which fixed a then-large
  overfit gap (0.25→0.20, EXP-002/003). A stronger α might regularize more but would slow convergence further
  against the fixed budget (the capacity-axis failure mode) and is unlikely to clear +0.1 given the α=0.2 null.
  This makes the **augmentation axis** (beyond Cutout) effectively exhausted too — an eighth axis at the plateau.
- **Key Learning**: Mild Mixup (α=0.2) stacked on Cutout adds no accuracy to the converged k=4 net (fair 88-epoch
  run, 95.86 ≈ compiled-k4 95.92); the model is already regularization-saturated, so a complementary augmentation
  yields diminishing returns rather than a gain.

## Verification
- **Conditions**: Cond 1 (clean completion in budget) PASS; Cond 2 (≥96.10) **FAIL** (95.86); Cond 3 skipped.
- **Review Notes**: Trustworthy — clean single run, frozen eval, seed 42, eval once/epoch (88=88), num_params
  UNCHANGED (4,299,866) confirms the parameter-free augmentation (no architecture change), compile is
  execution-only with EXP-007 null standalone effect → the null is attributable to Mixup, not under-training (88
  epochs, fully converged) or compile. The elevated test loss is a known Mixup soft-target artifact, not a metric
  problem (the metric is accuracy). No reward-hacking surface. −0.14pp within noise.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, trustworthy, well-trained run; primary metric below the +0.1 bar (cond 2 failed).

## Unexplored Avenues
- **Stronger Mixup (α=0.5–1.0) or CutMix** — more aggressive interpolation could regularize harder, but slows
  convergence more against the fixed budget; given the α=0.2 null on an already-regularized model, low expected
  value. Low priority.
- **Per-channel input std normalization** (fix `std=(1,1,1)`) — near-free, untried, but BN likely absorbs it.
  Cheapest remaining clean probe. Low priority.
- **LR-schedule / optimizer micro-tuning on k=4** (peak-LR, warmup) — only WD was swept (EXP-005). A genuinely
  untried recipe knob; a single run tests one sweep point. Low-medium priority.
- The augmentation idea (Cutout + a complementary aug) is now effectively exhausted at this budget.

## Next Steps
1. **Per-channel input std-norm** (fix `std=(1,1,1)` → CIFAR std) — *low confidence*; near-free, textbook, untried,
   though BN likely absorbs it. Cheapest remaining clean probe. *Best next experiment* (low cost, clean attribution).
2. **LR-schedule micro-tuning** (e.g. peak-LR 0.2→0.3 or warmup fraction) — *low-medium confidence*; an untried
   optimization knob on the converged sweet spot.
3. **Accept 96.0% as a hard plateau** — *strategic*; EIGHT axes now exhausted (width, weight-decay-regularization,
   weight-averaging, training-length, channel-attention, compiled-capacity, activation, and augmentation-beyond-
   Cutout). Remaining moves are noise-scale. After the two cheap probes above, declaring the plateau is the honest
   scientific conclusion for this model/recipe/budget.

## Exit Action Results
- None defined for this goal — skipped.
