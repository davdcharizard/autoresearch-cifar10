# Report EXP-023: Lower label smoothing (LABEL_SMOOTHING 0.1 → 0.05)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md
- **Plan**: plans/plan-023.md
- **Log**: logs/exp-log-023.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within a fixed 300s budget on one H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); bar = **96.32%** (+0.1pp).

## Idea & Hypothesis
Chosen idea: reduce label smoothing 0.1→0.05 (single constant). Selected because the project's strongest, multiply-confirmed insight is that the recipe is **convergence-bound, not overfit-bound** (every add-a-regularizer move — WD↑/Mixup/CutMix/dropout — regressed or nulled), so REDUCING a regularizer is the indicated direction, and LS is the one recipe regularizer never swept. Hypothesis: less target over-softening lets the model commit to sharper predictions within the convergence-bound budget → clears 96.32; graceful no-improvement if LS top-1 effects are within noise.

## Approach
Single-line change: `LABEL_SMOOTHING = 0.1 → 0.05` (train.py L27). Everything else identical to the EXP-012 baseline. Zero compute/param change. No deviations from plan. The plan flagged a key measurement caveat: LS adds a fixed offset to cross-entropy, so test loss is NOT comparable across LS values — judge on best_test_acc only.

## Execution
Single run, no retries. Clean startup (params 4,299,866 unchanged, clean compile, no NaN). Completed 91 epochs (= baseline, throughput-neutral) in 403.3s, peak VRAM 453.8MB.

## Results
- **Primary metric**: best_test_acc = **96.03%** (baseline: 96.22, delta: **−0.19pp**, −0.20%)
- **Observations**: final_test_loss = 0.1564 (< baseline 0.195) — but this is purely the LS-offset artifact (lower LS ⇒ lower CE), NOT a quality gain, as the plan pre-empted. Best epoch hovered ~96.0 across the late schedule; the run never reached the 96.2 region.
- **Analysis**: Hypothesis REFUTED — reducing LS slightly HURT top-1 (−0.19pp, ~within the 0.2pp noise band but on the wrong side and below baseline). So 0.1 label smoothing is near-optimal/load-bearing for top-1, and the "reduce a regularizer" prescription does NOT pan out via LS. This refines the convergence-bound picture: the recipe's regularizer *values* (WD 1e-4, Cutout 16, LS 0.1) are each near interior optima — neither adding regularization (EXP-005/011/018/022, all regress/null) NOR moderately reducing it (LS here) helps. The recipe is simultaneously convergence-bound (can't absorb more regularization) and well-tuned (can't shed it either). This brackets LS the same way EXP-013/021 bracketed Cutout and EXP-016/017 bracketed LR-peak.
- **Key Learning**: Reducing LS 0.1→0.05 slightly hurt (96.03, −0.19pp); 0.1 is near-optimal and the "reduce a regularizer" direction doesn't help via LS — the recipe's regularizer values are all well-tuned interior optima.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) FAILED (96.03); Conds 2–3 skipped per protocol (would pass — clean 403.3s run, train.py-only, params 4,299,866, eval-count 91 == epochs).
- **Review Notes**: Results trustworthy — clean exit, params/scope intact, single-constant change, throughput-neutral (91 ep = baseline so no epoch confound). The lower CE loss was correctly anticipated as an LS artifact, not a quality signal — no false-positive risk. No integrity concerns.
- **Verdict**: no-improvement
- **Verdict Basis**: necessary condition failure (primary metric below bar and baseline).

## Unexplored Avenues
- **LS = 0.0 (remove entirely)**: a more extreme step in the same direction. Given 0.05 already underperformed 0.1, going further to 0.0 is very likely to hurt more (the curve points toward 0.1 being optimal). Low value — would only confirm the interior optimum.
- The label-smoothing axis is now bracketed/settled (0.1 best vs 0.05 worse), consistent with the LR-peak (EXP-016/017) and Cutout-size (EXP-013/021) interior-optimum findings. All scalar recipe hyperparameters are now confirmed near-optimal.

## Next Steps
- **Per-channel input std-normalization** (std=(1,1,1)→true CIFAR std) — the last untried convergence-neutral single-knob probe; confidence LOW it gains (Conv→BN almost certainly absorbs the rescale → expected null), but it cleanly closes the input-normalization axis in one run. (medium confidence clean null; low confidence gain.)
- **BlurPool / anti-aliased downsampling** (Zhang 2019; brainstorm-023 Idea 3) — the one remaining radical convergence-neutral *generalization* mechanism (no stochastic penalty); higher ceiling but carries the EXP-015 compile-graph epoch-cost/attribution risk. The best candidate if a genuine architectural lever is wanted after the cheap probes. (low-medium confidence; implementation + confound risk.)
- After std-norm (and optionally BlurPool), ~16 axes are exhausted and every scalar hyperparameter is bracketed — the honest call is the **96.22 plateau is convergence/generalization-bound at fixed k=4 capacity in 300s**. (high confidence the plateau is real.)

## Exit Action Results
<!-- No exit actions defined for this goal. -->
