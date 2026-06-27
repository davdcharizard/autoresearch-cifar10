# Report EXP-029: Clean-data BN running-stat recalibration before every eval
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-029.md
- **Plan**: plans/plan-029.md
- **Log**: logs/exp-log-029.md

## Goal

Maximize CIFAR-10 best_test_acc (%) within the fixed 300s training budget. Baseline 96.71 @ 1990397; bar 96.81. σ calibration (EXP-027): baseline mean ≈96.57, σ ≈0.16 ⇒ true effects ≥ +0.3 needed. Question tested: is the augmented-train/clean-test distribution mismatch in the BN running stats the evaluator consumes worth several tenths — i.e., was the BN-alignment share of EXP-025's measured +0.35 switch boost recoverable without that experiment's clean-data-training failure mode?

## Idea & Hypothesis

The recipe trains on heavily augmented batches (crop pad-4, flip, TrivialAugmentWide, RandomErasing) but is tested clean; the BN running stats consumed at eval estimate the AUGMENTED distribution. Idea: before every per-epoch eval, re-estimate all BN running stats on 8,192 clean (test-transform) train images — forward-only, no_grad, momentum=None cumulative average — leaving training structurally untouched (training never reads running stats). Anchors: AdaBN (eval-domain stats), SWA's BN re-estimation, and EXP-025's +0.35-within-1-epoch alignment boost. Hypothesis: trajectory-wide upward shift; plateau ≥ +0.25 over the baseline mean; best ≥ 96.81. Pre-registered alternative: a shift < +0.1 would mean EXP-025's boost was feature adaptation, not stat alignment.

## Approach

train.py only (+47 lines): constants (16 batches × 512), startup build of GPU-resident clean batches (first 8,192 train images, fixed slice, ~100MB VRAM, outside the timed budget), `recalibrate_bn` helper (reset stats → momentum None → 16 no-grad bf16 train-mode forwards through eager base_model → restore momentum), one-line call before the existing eval. Diff-verified: nothing inside the timed step body. Key decisions: eager base_model (no inductor guard churn), momentum restored inside the helper, fixed slice (no seed surface).

## Execution

One run, rc=0, no retries. All gates quiet: dt 22.7ms ≈ baseline, 137 epochs (vs 137.3 expected from throughput — exact), startup 16.2s (+3.5 clean-tensor build), VRAM 1712.8 (+100 as predicted), recalib wall ≈ +48s (total 541.2s ≤ 600). Post-hoc profile showed 4/263 windows >27ms (letter of the ≤2 rule exceeded); NO rerun — documented judgment: epochs matched expectation exactly, train-loss trace family-identical, and the −10.93 miss is in EVAL accuracy, a quantity GPU contention cannot influence; the rerun rule exists to prevent contention-induced false failures near the bar.

## Results

- **Primary metric**: 85.78 (baseline: 96.71, delta: −10.93, −11.30%)
- **Observations**: Training was provably untouched (train losses and dt byte-similar to the baseline family; 137 epochs). Every recalibrated eval reads far below family: ep1 −4, ep30 −15, ep60 −16.6, ep90 −16.3, ep110 −13.2, converged plateau 85.5–85.8 (−10.9) with test_loss 0.4916 vs ~0.185. The gap PEAKS mid-training (high-LR phase, maximum stat sensitivity) and shrinks as the anneal stabilizes activations — but never closes. Implementation was verified correct (sane ep1, smooth trajectory, tight final-7 spread 0.16 — this is a converged model evaluated under different normalization constants, not a corrupted one).
- **Analysis**: The hypothesis is refuted with a large inverted sign, and the refutation is mechanistically clean because training was isolated from the change. The network's downstream weights and BN affines are calibrated to the normalization constants present in every training forward pass — the AUGMENTED batch statistics. Eval fidelity therefore requires REPRODUCING training-time constants, not matching the eval-input distribution: clean-data constants mis-scale every one of the 20 BN layers' outputs relative to what downstream layers expect, and the perturbation compounds. The augmented-EMA running stats are not a stale approximation to be "fixed" — they are part of the learned function. This retro-explains EXP-025's +0.35: that boost required the WEIGHTS to adapt under clean batch stats (training on clean data), i.e., it was feature adaptation, exactly the pre-registered alternative. It also explains why SWA re-estimates BN with the augmented train loader. The augmented↔clean activation-statistics gap this measures is enormous (~11pp of function damage) — heavy augmentation (TA+RE) creates a deeply augmentation-specific internal calibration.
- **Key Learning**: BN running stats are part of the learned function, calibrated jointly with the weights to the augmented training distribution — eval must reproduce training-time normalization constants, and any eval-time stat substitution is function damage, not alignment.

## Verification

- **Conditions**: condition 1 failed (best_test_acc 85.78 < 96.81); conditions 2–3 skipped per first-failure-stop (both would pass: 541.2s ≤ 600; eval_lines 137 = num_epochs)
- **Review Notes**: results confirmed trustworthy — params exact, epochs at throughput expectation, train-side signature baseline-identical isolating the effect to the eval-consumed stats; profile deviation (4/263 slow windows) documented and verdict-independent (eval-side deficit is contention-immune). No false-failure risk: the deficit is 68σ.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure — valid, clean, conclusively-explained negative; no hard-constraint violations (train.py only, GPU 0, eval once/epoch, ≤600s, no test data touched by recalibration)

## Unexplored Avenues

- **Recalibration on AUGMENTED train data**: would test pure "stat freshness" rather than distribution alignment — but with BN momentum 0.1 (EMA half-life ~7 batches) the running stats are already ~0.15s fresh; expected effect ≈ 0. Not worth a loop.
- **Mixed-distribution recalibration (interpolate augmented/clean stats)**: any point on that line is bracketed between 96.71 (pure augmented, baseline) and 85.78 (pure clean); monotonicity strongly implied by the mechanism — the baseline endpoint is the optimum. Closed by bracketing logic.
- **Making train-test stats agree by REDUCING augmentation** instead: that is the augmentation-pressure axis, closed bidirectionally (EXP-009/013/015/025).

## Next Steps

1. **Re-scan untouched interaction space with the "calibrated-function" lens** (high confidence in necessity): EXP-029 sharpens the recipe-as-local-optimum picture — even the eval-side buffers are jointly calibrated. Remaining ideas must add capability without perturbing ANY calibrated quantity: heat, epochs, numerics, noise, and now normalization constants.
2. **Concat avg+max pooling head** (medium-low): the one untried micro-architecture change with short-budget CIFAR pedigree (cifar10-fast); nearly free in dt; magnitude prior +0.1–0.2 is sub-screen but the candidate pool is thin.
3. **Objective shaping beyond LS (e.g., logit normalization / temperature on the CE)** (low): the only named-open axis never probed; interacts with LS 0.1 (calibrated); screen hard against the heat law before planning.

## Exit Action Results
