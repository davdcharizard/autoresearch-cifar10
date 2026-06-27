# Experiment Report EXP-025: Final-phase clean-data alignment (FixRes-style aug-off tail)

- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md
- **Plan**: plans/plan-025.md
- **Exp-log**: logs/exp-log-025.md
- **Verdict**: **no-improvement** (95.84 vs baseline 96.71, Δ −0.87)

## Goal

Maximize CIFAR-10 `best_test_acc` (%) of the wide ResNet-20 within the fixed 300s timed training budget; higher is better. Baseline at experiment time: **96.71** @ 1990397 (EXP-006 recipe). Success bar: ≥ 96.81 (+0.1pp absolute).

## Idea & Hypothesis

**Idea**: First out-of-recipe intervention after recipe-space closure (EXP-024) — change the time-structure of the training DISTRIBUTION rather than any constant. Train the final 15% of the budget (ALIGN_FRAC = 0.85, ~21 epochs) on the same train split but with EXACTLY the evaluator's transform (ToTensor + Normalize, no crop/flip/TrivialAugment/RandomErasing), so weights and BN running stats align to the distribution the frozen evaluator measures. Mechanism grounding: FixRes (Touvron et al. 2019, arXiv 1906.06423 — end training on the test distribution) + BN-stat recalibration (running stats under heavy aug are biased vs clean images).

**Hypothesis**: Signatures baseline-identical; transient dip ≤1pp for 1–2 post-switch epochs; then a plateau ABOVE the baseline family — best_test_acc ≥ 96.81 with final-7 median ≥ 96.6. A converged miss kills the alignment mechanism class for this setup.

## Approach

Three-part diff to train.py (+29/−1), everything else byte-identical: (1) `ALIGN_FRAC = 0.85` constant; (2) a second dataset/loader (`clean_set`/`clean_loader`) with the exact Eval transform and identical DataLoader kwargs; (3) epoch-level source switch in the while loop keyed to `total_training_time >= ALIGN_FRAC * TIME_BUDGET_S`. The timed step body, schedule, optimizer, compiled graph (same input signature), and eval were untouched. Loader-side change only — per-image CPU cost DROPS in the tail (transforms removed), the measured-safe direction per EXP-013.

## Execution

One run, pristine first try (task bobi9piiy): launched 2026-06-10T13:04:18Z via the standard composite (GPU-0 pre-check → launch → inline 15s watchdog), TRAIN_EXIT rc=0 at 13:12:49Z. No retries, no contention kills, no errors. Post-hoc profile: **0 of 267 windows >30ms, mean 22.4ms, expected epochs 139.1 vs 139 actual** — not even the documented one-window switch artifact appeared (clean-loader spin-up fully absorbed outside dt). Signatures baseline-exact: dt 22.4ms, VRAM 1613.0MB, params 4,286,026, startup 13.2s, total 487.8s, 13406 steps.

## Results

**best_test_acc 95.84 | final 95.76 | final_test_loss 0.2185** — the worst miss since EXP-018 (−0.99), and the twentieth consecutive non-improvement (EXP-007…025).

The trajectory tells the whole story. The switch fired as designed at ~step 11500 (~ep118): train loss dropped 0.799→0.572 in one watchdog window — clean data is much easier — and then FROZE at 0.5033 for the remaining ~1900 steps. Test accuracy rose 95.45→95.84 within ~5 post-switch epochs (this is the predicted BN/alignment boost, ~+0.35 over the pre-switch level), then hard-flatlined at 95.73–95.84 for the final ~16 epochs while **test_loss ROSE monotonically 0.2096→0.2185**. Train loss frozen at floor + test loss rising + test acc flat = the model overfit the 50k un-augmented images in the tail and stopped generalizing.

The comparison that kills the mechanism: over the same final ~21 epochs, the baseline's AUGMENTED cosine tail climbs ~+1.3pp (family ~95.3 at ep118 → 96.6–96.7 at ep139). The clean tail delivered its one-time +0.35 alignment boost and then forfeited the entire remaining climb. Net: −0.87. The augmented data in the tail is not a distribution-discrepancy cost to be removed — it is the LOAD-BEARING pressure that drives the final ~1pp of the plateau's rise. With lr ≤ 0.022 in the tail, WD + LS alone were nowhere near enough to prevent clean-set overfitting.

Why FixRes does not transfer here: in FixRes the train→test gap is a RESOLUTION/crop-statistics mismatch that fine-tuning at test preprocessing genuinely closes, on top of a model trained to convergence. Here there is no resolution gap (32×32 both sides), the "gap" is the augmentation itself, and the budget regime means the tail is not a post-convergence fine-tune — it IS the convergence phase. Removing augmentation during the convergence phase is just a late pressure-down, and the pressure axis's verdict (EXP-015: full-run pressure-down −0.30) extends to time-varying doses with an even worse outcome because the tail is where the max-statistic harvests.

Hypothesis validation: signatures predicted correctly ✓; sharp train-loss drop predicted correctly ✓; BN boost appeared (visible, ~+0.35) ✓; plateau-above-family prediction REFUTED — the plateau formed ~0.9pp BELOW the family. The mechanism class (final-phase distribution alignment) is dead for this recipe.

## Verification

- **Pre-condition (contention profile)**: PASS — 0/267 windows >30ms, epochs 139 within ±3 of expected 139.1. Run analyzable.
- **Condition 1 (best_test_acc ≥ 96.81)**: **FAIL** — 95.84. Genuine converged result: full-length run, clean profile, eval count correct, params 4,286,026. Not an infra false-failure.
- **Conditions 2–3**: skipped per first-failure-stop; both would have passed (rc=0 @ 487.8s ≤ 600; eval_lines 139 = num_epochs).
- **Verdict basis**: valid run, verification condition failed → `no-improvement`, metric 95.84.

## Key Learning

The augmented tail is load-bearing: under a time-budget regime where the cosine tail IS the convergence phase, the final ~1pp of plateau rise comes from continued training pressure (augmentation), not from train→test alignment. Swapping the tail to clean data buys a one-time ~+0.35 BN-alignment boost and then caps the plateau via immediate overfitting (train loss floors, test loss rises). FixRes-class end-on-test-distribution fine-tuning requires a converged model and a real preprocessing gap (e.g. resolution) — neither exists here. Late pressure-down loses worse than constant pressure-down.

## Unexplored Avenues

- **Shorter alignment (ALIGN_FRAC 0.95–0.97, ~4–7 epochs)**: could capture the +0.35 BN boost before overfitting sets in — but the boost was measured from a level already ~0.9pp below where the augmented tail would have been by then, so even a perfectly timed switch likely nets negative. Low promise; only worth it as a 2–3-epoch terminal flick if nothing else remains.
- **BN-stat-only recalibration (no weight updates)**: forward passes in train mode on clean data for ~1 epoch at lr=0 (or momentum-free stat refresh) at the very end — captures the alignment boost with zero overfitting risk. Costs ~2s of budget. The boost observed here (+0.35 from a depressed level) suggests the pure-BN component is small but possibly positive from the TOP of the augmented plateau. The cleanest surviving fragment of this mechanism class.
- **Mixing instead of switching**: interleave clean batches at low ratio throughout — but this is a constant pressure-down (EXP-015 territory), expected negative.

## Next Steps

1. **BN running-stat recalibration as a terminal micro-phase** (forward-only, ~1 epoch of clean data at the very end of the budget, no optimizer steps) — the only fragment of the alignment mechanism that cannot overfit; tests whether the +0.35-class boost exists from the top of the augmented plateau. Confidence: medium-low.
2. **Objective shaping: LS anneal 0.1→0 over the final 15%** (brainstorm-025 idea 2) — the objective-side late-phase change; weaker mechanism (calibration ≠ accuracy) and now shadowed by this result's lesson that late pressure-drops lose. Confidence: low.
3. **Baseline variance replicate** (brainstorm-025 idea 3) — measurement-only, calibrates the noise floor under twenty misses spanning −0.05…−0.99; cannot move the baseline by construction. Worth running when intervention candidates thin out further. Confidence: high (as measurement), zero (as improvement).
