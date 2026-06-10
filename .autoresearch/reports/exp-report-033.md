# Report EXP-033: Augmentation cooldown (disable TrivialAugment + Cutout for the final ~15%)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-033.md
- **Plan**: plans/plan-033.md
- **Log**: logs/exp-log-033.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within the fixed 300s training-compute budget on a single H20, editing only `train.py`. Baseline = **96.22%** (EXP-012, commit 6c417a4); pass bar = baseline + 0.1 = **96.32%**.

## Idea & Hypothesis
Chosen idea: a **time-varying augmentation schedule** — keep the full pipeline (RandomCrop+Flip+TrivialAugment+Cutout) for the first 85% of the budget, then for the final 15% disable the strong distribution-shifting augs (TA + Cutout), keeping only RandomCrop+Flip. This is the YOLOX "close mosaic" principle ported to CIFAR. Hypothesis: the low-LR cosine tail becomes a clean-data fine-tune that aligns both the weights AND the BN running statistics with the clean test distribution, lifting top-1 above the 96.32 bar at an unchanged ~91 epochs / dt~8ms / 4,299,866 params. The augmentation SCHEDULE was the one untouched lever in 33 experiments (every prior run used a static pipeline).

## Approach
Four localized edits to `train.py` (architecture untouched): (1) added `COOLDOWN_FRAC = 0.15`; (2) added a second CPU transform `train_tf_clean` = full pipeline minus `TrivialAugmentWide()`; (3) added an `aug_cooled` flag and an epoch-boundary switch that, once `total_training_time/TIME_BUDGET_S ≥ 0.85`, mutates `train_set.transform = train_tf_clean` (propagates to the next epoch's freshly-forked dataloader workers), sets the flag, and prints an observable marker; (4) gated the in-loop `cutout_batch` behind `if not aug_cooled` so TA and Cutout switch off together. Compute-neutral, hierarchy- and recipe-preserving. Everything else unchanged (PEAK_LR 0.2, cosine-to-0, WD 1e-4, LS 0.1, seed 42, torch.compile).

## Execution
Single run, no retries, exit 0 in 411s wall (300s training). The cooldown marker fired exactly once at `ep 77 frac 0.85` (as designed). Convergence tracked baseline normally through the full-aug phase (ep40 91.0%, ep70 94.9%, ep76 95.43% = last full-aug epoch), dt steady at 8ms throughout including post-cooldown (disabling the CPU TA op did not change the GPU-bound step time). No NaN, no errors.

## Results

- **Primary metric**: best_test_acc = **96.10%** @ ep86 (baseline 96.22%, delta **−0.12pp, −0.12%**); bar 96.32 not cleared.
- **Observations**: The clean-data cooldown produced a steep tail climb then a plateau-and-decline: ep76(full-aug)=95.43 → ep78=95.96 → ep82=96.01 → ep84=96.06 → **ep86=96.10 (peak)** → ep87=96.04 → ep88=96.00 → ep90=96.01. num_epochs 90 (≈ baseline ~91), dt ~8ms, params 4,299,866, final_test_loss 0.2000 (≈ baseline 0.195). peak_vram 453.8 MB.
- **Analysis**: The hypothesis is **partially supported but the variant did not clear the bar**. The cooldown mechanism worked and the tail climbed steeply over the first ~9 cooldown epochs — but two facts qualify the result: (1) the loss did NOT improve (0.2000 ≈ 0.195), and (2) the tail **peaked at ep86 then declined** for the final ~4 epochs. A standard cosine-to-0 tail is monotonically improving (LR→0 always sharpens the final epochs); a peak-then-decline is the signature of the clean-data phase beginning to **mildly overfit / saturate** once augmentation has been off for ~9 epochs. So the 15% window is too LONG: it captures the distribution-alignment lift but then overshoots into clean-tail overfitting, and the net peak (96.10) landed just below the 96.22 baseline. Because the pre-cooldown trajectory was itself ~baseline-or-slightly-below this run (ep76 95.43%), and there is no same-seed no-cooldown control, part of the 95.43→96.10 climb is ordinary cosine-anneal lift conflated with the cooldown effect — the clean attributable signal is the SHAPE (peak-then-decline), not a clean +Δ over baseline. Fits the firmly-established generalization-bound-at-fixed-capacity plateau: the lever moved the tail but not the ceiling.
- **Key Learning**: Augmentation cooldown (disabling TA+Cutout in the low-LR tail) measurably reshapes the final-phase trajectory — a steep clean-data climb that PEAKS ~9 epochs in then declines — but a 15% window overshoots into clean-tail saturation and lands at 96.10, just below baseline; the peak-then-decline points to a shorter window as the refinement.

## Verification

- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) **FAILED** — 96.10 < 96.32 (and −0.12pp below the 96.22 baseline). Cond 2 (clean completion <600s, 0 Traceback) passed. Cond 3 (only train.py; params 4,299,866 unchanged; eval-count 90 == epochs; core torch; seed 42) passed — no constraint violated.
- **Review Notes**: Results trustworthy. The cooldown fired once at the correct fraction (ep77/0.85), throughput was neutral (90 ep, dt 8ms) → a clean, fair, cleanly-attributable test of the 15%-cooldown variant. No parsing/stale-output/integrity concerns; the intervention is on the data-augmentation schedule (the goal's intended class), not a measurement-gap exploit.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid, trustworthy result; primary verification condition (clear the bar) failed; no hard constraint violated.

## Unexplored Avenues
- **Shorter cooldown window (~8-10%)** — the highest-value follow-up. The tail PEAKED at ep86 (~9 cooldown epochs in) then declined, indicating the 15% window overshoots into clean-tail saturation. A shorter window (e.g. COOLDOWN_FRAC 0.08-0.10, ending near the observed peak) would capture the distribution-alignment lift without the late overfit — directly motivated by THIS run's trajectory, and the augmentation-schedule axis is NOT exhausted by this single point.
- **Milder cooldown: drop ONLY TrivialAugment, KEEP Cutout** — Cutout is a label-preserving occlusion (not a distribution shift like TA's solarize/posterize/shear). Keeping Cutout through the tail may retain enough regularization to prevent the clean-tail overfit while still removing the strong distribution shift.
- **BN-only recalibration tail** — instead of training on clean data, freeze weights and only recompute BN running stats on clean (un-augmented) data for a few hundred steps at the end (cf. SWA's BN-update). Isolates the BN-distribution-alignment component from the weight-fine-tuning component.

## Next Steps
- **Re-run with a shorter cooldown (COOLDOWN_FRAC ≈ 0.08-0.10)** — confidence: medium-high. The peak-then-decline trajectory is direct evidence the window is too long; a shorter one targeting the ep86-equivalent peak is the obvious next probe and the augmentation-schedule axis remains open.
- **Variant: keep Cutout, drop only TA in the tail** — confidence: medium. Tests whether retaining occlusion regularization prevents the clean-tail saturation.
- **If both cooldown variants stall below the bar, treat the augmentation-schedule axis as closed** and accept the 96.22 plateau as the k=4/300s ceiling — confidence: medium that the plateau is real (now reinforced by 34 experiments across ~25 axes).

## Exit Action Results
<!-- No exit actions defined for this goal. -->
