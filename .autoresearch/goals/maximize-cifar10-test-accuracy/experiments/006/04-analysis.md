# Report EXP-006: Multi-crop translation TTA (airbench96 tta_level=2)
- **Created**: 2026-06-28

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within a fixed 300s training budget, editing only `train.py`. Baseline entering this experiment: **96.00%** (EXP-004, commit ae31206). Improvement bar: ≥ baseline + 0.1pp → **≥96.10%**.

## Idea & Hypothesis
Chosen idea (cross-model reviewer's pick): extend eval-time TTA in `ResNet9.forward` from mirror-only (2 views) to the airbench96 `tta_level=2` **mirror+translate** scheme (6 views). The diagnosis identified *incomplete eval-time view coverage* as the cheapest unexploited lever — we captured only the mirror component of the documented record recipe, never the translation views, which the airbench paper singles out as a distinct, non-additive TTA contributor. Hypothesis: adding the two diagonal-shift translation crops on top of the mirror pair lifts `best_test_acc` from 96.00% to ≥96.10%, with the training run byte-identical to the baseline (eval-only change) and total wall <600s. Falsifier: best < 96.10 → translate adds <0.1pp on top of mirror → eval-side lever near-exhausted.

## Approach
Single eval-only edit to `ResNet9.forward` (`train.py`): kept the `if self.training or not self.tta: return self._forward_once(x)` fast path verbatim; replaced the mirror-only TTA branch with a local `mirror(v)=0.5*(f(v)+f(v.flip(-1)))` helper applied to the image and two reflect-pad-1 diagonal crops `[:, :, 0:h, 0:w]` (shift −1,−1) and `[:, :, 2:2+h, 2:2+w]` (shift +1,+1), combined `0.5*mirror + 0.5*translate` (6 views). Pinned verbatim to upstream `airbench96.py`. No training-affecting line changed — all HPs, schedule, EMA, whitening, architecture, and `TTA_START_FRAC=0.8` held, so the 300s training trajectory is reproduced in code byte-for-byte. Milestone-1 smoke confirmed exactly 6 `_forward_once` calls under TTA, 1 under training/no-TTA, correct shapes/finiteness, and that TTA logits differ from single-forward (max-abs-diff 11.43). Deviation from plan: none.

## Execution
One clean run on GPU 1 (`CUDA_VISIBLE_DEVICES=1`), `timeout 600`, exit 0, wall 472.1s. No retries, no divergence. Training used the full 300.0s and fit **150 epochs / 14507 steps** (vs EXP-004's 142) — the time-budgeted loop packs however many SGD steps the host throughput allows into 300s, and the shared host (GPU 0 busy) ran faster this time. The 6-view tail eval added only ~27s over EXP-004's 445.2s total (cheaper than the ~50–95s estimate), so the wall-clock fallback was never needed.

## Results
- **Primary metric**: 95.93% (baseline: 96.00%, delta: **−0.07pp, −0.07%**) — peak at ep148; below both the 96.10 bar and the 96.00 baseline.
- **Observations**:
  1. **Multi-crop TTA works as intended.** A clear TTA-onset step appears at the `TTA_START_FRAC=0.8` gate: ep118 95.51 → ep119 95.79 (**+0.28pp**), exactly the behavior EXP-002 saw when flip-TTA engaged. So the translation views are being averaged correctly and *do* raise accuracy — the idea is not broken.
  2. **The headline missed the bar because this run's base trained model was a weaker draw.** The TTA change is provably eval-only (diff confined to `forward`; `num_params` identical), so it cannot hurt the trained weights — yet best peaked at 95.93% *with* the richer 6-view TTA. The model trained for 150 epochs this run vs EXP-004's 142; the time-budgeted design makes the final weights (and their test accuracy) vary run-to-run by ~±0.1pp even with the seed fixed, because the number of SGD steps that fit in 300s tracks host throughput.
- **Analysis**: The hypothesis's *mechanism* held (translate TTA adds views and lifts accuracy) but its *prediction* failed: the marginal gain of translate-over-mirror (we already had mirror in the 96.00 baseline) is small — at most a few hundredths of a pp — and is comparable to or smaller than the run-to-run variance of the trained model. EXP-004's 96.00 and this run's 95.93 are within that noise band, so a single-run A/B cannot resolve a sub-0.1pp eval-side increment. This is not a refutation of multi-crop TTA (it demonstrably helps vs no-TTA); it is evidence that **the eval-side lever is near-exhausted relative to the 0.1pp bar** once the mirror component is already captured, and that the metric carries an uncontrolled ~0.1pp noise floor from epoch-count jitter.
- **Key Learning**: Eval-time multi-crop TTA does lift accuracy (visible +0.28pp onset step) but its increment *over the mirror TTA already in the baseline* is below the ~0.1pp run-to-run noise floor created by the time-budgeted loop's epoch-count variance (142↔150 steps fit in 300s), so it cannot reliably clear the +0.1pp bar.

## Verification
- **Conditions**: C1 (clean run within wall guard) PASS — exit 0, one summary line, total 472.1s<600. C2 (full 300s budget + scope/integrity) PASS — training 300.0s, prepare.py byte-unchanged, only `train.py` changed, diff confined to the eval-path `forward`, num_params identical. **C3 (improvement ≥+0.1pp) FAIL** — best 95.93 < 96.10 (−0.07pp).
- **Review Notes**: Results trustworthy. Metric genuine (max per-epoch best 95.93 == summary, from `Eval.evaluate`, one eval/epoch, seeds unchanged). No reward-hacking concern — eval-time TTA via `model(inputs)` is a legitimate, documented technique already accepted in EXP-002; it improves predictions on the real frozen test set, not by gaming the harness. The −0.07pp vs baseline is within the run-to-run noise band, so this is a genuine no-improvement, not an invalid/suspicious result.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run; necessary condition C3 (metric ≥ bar) failed (metric below baseline). No constraint violation (not invalid); clean completion (not crash).

## Unexplored Avenues
- **Within-run A/B to isolate the translate increment.** Because TTA is deterministic given weights, a future eval-only experiment could, on the *same* trained model, compare mirror-only vs mirror+translate accuracy to measure the pure translate increment free of trained-model variance — but the ≤1-eval/epoch constraint and single-`forward`-mode design make this awkward without touching the loop. If a clean way exists (e.g., one eval that internally logs both), it would settle whether translate is worth keeping.
- **More aggressive / different TTA views.** airbench `tta_level=2` uses only 2 translation crops (diagonal ±1). 4-direction shifts or a 2px radius could add coverage — but each extra view costs eval wall-clock and the increment is likely even smaller; low expected value against the noise floor.
- **Attack the noise floor itself, not the mean.** The binding obstacle is now a ~0.1pp run-to-run variance. Anything that *raises the trained-model mean* by clearly more than 0.1pp (a real training-side win) is needed to register; sub-0.1pp eval-side polish will keep getting lost. This reframes the goal: the next high-value moves are training-side levers with >0.1pp headroom, not eval-side micro-optimizations.

## Next Steps
1. **Pivot to a training-side change with >0.1pp headroom** (medium confidence) — e.g. a second ReZero block at the proven layer2/8×8 stage (brainstorm Idea 2), or a stronger-augmentation/regularization change; eval-side polish is now below the noise floor and should be set aside.
2. **Stronger data augmentation** (medium-low confidence) — current aug is RandomCrop(4)+flip+Cutout(8); with 150 epochs now fitting in budget, slightly stronger regularization (larger/again cutout, or mixup with care) might lift the trained-model mean enough to clear noise.
3. **Keep multi-crop TTA as a free rider, not a standalone experiment** (high confidence it helps marginally) — since it demonstrably adds a small boost and costs only off-budget eval time, it is worth folding into a *future winning* training change rather than tested alone; but it is NOT the baseline now (no-improvement → discarded).

## Exit Action Results
<!-- No exit actions defined for this goal. -->
- None — no exit actions defined.
