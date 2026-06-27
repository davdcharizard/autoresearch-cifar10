# Experiment Log: EXP-041 — Derandomized alternating horizontal flip (shared-memory epoch tensor)

## Execution

- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-041.md
- **Plan**: plans/plan-041.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-041
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed (run clean; verification Condition 1 failed on merits — 96.49 < 96.81)

## Implementation Notes

### Summary
Milestone 1 as planned (28 insertions / 3 deletions, train.py only): added `from PIL import Image`; new `AlternatingFlipCIFAR10(datasets.CIFAR10)` whose `__getitem__` mirrors the torchvision parent but applies `img.transpose(Image.FLIP_LEFT_RIGHT)` iff `(int(epoch_box[0]) + index) % 2 == 0` before the transform stack; removed `RandomHorizontalFlip` from `train_tf`; created `epoch_box = torch.zeros(1, dtype=torch.int64).share_memory_()` and passed it into the dataset; the training loop sets `epoch_box[0] = epoch` right after `epoch += 1` (before the epoch's loader iterator is created). All three pre-launch sanities passed on CPU: (A) flip schedule exact — flips iff `(e+i)%2==0`, consecutive epochs mirrored, checked at i ∈ {0,1,7,100}; (B) worker propagation — with `num_workers=2, persistent_workers=True`, a batch fetched at epoch_box=0 equals the flip of the batch fetched after setting epoch_box=1 (`torch.equal(e0.flip(-1), e1)`); (C) params 4,286,026 unchanged.

### Surprises & Discoveries
None — the shared-memory propagation worked first try (Linux fork start method; `share_memory_()` storage is inherited by forked workers and writes propagate live). Sanity B's design needed one care point: the first DataLoader iterator must be drained before re-iterating, so the second pass cleanly observes the new epoch value.

### Decisions
- Flip applied BEFORE RandomCrop (it lives in the dataset, pre-transform) where baseline had flip after crop — distributionally equivalent (crop offsets are symmetric), noted per plan edge-case (i).
- Kept the flip marginal at exactly 50% per epoch by parity of (epoch + index) — no strength change, so the augmentation dose-response law is untouched.

## Run Log

### Run 1
- **Description**: Full budget-matched run of the alternating-flip variant on GPU 0 via `/tmp/exp041_composite.sh` (exp039 baseline-threshold watchdog: contention >27ms, startup tick 10, NaN/divergence/wall guards). Execution signatures expected byte-identical to baseline family (dt 22.3–22.4ms, 139 epochs/evals, params 4,286,026, VRAM 1613MB, total ~480–500s). Hypothesis: removing per-image orientation-coverage variance raises the converged plateau if coverage imbalance is a real error term at 139 epochs; the mechanism predicts any benefit shows EARLY (ep5/10/20) where coverage deficits are largest. Falsified by a plateau in the baseline band.
- **Job ID / PID**: background task bimgh8fcf (`/tmp/exp041_composite.sh`)
- **Log file**: run.log (project root); watchdog via composite stdout (task output file)
- **WandB**: N/A
- **Status**: completed (rc=0, no watchdog trigger)
- **Started**: 2026-06-10 20:39:18 (gates clear at poll 1: apps=0, load=5)
- **Ended**: 2026-06-10 ~20:47:10 (PROC_EXITED at tick 32; total_seconds 473.4)
- **Observations**: Pristine run: watchdog windows 21.8–22.7ms across 30 ticks, slow_streak never above 0 (task bimgh8fcf); 200-step quantization-safe windows: 67, mean 22.30ms, max 22.5, 0 > 27ms. Startup 9.4s (warm compile cache), VRAM 1613.0MB, 139 epochs / 13,453 steps — signatures byte-identical to family as predicted (zero-dt claim confirmed: the alternation costs nothing).
- **Key Metrics**: best_test_acc 96.49 (ep136), final 96.41, final_test_loss 0.1856 (family ~0.185), training_seconds 300.0, total 473.4s. Hot phase: ep3 47.06 / ep5 66.57 (family ~64) / ep10 74.96 (~75) / ep20 81.40 (~79) — at-or-slightly-above family early, consistent with a small coverage benefit in the low-epoch regime where the mechanism is strongest. Plateau (last 15): mean 96.273, min 95.74, max 96.49, spread 0.75 vs family ~96.5/±0.15 — the early benefit did NOT compound; the plateau reads mildly depressed with ~5× scatter. best 96.49 sits at the low edge of the baseline band (96.4–96.7).

## Experimental Adjustments

(none yet)

## Errors & Dead Ends

(none yet)

## Verification Results

### Conditions Checked

First-failure-stop per plan-041 § Verification Protocol; baseline at verification time 96.71 (bar 96.81).

- **Pre-condition — run integrity**: PASS. Profile pristine: 200-step quantization-safe windows (67) mean 22.30ms, max 22.5, 0 > 27ms; num_epochs 139 ∈ [135, 143]. Integrity: num_params 4,286,026 ✓, training_seconds 300.0 ✓, eval-line count 139 == num_epochs ✓. Source: run.log greps above; composite stdout (task bimgh8fcf).
- **Condition 1 — best_test_acc ≥ 96.81**: **FAIL on merits.** `grep "^best_test_acc:" run.log` → 96.49 < 96.81. Clean profile, full epoch count, family signatures — no false-failure risk. Verification stopped per first-failure-stop.
- **Condition 2 — completes within budget**: skipped per protocol (incidental: rc=0, total 473.4 ≤ 600).
- **Condition 3 — validation ≤ once/epoch**: skipped per protocol (incidental: 139 = 139).

**Diagnostics**: hot phase at-or-slightly-above family (ep5 66.57 vs ~64; ep20 81.40 vs ~79) — weak early coverage benefit, as the mechanism predicts for the low-epoch regime; plateau last-15 mean 96.273 with spread 0.75 (family ~96.5/±0.15) — the benefit does not compound and the plateau reads mildly depressed with elevated eval-to-eval scatter; final_test_loss 0.1856 = family. A plausible mechanism for the scatter: at every eval the network's most-recent exposure to each image's orientation is DETERMINISTIC by parity (alternating sets), so consecutive-epoch BN/weight states differ systematically where iid flip averages this out — derandomization trades sampling variance for periodic structure the eval samples at epoch boundaries.

### Informational Metrics

- peak_vram_mb: 1613.0 (unchanged)
- num_epochs: 139 (family-equal)
- num_params: 4,286,026 (unchanged)

## Human Notes

(autopilot — none)
