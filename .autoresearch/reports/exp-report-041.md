# Experiment Report: EXP-041 — Derandomized alternating horizontal flip (shared-memory epoch tensor)

- **Date**: 2026-06-10
- **Verdict**: no-improvement
- **Primary metric**: best_test_acc = **96.49%** (baseline 96.71, bar 96.81, delta −0.22; at the low edge of the baseline band 96.4–96.7)
- **Branch**: autoresearch/exp-041 (discarded)
- **Artifacts**: brainstorm/brainstorm-041.md · plans/plan-041.md · logs/exp-log-041.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc %, higher is better) within the fixed 300s charged training budget, modifying only `train.py`. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

## Idea & Hypothesis
**Idea**: The last unmeasured mechanism class — data order/coverage. Replace iid RandomHorizontalFlip with airbench's alternating flip (image i flipped iff (epoch+i) % 2 == 0), removing per-image orientation-coverage variance (σ≈4.2% under iid at 139 epochs → exactly 0) at zero cost in every closed currency. The previously blocking implementation problem (per-epoch state does not propagate to persistent DataLoader workers) was solved with a shared-memory int64 tensor written by the main loop and read live by forked workers, preserving epoch semantics exactly (139 epochs / 139 evals).

**Hypothesis**: Coverage derandomization raises the converged plateau if residual orientation-coverage imbalance is a real error term at 139 epochs; benefit should appear first in the early epochs (the airbench regime). Falsified by a plateau in the baseline band.

## Approach
train.py only (28 insertions / 3 deletions): `AlternatingFlipCIFAR10(datasets.CIFAR10)` subclass applying the parity flip pre-transform; `RandomHorizontalFlip` removed from the compose; `epoch_box = torch.zeros(1, dtype=torch.int64).share_memory_()` updated at each epoch top. Flip marginal stays 50%/epoch — augmentation strength, CPU cost, loader timing, and noise statistics unchanged. Three pre-launch CPU sanities passed: exact flip schedule, **worker propagation** (`torch.equal(e0.flip(-1), e1)` with persistent workers across an epoch_box update), params 4,286,026.

## Execution
Single pristine run (gates clear poll 1; launched 20:39:18; rc=0; total 473.4s; no watchdog trigger). Signatures byte-identical to family: 200-step windows mean 22.30ms, max 22.5, 0 >27ms; 139 epochs / 13,453 steps; VRAM 1613.0MB; startup 9.4s. The zero-cost claim was confirmed exactly. No retries, no adjustments, no errors.

## Results
- **best 96.49 (ep136), final 96.41, final_test_loss 0.1856 — falsification branch.** The bar-pass scenario (coverage imbalance ≥ +0.25) is refuted.
- **The mechanism's predicted early benefit appeared — weakly — and never compounded.** Hot phase at-or-above family (ep5 66.57 vs ~64; ep20 81.40 vs ~79): exactly where coverage deficits are largest (few flips sampled per image), balanced exposure helps a little, consistent with airbench's regime calibration. By the plateau the iid baseline has sampled ~70 flips per orientation per image and the deficit vanishes.
- **Derandomization is not free at the eval boundary — it trades sampling variance for periodic structure.** Plateau last-15: mean 96.273, spread 0.75 (family ~96.5 / ±0.15). At every eval, each image's most-recent orientation is DETERMINISTIC by parity (the flipped half alternates wholesale between consecutive epochs), so consecutive end-of-epoch weight/BN states differ systematically where iid flip averages the orientation mixture out. The result is elevated eval-to-eval scatter and a mildly depressed plateau mean — the same lesson as EXP-038/039 (BN constants) from a new direction: the eval samples the training process at epoch boundaries, and any mechanism that makes epoch boundaries SPECIAL (stale constants, parity structure) degrades exactly the statistic being harvested.
- **Trajectory fit**: 36th consecutive non-improvement. The data-order/coverage class is now measured and closed; with it, every mechanism class the program identified has at least one measured experiment.

## Verification
First-failure-stop per plan-041. Pre-condition: profile pristine (mean 22.30ms, 0 slow >27 on quantization-safe windows), 139 epochs ✓. Integrity: params 4,286,026 ✓, training_seconds 300.0 ✓, eval_lines 139 = num_epochs ✓. **Condition 1 FAILED on merits: 96.49 < 96.81.** Conditions 2–3 skipped per protocol (incidental: rc=0, 473.4s ≤ 600; 139 = 139). No false-failure risk: clean profile, exact signatures, and the diagnostic suite (early benefit, plateau scatter, family-equal test_loss) forms a coherent mechanism story. Verdict: **no-improvement**.

## Unexplored Avenues
- **Within-epoch interleaved alternation** (flip parity varying by batch rather than epoch, e.g. (step+i)%2): would destroy the epoch-boundary parity structure that caused the scatter while keeping exact coverage — but the measured EARLY benefit was already sub-σ and the plateau cost is what needs removing; expected best case ≈ baseline mean, sub-bar. Low value.
- **Paired-sample alternation** (i and its flip in the SAME batch, batch 256 pairs): changes effective batch composition and gradient noise (the same image twice per step) — collides with the gradient-noise law; closed by screen.
- The shared-memory epoch tensor pattern itself is validated reusable engineering for any future idea needing live per-epoch state in persistent workers (e.g., epoch-keyed augmentation schedules — though time-varying augmentation is itself a closed axis).

## Next Steps
1. **Record the data-order/coverage class as measured-closed** and the epoch-boundary lesson (mechanisms that make epoch ends special degrade the harvested statistic) as a unifying pattern with EXP-038/039 (high confidence).
2. **Every identified mechanism class now has a measurement.** The program is at a fully-measured local optimum with bar = mean + 1.5σ; the directive's escalation path continues — next candidates must come from genuinely novel mechanism construction (recombination of validated parts under the laws) rather than any standard menu (medium confidence in framing).
3. **Protocol option for a future promising mechanism**: pre-registered replicate pair to halve effective σ before judging a mid-band read (carried from exp-report-040; medium confidence).

## Key Learning
Derandomizing augmentation coverage helps exactly where its anchors say (early, when per-image samples are few — ep5/20 ran above family) and buys nothing at a 139-epoch plateau, where iid sampling has already converged the coverage; worse, alternation makes epoch boundaries structurally special (each image's last-seen orientation is deterministic by parity), raising end-of-epoch eval scatter 5× and mildly depressing the plateau — the third independent demonstration (after EXP-038/039's BN-constants results) that the best-over-evals statistic punishes any mechanism that perturbs the state specifically at epoch boundaries. Data-order/coverage class closed; the program has now measured every mechanism class it identified.
