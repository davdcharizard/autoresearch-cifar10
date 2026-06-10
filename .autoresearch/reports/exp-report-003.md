# Report EXP-003: Vectorized GPU Cutout (recover throughput)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md
- **Plan**: plans/plan-003.md
- **Log**: logs/exp-log-003.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s budget, editing only `train.py`. Higher is better.
Baseline at experiment time: **95.42%** (EXP-002). Success bar: ≥ 95.52.

## Idea & Hypothesis
Chosen idea: move Cutout from a per-sample CPU transform to a vectorized GPU batch op. EXP-002 showed Cutout
helped (95.42%) but its per-sample `torch.randint().item()` in dataloader workers throttled throughput
(79→54 epochs). Hypothesis: an equivalent near-zero-cost batched GPU mask restores ~79 epochs, and the extra
training of the already-regularized model lifts best_test_acc above 95.42% (expected ~95.6–96%).

## Approach
`train.py`-only change. Replaced the `Cutout` transform class with `cutout_batch(x, size)` — builds a
`(B,H,W)` boolean hole mask via broadcast `arange` comparisons and `masked_fill`s the batch to 0 (no `.item()`
sync, no per-sample loop). Applied in the training loop right after the device transfer, before the autocast
forward. Cutout semantics identical (one random ≤16×16 border-clipped hole per image). Model (k=4), recipe,
seed all unchanged. Single run, no retries.

## Results
- **Primary metric**: **96.00%** (baseline: 95.42%, delta: **+0.58 pp**, +0.61%)
- **Observations**: best 96.00 @ epoch 73; final 95.85%; final_test_loss **0.204** (< EXP-002's 0.217).
  Throughput fully recovered: **77 epochs / 29,931 steps** vs EXP-002's 54 / 21,052 — back to EXP-001's level,
  confirming the dataloader bottleneck was the cause. Crossed the **96%** mark. Trajectory: 91.73 → 92.06 →
  94.90 → 95.42 → 96.00.
- **Analysis**: Hypothesis confirmed end-to-end. The early-run probe showed throughput restored (step 3100 @
  10.6% vs EXP-002's 1950), and the recovered ~23 epochs converted directly into +0.58 pp. This validates the
  goal-learnings note that the Cutout cost was a pure implementation artifact, and reaffirms the project insight
  that the 300s wall-clock is the binding budget — efficient kernels/augmentation translate straight into
  accuracy. The GPU-Cutout speedup now also benefits every future experiment (e.g. makes k=6 / heavier aug viable).
- **Key Learning**: Eliminating the Cutout dataloader bottleneck (per-sample CPU → vectorized GPU mask) recovered
  54→77 epochs and lifted acc 95.42→96.00% (+0.58) — under a wall-clock budget, augmentation/kernel efficiency
  *is* an accuracy lever.

## Verification
- **Conditions**: all passed (clean completion in budget; 96.00 ≥ 95.52; only train.py changed, eval once/epoch,
  no new deps, seed unchanged).
- **Review Notes**: Trustworthy. The gain came from a legitimate efficiency improvement that lets the same
  recipe train longer within the fixed budget — not eval tampering or seed hacking (single fixed-seed run, eval
  frozen). Lower eval loss + more epochs corroborate the accuracy gain. Not reward hacking: benefit is genuine
  generalization and would survive benchmark recomposition.
- **Verdict**: improvement
- **Verdict Basis**: all necessary conditions passed + primary metric improved by +0.58 pp (above +0.1 bar).

## Unexplored Avenues
- **Push width to k=6/8 now that throughput is restored**: capacity scaling is back on the table with a fast
  pipeline and ~77-epoch headroom; VRAM still ~free.
- **Weight decay 1e-4 → 5e-4 (WRN standard)**: stack regularization on the now-efficient pipeline.
- **Tune Cutout strength / probability**: with 77 epochs a slightly larger hole or p<1 could shift the bias-variance point.
- **mixup / random-erasing variants**: additional augmentation now affordable at full throughput.
- **Larger batch + LR scaling**: more GPU utilization; though EXP-000 showed extra steps help only when not
  capacity/aug-bound — revisit after capacity changes.

## Next Steps
1. **Increase width to k=6** — *medium-high confidence*; capacity was the dominant lever (EXP-001) and the
   pipeline is now fast enough to train a bigger model for enough epochs. Watch epoch count.
2. **Weight decay 5e-4** — *medium confidence*; cheap WRN-standard regularizer, isolated test.
3. **Tune Cutout (size/probability) or add mixup** — *medium confidence*; further regularization at full throughput.

## Exit Action Results
- None defined for this goal — skipped.
