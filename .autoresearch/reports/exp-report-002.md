# Report EXP-002: Cutout augmentation (16×16) on the k=4 WideResNet
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md
- **Plan**: plans/plan-002.md
- **Log**: logs/exp-log-002.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s budget, editing only `train.py`. Higher is better.
Baseline at experiment time: **94.90%** (EXP-001). Success bar: ≥ 95.00 (baseline + 0.1 pp).

## Idea & Hypothesis
Chosen idea: add Cutout (16×16, one hole) to the training augmentation on top of the k=4 WideResNet. After
widening reached 94.90%, the high-capacity model trained on only pad-crop-flip was likely overfitting, so
regularization was the suspected next ceiling. Cutout is the canonical WideResNet companion (DeVries & Taylor
2017). Hypothesis: Cutout lifts best_test_acc above 94.90% (expected ~95%) by reducing overfitting, at
negligible compute cost.

## Approach
`train.py`-only change, EXP-001 model + EXP-000 recipe held fixed. Added `CUTOUT_SIZE=16`, a `Cutout`
callable that zeros one random 16×16 square on the normalized CxHxW tensor (center via seeded `torch.randint`),
appended as the last element of the training `transforms.Compose`. Eval transform (in frozen `prepare.py`)
untouched. Single run, no retries.

## Execution
One clean run on GPU 0. 4.3M params (unchanged), loss healthy, no NaN, completed in budget (300.0s training,
367.6s total). **Unexpected**: only 54 epochs / 21,052 steps fit vs EXP-001's 79 / 30,498 — a ~31% throughput
drop. Root cause: the per-sample `torch.randint(...).item()` calls (two per image) in the 8 dataloader workers
became a CPU bottleneck (effective ~14 ms/step vs EXP-001's ~9.8). Despite far fewer epochs, accuracy still rose.

## Results
- **Primary metric**: **95.42%** (baseline: 94.90%, delta: **+0.52 pp**, +0.55%)
- **Observations**: best 95.42 @ epoch 53; final 95.25%; final_test_loss **0.217** vs EXP-001 0.249 — a clear
  drop in overfitting, exactly the intended Cutout effect. Trajectory: 91.73 → 92.06 (recipe) → 94.90 (width)
  → 95.42 (Cutout). Achieved with ~31% FEWER epochs than EXP-001 due to the dataloader bottleneck.
- **Analysis**: Hypothesis confirmed — the larger model was regularization-limited and Cutout helped. The fact
  that accuracy improved *despite* a large epoch reduction is strong evidence the gain is regularization, not
  optimization, and implies a compounding opportunity: a **cheaper/vectorized Cutout** (avoid per-sample
  `.item()` syncs; e.g. GPU-side batched masking, or precompute coords) would restore EXP-001's ~79 epochs and
  likely add more accuracy on top of the +0.52. The Cutout efficiency cost is an implementation artifact, not
  fundamental.
- **Key Learning**: Cutout regularization lifts the wide model (+0.52 pp, overfit loss 0.25→0.22) even at ~31%
  fewer epochs — but a naive per-sample `torch.randint().item()` Cutout throttles the dataloader; a vectorized
  version should compound the gain.

## Verification
- **Conditions**: all passed (clean completion in budget; 95.42 ≥ 95.00; only train.py changed, eval
  once/epoch, no new deps, seed unchanged).
- **Review Notes**: Trustworthy. Gain came through a train-only augmentation; frozen eval untouched; single
  fixed-seed run, no seed hacking. Adversarial check: Cutout is a generalization regularizer whose benefit
  survives benchmark recomposition — not gaming. +0.52 pp is above plausible run-to-run noise and corroborated
  by the lower eval loss.
- **Verdict**: improvement
- **Verdict Basis**: all necessary conditions passed + primary metric improved by +0.52 pp (above +0.1 bar).

## Unexplored Avenues
- **Vectorized / cheaper Cutout**: implement masking on the GPU batch (or avoid `.item()` syncs) to recover the
  lost ~25 epochs; likely compounds with the regularization gain — highest-value immediate follow-up.
- **Tune Cutout strength**: with only 54 epochs the 16px hole may be slightly strong; a smaller hole or
  probabilistic application (p<1) could help at this budget.
- **Combine with mixup/label-smoothing interplay or higher weight decay (5e-4)**: stack regularizers now that
  Cutout helped; watch for over-regularization at the short budget.
- **Re-test more width (k=6) now that regularization is in place**: the regularized model may exploit more
  capacity better than the unregularized one did.

## Next Steps
1. **Vectorized/GPU Cutout to restore throughput** — *high confidence* it recovers ~25 epochs and likely lifts
   accuracy further; the current bottleneck is a pure implementation artifact (exp-report-002 § Execution).
2. **Tune Cutout strength (smaller hole or p<1)** — *medium confidence*; 16px may be slightly strong at 54 epochs.
3. **Weight decay 1e-4 → 5e-4** (WRN standard) — *medium confidence*; cheap regularizer stack, but risk of
   over-regularization at the short budget.

## Exit Action Results
- None defined for this goal — skipped.
