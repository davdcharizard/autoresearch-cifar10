# Report EXP-005: Weight decay 1e-4 → 5e-4 (k=4 + Cutout)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md
- **Plan**: plans/plan-005.md
- **Log**: logs/exp-log-005.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s budget, editing only `train.py`. Baseline 96.00%
(EXP-003). Success bar ≥ 96.10.

## Idea & Hypothesis
Raise WEIGHT_DECAY 1e-4→5e-4 (WRN standard) on the k=4 + Cutout model. Hypothesis: stronger L2 regularization
of the 4.3M-param model lifts acc above 96.00% (~+0.1–0.4 pp), throughput-neutral.

## Approach
One-line change `WEIGHT_DECAY` 1e-4→5e-4; all else fixed. Single run.

## Results
- **Primary metric**: **96.05%** (baseline 96.00%, delta **+0.05 pp** — below the +0.1 bar)
- **Observations**: final_test_loss **0.1956** (vs EXP-003 0.204) — stronger WD measurably reduced overfitting,
  the intended mechanism. But test accuracy moved only +0.05 (within run-to-run noise). This run fit **65
  epochs** (vs EXP-003's 77): per-step time ran ~17ms vs ~10ms. WD is compute-neutral and GPUs were idle at
  post-run check, so the shortfall is transient throughput jitter (possibly transient host/dataloader
  contention) — not caused by the WD change, but it may have suppressed a small WD benefit.
- **Analysis**: WD 5e-4 helps the loss/regularization axis but does not move accuracy past the bar — the
  regularization axis (Cutout already in place + label smoothing) appears near-saturated for this model at this
  budget. The +0.05 is within the noise band implied by EXP-005's own 12-epoch shortfall, so WD 5e-4 is best
  read as "neutral-to-marginally-positive, not a clear win." Trajectory unchanged: best stays 96.00% (EXP-003).
- **Key Learning**: WD 5e-4 lowers eval loss (0.204→0.196) but not accuracy past baseline (+0.05, sub-bar);
  the regularization axis is near-saturated. Gains at 96% are now noise-scale — need a different axis or
  variance-aware evaluation.

## Verification
- **Conditions**: Cond 1 (clean completion) PASS; Cond 2 (≥96.10) **FAIL** (96.05); Cond 3 skipped.
- **Review Notes**: Trustworthy clean run, eval frozen, fixed seed. Genuine no-improvement (sub-bar). The
  65-epoch throughput confound is noted but does not change the verdict (we don't chase noise via re-runs —
  that risks seed/variance hacking).
- **Verdict**: no-improvement
- **Verdict Basis**: valid run, primary metric below the +0.1 bar (verification condition 2 failed).

## Unexplored Avenues
- **WD 3e-4 (intermediate)** or WD 5e-4 re-run at full ~77 epochs: the loss improved, so a milder WD or a
  clean full-epoch run might clear the bar — but expected gain is noise-scale; low priority.
- **A different axis entirely**: gains at 96% are now ≤ run-to-run noise on the regularization/recipe axes.
  Candidates that could move accuracy more than noise: (a) test-time augmentation is disallowed (eval frozen);
  (b) a fundamentally better-utilizing architecture at k=4 budget (e.g. squeeze-excite, better stem/activation);
  (c) longer effective training via faster steps (larger batch + LR scaling to raise steps/s).
- **Peak-LR sweep** on k=4: LR 0.2 was never tuned for the wide net; a better peak could matter more than WD.
- **Variance-aware bar**: at 96% the +0.1 bar is near the noise floor — future ideas should target ≥ ~0.3 pp
  expected gains to stand out (process note, not a code change).

## Next Steps
1. **Peak-LR tuning on k=4 (e.g. 0.3 or 0.15)** — *medium confidence*; LR is the least-tuned recipe knob and
   can outweigh WD. Cheap, throughput-neutral.
2. **Throughput-up to buy epochs (larger batch 256 + LR scale)** — *medium confidence*; more steps of the
   regularized model; EXP-003 showed extra epochs help when not over-regularized.
3. **Architectural efficiency at k=4 (e.g. SE blocks / better activation)** — *medium confidence*; a new axis
   since width (capacity) and WD/Cutout (regularization) are near their local ceilings.

## Exit Action Results
- None defined for this goal — skipped.
