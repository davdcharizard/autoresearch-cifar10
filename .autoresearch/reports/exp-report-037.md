# Experiment Report EXP-037

**Date**: 2026-06-09
**Verdict**: no-improvement
**Metric**: best_test_acc = 96.04% (baseline 96.22%, bar 96.32%; Δ −0.18pp)

## Goal
Maximize CIFAR-10 ResNet-20 `best_test_acc` (%) within a fixed 300s training-compute
budget on a single H20, editing only `train.py`. Higher is better. Baseline 96.22%
(EXP-012, commit 6c417a4); success bar = baseline + 0.1 = 96.32%.

## Idea & Hypothesis
**Chosen idea**: switch the train-only `RandomCrop(32, padding=4)` from default
zero-padding (`padding_mode='constant'`) to `padding_mode='reflect'`. After the
input-std-normalization lead was killed as infeasible (frozen eval pins std=(1,1,1)),
reflect-padding was the best feasible remaining lever: compute-neutral, train-only (eval
does not crop → no train/eval mismatch), genuinely untried, and mechanistically distinct
from every closed augmentation axis — it targets augmentation *quality* (border
statistics), not strength or policy.

**Hypothesis**: reflecting (vs zero-filling) the 4-px crop border removes the artificial
black wedges that zero-padding injects into translated training crops, tightening the
train/test distribution match and marginally lifting best_test_acc. Honest most-likely
outcome: within-noise (~96.1–96.3), since the net is regularization-saturated and BN may
absorb the thin-border effect.

## Approach
Single-argument edit to `train.py` L158: added `padding_mode="reflect"` to the existing
`RandomCrop`. No other change — all hyperparameters (PEAK_LR 0.2, batch 128, WD 1e-4,
label smoothing 0.1, Cutout 16, TrivialAugmentWide, cosine-to-0 LR, Nesterov m0.9, seed
42, 300s budget, torch.compile reduce-overhead) held fixed to isolate the
augmentation-quality lever. AST + diff-scope checks passed (train.py only).

## Execution
One run, no retries, exit 0. Background local run, `CUDA_VISIBLE_DEVICES=0 uv run
train.py > run.log 2>&1`. dt held at 8ms for 677/712 sampled steps (34×9ms, 1×11ms) —
throughput identical to baseline. Test acc tracked the baseline trajectory throughout
(ep50 91.59%, ep90 95.96%, ep92 96.04%). No errors/NaN.

Final metrics: best_test_acc 96.04% | final_test_acc 96.04% | final_test_loss 0.1960 |
num_epochs 92 | num_steps 35604 | training_seconds 300.0 | total_seconds 421.1 |
startup 3.4s | peak_vram_mb 453.8 | params 4,299,866.

## Results
Reflect-padding produced **96.04% (−0.18pp vs baseline 96.22)** — a clean within-noise
null on the negative side. The run was throughput-neutral (92 epochs vs baseline ~91,
dt 8ms) and converged (final_test_loss 0.196 ≈ baseline 0.195), so this is NOT an
underfit or a compute-wall confound — the augmentation-quality change simply did not move
top-1 (and if anything nudged it slightly negative within the run-to-run noise band).

This is consistent with the plateau diagnosis: the net is **regularization-saturated** at
fixed k=4 capacity. The hypothesized mechanism (black-border artifact in zero-padded
crops) is real but second-order — the artificial wedge occupies a thin frame of a minority
of crops, and is followed immediately by conv1→BatchNorm, which evidently absorbs the
distributional difference. The result confirms (yet again) the **polish-vs-top1** pattern
in a new guise: a defensible, mechanism-backed, compute-neutral refinement that does not
translate into top-1 because the bottleneck is generalization capacity, not augmentation
border fidelity. It fits the experiment trajectory — the crop-padding-mode sub-lever joins
the ~28 closed axes. The augmentation axis (strength EXP-013/021, policy EXP-014, mixing
EXP-011/018, cooldown EXP-033/034/035, and now border-quality EXP-037) is comprehensively
closed.

## Verification
- **Cond 1 — primary metric clears bar (`>= 96.32`)**: FAILED — 96.04%.
- **Cond 2 — clean completion within budget**: PASSED — summary printed, total_seconds
  421.1 < 600, training_seconds 300.0, exit 0.
- **Cond 3 — no hard-constraint violations**: PASSED — diff = train.py only; params
  4,299,866 unchanged; eval-line count 92 == num_epochs 92 (≤1 eval/epoch); no new deps;
  seed 42; prepare.py/eval untouched.

Results trustworthy: metric plausible (on-trajectory), converged loss, throughput-neutral,
no integrity concerns. **Verdict: no-improvement** (condition 1 failed on its merits).

## Unexplored Avenues
- The crop-padding-mode sub-lever is exhausted (constant vs reflect both tested; reflect
  ≤ constant here). `symmetric`/`edge` modes are minor variants of reflect with no reason
  to differ — not worth a run. **Augmentation-quality axis closed.**
- The genuinely-open frontier remains **capacity that doesn't trip the compute wall**: the
  net is generalization-bound at k=4, but every capacity add tried (k>4, SE, preact,
  ResNet-D, BlurPool, SAM) either cost epochs or didn't transfer to the shallow 32×32 net.
  An untried angle: **test-time augmentation (TTA)** at eval — but eval is frozen, so TTA
  is infeasible (would require editing Eval.evaluate). Confirmed dead.
- **Knowledge distillation from a self-ensemble / snapshot** within the 300s budget is
  compute-walled (needs a teacher pass). Likely dead but not explicitly tested.

## Next Steps
1. **Self-distillation / snapshot-ensemble logit averaging within budget** (low confidence)
   — average predictions across the last K epochs' BN states; but eval is frozen
   (single-model `evaluate(model)`), so any ensemble must collapse into one weight set →
   reduces to weight-averaging, already closed (EMA/SWA, EXP-019). Likely dead.
2. **Revisit the strongest near-miss (Gradient Centralization, EXP-030) more carefully**
   (medium confidence) — GC tied baseline + improved loss DESPITE a 3-epoch handicap from
   the un-fused Python loop; a fully `torch._foreach_`-vectorized GC that recovers those
   epochs is the single most promising untried throughput-neutral optimizer lever. This
   was flagged as the EXP-031 lead and never executed. **Highest-value next probe.**
3. **More radical: a fundamentally different capacity allocation** (low-medium confidence)
   — e.g. a shallower-but-wider or pyramidal channel schedule at matched params/FLOPs, to
   test whether the k=4 {64,128,256} allocation is itself suboptimal rather than the
   capacity level. Higher risk (architecture-transfer + params change) but addresses the
   actual generalization bound rather than another polish lever.

## Key Learning
Reflect vs zero crop-border padding is a real-but-second-order augmentation-quality change
that BatchNorm absorbs: 96.04% (−0.18pp), throughput-neutral and converged. Augmentation
axis (strength/policy/mixing/cooldown/border) now comprehensively closed — the bound is
generalization capacity, not augmentation fidelity.
