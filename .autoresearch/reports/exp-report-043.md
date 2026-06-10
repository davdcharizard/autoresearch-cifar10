# Report EXP-043: AdamW optimizer-family swap

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-043.md
- **Plan**: plans/plan-043.md
- **Log**: logs/exp-log-043.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within the fixed 300s training-compute budget on a
single H20, editing only `train.py`. Baseline = **96.22%** (EXP-012, commit 6c417a4); bar = **96.32**.
This loop probed the last major untested axis: the optimizer FAMILY (adaptive AdamW vs SGD+Nesterov).

## Idea & Hypothesis
Chosen from brainstorm-043. All 44 prior experiments used `optim.SGD(lr=0.2, momentum=0.9, nesterov=True,
wd=1e-4)`; the three "optimizer" experiments (GC EXP-030/031, SAM EXP-036, PolyLoss EXP-041) only modified
SGD's gradients/objective and were loss-only or regressions. Swap to `AdamW(lr=2e-3, betas=(0.9,0.999),
eps=1e-8, weight_decay=0.05)` — decoupled WD (Loshchilov & Hutter, ICLR 2019) — keeping the cosine+warmup
schedule, TA+Cutout, label smoothing, seed, compile, and frozen eval identical. Hypothesis: adaptive
per-parameter steps converge faster in 300s and lift best_test_acc above 96.32. Honest prior: the
axis-independent polish-vs-top1 wall (project-insights L61) explicitly covers the optimizer axis, and the
well-known adaptive-generalization gap on CIFAR ConvNets makes a small regression/null most likely; the
probe's primary value is map-completion (closing the optimizer-family question).

## Approach
Three edits to `train.py` (scope-clean): `PEAK_LR` 0.2→2e-3, `WEIGHT_DECAY` 1e-4→0.05, and the optimizer
constructor SGD→AdamW. `MOMENTUM=0.9` left defined but unused. Literature-standard AdamW-from-scratch config
chosen so a regression is a real verdict, not a tuning miss. num_params unchanged (4,299,866). No deviations.

## Execution
Single clean run on idle GPU 0 (both H20s idle at launch, GPU0 uncontended throughout). AdamW trained
**stably** — no divergence despite the adaptive optimizer (5% warmup + label smoothing guarded the early
steps): smooth convergence ep1 47.0% → ep41 92.3 → ep71 95.4 → flat 95.7-95.9 tail (ep81-91). dt steady 8ms
(630×8ms + 70×9ms) → 91 epochs = baseline-equivalent → a fair, throughput-neutral test. total_seconds 408.4,
exit 0, peak_vram 470 MB (≈ baseline; AdamW's two moment buffers are small relative to activations).

## Results
- **Primary metric**: best_test_acc **95.87%** (baseline 96.22, delta **−0.35pp**, −0.36% — a small
  regression, slightly beyond the ±0.15–0.25pp noise band).
- **Observations**:
  - **AdamW is throughput-free**: dt steady 8ms = baseline. The optimizer-family change adds no measurable
    per-step cost — a fully fair, epoch-neutral test (91 ep ≈ baseline ~91).
  - **Stable convergence, no divergence**: lr=2e-3 + warmup + LS kept AdamW well-behaved; the run reached a
    flat, converged tail (not an underfit). So −0.35pp is a genuine generalization deficit, not instability
    or under-training.
  - final_test_loss 0.2023 > baseline 0.195 — AdamW gave neither a top-1 nor a loss win.
- **Analysis**: Hypothesis falsified, as the polish-wall prior predicted. This is the textbook
  **adaptive-optimizer generalization gap**: AdamW's per-parameter adaptive step sizes find a slightly
  worse-generalizing minimum than well-tuned SGD+Nesterov on this CIFAR ConvNet, even with decoupled WD and
  an identical strong-aug recipe (the gap is documented to narrow with strong aug / long schedules but not
  to close on CIFAR conv nets at ~90 ep). Converged cleanly to 95.87 — so the deficit is generalization, not
  optimization speed (AdamW did NOT under-converge; it reached a flat tail at a fair 91 ep). Confirms the
  axis-independent polish wall extends to a whole-optimizer-family change, not just SGD-gradient/objective
  modifications (GC/SAM/PolyLoss): **changing HOW the model optimizes does not raise the generalization
  ceiling here.** The well-tuned SGD recipe (peak 0.2, Nesterov, cosine-to-0) remains the best optimizer.
- **Key Learning**: AdamW (lr 2e-3, decoupled wd 0.05) — the first optimizer-FAMILY change in 44 experiments
  — trains stably and throughput-neutrally (8ms/91ep) but regresses −0.35pp (95.87) via the adaptive
  generalization gap. Tuned SGD+Nesterov is the better optimizer here; the optimizer-family axis is CLOSED.

## Verification
- **Conditions**: NECESSARY primary-metric condition FAILED (95.87 < bar 96.32, < baseline 96.22). Clean-
  completion and no-constraint-violation conditions passed (total 408.4s < 600, exit 0, diff = train.py only,
  eval 91 == num_epochs 91 ≤1/epoch, seed 42, AdamW is in torch.optim — no new deps, num_params unchanged).
- **Review Notes**: Trustworthy — clean uncontended 8ms/91-ep run, stable convergence, metric cross-
  consistent with the loss/dt/trajectory evidence. No integrity concern (a standard optimizer swap; eval
  untouched). Not a near-miss requiring a re-run.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, fair, throughput-neutral run; necessary primary-metric condition failed (mild
  regression from the adaptive generalization gap).

## Unexplored Avenues
- **AdamW LR/WD sweep (lr 1e-3/3e-3, wd 0.02/0.1)**: could shrink the −0.35pp gap toward zero, but the best
  plausible outcome is matching (not beating) tuned SGD — Adam-family rarely exceeds well-tuned SGD on CIFAR
  conv nets. Low value; not worth burning loops on optimizer tuning.
- **Other optimizer families (LAMB, Lion, Adan, Lookahead-on-SGD)**: all sit in the same optimization-not-
  generalization class the polish wall covers; the prior is the same (loss/convergence, not top-1). Low value.
- The takeaway: the optimizer axis (family AND gradient/objective modifications) is now comprehensively
  mapped and closed — optimization changes do not move the top-1 plateau on this generalization-bound net.

## Next Steps
- **Treat the optimizer axis as fully CLOSED** (confidence: high): AdamW (family) joins GC/SAM/PolyLoss
  (SGD-modifications) — no optimizer change moves top-1 here. Do not retry optimizer swaps or LR/WD sweeps.
- **Remaining moves are radical iso-dt architecture gambles** (confidence: low): with capacity, augmentation,
  schedule, optimizer, objective, weight-averaging, classifier-head, and intermediate-feature-routing all
  closed, the only untested territory is a different architecture FAMILY at iso-dt (e.g. ResNeXt grouped
  convs) — but iso-FLOPs≠iso-dt here (EXP-038) makes a clean test unlikely; high risk of a dt-confounded
  regression. The next loop should either attempt one such gamble with careful dt control, or document the
  96.22 plateau as the robust ceiling.
- **The 96.22 plateau is now very strongly confirmed as the robust ceiling** for this k=4 ResNet-20 at 300s
  (confidence: high): 33 consecutive no-improvements across every major axis.

## Exit Action Results
<!-- No exit actions defined in the goal file. -->
- None defined.
