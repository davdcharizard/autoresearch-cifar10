# EXP-012: TrivialAugmentWide added to the train pipeline (kept with Cutout) + compile enabler

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md
- **Plan**: plans/plan-012.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-012
- **Commit**: 6c417a4 (on autoresearch/exp-012; merged to autoresearch/dev)
- **PR**: (none — no git remote by design; commits kept local per TASK.md Git Rules)
- **Outcome**: completed → verdict **improvement** (96.22 ≥ 96.10; +0.22pp over baseline 96.00)

## Implementation Notes

### Summary
`train.py`-only edits (Milestone 1): (1) inserted `transforms.TrivialAugmentWide()` into `train_tf` between
`RandomHorizontalFlip()` and `ToTensor()` — it operates on the PIL image (CIFAR10 yields PIL), applying one random
op per image at a uniformly random strength; Cutout(16) is retained (it runs later, GPU-side in the loop), giving
the canonical TA+Cutout pairing. (2) Added `compiled_model = torch.compile(model, mode="reduce-overhead")` right
after the `num_params` print and routed the training forward through `compiled_model(inputs)`; eval stays on the
eager `model` handle (EXP-007/008/010/011 validated pattern). Ruff clean, `git diff` = train.py only (10 insertions,
1 deletion). Param count expected 4,299,866 (UNCHANGED — TA and compile add no parameters).

### Surprises & Discoveries
- (none at implementation time — `TrivialAugmentWide` confirmed importable in torchvision 0.24.1, no new dependency.)

### Decisions
- **TA placed before ToTensor (on PIL)**: TrivialAugmentWide expects a PIL Image or uint8 tensor; placing it after
  the geometric PIL transforms and before ToTensor keeps the input type valid and matches the standard recipe order.
- **TA + Cutout both kept**: the canonical SOTA CIFAR-WRN config combines auto-augmentation with Cutout; testing TA
  alone would discard the proven Cutout gain. The experiment tests TA *as an addition*, the realistic strong-aug recipe.
- **Compile included**: TA adds CPU augmentation cost that could starve the launch-bound GPU; compile (EXP-007,
  ~30% throughput, null standalone accuracy effect) buys epoch headroom so the run stays converged and any gain is
  attributable to TA, not compile.

## Experimental Adjustments

(none yet)

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID; local background run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08 (exit 0)

Description:
- Running the EXP-003 k=4 + Cutout recipe with TrivialAugmentWide added to the train pipeline, compiled
  (reduce-overhead). Tests whether a strong, diverse auto-augmentation policy (photometric+geometric, orthogonal to
  Cutout's occlusion) improves generalization at fixed capacity. Expect params 4,299,866 (unchanged), clean compile,
  and — the key uncertainty — enough realized epochs (≳75) to be a fair converged test rather than throughput-starved.
  Success = best_test_acc ≥ 96.10. NOTE: train loss may read higher under stronger aug; judge health by eval acc.

Observations:
- Clean startup: **params 4,299,866 (UNCHANGED)** — confirms TA + compile add no parameters (run.log head). Clean
  compile, no traceback, no NaN.
- **Steady-state dt = 8ms/step (~15,300 img/s)** from step ~500 — IDENTICAL to compiled-k4 (EXP-007). The key
  budget de-risk: TrivialAugment's per-sample CPU cost does NOT starve the launch-bound GPU (8 workers keep up), so
  the run is tracking toward ~89 epochs ⇒ a FAIR, fully-converged test, NOT throughput-starved (source: run.log,
  step 500-650 progress lines).
- Train loss reads higher/noisier under TA (1.77–1.83 @ step 500-650, vs a non-TA run) — expected from stronger
  augmentation, no NaN/divergence. ep1 eval 42.78% (early, normal warmup).

Key Metrics:
- **best_test_acc: 96.22%** @ ep 87 — ABOVE baseline 96.00 (+0.22pp) AND the 96.10 bar (run.log summary). First
  improvement since EXP-003.
- **num_epochs: 91** / num_steps 35,302 — fair, fully-converged test (eval count 91 == num_epochs ⇒ eval once/epoch);
  NOT epoch-starved. TA got a fair shot AND beat baseline.
- **final_test_loss: 0.1950** — LOWER than baseline 0.204 and compiled-k4 0.208 (EXP-007). Loss↓ AND acc↑ together
  ⇒ a genuine generalization improvement, not a noise spike or a soft-target artifact (contrast EXP-011 Mixup, where
  loss ROSE). final_test_acc 96.18; late evals cluster 96.12–96.22 (ep 87–91) — stable, not a lone lucky epoch.
- num_params 4,299,866 (UNCHANGED — TA + compile parameter-free). peak_vram 453.8 MB. dt 8ms (~15,300 img/s ≈
  compiled-k4) — TA did not starve the GPU.
- Attribution: compiled-k4 ≈ baseline null (EXP-007, 95.92/0.208) ⇒ the +0.30pp over compiled-k4 and −0.013 loss are
  attributable to TrivialAugment, not compile.

## Verification Results

### Conditions Checked

- **Cond 1 — clean completion within budget**: PASS. best_test_acc 96.22% present, total_seconds 406.6 < 600, no
  traceback (run.log; tracebacks=0).
- **Cond 2 — metric ≥ 96.10**: **PASS**. 96.22 ≥ 96.10 (also > 96.00 baseline, +0.22pp). → improvement.
- **Cond 3 — no constraint violations**: PASS. `git diff --name-only` = train.py only; num_params UNCHANGED
  (4,299,866 ⇒ TA + compile add no parameters, no architecture change); seed 42 intact; eval count 91 == num_epochs
  (eval once/epoch — no eval-frequency hacking). No reward-hacking surface (TA is a standard input augmentation; the
  gain would survive any benchmark recomposition).

### Informational Metrics

- num_epochs 91 / num_steps 35,302 (fair converged test, not epoch-starved). final_test_loss 0.1950 (BELOW baseline
  0.204 / compiled-k4 0.208 — corroborates the accuracy gain). peak_vram_mb 453.8. img/s ~15,300 (dt 8ms ≈ compiled-k4).

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
