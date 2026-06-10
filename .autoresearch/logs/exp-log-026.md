# EXP-026: Bag-of-Tricks free convergence bundle (zero-init residual γ + no-bias-decay)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md
- **Plan**: plans/plan-026.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-026
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (run clean & compute-neutral; verification Cond 1 failed → no-improvement; within-noise NULL; verdict rendered in analyze)

## Implementation Notes

### Summary
Milestone 1: two compute/param-neutral edits in `train.py`. (1) Zero-init residual γ — after `self.apply(self._weights_init)` in `ResNet.__init__`, a loop sets `init.zeros_(m.bn2.weight)` for every `BasicBlock` (each residual branch outputs 0 at init → block starts as identity). (2) No-bias-decay — replaced the single-group `optim.SGD(model.parameters(), weight_decay=1e-4)` with a two-group SGD: weight_decay=1e-4 for ndim≥2 (conv/linear weights), 0.0 for ndim≤1 (BN γ/β + fc bias). Smoke test (`uv run python`) confirmed: num_params 4,299,866 unchanged, bn2.γ all-zero across all 9 BasicBlocks, optimizer has 2 groups (WD [1e-4, 0.0], 23+45 param tensors) covering all 4,299,866 params. git diff = train.py only; AST clean.

### Surprises & Discoveries
None. The single-group→two-group optimizer change is transparent to the training loop's LR update (`for pg in optimizer.param_groups: pg["lr"]=lr`) and LR readout (`param_groups[0]["lr"]`), both of which remain correct. Param split: 23 weight tensors (decay) vs 45 BN/bias tensors (no-decay).

### Decisions
Used `p.ndim <= 1` as the no-decay predicate (captures BN γ/β and the fc bias; conv weights have bias=False so no conv biases exist). Bundled both tricks in one run to maximize the chance of clearing the +0.1pp bar on a shallow net where each alone is likely marginal; a gain triggers a one-run ablation next loop (recorded in brainstorm/report).

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — background bash task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-09

Description:
- Runs the EXP-012 recipe (LR 0.2, batch 128, TA+Cutout(16), cosine-to-0, ~91 epochs) with two Bag-of-Tricks free convergence levers added: zero-init residual γ (identity-init blocks) and no-bias-decay (WD only on conv/linear weights). Both are compute/param-neutral, so the run MUST stay at ~91 epochs / ~8ms/step / 4,299,866 params — a fair throughput-neutral test (no epoch-wall/update-collapse confound). Expected: a small convergence-quality gain that, if it clears the noise floor (~0.2pp), pushes best_test_acc above the 96.32 bar; otherwise a within-noise null indicating the free tricks are too marginal on this shallow net.

Observations:
- **Compute-neutral CONFIRMED**: dt steady ~8ms (matches baseline), 390 batches/epoch (batch 128), **93 epochs** / 35,913 steps (vs baseline 91 / ~35,490 — within run-to-run jitter, the changes add zero FLOPs), params 4,299,866 unchanged, peak VRAM 453.8MB (≈ baseline). Clean run: no Traceback, no NaN, total_seconds 404.8 < 600 (source: run.log summary + step samples).
- **Within-noise NULL**: best_test_acc **96.18%** vs baseline 96.22 = **−0.04pp** — statistically indistinguishable (goal-learnings High: sub-~0.2pp deltas are noise at this budget). Below the 96.32 bar.
- Interesting secondary: final_test_loss **0.1899 < baseline 0.195** (slightly LOWER) — the free tricks gave a tiny loss/calibration benefit that did NOT convert to top-1, echoing the SWA "loss↓ but top-1 flat" signature (EXP-019/020). Consistent with the bundle being mechanistically real but marginal on this shallow 9-residual-block net (source: run.log `final_test_loss: 0.1899`).

Key Metrics:
- best_test_acc: 96.18% (source: run.log `best_test_acc:` line)
- num_epochs: 93 (vs baseline 91 — compute-neutral confirmed); num_steps: 35,913 (vs ~35,490)
- dt: ~8ms (matches baseline — compute-neutral); final_test_loss: 0.1899 (vs 0.195, slightly LOWER); final_test_acc: 96.18%; total_seconds: 404.8; peak_vram_mb: 453.8; num_params: 4,299,866 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAILED**. best_test_acc = 96.18% < 96.32 (−0.04pp vs baseline 96.22, within ~0.2pp noise floor). (source: run.log `best_test_acc: 96.18%`)
- **Cond 2 — clean completion within budget**: skipped — aborted after Cond 1 failed. (Would pass: total_seconds=404.8 < 600, Traceback=0, summary present.)
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 1 failed. (Would pass: git diff = train.py only, num_params=4,299,866 unchanged, eval-count=93 == num_epochs=93, only core-torch init.zeros_/SGD param-groups added / no new deps, seed 42 intact.)

Verdict basis: first necessary condition failed → no-improvement. CLEAN compute-neutral run (dt 8ms, 93 ep, params/VRAM unchanged) → a trustworthy within-noise NULL, not a confounded result.

### Informational Metrics

- Not collected per protocol (only when all conditions pass). For the record: num_epochs=93 (vs 91, compute-neutral), num_steps=35,913, dt ~8ms (matches baseline), final_test_loss=0.1899 (vs 0.195 — slightly LOWER, loss-not-top1 signature), peak_vram_mb=453.8 (unchanged).

## Errors & Dead Ends

<!-- none -->

## Human Notes

> (none — autopilot)
