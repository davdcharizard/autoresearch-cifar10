# Experiment Log EXP-042: Deep supervision — auxiliary layer2 classifier with decayed aux loss

## Execution
- **Created**: 2026-06-09
- **Brainstorm**: brainstorm/brainstorm-042.md
- **Plan**: plans/plan-042.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-042
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed (clean run; verdict no-improvement — see Verification Results)

## Implementation Notes

### Summary
Implemented the plan's deep-supervision changes in `train.py` only, in three edits mapping to Milestone 1:
(1) added `LAMBDA_AUX = 0.3` after `CUTOUT_SIZE`; (2) added `self.aux_fc = nn.Linear(w2, num_classes)`
(w2 = 32*k = 128) to `ResNet.__init__` before `self.apply(self._weights_init)` so it gets the same
Kaiming-normal init as `fc`; (3) branched `ResNet.forward` on `self.training` to compute an auxiliary
logit from `F.adaptive_avg_pool2d(layer2_out,1).flatten(1)` and return `(main, aux)` in training / `main`
only at eval; (4) changed the training-loop loss to `loss_main + lam*loss_aux` with
`lam = LAMBDA_AUX*(1 - total_training_time/TIME_BUDGET_S)` (same elapsed-time fraction as the LR schedule).
AST parses, `ruff check` passes, `git diff --stat` shows only train.py (+26/−4).

### Surprises & Discoveries
- The eval path is safe by construction: `evaluator.evaluate` uses the EAGER `model` handle (not
  `compiled_model`) and calls `model.eval()` → `self.training=False` → forward returns a single tensor,
  so the frozen harness sees the unchanged main head. `compiled_model` is only ever called in the training
  loop with `self.training=True`, so torch.compile only traces the tuple-returning training branch.

### Decisions
- **Decayed λ (0.3→0) rather than fixed 0.3**: ensures the final, evaluated iterates optimize the pure main
  objective, sidestepping the regularizer-underfit wall (dropout EXP-022, SAM EXP-036) and any aux-induced
  distortion of the main head at convergence. Tied to `total_training_time` (pre-step value), identical to
  the LR `frac`, so the two schedules are phase-consistent.
- **Aux head on layer2 (not layer1)**: layer2 (after 6 of 9 blocks, 128-ch, 16×16) supervises mid-level
  features — the standard single-aux-head placement; deeper than layer1 (more semantic) but still leaving
  the final stage to specialize for the main head.
- **`loss.item()` display** now reflects the combined training loss (informational only; final_test_loss
  from the frozen eval is the comparable loss metric).

## Run Log

### Run 1
- **Description**: First and intended run of EXP-042 on idle GPU 0 (both H20s confirmed 0 MiB / 0% util at
  launch). Runs the deep-supervision recipe (aux layer2 head, λ 0.3→0) under the otherwise-baseline
  TA+Cutout recipe to test whether auxiliary mid-level supervision lifts best_test_acc above the 96.32 bar
  at a throughput-neutral ~88–94 epochs. Expected: a within-noise-to-modest effect (deep supervision is
  depth-scaling; net is shallow). Watching dt for throughput-neutrality (target ~8ms).
- **Job ID / PID**: (local background)
- **Log file**: run.log
- **WandB**: n/a
- **Status**: completed — DISCARDED (throughput-confounded)
- **Started**: 2026-06-09
- **Observations**: Ran clean on idle GPU 0 (GPU1 idle, no contention; GPU idle again post-run, so the
  dt regression was NOT external contention). But dt was bimodal/interleaved from ep1 — majority ~14ms
  (161 steps) / 16ms (92), only 98 steps @8ms — vs baseline steady 8ms. Steady ~14ms ⇒ CUDA-graph capture
  defeated by the `if self.training` branch returning tuple-vs-tensor. Throughput collapsed → only 55
  epochs (vs baseline ~91) → severe under-train. dt timeline interleaved throughout (ep1→ep51), confirming
  intrinsic-to-the-implementation, not mid-run contention.
- **Key Metrics**: best_test_acc 93.36% (−2.86pp), final_test_loss 0.2645, 55 ep, 21178 steps,
  peak_vram 495.5MB, total 389.2s. Source: run.log summary + `tr '\r' '\n' < run.log | grep dt`.

## Experimental Adjustments
- **Run 1 → Run 2 (throughput fix)**: Per the EXP-030→EXP-031 precedent (re-test throughput confounds
  before concluding) and project-insights L77/L108, Run 1's dt regression made it an unfair test of deep
  supervision (55 ep vs 91). Fix: removed the data-dependent `if self.training` branch. `forward` is now
  byte-identical to baseline (single-tensor main path → clean reduce-overhead CUDA-graph → eval unchanged);
  added a separate `forward_train` that ALWAYS returns `(main, aux)` (stable output structure) and is the
  compiled training target (`torch.compile(model.forward_train, mode="reduce-overhead")`). Goal: restore
  dt≈8ms to isolate the true intrinsic cost of the aux backward and give a fair top-1 test.

### Run 2
- **Description**: Compile-stable re-run of EXP-042 on idle GPU 0 (both H20s 0 MiB/0% at launch). Same
  deep-supervision recipe (aux layer2 head, λ 0.3→0) but with the throughput fix above. Tests whether
  restoring CUDA graphs returns dt to ~8ms (isolating compile-artifact vs intrinsic aux-backward cost) and
  whether a fairer-epoch run lifts best_test_acc toward the 96.32 bar.
- **Job ID / PID**: local background (b5l5v3kfy)
- **Log file**: run.log
- **Status**: completed — CLEAN, FAIR (reported run)
- **Started**: 2026-06-09
- **Observations**: Throughput fix CONFIRMED — dt steady 8ms from ep1 (613×8ms + 81×9ms + 1×11ms), exactly
  baseline. So Run 1's 14ms was 100% the CUDA-graph-breaking `self.training` branch, NOT intrinsic
  aux-backward cost (the aux head fuses essentially free). 90 epochs = baseline-equivalent ~91 → a fair,
  throughput-neutral test. GPU0 uncontended throughout (a GPU1 neighbor appeared only post-run). Eval = 90
  lines == num_epochs (≤1/epoch).
- **Key Metrics**: best_test_acc 95.91% (baseline 96.22, −0.31pp), final_test_acc 95.91, final_test_loss
  0.2026 (vs baseline 0.195 — slightly WORSE), 90 ep, 34795 steps, peak_vram 453.8MB, total 408.7s, exit 0.
  Source: run.log summary + dt distribution.

## Errors & Dead Ends
(none yet)

## Verification Results
### Conditions Checked
- **Cond 1 — primary metric ≥ baseline+0.1 (NECESSARY): FAILED.** best_test_acc = **95.91** < bar
  **96.32** (baseline 96.22 + 0.1); also below baseline by −0.31pp. Source: run.log `best_test_acc:`.
  → As the necessary primary-metric condition failed, the verdict is **no-improvement**.
- **Cond 2 — clean completion within budget: PASS.** Summary block printed, exit 0, total_seconds 408.7
  < 600, training_seconds 300.0, num_epochs 90. (Recorded for completeness.)
- **Cond 3 — no constraint violations: PASS.** `git diff --stat` = train.py only (prepare.py/eval
  untouched); eval lines 90 == num_epochs 90 (≤1/epoch); seed 42 unchanged (`torch.manual_seed(42)`);
  no new deps; no seed hacking. (Recorded for completeness.)

### Informational Metrics
- peak_vram_mb: 453.8 (slightly below baseline ~491; aux head adds little).
- num_epochs / num_steps: 90 / 34795 — throughput-neutral (baseline ~91), confirming a FAIR test.
- dt distribution: 613×8ms + 81×9ms + 1×11ms (steady 8ms = baseline; the compile fix fully restored it).
- final_test_loss: 0.2026 (vs baseline 0.195 — slightly WORSE; not even a polish/loss win).
