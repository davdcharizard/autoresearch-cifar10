# Experiment Log EXP-063: Augmentation cooldown @0.10 on the AugMix-p0.5 best recipe

## Execution
- **Created**: 2026-06-09
- **Brainstorm**: brainstorm/brainstorm-063.md
- **Plan**: plans/plan-063.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-063
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan Milestone 1 as three edits to train.py: (1) added `COOLDOWN_FRAC = 0.10` constant; (2) after `train_loader`, added a clean `cooldown_tf` (RandomCrop+Flip+ToTensor+Normalize — `train_tf` minus the AugMix RandomApply) and a `cooldown_loader` over a second CIFAR10 instance (`download=False`); (3) in the training loop, compute `cooldown_active = (total_training_time/TIME_BUDGET_S) >= (1.0 - COOLDOWN_FRAC)` once per epoch, select `cooldown_loader` when active, gate the GPU `cutout_batch` call behind `if not cooldown_active:`, and print a one-time `>> AUG COOLDOWN fired` marker. All else byte-identical to EXP-054.

### Surprises & Discoveries
None. Smoke checks: AST OK; `git diff --name-only` == train.py only; programmatic confirmation that `cooldown_tf` contains no AugMix, the `evaluator.evaluate(model, device)` eval call is unchanged, and `COOLDOWN_FRAC == 0.10`.

### Decisions
Used a second loader rather than mutating `train_set.transform` mid-run — cleaner and avoids any ambiguity about whether a transform mutation propagates to freshly-spawned dataloader workers. The Cutout gate is a host-level Python branch on `cooldown_active`, applied to the input tensor BEFORE `compiled_model(inputs)`, so the compiled forward's input shape/dtype (128,3,32,32 bf16) is unchanged → no recompile and no cudagraph break (contrast EXP-042, which branched inside the compiled forward). Cooldown removes tail augmentation work → wall-neutral-to-faster (no EXP-061 wall-overrun risk).

## Run Log

### Run 1
- **Description**: Aug cooldown @0.10 on the EXP-054 AugMix-p0.5 best. For the final 10% of the 300s budget, AugMix (clean loader) and GPU Cutout (gated) are disabled so the model weights + BN running stats jointly re-adapt to the near-clean eval distribution. Tests whether this train→eval distribution-matching tail-climb (EXP-034 precedent: +0.21 climb, +0.04 net on TA; EXP-061 joint-adaptation mechanism) clears the 96.55 bar on the AugMix recipe. Expected: near-noise on a deeply-mapped plateau (50%-subset AugMix has a smaller train→eval gap than full-coverage TA). Launched on idle GPU 1 (GPU 0 has foreign proc PID 1314331).
- **Job ID**: (local, background bash)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09
- **Key Metrics**: best_test_acc 96.31% | final_test_loss 0.2011 | total_seconds 574.1 (faster than EXP-054's 593s — cooldown lightens the tail, as predicted) | num_epochs 91 | num_steps 35323 | num_params 4,299,866 | peak_vram_mb 453.8. dt: 608×8ms + 96×9ms + 1×10ms + 1×30ms (compile warmup) — uncontended, throughput identical to EXP-054. 0 NaN/error.
- **Observations**: Cooldown fired ep83/frac0.907 (marker present → mechanism engaged). Tail-climb signature confirmed: pre-cooldown base 96.09 (ep82) → 96.27 (ep84) → 96.31 (ep86), a +0.22 clean-tail climb — matches EXP-034's +0.21. But the pre-cooldown base (96.09) sat below where EXP-054's full-aug trajectory reaches (96.45), so the net result (96.31) fell short. Final-eval loss 0.2011 slightly above EXP-054's 0.1968 (clean-tail re-adaptation didn't lower loss vs full-aug).

## Experimental Adjustments
(none)

## Errors & Dead Ends
(none)

## Verification Results

### Conditions Checked
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **96.31%** < 96.55. **FAILED.** (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: not formally evaluated (aborted after condition 1), but for the record all pass: total_seconds 574.1 < 600 ✓, num_params 4,299,866 ✓, num_epochs 91 ✓, 0 NaN/error ✓.
3. **Necessary condition 3 — no hard-constraint violation**: not formally evaluated (aborted), but for the record: `git diff --name-only` == train.py only ✓; eval once/epoch on eager model unchanged ✓; cooldown-fired marker present (ep83/frac0.907) ✓ — mechanism genuinely engaged, not a silent no-op; uncontended dt (608×8ms) ✓.

**Verdict**: no-improvement — valid, in-budget, uncontended run that missed the accuracy bar (96.31 < 96.55), a −0.14pp regression vs baseline 96.45. The cooldown mechanism worked exactly as documented (real +0.22 clean-tail climb, matching EXP-034's +0.21) but climbed from a pre-cooldown base (96.09) that sat below where EXP-054's full-augmentation trajectory reaches (96.45). Sacrificing the final ~9 epochs of augmented training for clean re-adaptation is net-negative on the AugMix recipe — the same outcome cooldown gave on the TA recipe (EXP-034: +climb, net below the no-cooldown baseline).

