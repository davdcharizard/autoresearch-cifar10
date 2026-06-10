# EXP-015: Pre-activation (true-WRN) BasicBlocks — BN→ReLU→conv, bare-conv shortcuts, final BN→ReLU

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-015.md
- **Plan**: plans/plan-015.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-015
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Converted the post-activation ResNet-v1 `BasicBlock` to the canonical pre-activation
form (He et al. 2016 / WideResNet): `bn1` resized to `in_channels`; `forward` =
`out = relu(bn1(x))` → shared pre-activated input feeds both `conv1` and the projection
shortcut → `conv2(relu(bn2(out)))` → `out += shortcut`, with NO post-add ReLU. The
downsample shortcut became a bare 1×1 conv (no BN), stored as `None` for identity
blocks. In `ResNet`, dropped the stem `bn1` (stem is now bare `conv1`), added
`self.bn_final = BatchNorm2d(w3)`, and updated `forward` to `conv1 → layer1..3 →
relu(bn_final) → avgpool → fc`. This maps to plan-015 Milestone 1 (code change +
local checks). `_weights_init`, compile, and eval handle are unchanged.

### Surprises & Discoveries
None during implementation — the existing code structure mapped cleanly onto the
pre-activation rewrite. The identity-block shortcut went from `nn.Identity()` to
`None` (a small simplification that avoids an unnecessary module call in forward).

### Decisions
- Used `self.shortcut = None` for identity blocks rather than `nn.Identity()`, so the
  forward path passes the raw `x` (true identity) instead of the pre-activated input —
  this is the canonical pre-act identity path (gradient flows unimpeded through the
  add). Projection blocks apply the bare 1×1 conv to the *pre-activated* input
  `relu(bn1(x))`, per He 2016.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID recorded below)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08

Description:
- Running the EXP-015 pre-activation conversion at the EXP-012 baseline recipe (k=4,
  batch 128, peak LR 0.2 cosine, Nesterov, WD 1e-4, LS 0.1, Cutout(16), TrivialAugment,
  torch.compile, seed 42) on a single H20. Pre-activation is compute-neutral (same FLOPs)
  so we expect ~8ms/step and ~88–91 epochs, matching EXP-012's budget. Hypothesis: the
  cleaner identity/gradient path lifts best_test_acc above the 96.32 bar with a
  corroborating final_test_loss ≤ 0.195. A null (≈96.22, loss ≈0.195) would settle the
  block-ordering axis on this shallow k=4 ResNet-20.

Observations:
- Clean startup: `params: 4,298,970` (−896 from baseline 4,299,866 — exactly the expected
  BN-restructuring delta; not a capacity change), clean compile, no traceback (source: run.log L1-4).
- Steady-state throughput ~8ms/step at fast steps but with frequent jitter to 13–16ms (TA CPU
  augmentation jitter, same pattern as EXP-012) (source: run.log step 00050–00350).
- Healthy training: loss descended normally, ep 1 test_acc 44.42%, no NaN (source: run.log eval ep 1).
- **Ran only 78 epochs / 30,246 steps** — notably FEWER than EXP-012's 91 epochs. Pre-activation
  was NOT throughput-neutral in practice here (avg ~12,900 img/s vs EXP-012 higher): the extra
  final BN→ReLU + restructured block likely produced a less-efficient compile graph / more launch
  overhead. This is a fairness confound (fewer SGD steps) but does not change the verdict.
- Run exited 0, total_seconds 403.4 < 600 (source: run.log final summary, background task exit 0).

Key Metrics:
- best_test_acc: 95.85% @ ep 77 (source: run.log L`best_test_acc`)
- final_test_acc: 95.77% @ ep 78 (source: run.log)
- final_test_loss: 0.2012 @ ep 78 (source: run.log) — vs EXP-012's 0.195
- num_epochs: 78 | num_steps: 30,246 | num_params: 4,298,970 | training_seconds: 300.0 | total_seconds: 403.4

## Verification Results

### Conditions Checked
- **Cond 1 — clean completion within budget**: PASS. `best_test_acc` and `total_seconds`
  present; total_seconds 403.4 < 600; Traceback count 0 (source: run.log final summary).
- **Cond 2 — primary metric clears bar**: **FAIL**. best_test_acc = 95.85% < 96.32 bar
  (baseline 96.22 + 0.1). Δ = −0.37pp vs baseline. → verdict no-improvement. (Decisive condition.)
- **Cond 3 — no constraint violations**: skipped — not reached after Cond 2 failed. (For the
  record: scope was clean — `git diff --name-only` = train.py only; eval-line count 78 ==
  num_epochs 78 (eval once/epoch); num_params 4,298,970 ≈ 4.30M sane BN-restructuring delta;
  seed 42 intact.)

### Informational Metrics
- Not collected (only collected when all necessary conditions pass). For reference:
  num_epochs 78 (vs EXP-012's 91 — throughput confound), final_test_loss 0.2012 (> EXP-012's 0.195).

## Errors & Dead Ends

## Human Notes

> (none)
