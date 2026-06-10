# EXP-051: LayerScale — learnable per-channel residual-branch scaling

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-051.md
- **Plan**: plans/plan-051.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-051
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan-051 Milestone 1: two edits to `BasicBlock` in `train.py`. (1) `__init__`: added `self.layer_scale = nn.Parameter(torch.full((out_channels, 1, 1), 0.1))` — a per-channel learnable scale, init 0.1 (CaiT LayerScale). (2) `forward`: inserted `out = out * self.layer_scale` between the residual branch (`bn2(conv2)`) and the `+= shortcut(x)` add. Smoke test passed: AST clean, diff = `train.py` only, num_params 4,301,210 (= baseline 4,299,866 + 1,344 LayerScale scalars), 9 `layer_scale` params sized {64×3, 128×3, 256×3}, all init 0.1. `_weights_init` only touches Conv2d/Linear (isinstance), so layer_scale keeps its 0.1 init (confirmed).

### Surprises & Discoveries
- None. The only console noise during smoke test was a UserWarning from my debug `float(param[0])` print (converting a requires_grad tensor to scalar) — not from the model code.

### Decisions
- Init 0.1 (not the tiny 1e-4 CaiT uses for very deep nets): on a shallow 9-block net, too-small an init would over-suppress residual branches and need many epochs to recover (underfit risk); 0.1 is the CaiT default for ≤18 blocks.
- Per-channel `(C,1,1)` shape (broadcasts over N,H,W), channels_last-safe; a static elementwise multiply → expected CUDA-graph-safe / dt-neutral.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID at launch — background Bash)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (401.9s total wall)

Description:
- Running the baseline k=4 ResNet-20 recipe with a CaiT LayerScale (learnable per-channel residual scale, init 0.1) added to every BasicBlock, on idle GPU 1. Tests whether a learnable residual-magnitude DOF helps generalization at fixed capacity. Expect throughput-neutral (dt 8ms, ~91 ep, no CUDA-graph break) and best_test_acc near baseline (EXP-026 zero-init-γ predicts a likely null — "needs depth" on this shallow net). Clears the bar (96.32) only if the modern LayerScale formulation helps where zero-init-γ didn't.

Observations:
- **NOT throughput-free — the LayerScale multiply added ~1ms/step (dt 8→9ms, mixed 314×8ms / 346×9ms)**, the same partial-fusion cost as SiLU (EXP-028). The `(C,1,1)` pointwise multiply did not fully fuse into the surrounding conv/BN/relu under inductor. Result: num_epochs 85 (vs baseline 91), num_steps 33,140 (vs ~35.5k) — a mild ~6-epoch under-train confound (NOT a CUDA-graph break, which would be 14-16ms). (source: run.log, dt extraction)
- **REGRESSED −0.39pp**: best_test_acc 95.83 < baseline 96.22. final_test_loss 0.1987 ≈ baseline 0.195 (no loss gain either). test_acc plateaued in the mid-95.8s in the tail (ep82 95.63 → ep85 95.78, best 95.83 @ earlier) — not still climbing, so not severely under-trained, but the ~6 lost epochs contribute. (source: run.log summary + last evals)
- **The residual-scaling DOF did not help** — consistent with EXP-026 (zero-init-γ, "needs depth") now confirmed for the modern LayerScale formulation too. The −0.39pp is a mix of (a) no benefit from the lever and (b) the ~6-epoch cost of the unfused multiply. Either way LayerScale does not help on this shallow net. (source: run.log)
- Clean completion: training_seconds 300.0, total_seconds 401.9 < 600, peak_vram 522.1 MB, no NaN/traceback. (source: run.log summary)

Key Metrics:
- best_test_acc: 95.83% (−0.39pp vs baseline 96.22; −0.49pp vs bar 96.32); final_test_acc 95.78% @ ep85; final_test_loss 0.1987
- num_epochs: 85; num_steps: 33,140; training_seconds: 300.0; total_seconds: 401.9; peak_vram_mb: 522.1; num_params: 4,301,210 (baseline + 1,344 LayerScale scalars)
- dt: mixed 8-9ms (314×8ms, 346×9ms) — ~1ms unfused-multiply cost → 85 ep (vs baseline 91)

## Verification Results

### Conditions Checked
1. **Primary necessary condition (`best_test_acc ≥ 96.32`)** — FAIL. best_test_acc 95.83 < 96.32 (−0.49pp vs bar, −0.39pp vs baseline). → no-improvement. (source: run.log `best_test_acc: 95.83%`)
2. **Run completes cleanly within budget** — PASS. Summary printed; total_seconds 401.9 < 600; training_seconds 300.0; num_params 4,301,210 (= baseline + 1,344 LayerScale); no crash/NaN. (source: run.log summary)
3. **No hard-constraint violations** — PASS. `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps; seed 42 unchanged; no seed hacking. (source: git diff)

Verdict: **no-improvement** (primary condition fails, −0.39pp; clean valid run, mild ~6-epoch dt confound). Run completed cleanly → Outcome = completed.

### Informational Metrics
- num_epochs / num_steps: 85 / 33,140 (mild reduction from the ~1ms unfused-multiply cost).
- peak_vram_mb: 522.1 (slightly higher than baseline ~454 — LayerScale activations/grads).
- final_test_loss: 0.1987 (≈ baseline 0.195; no loss gain — not even a polish win).
- dt: 8-9ms mixed (NOT throughput-free as hoped — the multiply partially unfused).

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
