# EXP-047: Ghost BatchNorm — implicit regularization via small-sub-batch statistics

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-047.md
- **Plan**: plans/plan-047.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-047
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the GhostBatchNorm2d swap per plan-047 (Milestone 1). Added `GHOST_SPLITS = 4` to the hyperparameter block and a `GhostBatchNorm2d(nn.BatchNorm2d)` subclass above `BasicBlock`. The training-mode forward splits ONLY the outer batch dim via `x.view(s, N//s, C, H, W)` (a valid view on channels_last — deliberately NOT the channel-fold `view(N/s, C*s, H, W)` trick, which would force an NCHW copy), normalizes each of the s=4 ghost groups (size 32) by its own biased mean/var over `(group, H, W)`, applies the shared affine, and reshapes back to `(N,C,H,W)`. Running stats update IN-PLACE (`.mul_().add_()`) from FULL-batch statistics under `torch.no_grad()` — the same in-place pattern stock nn.BatchNorm2d uses under reduce-overhead, so CUDA-graph-safe. Eval-mode forward delegates to `F.batch_norm(..., training=False, ...)`, byte-identical to standard BN. Swapped all 5 BN sites: stem `bn1`, `BasicBlock.bn1`/`bn2`, and the downsample-shortcut BN. AST parses; no `nn.BatchNorm2d` instantiations remain; `_weights_init` still only inits Conv2d/Linear so GhostBN keeps BN default init (weight=1,bias=0,running_mean=0,running_var=1); compile mode stays `reduce-overhead`; optimizer/schedule/aug/seed/batch all unchanged.

### Surprises & Discoveries
- **GhostBN cost a steady ~1ms/step (8→9ms)** — NOT a CUDA-graph break (dt dist: 589×9ms, 13×10ms, 1×16ms[compile]; a graph break would be ~14-16ms throughout, cf. EXP-042). The graph stayed intact (static shapes worked), but the manual per-ghost + full-batch reductions add real work → epochs 91→78 (−14%). So GhostBN is NOT perfectly throughput-neutral with this manual implementation.
- **Markedly slower early convergence**: ep1 19.98% / ep2 24.13% / ep3 31.24% (vs baseline-kernel EXP-046 ep1 45.7% / ep2 54.96%). The noisier ghost-32 statistics perturb early optimization at high LR. It climbs (ep6 46.5% → ep7 56.9% → ep8 59.3%) but never catches up.

### Decisions
- Used FULL-batch stats (not ghost-averaged) for the running-buffer update — gives a cleaner population estimate for the (unchanged) eval path; the regularization comes from the per-ghost NORMALIZATION during training, while eval should still see the true population statistics.
- Computed normalization stats manually (per-ghost mean/var) rather than via the cudnn channel-fold `F.batch_norm` trick, because that trick is invalid on channels_last tensors and would force a layout copy (dt risk). The manual path adds a second (full-batch) reduction for the running update; the 3×3 convs dominate per-step time so dt is expected to stay ~8ms — gated explicitly in Milestone 3.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (394.8s total wall)

Description:
- Running the baseline k=4 ResNet-20 recipe with all BatchNorm2d replaced by GhostBatchNorm2d (num_splits=4, ghost size 32) on idle GPU 1 (foreign job isolated on GPU 0, no contention). Tests whether normalization-noise regularization (the one untouched accuracy axis) lifts top-1 ≥96.32, or lands within noise (closing the axis).

Observations:
- **Banner confirmed param-neutral**: `ResNet-20 | params: 4,299,866` — GhostBN adds no parameters. (source: run.log head)
- **dt steady 9ms (NOT a graph break)**: distribution 589×9ms, 13×10ms, 1×16ms (the single compiled step). A reduce-overhead CUDA-graph break would show ~14-16ms throughout (EXP-042) — it did not, so the static-shape design held and the graph stayed intact. The ~1ms premium over baseline 8ms is the manual per-ghost + full-batch reduction cost. (source: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`)
- **Epochs trimmed by the dt premium**: num_epochs 78 (vs baseline ~91, −14%), num_steps 30,165 — the 9ms dt cut ~13 epochs from the 300s budget → partial epoch-wall underfit.
- **Slow early convergence + worse final loss**: ep1 19.98% (vs ~45.7% baseline), recovering to ep8 59.3% but plateauing; final_test_loss 0.2197 > baseline 0.195 → the run is both underfit AND the GhostBN noise did not improve generalization (loss is worse, not just top-1). best_test_acc 95.16 reached late (best held from ~ep70s; ep76-78 all ~95.0-95.16). (source: run.log eval lines + summary)
- Clean completion: training_seconds 300.0, total_seconds 394.8 < 600, startup 2.9s, peak_vram 520.8 MB (≈baseline), no NaN/traceback. (source: run.log summary)

Key Metrics:
- best_test_acc: 95.16% (−1.06pp vs baseline 96.22); final_test_acc 95.04% @ ep78; final_test_loss 0.2197
- num_epochs: 78; num_steps: 30,165; training_seconds: 300.0; total_seconds: 394.8; startup_seconds: 2.9; peak_vram_mb: 520.8; num_params: 4,299,866 (unchanged)
- dt: steady 9ms (589×9ms, 13×10ms, 1×16ms-compile) — ~1ms premium over baseline 8ms → epochs 91→78

## Verification Results

### Conditions Checked
1. **Run completes cleanly within budget** — PASS. Summary printed; total_seconds 394.8 < 600; training_seconds 300.0; startup 2.9s; num_params 4,299,866 (unchanged); no crash/NaN. (run.log summary)
2. **Throughput-neutrality gate (de-confound)** — PARTIAL. dt steady 9ms (NOT a graph break — would be ~14-16ms; static-shape design held), but ~1ms above baseline 8ms → num_epochs 78 vs ~91 (−14%). The accuracy comparison is therefore mildly epoch-confounded (a baseline-BN run at 78 ep would also underfit somewhat), but dt stayed within the plan's ≤9ms acceptable band so this is a VALID research result, not an implementation confound requiring retry. Note for the verdict: the final loss is WORSE (0.2197 > 0.195) and early convergence slower, so GhostBN's net effect is negative beyond the pure epoch reduction.
3. **Primary necessary condition (`best_test_acc ≥ 96.32`)** — FAIL. best_test_acc 95.16 < 96.32 (−1.06pp vs baseline). → no-improvement.
4. **No hard-constraint violations** — PASS. `git diff --name-only` = `train.py` only; prepare.py/eval untouched; evaluate() once/epoch (loop unchanged); no new deps (GhostBN uses only torch); seed 42 unchanged; GhostBN is a deterministic architectural change (no added RNG) → no seed hacking.

Verdict: **no-improvement** (primary condition fails, −1.06pp regression). Run completed cleanly → Outcome = completed.

### Informational Metrics
- num_epochs / num_steps: 78 / 30,165 (the 9ms dt premium cut ~13 epochs vs baseline ~91).
- peak_vram_mb: 520.8 (≈ baseline; GhostBN's 5-D view/reductions add trivial memory).
- final_test_loss: 0.2197 (WORSE than baseline 0.195) — GhostBN noise did not improve generalization even setting aside the epoch reduction.
- dt: steady 9ms (vs baseline 8ms).

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
