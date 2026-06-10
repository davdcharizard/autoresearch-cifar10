# EXP-027: ResNet-D downsample (avgpool-2 + 1×1-stride-1 shortcut)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md
- **Plan**: plans/plan-027.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-027
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented Milestone 1 per plan-027: replaced the lossy stride-2 1×1 shortcut in `BasicBlock.__init__` (train.py L80-86) with the ResNet-D form. The branch logic is now three-way: `stride != 1` → `AvgPool2d(2,stride=2) → Conv2d(in,out,1,stride=1,bias=False) → BatchNorm2d(out)` (the 2 downsample blocks); `in≠out` only (stride==1) → plain stride-1 1×1 + BN (layer1 block0 channel-change); else `Identity` (the 6 non-downsample blocks). No config changes. Smoke test passed: AST parse clean; `num_params == 4,299,866` (unchanged — avgpool is param-free and the 1×1 conv shape is identical); forward (2,3,32,32)→(2,10) with no shape error (spatial alignment 32→16→8 holds between the main 3×3-stride-2 path and the avgpool-2 shortcut). Module inspection confirmed exactly layer2.0 and layer3.0 take the ResNet-D path; layer1.0 keeps the plain 1×1.

### Surprises & Discoveries
None. The three-way branch dropped in cleanly; the existing main-path conv1 is already 3×3 stride-2 (ResNet-B), so only the shortcut needed touching and spatial dims aligned on the first try.

### Decisions
- Kept the stride-1 channel-change block (layer1 block0) on a plain 1×1 (no avgpool) — ResNet-D's avgpool only applies where there is actual spatial downsampling (stride≠1); at stride 1 an avgpool(2,stride=2) would wrongly halve the spatial size. This matches the plan's three-way branch exactly.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID recorded at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A (no WandB in this project)
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- Running the full 300s-compute-budget training of the ResNet-D-shortcut k=4 WideResNet on a single H20. The hypothesis is that replacing the lossy stride-2 1×1 shortcut (which discards 3/4 of input pixels at the 2 downsample points) with an information-preserving avgpool-then-stride-1-1×1 improves generalization and lifts best_test_acc above the 96.32 bar. Critical secondary check: throughput-neutrality — ResNet-D adds ~zero FLOPs, so epochs should hold ~91 / dt ~8ms; if epochs drop below ~85 the result is compute-confounded (cf. EXP-024 BlurPool) and not a fair test.

Observations:
- Run exited 0, clean compile (no graph break on AvgPool2d), no NaN/Traceback (source: run.log; Traceback count 0).
- **REGRESSION**: best_test_acc 95.75% vs baseline 96.22 (−0.47pp), well below the 96.32 bar (source: run.log summary block).
- **Throughput-neutral / FAIR test**: num_epochs 89, num_steps 34562 → mean dt ≈ 8.7ms (baseline ~91 ep / 35500 steps / 8ms). Epochs 89 > 85, so NOT compute-confounded (unlike EXP-024 BlurPool's 77). The 2.6% step drop is the trivial avgpool cost; it does NOT explain a 0.47pp regression.
- **Loss also worse**: final_test_loss 0.2099 vs baseline 0.195 — unlike the convergence-polish nulls (EXP-006/019/020/026) which IMPROVED loss, ResNet-D made BOTH top-1 and loss worse. The avgpool low-pass on the 2 downsample shortcuts blurs discriminative high-frequency signal rather than helping.

Key Metrics:
- best_test_acc: 95.75% (source: run.log summary block)
- final_test_loss: 0.2099 (source: run.log)
- num_epochs: 89 | num_steps: 34562 | num_params: 4,299,866 | peak_vram_mb: 467.3 | total_seconds: 404.6 (source: run.log)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar**: FAIL. best_test_acc = 95.75% < 96.32 (baseline 96.22 + 0.1). The −0.47pp regression is decisive; per the plan, stop at first failure. (source: run.log summary block)
- **Cond 2 — clean completion within budget**: PASS (informational, recorded though Cond 1 already failed). Summary block printed, Traceback count 0, total_seconds 404.6 < 600. (source: run.log)
- **Cond 3 — no constraint violations**: PASS (informational). `git diff --name-only` = train.py only; num_params 4,299,866 unchanged; 89 evals for 89 epochs (≤1/epoch); no new deps (AvgPool2d/Conv2d/BatchNorm are core torch); seed 42 unchanged. (source: git diff, run.log)

**MANDATORY attribution note (epoch-wall + FLOPs-neutral-≠-wall-clock-neutral, EXP-015/024):** num_epochs 89, mean dt ≈ 8.7ms. Epochs held in the throughput-neutral band (89 vs ~91; >>85), so the regression is a FAIR test of the lossless-downsample hypothesis — NOT a compute confound like EXP-024 BlurPool (77 epochs). ResNet-D's avgpool downsample genuinely HURTS top-1 (and loss) on this shallow 2-downsample CIFAR net. Verdict: **no-improvement**.

### Informational Metrics

- peak_vram_mb: 467.3 (≈ baseline, as expected — no param change)
- num_epochs / num_steps: 89 / 34562 (vs baseline ~91 / ~35500 — throughput-neutral, slight avgpool overhead)
- final_test_loss: 0.2099 (WORSE than baseline 0.195 — ResNet-D degraded loss too)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
