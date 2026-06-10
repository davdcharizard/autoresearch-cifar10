# Plan EXP-027: ResNet-D downsample (avgpool-2 + 1×1-stride-1 shortcut)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md

## Milestones

### Milestone 1: Code changes implemented and passing local checks
- [ ] Edit `BasicBlock.__init__` shortcut branch (train.py L80-86): for the downsample case (stride≠1) use ResNet-D `AvgPool2d(2,stride=2) → Conv2d(in,out,1,stride=1,bias=False) → BatchNorm2d(out)`; keep a plain stride-1 1×1 for the stride-1 channel-change case (in≠out); keep Identity otherwise.
- [ ] `git diff --name-only` shows ONLY `train.py`.
- [ ] Smoke check (`uv run python`): build `ResNet(3,10,4)`, assert (a) `num_params == 4,299,866` unchanged (avgpool is param-free, 1×1 shape identical), (b) a forward pass on a (2,3,32,32) tensor returns (2,10) with no shape error (spatial alignment 32→16→8 holds between main path and ResNet-D shortcut). AST parse clean.

### Milestone 2: Experiment launched and confirmed running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Within ~60s: `run.log` shows `params: 4,299,866`, clean compile (no graph break on AvgPool2d), step lines appearing, no NaN.

### Milestone 3: Run completes; throughput-neutrality confirmed (CRITICAL)
- [ ] Run exits 0 and prints the summary block.
- [ ] **KEY: confirm throughput-neutral** — `num_epochs ≈ 91` and `dt ≈ 8ms`. Per the EXP-015 High insight (FLOPs-neutral ≠ wall-clock-neutral under torch.compile), a restructured graph can silently cost epochs even at ~equal FLOPs. If `num_epochs < ~85`, the result is COMPUTE-CONFOUNDED (like EXP-024) and any delta is not a clean test.
- [ ] `total_seconds < 600`.

## Code Changes
- **train.py — `BasicBlock.__init__` shortcut (L80-86)**: replace
  ```python
  if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels),
      )
  else:
      self.shortcut = nn.Identity()
  ```
  with
  ```python
  if stride != 1:
      # ResNet-D downsample (Bag of Tricks, He 2019): average-pool before a stride-1
      # 1x1 so the shortcut keeps all input pixels instead of dropping 3/4 (the lossy
      # stride-2 1x1). Near-compute-neutral: 1x1 FLOPs unchanged (runs at the pooled
      # resolution); avgpool is trivial. Main-path conv1 stays 3x3 stride-2 (untouched)
      # — this is the key difference from EXP-024 BlurPool, which moved the heavy 3x3
      # conv to stride-1 and cratered epochs.
      self.shortcut = nn.Sequential(
          nn.AvgPool2d(2, stride=2),
          nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
          nn.BatchNorm2d(out_channels),
      )
  elif in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
          nn.BatchNorm2d(out_channels),
      )
  else:
      self.shortcut = nn.Identity()
  ```
  Tests the hypothesis that an information-preserving residual downsample improves generalization. In our net only the 2 downsample blocks (layer2/layer3 first block, stride=2 & in≠out) take the ResNet-D path; layer1 block0 (stride=1, in≠out) takes the plain 1×1; the other 6 blocks are Identity. Risk/edge: spatial alignment — AvgPool2d(2,stride=2) maps 32→16 and 16→8, matching the main path's 3×3-stride-2 (padding=1) output; verified in the smoke test. The restructured shortcut graph is the throughput risk (EXP-015), addressed by the Milestone-3 epoch check.

## Configuration Changes
- (none — pure architecture change to the shortcut; LR 0.2, batch 128, warmup 0.05, WD 1e-4, LS 0.1, Cutout 16, TA, cosine-to-0, params 4,299,866 all unchanged)

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background.
- Resources: single NVIDIA H20 (GPU 0). VRAM ~0.5 GB (unchanged — no param change, tiny extra activation for the pooled shortcut).
- Estimated runtime: ~300s training compute + ~30s startup/compile + ~91 evals ≈ baseline ~380-405s total. Must stay < 600s.
- Log output: `run.log` in project root via redirection; source of truth.
- Tool skill: none (local run).

## Abort Criteria
- **NaN/inf loss** at any point → kill, treat as failed.
- **Loss not decreasing after warmup** → kill.
- **No output / hang**: no new step lines in `run.log` for >120s → kill.
- **Wall-clock runaway**: process running past ~580s → kill (<600s constraint).
- CUDA OOM (not expected) → kill, treat as failed.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22**, pass threshold **best_test_acc ≥ 96.32**. Run conditions in order; stop at first failure.

1. **Cond 1 — primary metric clears bar.** After completion:
   `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`
   PASS iff `best_test_acc ≥ 96.32`. Empty `best_test_acc:` ⇒ crash (`tail -n 50 run.log`) → crash verdict.
2. **Cond 2 — clean completion within budget.** PASS iff summary block printed, `grep -c Traceback run.log` == 0, `total_seconds < 600`.
3. **Cond 3 — no constraint violations.** PASS iff: `git diff --name-only` lists only `train.py`; `num_params == 4,299,866`; eval-count == num_epochs (`grep -c "eval ep" run.log` == num_epochs); no new deps (AvgPool2d/Conv2d/BatchNorm are core torch); seed 42 unchanged.

**MANDATORY attribution note (project's epoch-wall + FLOPs-neutral-≠-wall-clock-neutral insights, EXP-015/024 — High):** record `num_epochs` and mean `dt`. ResNet-D adds ~zero FLOPs, so epochs SHOULD be ~91 / dt ~8ms.
- epochs ~91 & dt ~8ms → throughput-neutral → the accuracy delta is a FAIR test of the lossless-downsample hypothesis (whatever the verdict).
- epochs < ~85 (dt risen) → COMPUTE-CONFOUNDED like EXP-024/015 (restructured graph less compile-efficient) → a regression cannot be attributed to ResNet-D's merit; note this explicitly in the report.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -a "^peak_vram_mb:" run.log` — expect ~0.5 GB.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 / ~35,500 (throughput-neutral check vs baseline).
- final_test_loss: `grep -a "^final_test_loss:" run.log` — compare to baseline 0.195.
