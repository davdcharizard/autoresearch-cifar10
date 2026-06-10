# Plan EXP-032: Multi-scale feature-aggregation classifier head
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-032.md

## Milestones

### Milestone 1: Code changes implemented and passing local checks
- [ ] **`ResNet.__init__`** — widen the classifier input to accept concatenated layer2+layer3 features. Change train.py L107 `self.fc = nn.Linear(w3, num_classes)` → `self.fc = nn.Linear(w2 + w3, num_classes)` (w2=32k=128, w3=64k=256 → `Linear(384, 10)`).
- [ ] **`ResNet.forward`** — pool layer2 AND layer3, concatenate, classify. Replace train.py L128-133:
  ```python
          out = self.layer1(out)
          out2 = self.layer2(out)                       # 128ch @16x16 (mid-level)
          out3 = self.layer3(out2)                      # 256ch @8x8  (high-level)
          p2 = F.adaptive_avg_pool2d(out2, 1).flatten(1)  # (B,128)
          p3 = F.adaptive_avg_pool2d(out3, 1).flatten(1)  # (B,256)
          out = torch.cat([p2, p3], dim=1)              # (B,384) multi-scale features
          return self.fc(out)
  ```
- [ ] `git diff --name-only` shows ONLY `train.py`.
- [ ] AST parse clean (`uv run python -c "import ast; ast.parse(open('train.py').read())"`).
- [ ] Smoke check (`uv run python`): build `ResNet(3,10,width_mult=4).cuda()`, assert (a) params == **4,301,146** (baseline 4,299,866 + 1280 from the wider fc), (b) one forward on a `(8,3,32,32)` channels_last cuda batch returns shape `(8,10)` with no error, (c) the fc weight has shape `(10, 384)`.

### Milestone 2: Experiment launched and confirmed running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Within ~60s: `run.log` shows `params: 4,301,146`, clean compile, step lines, no NaN, loss decreasing normally.

### Milestone 3: Run completes; throughput-neutrality confirmed
- [ ] Run exits 0 and prints the summary block.
- [ ] **Confirm throughput-neutral** — `num_epochs ≈ 91` and `dt ≈ 8ms` (the extra layer2 global-avg-pool on a 128×16×16 tensor + a 384-wide fc matmul are negligible vs the convs). If epochs drop materially the change is unexpectedly costly → note as a throughput confound.
- [ ] `total_seconds < 600`.

## Code Changes
- **train.py L107 (`ResNet.__init__`)**: `nn.Linear(w3, num_classes)` → `nn.Linear(w2 + w3, num_classes)`. Adds 1280 fc weights (+0.03%): params 4,299,866 → 4,301,146.
- **train.py L128-133 (`ResNet.forward`)**: keep references to both `layer2` output (`out2`, 128ch@16×16) and `layer3` output (`out3`, 256ch@8×8); global-avg-pool each to (B,128) and (B,256); `torch.cat` → (B,384); `self.fc`.

  **Why this tests the hypothesis**: the feature-aggregation / classifier-head axis is the one structural lever untouched in 32 experiments (every run pooled only layer3 → fc). Feeding the classifier multi-scale features (mid-level layer2 ⊕ high-level layer3) is a compute-neutral, integrity-clean inductive-bias change on the generalization side — it changes WHAT the classifier sees (can move top-1) without adding compute (no epoch wall, cf. project-insights High) and without the polish-vs-top1 trap (it is not an optimizer/averaging/init tweak). It also adds a direct gradient path to layer2.

  **Risks/edge cases**: (a) Directly supervising layer2 via the head could disrupt the tuned feature hierarchy → mild regression (the dominant risk). (b) Magnitude likely small on an already-good net → no-improvement within the ±0.2pp noise floor. (c) `torch.compile` recompiles for the new graph (one-time, charged to step-1 budget, negligible). (d) Compute-neutral → no epoch-wall confound; VERIFY epochs ~91. (e) **`num_params` legitimately changes to 4,301,146** — this is the intended head change, NOT a constraint violation (the goal sets no fixed param-count constraint; prior runs' "4,299,866 unchanged" was a per-experiment fairness check, not a hard cap). The verification param check below uses the NEW expected value.

## Configuration Changes
- (none — pure architecture change to the classifier head. PEAK_LR 0.2, WARMUP_FRAC 0.05, batch 128, WD 1e-4, LS 0.1, Cutout 16, TA, Nesterov, momentum 0.9, cosine-to-0, seed 42, torch.compile(reduce-overhead) all unchanged.)

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background, with a `Monitor` watch on run.log for per-epoch evals, the summary, and NaN/error/dt.
- Resources: single NVIDIA H20 (GPU 0). VRAM ~0.5 GB (one extra small pooled activation; ≈ baseline).
- Estimated runtime: ~380–405s total. Must stay < 600s.
- Log output: `run.log` in project root via redirection; source of truth.
- Tool skill: none (local run).

## Abort Criteria
- **NaN/inf loss** at any point → kill, treat as failed.
- **Shape/compile error** at step 1 (traceback in run.log, e.g. cat/fc dim mismatch) → kill, fix, re-launch (counts as the code-fix retry).
- **Loss not decreasing after warmup** → kill.
- **No output / hang**: no new step lines for >120s → kill.
- **Wall-clock runaway**: process past ~580s → kill.
- CUDA OOM (not expected) → kill, treat as failed.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22**, pass threshold **best_test_acc ≥ 96.32**. Run conditions in order; stop at first failure.

1. **Cond 1 — primary metric clears bar.** After completion:
   `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`
   PASS iff `best_test_acc ≥ 96.32`. Empty `best_test_acc:` ⇒ crash (`tail -n 50 run.log`) → crash verdict.
2. **Cond 2 — clean completion within budget.** PASS iff summary block printed, `grep -c Traceback run.log` == 0, `total_seconds < 600`.
3. **Cond 3 — no constraint violations.** PASS iff: `git diff --name-only` lists only `train.py`; `num_params == 4,301,146` (the intended +1280 multi-scale-head change; NOT 4,299,866); eval-count == num_epochs (`grep -c "eval ep" run.log` == num_epochs); no new deps (core torch only); seed 42 unchanged.

**MANDATORY attribution note (epoch-wall + FLOPs-neutral-≠-wall-clock-neutral, EXP-015/024/030/031):** record `num_epochs` and mean `dt`. The change is compute-neutral, so epochs SHOULD be ~91 / dt ~8ms.
- epochs ~91 & dt ~8ms → throughput-neutral → the accuracy delta is a FAIR test of the multi-scale head.
- epochs < ~85 (dt risen) → unexpectedly costly → COMPUTE-CONFOUNDED → note explicitly; a regression cannot be cleanly attributed to the head change's merit.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -a "^peak_vram_mb:" run.log` — expect ~0.5 GB.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 / ~35,000 (throughput-neutral check).
- final_test_loss: `grep -a "^final_test_loss:" run.log` — compare to baseline 0.195.
