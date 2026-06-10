# Plan EXP-026: Bag-of-Tricks free convergence bundle (zero-init residual γ + no-bias-decay)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md

## Milestones

### Milestone 1: Code changes implemented and passing local checks
- [ ] Edit `ResNet.__init__` (train.py, after `self.apply(self._weights_init)` at L108): add a loop zeroing `bn2.weight` (γ) for every `BasicBlock`.
- [ ] Edit the optimizer construction (train.py L192-198): split `model.parameters()` into two SGD param groups — `weight_decay=WEIGHT_DECAY` for ndim≥2 (conv/linear weights), `weight_decay=0.0` for ndim≤1 (BN γ/β and biases).
- [ ] `git diff --name-only` shows ONLY `train.py`.
- [ ] Smoke check (`uv run python`): build `ResNet(3,10,4)`, assert (a) `num_params == 4,299,866` unchanged, (b) every `BasicBlock.bn2.weight` is all-zeros at init, (c) the SGD optimizer has 2 param groups with weight_decay [1e-4, 0.0] and their param counts sum to the total. AST parse clean.

### Milestone 2: Experiment launched and confirmed running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Within ~60s: `run.log` shows `params: 4,299,866`, clean compile, step lines appearing, no NaN.

### Milestone 3: Run completes; compute-neutrality confirmed
- [ ] Run exits 0 and prints the summary block.
- [ ] Confirm compute-neutral: `num_epochs ≈ 91` and `dt ≈ 8ms` (must match baseline — these changes touch only init values and optimizer param-grouping, zero FLOPs added; a large epoch deviation would signal an unintended throughput change).
- [ ] `total_seconds < 600`.

## Code Changes
- **train.py — `ResNet.__init__`, after L108 `self.apply(self._weights_init)`**: add
  ```python
  # EXP-026: zero-init the last BN gamma in each residual block (Bag of Tricks, He 2019)
  # so each residual branch outputs 0 at init -> block starts as identity, easing early
  # optimization. Compute/param-neutral (gamma is an existing BN param).
  for m in self.modules():
      if isinstance(m, BasicBlock):
          init.zeros_(m.bn2.weight)
  ```
  Tests the identity-init half of the hypothesis. Risk: none functional — γ learns away from 0 during training; the 6 identity-shortcut blocks start as exact identity, the 3 projection blocks as relu(BN(proj(x))).
- **train.py — optimizer construction (L192-198)**: replace the single-group `optim.SGD(model.parameters(), lr=PEAK_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)` with a two-group split:
  ```python
  decay, no_decay = [], []
  for p in model.parameters():
      (no_decay if p.ndim <= 1 else decay).append(p)  # ndim<=1 = BN gamma/beta + biases
  optimizer = optim.SGD(
      [
          {"params": decay, "weight_decay": WEIGHT_DECAY},
          {"params": no_decay, "weight_decay": 0.0},
      ],
      lr=PEAK_LR,
      momentum=MOMENTUM,
      nesterov=True,
  )
  ```
  Tests the no-bias-decay half. The training loop's LR update (`for pg in optimizer.param_groups: pg["lr"] = lr`, L228-229) and the LR readout (`optimizer.param_groups[0]["lr"]`, L252) both remain correct with two groups. Risk: none — standard idiom; only the WD applied to ~few BN/bias params changes.

## Configuration Changes
- Zero-init residual γ: bn2.weight init 1.0 → 0.0 for all 9 BasicBlocks (init-only; learns freely after)
- Weight decay scope: WD 1e-4 applied to ALL params → applied ONLY to conv/linear weights (ndim≥2); BN γ/β + fc bias get WD 0.0
- (no change to LR 0.2, batch 128, warmup 0.05, momentum, label smoothing, Cutout, augmentation, schedule, params)

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background.
- Resources: single NVIDIA H20 (GPU 0). VRAM ~0.5 GB (unchanged from baseline — no architecture change).
- Estimated runtime: ~300s training compute + ~30s startup/compile + ~91 evals ≈ baseline ~380-400s total. Must stay < 600s.
- Log output: `run.log` in project root via redirection; source of truth (summary block at end).
- Tool skill: none (local run).

## Abort Criteria
- **NaN/inf loss** at any point → kill, treat as failed. (Zero-γ init can momentarily zero residual gradients but is standard and stable; not expected.)
- **Loss not decreasing after warmup** (smoothed loss higher at ~step 1500 than at warmup end) → kill.
- **No output / hang**: no new step lines in `run.log` for >120s → kill.
- **Wall-clock runaway**: process running past ~580s → kill (respect <600s constraint).
- CUDA OOM (not expected) → kill, treat as failed.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22**, pass threshold **best_test_acc ≥ 96.32**. Run conditions in order; stop at first failure.

1. **Cond 1 — primary metric clears bar.** After completion:
   `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`
   PASS iff `best_test_acc ≥ 96.32`. Empty `best_test_acc:` ⇒ crash (`tail -n 50 run.log`) → crash verdict.
2. **Cond 2 — clean completion within budget.** PASS iff summary block printed, `grep -c Traceback run.log` == 0, and `total_seconds < 600`.
3. **Cond 3 — no constraint violations.** PASS iff: `git diff --name-only` lists only `train.py`; `num_params == 4,299,866`; eval-count == num_epochs (`grep -c "eval ep" run.log` == num_epochs); no new deps (only `init.zeros_` + SGD param groups, core torch); seed 42 unchanged.

**Compute-neutrality attribution note**: record `num_epochs` and mean `dt`. These changes add ZERO FLOPs, so epochs MUST be ~91 and dt ~8ms (matching baseline). If epochs deviate materially, investigate before attributing any metric delta (per the project's "verify epoch count, not just FLOPs" insight, EXP-015). A clean ~91-epoch run makes the accuracy delta a fair test of the Bag-of-Tricks levers.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -a "^peak_vram_mb:" run.log` — expect ~0.5 GB (unchanged).
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 / ~35,500 (compute-neutral check vs baseline).
- final_test_loss: `grep -a "^final_test_loss:" run.log` — compare to baseline 0.195 (convergence-quality signal).
