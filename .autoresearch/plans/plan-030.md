# Plan EXP-030: Gradient Centralization (GC) on the SGD step
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-030.md

## Milestones

### Milestone 1: Code changes implemented and passing local checks
- [ ] Insert a Gradient Centralization step in the training loop between `loss.backward()` (train.py L237) and `optimizer.step()` (L238): for each `p` in `model.parameters()` with `p.grad is not None and p.grad.ndim > 1`, subtract the per-output-unit mean of the gradient over the fan-in dims: `p.grad.add_(-p.grad.mean(dim=tuple(range(1, p.grad.ndim)), keepdim=True))`.
- [ ] `git diff --name-only` shows ONLY `train.py`.
- [ ] Smoke check (`uv run python`): build `ResNet(3,10,4)`, run one forward+backward on a (8,3,32,32) batch, apply the GC loop, then assert (a) for every conv/linear weight grad the per-output-unit mean is ≈0 after centralization (`g.mean(dim=(1..))` ≈ 0, atol 1e-6), (b) 1-D params (BN γ/β, bias) grads are UNCHANGED (not centralized), (c) params == 4,299,866 unchanged. AST parse clean.

### Milestone 2: Experiment launched and confirmed running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Within ~60s: `run.log` shows `params: 4,299,866`, clean compile, step lines, no NaN, loss decreasing normally.

### Milestone 3: Run completes; throughput-neutrality confirmed
- [ ] Run exits 0 and prints the summary block.
- [ ] **Confirm throughput-neutral** — `num_epochs ≈ 91` and `dt ≈ 8ms` (GC adds a handful of tiny reductions per step; should be negligible). If epochs drop materially the GC loop is unexpectedly costly (e.g. a host-device sync) → note as a throughput confound.
- [ ] `total_seconds < 600`.

## Code Changes
- **train.py — insert GC between `loss.backward()` and `optimizer.step()` (L237-238)**:
  ```python
              loss.backward()
              # Gradient Centralization (Yong et al., ECCV 2020): centralize each conv/linear
              # weight gradient to zero-mean over its fan-in dims (per output unit). Constrains
              # the weight space + standardizes the gradient -> accelerates convergence and
              # regularizes the landscape, at ~zero compute. 1-D params (BN gamma/beta, bias)
              # are left untouched. Tests the untried optimizer/gradient-dynamics axis (EXP-030).
              for p in model.parameters():
                  g = p.grad
                  if g is not None and g.ndim > 1:
                      g.add_(-g.mean(dim=tuple(range(1, g.ndim)), keepdim=True))
              optimizer.step()
  ```

  Why this tests the hypothesis: GC modifies the gradient/update rule — the genuinely-unexplored optimizer/gradient-dynamics class (only WD ever swept). Its mechanism (constrain weight space + standardize gradient) simultaneously targets this recipe's two binding constraints (convergence-bound in the fixed budget AND the generalization gap at fixed capacity) at ~zero compute cost, so it sidesteps the epoch wall and is NOT a convergence-COSTING regularizer (the class that failed).

  Risks/edge cases: (a) `model.parameters()` is the eager handle; gradients flow into these tensors (the compiled model wraps the same params), so `p.grad` is correct — verified by the smoke test running a real backward. (b) GC must use `model.parameters()` not `compiled_model.parameters()` (same params, but use the canonical `model` handle already in scope). (c) The per-step reductions are tiny GPU ops with no host sync → throughput-neutral (verify via epoch count). (d) Applies to conv (4-D) and fc (2-D) weights; excludes BN γ/β & biases (1-D) per the paper. (e) Magnitude may be within the ~0.2pp noise floor on this shallow net → no-improvement (graceful failure).

## Configuration Changes
- (none — pure optimizer/gradient-step change. PEAK_LR 0.2, WARMUP_FRAC 0.05, batch 128, WD 1e-4, LS 0.1, Cutout 16, TA, Nesterov, momentum 0.9, cosine-to-0, params 4,299,866, seed 42 all unchanged.)

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background, with a `Monitor` watch on run.log for the summary + NaN/error.
- Resources: single NVIDIA H20 (GPU 0). VRAM ~0.5 GB (unchanged — GC is in-place on existing grads).
- Estimated runtime: ~380-405s total (compute-neutral). Must stay < 600s.
- Log output: `run.log` in project root via redirection; source of truth.
- Tool skill: none (local run).

## Abort Criteria
- **NaN/inf loss** at any point → kill, treat as failed.
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
3. **Cond 3 — no constraint violations.** PASS iff: `git diff --name-only` lists only `train.py`; `num_params == 4,299,866`; eval-count == num_epochs (`grep -c "eval ep" run.log` == num_epochs); no new deps (pure torch tensor ops); seed 42 unchanged.

**MANDATORY attribution note (epoch-wall + FLOPs-neutral-≠-wall-clock-neutral, EXP-015/024):** record `num_epochs` and mean `dt`. GC adds only tiny per-step reductions, so epochs SHOULD be ~91 / dt ~8ms.
- epochs ~91 & dt ~8ms → throughput-neutral → the accuracy delta is a FAIR test of GC.
- epochs < ~85 (dt risen) → the GC loop is unexpectedly costly (e.g. a hidden host sync) → COMPUTE-CONFOUNDED → note explicitly; a regression cannot be cleanly attributed to GC's merit.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -a "^peak_vram_mb:" run.log` — expect ~0.5 GB.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 / ~35,500 (throughput-neutral check).
- final_test_loss: `grep -a "^final_test_loss:" run.log` — compare to baseline 0.195.
