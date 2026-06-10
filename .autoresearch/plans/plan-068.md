# Plan EXP-068: Lookahead optimizer wrapper (k=5, α=0.5) around Nesterov SGD

- **Created**: 2026-06-10
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-068.md

## Closed-axis check
The optimizer axis is closed for FAMILY (AdamW EXP-043), GRAD-DYNAMICS (GC EXP-030/031, grad-clip EXP-064), and OBJECTIVE (SAM EXP-036, PolyLoss EXP-041) — but NOT for the META-WRAPPER class. Lookahead is mechanically distinct: a fast/slow weight split where the slow weights periodically pull the fast weights back AND re-enter training (fast←slow reset every k). This differs from the closed EMA/SWA family (EXP-006/019/020), which average only the EVAL-time weights and never feed back into the optimization trajectory. It does NOT contradict any High-importance insight (no added compute/layers/2nd-graph, no augmentation change). The EMA/SWA closure is a real headwind (weight-space averaging has been null here) and is acknowledged. cudagraph-safe: all Lookahead ops are eager `torch.no_grad()` in-place param copies BETWEEN graph replays (same pattern as the clean EXP-064 grad-clip), preserving param memory addresses like `optimizer.step()` does.

## Milestones

### Milestone 1: Code change + smoke
- [ ] train.py constants (near top, after LABEL_SMOOTHING/CUTOUT_SIZE): `LOOKAHEAD_K = 5`, `LOOKAHEAD_ALPHA = 0.5`.
- [ ] train.py after the optimizer is constructed (~L206): `slow_weights = [p.detach().clone() for p in model.parameters()]` (GPU, float32, same order as parameters()).
- [ ] train.py pre-loop counters (~L219): add `la_step = 0`.
- [ ] train.py after `optimizer.step()` (~L246): increment `la_step`; every `LOOKAHEAD_K` steps, under `torch.no_grad()`, for each `(p, s)` in `zip(model.parameters(), slow_weights)`: `s.add_(p.data - s, alpha=LOOKAHEAD_ALPHA)` then `p.data.copy_(s)`.
- [ ] train.py around the epoch-end eval call (`test_loss, test_acc = evaluator.evaluate(model, device)`): evaluate the SLOW weights faithfully — snapshot fast weights, load slow into the model params, eval, then restore fast so training continues from the fast trajectory. (See Code Changes for the exact block.)
- [ ] Smoke: `python -c "import ast; ast.parse(open('train.py').read())"` OK; `git diff --name-only` == train.py only; visually confirm slow-weight init is AFTER model construction and the every-K update is AFTER optimizer.step(); confirm eval save/restore brackets the evaluate() call.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (the per-step Lookahead op every 5th step is a cheap eager copy → negligible; must NOT show a cudagraph break to 14-16ms), no NaN, eval test_acc climbing normally (slow-weight eval). The every-5th-step copy adds a tiny amount to those steps' dt — acceptable as long as the steady-state stays ~8ms and epochs ≈ 91.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract metrics, compare to baseline 96.45 / bar 96.55. Expect ~88-91 ep (the periodic copy is near-free; eval-time save/restore is outside the Σdt timer).

## Code Changes
- **train.py constants**: add `LOOKAHEAD_K = 5` and `LOOKAHEAD_ALPHA = 0.5` near the other hyperparameters. Why: standard Lookahead defaults (Zhang et al. 2019: k=5–10, α=0.5).
- **train.py optimizer area (~L206)**: `slow_weights = [p.detach().clone() for p in model.parameters()]`. Why: the slow-weight set is the Lookahead iterate; initialized to the model's initial weights, same iteration order as the update loop (deterministic). Only PARAMETERS (not BN running-stat buffers) are tracked — BN buffers follow the fast weights via their own momentum, which is the standard Lookahead behavior.
- **train.py training loop after `optimizer.step()` (~L246)**:
  ```python
  la_step += 1
  if la_step % LOOKAHEAD_K == 0:
      with torch.no_grad():
          for p, s in zip(model.parameters(), slow_weights):
              s.add_(p.data - s, alpha=LOOKAHEAD_ALPHA)
              p.data.copy_(s)
  ```
  Why: every k=5 fast steps, pull slow toward fast by α=0.5 and reset fast←slow — the Lookahead update. In-place copies preserve param addresses (cudagraph-safe). Edge case: the gate is on a dedicated `la_step` counter (incremented only here) so the K-cadence is exact regardless of the existing `step` counter's timing.
- **train.py eval call (epoch end)**: replace the bare `test_loss, test_acc = evaluator.evaluate(model, device)` with:
  ```python
  fast_backup = [p.detach().clone() for p in model.parameters()]
  with torch.no_grad():
      for p, s in zip(model.parameters(), slow_weights):
          p.data.copy_(s)
  test_loss, test_acc = evaluator.evaluate(model, device)
  with torch.no_grad():
      for p, b in zip(model.parameters(), fast_backup):
          p.data.copy_(b)
  ```
  Why: faithfully evaluate the SLOW (Lookahead) iterate — the paper's reported gains are on slow weights — then restore the fast weights so training continues unperturbed. These copies are eager and OUTSIDE the Σdt timer (eval is untimed), so zero compute-budget cost; wall cost is ~negligible (two 4.3M-param copies/epoch). Risk: must restore fast weights correctly or training corrupts — the snapshot/restore is symmetric and bracketed tightly around the single evaluate() call.

## Configuration Changes
- `LOOKAHEAD_K = 5`, `LOOKAHEAD_ALPHA = 0.5` (standard defaults). All else byte-identical to EXP-054 (k=4 WideResNet-20, AugMix-p0.5, GPU Cutout16, cosine peak0.2/warmup0.05/Nesterov m0.9/WD1e-4/LS0.1, batch128, seed42, compile reduce-overhead). num_params unchanged (4,299,866) — slow_weights are buffers held in a Python list, NOT registered model parameters, so the reported param count is unchanged.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check nvidia-smi; relaunch on contention — Σdt budget REQUIRES an uncontended GPU).
- Estimated runtime: ~88-91 epochs, dt ~8ms, Σdt ~300s, wall ~593s (< 600s; the periodic copy is near-free w.r.t. Σdt; eval save/restore is untimed). Monitor the 600s wall (recipe is wall-tight, 3 prior breaches) — but this change adds ~no wall.
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf or eval test_acc not climbing by ep5 (would suggest broken fast/slow bookkeeping — e.g., fast weights not restored after eval → training corruption).
- dt elevated to ≥13ms sustained (cudagraph break from the in-place param ops — EXP-042 signature): kill, record.
- Runtime error in the Lookahead block or eval save/restore (shape/zip mismatch): capture traceback, treat as code error (1 fix-retry per execute skill).
- dt drifts ≫ 8ms (contention): kill, relaunch on clean idle GPU.
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:|^training_seconds:" run.log`; confirm summary printed, `training_seconds ≈ 300`, `total_seconds < 600`, `num_params == 4,299,866`, `grep -ciaE "nan|traceback|error" run.log` == 0. (If total_seconds breaches 600 by a small margin while training_seconds≈300 and metric trustworthy → no-improvement per EXP-061/065 precedent, NOT invalid.)
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only; prepare.py/eval untouched; evaluate() called once/epoch (the save/restore brackets a SINGLE evaluate() call — still one eval/epoch); no new deps; seed 42 unchanged; ran uncontended (steady ~8ms dt).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics / cudagraph-break abort.
- Timeout: 12 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- `num_epochs`/`num_steps`, `final_test_loss`, `peak_vram_mb`: `grep -aE "^num_epochs:|^num_steps:|^final_test_loss:|^peak_vram_mb:" run.log` — confirm ~88-91 ep (throughput ≈ unchanged; peak_vram rises modestly from the slow_weights + fast_backup copies, ~2× param memory ≈ +35MB, well within budget). Compare final_test_loss to EXP-054's 0.1968 (Lookahead's slow iterate is typically smoother → watch loss as the sensitive secondary signal).
