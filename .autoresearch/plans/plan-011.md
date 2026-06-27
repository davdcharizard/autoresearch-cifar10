# Plan EXP-011: EMA weight averaging for evaluation (decay 0.995)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md

## Milestones

### Milestone 1: EMA implemented in train.py, passing static checks
- [x] Branch `autoresearch/exp-011` created from `autoresearch/dev`
- [x] `import copy` added; `EMA_DECAY = 0.995` constant added to the hyperparameter block
- [x] EMA model created after optimizer setup: `copy.deepcopy(base_model)` (eager, channels_last preserved), `.eval()`, all params `requires_grad_(False)`; param/buffer lists cached for both models
- [x] Per-step update inserted after `optimizer.step()` and BEFORE `torch.cuda.synchronize()` (inside the timed region — honest dt accounting): `torch._foreach_lerp_(ema_params, model_params, 1.0 - EMA_DECAY)` + buffer copy
- [x] Eval switched: `evaluator.evaluate(ema_model, device)` replaces `evaluator.evaluate(base_model, device)` (still exactly once per epoch, still through the frozen `Eval`)
- [x] `uv run python -m py_compile train.py` exits 0

### Milestone 2: Run launched and early signals healthy
- [x] GPU 0 confirmed free via `nvidia-smi` before launch (freed 06:29:10 after foreign PID 1987359 exited; watcher b98xpp0uy)
- [x] `uv run train.py > run.log 2>&1` started in background (task b8lilmxl0); run.log being written
- [x] dt gate at step ~100: dt = 22ms at steps 700–750 (identical to baseline — EMA tax invisible at print granularity) → projects ~139 epochs
- [x] EMA eval sanity at epoch 5: ep 1 = 10.24% (init-dominated, expected), ep 5 = 47.38% — well above the 10.5% bug gate

### Milestone 3: Run completed within caps, metrics extracted
- [x] Run exits cleanly; total_seconds ≤ 600 (Run 3: 479.9s; Runs 1–2 were infra-contaminated by GPU/host contention and retried per protocol)
- [x] `grep "^best_test_acc:\|^peak_vram_mb:" run.log` returns both lines (96.46 / 1631.0)
- [x] num_epochs ≥ ~130 (Run 3: 139 — throughput byte-identical to baseline); 139 eval lines = 139 epochs

### Milestone 4: Verification rendered
- [x] Conditions evaluated per protocol: 1 PASS (479.9s clean), 2 FAIL (96.46 < 96.81), 3 skipped per first-failure stop; recorded in exp-log-011.md

## Code Changes
- **train.py** (only file; all other paths hook-protected):
  1. Top of file: `import copy` (stdlib, no new dependency).
  2. Hyperparameter block: add `EMA_DECAY = 0.995  # ~1.4-epoch half-life at 97 steps/epoch; EMA of weights for eval`.
  3. After the optimizer is constructed (and after `base_model`/`torch.compile` split — order vs the compile warmup is irrelevant since warmup never calls `optimizer.step()`):
     ```python
     ema_model = copy.deepcopy(base_model).eval()
     for p in ema_model.parameters():
         p.requires_grad_(False)
     ema_params = list(ema_model.parameters())
     ema_buffers = list(ema_model.buffers())
     live_params = list(base_model.parameters())
     live_buffers = list(base_model.buffers())
     ```
     `base_model` and the compiled `model` share storage, so `live_params` tracks training exactly.
  4. In the step loop, immediately after `optimizer.step()`, before `torch.cuda.synchronize()`:
     ```python
     with torch.no_grad():
         torch._foreach_lerp_(ema_params, live_params, 1.0 - EMA_DECAY)
         for b_ema, b in zip(ema_buffers, live_buffers):
             b_ema.copy_(b)
     ```
     `_foreach_lerp_(self, other, w)` computes `self += w*(other-self)` = `decay*ema + (1-decay)*live` — the standard EMA. Buffers (BN running_mean/var float32 + num_batches_tracked int64) are direct-copied, not averaged — standard practice and dtype-safe (`lerp` would fail on int64; plain `.copy_()` per buffer avoids any foreach dtype edge case; ~40 small tensors, negligible cost).
  5. Eval call: `test_loss, test_acc = evaluator.evaluate(ema_model, device)`.

  Why this tests the hypothesis: the ONLY behavioral change at eval time is which weights are scored — training dynamics, schedule, augmentation, and compile are byte-identical. Any metric delta is attributable to weight averaging (plus the small dt tax, observable in num_epochs).

  Risks/edge cases: (a) deepcopy must clone the EAGER `base_model`, never the compiled wrapper; (b) EMA update must touch `base_model`'s param list, not `model.parameters()` (same tensors, but the eager list is unambiguous); (c) `ema_model` stays in `.eval()` permanently — it is never forwarded during training so per-epoch `model.train()` doesn't affect it; (d) VRAM +~17MB (4.29M fp32 params) — trivial vs soft constraint.

## Configuration Changes
- EMA_DECAY: (new) -> 0.995 (97 steps/epoch ⇒ half-life ≈ 138 steps ≈ 1.4 epochs: long enough to average the ±0.1pp terminal eval noise over ~1.5 epochs of snapshots, short enough that the EMA tracks the cosine anneal and converges onto the final weights as LR→0; standard CIFAR-scale value)
- No other hyperparameter, schedule, augmentation, or architecture changes — single-variable experiment against baseline 1990397.

## Execution Environment
- Method: local — `uv run train.py > run.log 2>&1` from the project root, on GPU 0 only (confirm free via `nvidia-smi` first; if busy, wait — never use GPU 1)
- Resources: 1x H20 (GPU 0); VRAM ≈ baseline + 17MB
- Estimated runtime: ~480s total wall clock (300s timed training + ~140 evals + startup ≈ baseline's 480.8s; hard caps: 600s)
- Log output: all stdout/stderr to `run.log` in project root (no tee/stream per goal procedure); deleted after the experiment concludes
- Tool skill: none (local run; background Bash + Monitor watcher per established practice)

## Abort Criteria
- Loss NaN/inf at any step (mechanically impossible from an eval-side change — would indicate an implementation error touching live weights)
- EMA eval still ≤ 10.5% at epoch 5 or later (random-level past the init-dominated window ⇒ buffer/param tracking bug; kill, fix, relaunch — counts as the one code-error retry). Do NOT kill on a near-random epoch-1/2 eval — that is expected while the EMA is still init-dominated.
- dt-gate: at step ~100, projected epochs < 65 (dt > ~45ms) — not expected (budgeted dt ≤ 25ms); kill per protocol if hit
- No new output in run.log for > 3 minutes
- Total wall clock approaching 600s with the run still going (kill at 590s; the 10-minute cap is a hard failure boundary)

## Verification Protocol

### Verification Procedure
Run from the project root after the run exits. Baseline from `exp-index.sh baseline` = **96.71** (commit 1990397), direction higher; bar = 96.71 + 0.10 = **96.81**.

1. **Run completed within budget (≤ 600s total)** — `grep "^total_seconds:" run.log`; PASS if value ≤ 600 and the summary block exists (its presence proves a clean exit). Empty grep ⇒ crash ⇒ read `tail -n 50 run.log`, verdict path = crash. Timeout: n/a (post-hoc parse, <5s).
2. **best_test_acc ≥ 96.81** — `grep "^best_test_acc:" run.log`; PASS if the parsed value ≥ 96.81. Timeout: n/a.
3. **Validation at most once per epoch** — compare `grep -c "eval ep" run.log` with `grep "^num_epochs:" run.log`; PASS if eval-line count ≤ num_epochs. Structurally guaranteed (single eval call per epoch loop) but checked mechanically. Timeout: n/a.

First failed condition stops evaluation (goal rule). On all-pass: verdict improvement; insert index row; commit train.py on `autoresearch/exp-011`; merge to `autoresearch/dev` (no remote — skip push/PR per TASK.md, fields stay "(pending)"/"N/A"). On condition-2 failure with clean run: no-improvement; discard changes. Delete run.log either way once analysis has extracted what it needs.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~baseline + ~17MB
- num_epochs: `grep "^num_epochs:" run.log` — expect 133–139 (measures the EMA dt tax)
- num_params: `grep "^num_params:" run.log` — must equal baseline's 4,292,170 (architecture untouched)
