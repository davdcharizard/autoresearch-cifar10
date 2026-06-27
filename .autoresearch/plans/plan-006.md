# Plan EXP-006: torch.compile with pre-loop warmup
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md

## Milestones

### Milestone 1: Experiment branch + compile/warmup implemented
- [x] Create experiment branch `autoresearch/exp-006` from `autoresearch/dev`
- [x] In train.py: keep eager reference `base_model = model`, wrap `model = torch.compile(model)` after model creation
- [x] Insert compile-warmup block after optimizer creation, BEFORE `t_start_training`: 3 forward+backward passes on a synthetic batch with exact training config (model.train(), channels_last, bf16 autocast, label-smoothed CE), then `optimizer.zero_grad(set_to_none=True)`; `torch.cuda.synchronize()`; NO `optimizer.step()` anywhere in warmup
- [x] Change the eval call to `evaluator.evaluate(base_model, device)` so the frozen Eval path stays eager (no second compilation; weights/buffers shared with the compiled wrapper)
- [x] Static check passes: `uv run ruff check train.py`
- [x] Scope check: `git status --porcelain` shows only `train.py` modified

### Milestone 2: Experiment running on GPU 0
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [x] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Early signal: params 4,286,026 (unchanged); first step lines show dt ≤ ~25ms (compile working) and epoch-1 eval completes with acc ≥ 15%

### Milestone 3: Run completed and metrics extracted
- [x] Process exited; `grep "^best_test_acc:" run.log` non-empty
- [x] `total_seconds:` ≤ 600
- [x] Full summary block recorded into logs/exp-log-006.md, including `startup_seconds` (verifies compile cost landed in startup, not the training budget)

## Code Changes

- **train.py** (only file modified — hard constraint). Three edits:
  1. After `model = ResNet(...)`: `base_model = model` then `model = torch.compile(model)` (default/inductor mode — no mode argument).
  2. Warmup block placed after the `print(f"Batches per epoch: ...")` line and before `t_start_training = time.time()`:
     ```python
     # Compile warmup: one-time inductor compilation must land in startup, not
     # in the per-step timed budget. No optimizer.step() -> weights unchanged.
     warm_x = torch.randn(BATCH_SIZE, 3, 32, 32, device=device).to(
         memory_format=torch.channels_last
     )
     warm_y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)
     model.train()
     for _ in range(3):
         with torch.autocast("cuda", dtype=torch.bfloat16):
             warm_loss = F.cross_entropy(
                 model(warm_x), warm_y, label_smoothing=LABEL_SMOOTHING
             )
         warm_loss.backward()
     optimizer.zero_grad(set_to_none=True)
     torch.cuda.synchronize()
     del warm_x, warm_y, warm_loss
     ```
  3. Eval call: `evaluator.evaluate(model, device)` → `evaluator.evaluate(base_model, device)`.

  Why this tests the hypothesis: the only functional change is compiled execution of the training step; data, recipe, eval, and schedule are untouched. num_epochs and dt in the log directly measure the throughput delta.

  Risks/edge cases: (a) BN running stats see 3 random batches during warmup — decays by momentum 0.1 over ~11k subsequent steps, negligible by epoch 1 and irrelevant to train-mode forward passes; (b) warmup consumes CUDA RNG draws, shifting the random stream vs baseline — seed stays 42, no seed selection involved; (c) inductor graph break or compile error → crash at startup, caught by the 120s no-output abort; (d) recompile churn mid-run would show as periodic multi-second dt spikes in step lines — diagnose, do not silently accept.

## Configuration Changes
- None — zero hyperparameter changes (pure systems experiment; single-variable: compiled vs eager training step)

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM ~1.7GB (compile adds workspace)
- Estimated runtime: ~8–9.5 min total: 300s training + 60–120s compile (startup) + ~130–160 evals (~2 min) — under the 600s cap with margin ~30–90s; monitor
- Log output: stdout/stderr → `run.log`; metrics via grep; delete run.log after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill, treat as failure (hard constraint)
- No output in run.log within 120s of launch (params line prints before compile, so silence past that means hang) → kill and diagnose
- No `eval ep   1` line within 300s of launch (compile hang) → kill and diagnose
- Training loss NaN/inf, or epoch-1 eval accuracy < 15% → kill, diagnose
- Step lines show dt ≥ ~27ms sustained (i.e., no speedup vs eager): do NOT kill — the throughput datapoint is the experiment's information value; let it finish
- Empty `grep "^best_test_acc:" run.log` after exit → crash; read `tail -n 100 run.log` for the inductor traceback

## Verification Protocol

### Verification Procedure
Run from project root after process exit. Baseline via `exp-index.sh baseline` = **96.23** (commit 1174e0d) at planning time → pass threshold **≥ 96.33** (+0.1 pp).

1. **Run completes without crashing within budget (≤ 10 min total)**
   - Command: `grep "^best_test_acc:\|^total_seconds:" run.log`
   - Pass: `best_test_acc:` present AND `total_seconds:` ≤ 600. Timeout: kill if alive >10 min → fail.
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.33)**
   - Command: `grep "^best_test_acc:" run.log` → parse; compare against fresh `exp-index.sh baseline` at verification time.
   - Pass: value ≥ 96.33. Evaluation stops at first failure.
3. **Validation at most once per epoch**
   - Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log`
   - Pass: eval-line count ≤ num_epochs.

Budget-integrity check (analysis-stage, not a pass/fail condition): `grep "^startup_seconds:" run.log` should show ~60–120s (compile absorbed in startup) and `training_seconds:` ≈ 300 — confirms the timed budget was not consumed by compilation.

### Informational Metrics (Optional)
- num_epochs: `grep "^num_epochs:" run.log` — THE throughput datapoint (eager baseline: 114; hypothesis: ≥ 130); decides viability of the future aligned-width push regardless of verdict
- startup_seconds: `grep "^startup_seconds:" run.log` (expect 60–120 — compile cost location proof)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~1.6–2GB)
- num_params: `grep "^num_params:" run.log` (expect 4,286,026 — unchanged)
