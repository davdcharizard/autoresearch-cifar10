# Plan EXP-007: torch.compile (reduce-overhead) to buy more epochs
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md

## Milestones

### Milestone 1: Code changes implemented and parse-clean
- [x] Edit `train.py` only: add `compiled_model = torch.compile(model, mode="reduce-overhead")` after the model
      is on device; use `compiled_model(inputs)` for the training forward; keep eval on the eager `model`
- [x] `python -c "import ast; ast.parse(open('train.py').read())"` passes
- [x] `uv run ruff check train.py` passes (only train.py changed: +7/-1; eval line still eager `model`)
- [x] (compile smoke-tested already in planning: reduce-overhead = 8.1ms vs eager 9.4ms, ~11.6s compile cost, no errors)

### Milestone 2: Run launched and confirmed training
- [x] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] `run.log` shows `Device: cuda`, `num_params 4,299,866` (UNCHANGED), steady-state `dt`=8ms (down from
      ~10–11ms), ~15,200 img/s (up from ~11,600), loss decreasing 1.43→1.33, no NaN, no recompile/errors

### Milestone 3: Run completes within budget and throughput improved
- [x] `run.log` contains `best_test_acc:` summary (95.92%); `total_seconds` < 600 (402.9)
- [x] `num_epochs`=89 (vs EXP-003's 77, +12 / +28% steps) — compile bought epochs net of its cost. Mechanism ✓

### Milestone 4: Verification verdict
- [ ] FAIL: `best_test_acc` 95.92 < 96.10 (and < 96.00 baseline) → no-improvement. Extra epochs did NOT raise acc.
- [~] skipped (aborted after metric failure)

## Code Changes
All changes confined to `train.py` (only editable file; `prepare.py` hook-protected). The k=4 WideResNet, Cutout,
and the FULL optimization recipe (bf16, channels_last, time-fraction cosine, Nesterov, label smoothing, batch
128, WD 1e-4, PEAK_LR 0.2, seed 42) are held **byte-for-byte FIXED** — the only change is that the training
forward runs through a compiled copy of the model. Identical math, computed with fewer kernel launches → faster
steps → more epochs in 300s. Cleanest possible "more epochs of the proven recipe" test (EXP-003 mechanism).

- **train.py — compile the model** (right after `model = ResNet(...).to(device, channels_last)` and the
  `num_params` print):
  ```python
  compiled_model = torch.compile(model, mode="reduce-overhead")
  ```
  *Why `reduce-overhead`*: smoke-tested in planning — default mode gave only 1.03× (net loss after the ~13.6s
  compile cost), while `reduce-overhead` (CUDA graphs) gave **8.1ms vs 9.4ms eager (1.16×)** with ~11.6s compile
  cost, which repays itself over 300s. This is the launch-bound-appropriate mode (CUDA graphs amortize launch
  overhead). No new dependency — `torch.compile` is core torch (2.9.1+cu128).

- **train.py — train through the compiled model**: in the training loop, change the forward
  `outputs = model(inputs)` → `outputs = compiled_model(inputs)` (still inside the existing
  `torch.autocast(... bfloat16)` block). The optimizer is built on `model.parameters()`; `compiled_model` shares
  those exact parameters (it wraps `model` via `._orig_mod`), so `optimizer.step()` updates them and the compiled
  forward sees the updates. *Why*: routes only the hot training forward/backward through the fused/graphed path.

- **train.py — eval stays on the eager model (UNCHANGED)**: the per-epoch eval call remains
  `evaluator.evaluate(model, device)` (the original eager handle). *Why*: eval batch size (256) and the variable
  last test batch differ from the training shape; evaluating the eager `model` (same shared weights) avoids
  triggering eval-time recompiles while reporting the identical trained weights. The compile cost is paid once on
  the training shape only.

## Configuration Changes
- (no hyperparameter changes) — `torch.compile(mode="reduce-overhead")` wraps the training forward; all recipe
  knobs UNCHANGED: WIDTH_MULT 4, Cutout(16), PEAK_LR 0.2, WD 1e-4, label smoothing 0.1, batch 128, bf16,
  channels_last, Nesterov, cosine schedule, MAX_STEPS 10_000_000, seed 42, eval frozen.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background)
- Resources: 1× NVIDIA H20 (GPU 0; GPU 1 free). VRAM ≈ EXP-003 (~500 MB; CUDA graphs add a modest static-buffer
  pool, still far below the 98 GB ceiling).
- Estimated runtime: ~300s training (incl. ~12s one-time compile charged to the budget) + ~60–80s startup/eval ≈
  6–8 min. Expect num_epochs to rise to ~83–90 (vs EXP-003's 77) from the ~16% step speedup net of compile cost.
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:|^final_test_loss:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` or diverging (rising) for sustained steps after warmup → kill, treat as crash.
- Python traceback in `run.log` (compile failure, graph-break error, empty `best_test_acc:` at end) → crash;
  inspect `tail -n 50 run.log`. (Compile already smoke-tested clean in planning, so a failure here is unexpected.)
- Recompile spam (repeated multi-second stalls / "recompiling" messages mid-training) sharply cutting throughput
  → record; if it pushes wall-clock toward the limit, treat as failure.
- No new output in `run.log` for > 3 min while training should be active (allow for the one-time compile) → kill (hang).
- Total wall-clock > 10 min → kill, treat as failure.

## Verification Protocol

### Verification Procedure
Run from project root after completion. Baseline = **96.00** (`exp-index.sh baseline`); success bar = **96.10**.

1. **Clean completion within budget** (necessary condition: runs cleanly in budget):
   - `grep -aE "^best_test_acc:|^total_seconds:" run.log`; `tail -n 50 run.log` for tracebacks.
   - PASS if `best_test_acc:` present (non-empty), `total_seconds < 600`, no traceback. Timeout: 10 min.
2. **Metric improvement** (necessary condition: `best_test_acc` ≥ baseline + 0.1):
   - Parse `best_test_acc`. PASS if `best_test_acc >= 96.10`. FAIL (→ no-improvement) otherwise.
3. **No constraint violations** (necessary condition: no constraint violations):
   - `git diff --name-only autoresearch/dev` shows only `train.py`; no diff on `pyproject.toml`/`uv.lock`
     (torch.compile is core torch — no new deps); eval-line count == num_epochs (eval once/epoch); seed unchanged
     (`grep manual_seed train.py` → 42).
   - PASS if all hold; else → invalid.
   All necessary conditions must PASS; stop at first failure.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — **KEY signal**: expect an increase vs
  EXP-003's 77 (target ~83–90), confirming compile bought epochs net of its one-time cost. If epochs did NOT
  rise, the speedup was eaten by compile/recompile overhead (records the mechanism for analysis).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare to EXP-003's 0.204 (more epochs of a
  regularized model should not over-fit; watch for train/test divergence).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` — expect ~500–700 MB (CUDA-graph static pool).
- num_params: `grep -aE "^num_params:" run.log` — must be **4,299,866 (unchanged)** — compile changes execution,
  not the model.
