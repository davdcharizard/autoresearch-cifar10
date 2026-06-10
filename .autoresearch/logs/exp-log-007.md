# EXP-007: torch.compile (reduce-overhead) to buy more epochs

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md
- **Plan**: plans/plan-007.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-007
- **Commit**: (none — no-improvement, changes discarded)
- **PR**: (none — no-improvement)
- **Outcome**: completed (clean run; verification cond 2 failed → no-improvement verdict in analyze)

## Implementation Notes

### Summary
Two edits to `train.py` only (Milestone 1). (1) After the model is placed on device (channels_last) and
`num_params` printed, added `compiled_model = torch.compile(model, mode="reduce-overhead")`. (2) Changed the
training-loop forward from `outputs = model(inputs)` to `outputs = compiled_model(inputs)` (still inside the
existing bf16 autocast block). The per-epoch eval call is left UNCHANGED on the eager `model` handle (shared
weights with the compiled wrapper via `._orig_mod`), so eval — which uses a different batch size and a variable
last batch — does not trigger recompiles. The optimizer is built on `model.parameters()`, which `compiled_model`
shares, so `optimizer.step()` updates the same weights the compiled forward reads. Parse-clean, ruff clean, diff
is train.py-only (+7/-1).

### Surprises & Discoveries
- **Default-mode `torch.compile` is net-negative here** (smoke-tested in planning): only 1.03× (9.4→9.1ms) for a
  ~13.6s compile cost → would *reduce* epochs. `reduce-overhead` (CUDA graphs) is the launch-bound-appropriate
  mode: 8.1ms vs 9.4ms eager (1.16×) for ~11.6s compile cost. The plan was set to `reduce-overhead` accordingly.
- Larger-batch smoke test confirmed the launch-bound diagnosis (us/img drops 73.4→65.8→60.9 at batch
  128→256→512) but batch scaling is deferred (LR confound) — this experiment isolates the compile lever only.

### Decisions
- **mode="reduce-overhead"** (not default): chosen from the smoke-test numbers above — it is the only mode with a
  positive net throughput effect within the 300s budget.
- **Compile cost charged honestly to the budget**: compilation happens on the first training step(s) inside the
  timed loop (counted in `total_training_time`); no pre-loop warmup trick. The experiment is the honest test of
  whether the speedup repays the compile cost in 300s.
- **Eval on eager `model`**: avoids eval-time recompiles on the differing eval batch shapes while reporting the
  identical trained weights.

## Experimental Adjustments

(none yet)

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID; local background run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08 (exit 0)

Description:
- Running the EXP-003 recipe (k=4 WideResNet + GPU Cutout, all hyperparameters identical) with the sole change
  that the training forward executes through a `reduce-overhead`-compiled copy of the model. Expect a clean run
  within 300s, `num_params` unchanged at 4,299,866, steady-state dt ~8ms (down from ~10–11ms) after a one-time
  ~12s compile, and num_epochs to rise above EXP-003's 77 (target ~83–90). The hypothesis is that those extra
  epochs of the identical 96.00% recipe lift `best_test_acc` past the 96.10 bar.

Observations:
- Clean startup, `num_params: 4,299,866` (unchanged — compile changes execution, not the model) (run.log L1-2).
- Compile succeeded with no graph breaks/recompiles/errors. Steady-state **dt = 8ms/step, ~15,200 img/s** (up
  from EXP-006's ~10–11ms / ~11,600 img/s) — a ~31% full-loop throughput gain, larger than the 1.16× isolated-
  forward smoke test (the real loop's eager overhead is also reduced) (run.log, ep 3–4 step lines).
- Loss decreasing normally (1.43→1.33 by step 1300), no NaN; eval ep 3 = 61.79% (normal early eager-model acc).
- At 8ms/step the run is tracking toward ~90+ epochs vs EXP-003's 77 — the throughput gain materialized.

Key Metrics:
- best_test_acc: **95.92%** @ best over 89 epochs — BELOW the 96.00 baseline and the 96.10 bar (run.log summary).
- num_epochs: **89** / num_steps: **34,523** — vs EXP-003's 77 / ~27,020. Compile bought **+12 epochs (+28%
  steps)** net of its one-time cost. The throughput lever WORKED.
- dt steady 8ms, ~15,400 img/s throughout (run.log step lines); training_seconds 300.0; total_seconds 402.9.
- final_test_acc 95.81; final_test_loss **0.2081** (≈ EXP-003's 0.204 — no overfitting, but no gain either).
- peak_vram_mb 453.8 (≈ EXP-003); num_params 4,299,866 (unchanged — compile changes execution, not the model).
- Late evals stable 95.81–95.92 (ep 84–89) — converged plateau; extra epochs did not lift the ceiling.

## Verification Results

### Conditions Checked

- **Cond 1 — clean completion within budget**: PASS. best_test_acc present (95.92%), total_seconds 402.9 < 600,
  no traceback, no recompile/graph-break spam (run.log).
- **Cond 2 — metric ≥ 96.10**: **FAIL**. 95.92 < 96.10 (also < 96.00 baseline). → no-improvement.
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 2 per protocol.

### Informational Metrics

- Not formally collected (cond 2 failed). Recorded above: num_epochs 89 (+12 vs EXP-003 — mechanism succeeded),
  final_test_loss 0.2081 (no overfit, no gain), num_params unchanged, peak_vram 453.8 MB.

## Errors & Dead Ends

(none)

## Human Notes

> (none — autopilot)
