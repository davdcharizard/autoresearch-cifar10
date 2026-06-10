# Plan EXP-009: Compiled k=5 WideResNet (capacity, threading the k4–k6 gap)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md

## Milestones

### Milestone 1: Code changes implemented and parse-clean
- [ ] Edit `train.py` only: set `WIDTH_MULT = 5`; add `compiled_model = torch.compile(model, mode="reduce-overhead")`
      after the model is on device + `num_params` printed; route the training forward through `compiled_model`;
      keep eval on the eager `model` (`evaluator.evaluate(model, device)` UNCHANGED).
- [ ] `python -c "import ast; ast.parse(open('train.py').read())"` passes; `uv run ruff check train.py` passes.
- [ ] Sanity: param count prints **6,712,314** (= k=5; +56.1% vs k=4's 4,299,866); `git diff --stat` shows only
      `train.py` changed; eval line still uses eager `model`; `grep manual_seed train.py` → 42 (no seed change).

### Milestone 2: Run launched and confirmed training
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background.
- [ ] `run.log` shows `Device: cuda`, `num_params 6,712,314`, clean compile (no graph breaks/recompile spam),
      steady-state dt printed, loss decreasing, no NaN. Record steady-state dt + img/s + projected epochs.

### Milestone 3: Run completes within budget and summary emitted
- [ ] `run.log` contains the `best_test_acc:` summary; `total_seconds` < 600.
- [ ] Record `num_epochs` — the KEY confound signal: ≥~55 epochs ⇒ fair test; collapse toward ~35 (EXP-004 eager
      k=6 territory) ⇒ epoch-starved, result confounded (note for analysis, not an abort).

### Milestone 4: Verification verdict
- [ ] Apply the Verification Protocol below. PASS all three necessary conditions ⇒ improvement; Cond-2 fail
      (best_test_acc < 96.10) ⇒ no-improvement; constraint breach ⇒ invalid; crash/empty summary ⇒ crash.

## Code Changes
All changes confined to `train.py` (only editable file; `prepare.py` hook-protected). The full EXP-003 recipe stays
FIXED (bf16, channels_last, time-fraction cosine peak 0.2 / 5% warmup, Nesterov, label smoothing 0.1, batch 128,
WD 1e-4, Cutout(16) GPU-vectorized, seed 42). Two coupled changes — width (the accuracy intervention) + compile
(the validated throughput enabler that keeps k=5's epoch count viable):

- **train.py — hyperparameter `WIDTH_MULT`**: `4 → 5`. Stages become {80,160,320}; params 4,299,866 → **6,712,314**
  (+56.1%). *Why*: capacity is the project's dominant lever (+2.84pp at k=1→k=4, EXP-001); k=5 is the untested
  width between the k=4 sweet spot and the compute-bound k=6 cliff (EXP-004). No other architecture change — the
  `ResNet(width_mult=...)` plumbing already supports arbitrary k.

- **train.py — compile + eval split** (identical pattern to EXP-007/008, the validated enabler): after the model is
  on device and `num_params` printed, add `compiled_model = torch.compile(model, mode="reduce-overhead")`; change
  the training forward from `outputs = model(inputs)` to `outputs = compiled_model(inputs)`; keep eval on the eager
  `model` (`evaluator.evaluate(model, device)` UNCHANGED, avoids recompiles on the 256-batch eval shape). *Why*:
  k=5 eager is ~1.56× the FLOPs of k=4 (~10ms→~15ms eager); compile recovers ~30% (EXP-007) → projected ~11–12ms
  → ~55–65 epochs, a fair (not epoch-starved) test of the added capacity rather than the EXP-004 starvation trap.

**Attribution note**: EXP-007 established compiled-k4 alone = 95.92 ≈ baseline (null standalone accuracy effect),
so any gain over ~96.0 here is attributable to the k=5 width, not the compile.

**Risks/edge cases**: (a) compile cost ~20s is charged to the 300s budget (already true in EXP-007/008, accounted
in epoch projections); (b) if k=5 throughput comes in worse than projected (toward ~18ms → ~45 epochs), the run is
mildly epoch-starved — diagnosable via num_epochs, not a crash; (c) VRAM rises modestly (k=4 peaked ~456–490 MB of
98 GB; k=5 still trivial) — soft constraint, fine.

## Configuration Changes
- WIDTH_MULT: `4 → 5` (stages {80,160,320}, 6,712,314 params)
- Execution: training forward via `torch.compile(model, mode="reduce-overhead")`; eval on eager `model`
- ALL else UNCHANGED: NUM_BLOCKS 3, Cutout(16), PEAK_LR 0.2, WARMUP_FRAC 0.05, WD 1e-4, label smoothing 0.1,
  batch 128, MOMENTUM 0.9 + Nesterov, bf16 autocast, channels_last, cosine schedule, MAX_STEPS 10_000_000, seed 42,
  eval harness frozen.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background via Bash run_in_background).
- Resources: 1× NVIDIA H20 (GPU 0; GPU 1 free). VRAM well under 1 GB of 98 GB.
- Estimated runtime: ~300s training (incl. ~20s one-time compile charged to budget) + ~60–90s startup/eval ≈
  6–8 min total. Expect num_epochs ~55–65.
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^final_test_loss:|^num_params:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` or diverging after warmup → kill, treat as crash.
- Python traceback in `run.log` (compile failure, OOM, shape error, empty `best_test_acc:`) → crash; `tail -n 50 run.log`.
- Recompile spam / repeated multi-second stalls mid-training that sharply cut throughput → record; if it pushes
  wall-clock toward the 10-min limit, treat as failure.
- No new output in `run.log` for > 3 min while training (allow for the one-time ~20s compile) → kill (hang).
- Total wall-clock > 10 min → kill, failure.
- num_epochs collapsing well below ~50 (toward EXP-004's 35) → NOT an abort, but record: k=5 more compute-bound
  than projected, benefit likely epoch-masked (key note for analysis).

## Verification Protocol

### Verification Procedure
Run from project root after completion. Baseline = **96.00** (`exp-index.sh baseline`); success bar = **96.10**.

1. **Clean completion within budget** (necessary cond 2): `grep -aE "^best_test_acc:|^total_seconds:" run.log`;
   `tail -n 50 run.log` for tracebacks. PASS if `best_test_acc:` present, `total_seconds < 600`, no traceback.
   Timeout 10 min. FAIL → crash.
2. **Metric improvement** (necessary cond 1): parse `best_test_acc` from the summary. PASS if
   `best_test_acc >= 96.10` (= baseline 96.00 + 0.1). Else → no-improvement. Stop here on fail.
3. **No constraint violations** (necessary cond 3): `git diff --name-only autoresearch/dev` = only `train.py`;
   no `pyproject.toml`/`uv.lock` diff (compile = core torch, no new dep); eval-line count (`grep -c "eval ep" run.log`)
   == num_epochs (eval once/epoch); seed unchanged (`grep manual_seed train.py` → 42). PASS if all hold, else → invalid.
   All necessary conditions must PASS; stop at first failure.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — **KEY confound signal** (fair test ⇒
  ~55–65; starvation ⇒ toward ~35). Interpreting any no-improvement near 96.0 requires this.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — vs EXP-003's 0.204 / compiled-k4's 0.208 (does the
  added capacity reduce loss?).
- num_params: `grep -aE "^num_params:" run.log` — expect 6,712,314 (confirms k=5).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` (soft-constraint awareness).
