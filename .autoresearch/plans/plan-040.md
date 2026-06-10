# Plan EXP-040
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md
**Brainstorm**: brainstorm/brainstorm-040.md

## Summary
Enable cuDNN convolution-algorithm autotuning by setting **`torch.backends.cudnn.benchmark = True`** once
at the start of training (before model construction / `torch.compile`). Input shapes are fixed
(128×3×32×32, `drop_last=True`), so cuDNN benchmarks each conv shape's algorithms once and caches the
fastest. Everything else in the recipe is unchanged. The time-fraction LR schedule
(`total_training_time / TIME_BUDGET_S`) anneals fully regardless of throughput, so any per-step dt
reduction simply fits MORE epochs into the fixed 300s budget. This is the first probe of the only untried
axis after 40 experiments — per-step THROUGHPUT under the dt-gated budget — and a clean test of the
pivotal open question: is the net convergence-bound at ~91 ep (more epochs lift top-1) or epoch-saturated?

## Baseline (from experiment index)
- best_test_acc baseline = **96.22%** (commit 6c417a4, EXP-012); bar = **96.32** (baseline + 0.1).
- Reference run shape: ~91 epochs, dt ~8ms/step (uncontended), final_test_loss ~0.195, params 4,299,866.

## Hypothesis
`cudnn.benchmark` selects faster conv algorithms for the fixed shapes → steady-state dt drops below ~8ms
→ more than ~91 epochs fit in 300s. If the net is convergence-bound at this operating point, the extra
epochs lift best_test_acc above 96.32. Honest most-likely outcome: small dt change (compile may already
be near-conv-optimal) → modest gain if convergence-bound, else within-noise null. INFORMATIVE in all
outcomes: (a) dt↓ & acc↑ ⇒ convergence-bound confirmed + improvement; (b) dt↓ & acc flat ⇒ epoch-saturated
(closes throughput→epochs); (c) dt unchanged ⇒ compile already conv-optimal (closes cuDNN-autotune
sub-lever).

## Milestones

### Milestone 1 — Code change implemented and passing local checks
- [ ] Edit `train.py`: add `torch.backends.cudnn.benchmark = True` inside `main()` immediately after the
      `device = torch.device(...)` / `print(f"Device: {device}")` lines (≈L150), before model construction.
- [ ] AST check: `uv run python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
- [ ] Flag-set check: `uv run python -c "import train, torch; train  # import side-effect-free; flag is set inside main()"`
      and grep that the line is present: `grep -n "cudnn.benchmark" train.py` → exactly one line, `= True`.
- [ ] Diff scope check: `git diff --name-only` lists **only** `train.py`.

### Milestone 2 — Experiment running (uncontended GPU)
- [ ] Confirm a GPU is idle (`nvidia-smi`: util ~0%, mem <700MiB) — shared H20 node intermittently
      saturated by another user's jobs (infra-errors). Both GPUs were idle as of the EXP-039 run.
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Confirm `run.log` shows `Device: cuda`, a `params: 4,299,866` line (UNCHANGED — model untouched),
      first eval line; check early steady-state dt.

### Milestone 3 — Run completed and verified
- [ ] Run exits 0, prints full summary block (`best_test_acc:` … `num_params:`).
- [ ] **Clean/fair run**: GPU uncontended for the whole run (dt distribution dominated by one steady
      value; no sustained 20-40ms contention band). Record num_epochs and the steady dt.
- [ ] Extract metrics, compare best_test_acc to bar 96.32 / baseline 96.22.
- [ ] **Diagnostic** (the point of this experiment): compare num_epochs and steady dt to baseline
      (~91 ep / ~8ms). dt↓ ⇒ cuDNN found faster algos; epochs↑ ⇒ more training.
- [ ] Confirm clean completion (<600s wall, eval_count == num_epochs, only train.py changed, seed 42,
      num_params 4,299,866 unchanged).

## Code Changes

**File: `train.py` (single line added — the ONLY change)**
- **Inside `main()`, right after `print(f"Device: {device}")` (≈L150)**, add:
  ```python
  # cuDNN convolution-algorithm autotuning: input shapes are fixed (128×3×32×32, drop_last=True),
  # so cuDNN benchmarks conv algorithms once and caches the fastest per shape — a standard,
  # crash-safe throughput lever for launch-bound nets (EXP-040). Lowers per-step dt → more epochs
  # fit in the fixed 300s budget (same mechanism as the accepted torch.compile enabler, EXP-007).
  torch.backends.cudnn.benchmark = True
  ```
- **Why it tests the hypothesis**: it reduces per-step GPU wall-clock without touching the model, data,
  optimizer, schedule, or eval — so any change in num_epochs is purely throughput, and any change in
  best_test_acc at more epochs isolates the convergence-bound question. Placed before model construction
  and `torch.compile` so the flag is active when cuDNN algorithms are first selected.
- **Risks/edge cases**: (a) benchmark mode adds a one-time per-shape autotuning cost in the first few
  steps (negligible, a few steps of ~300s budget); shapes are fixed so no re-benchmarking thrash.
  (b) With `torch.compile(reduce-overhead)` already active, inductor may have already chosen conv algos
  → the flag could be a dt-neutral no-op (clean null, no confound). (c) No determinism concern for the
  metric: seed 42 is unchanged; benchmark-mode algorithm selection does not constitute "seed hacking"
  (it does not alter the data order or RNG stream — it selects kernels). (d) No new deps, no eval change,
  params unchanged (4,299,866).

## Configuration Changes
One global PyTorch backend flag: `torch.backends.cudnn.benchmark` False(default)→True. No hyperparameter,
model, data, optimizer, schedule, or eval change. (PEAK_LR 0.2, batch 128, WD 1e-4, LS 0.1, Cutout 16,
TrivialAugmentWide, cosine-to-0 LR, Nesterov m0.9, seed 42, 300s budget, torch.compile reduce-overhead,
widths {64,128,256} — all unchanged.)

## Execution Environment
- **Method**: local — `CUDA_VISIBLE_DEVICES=<idle_gpu> uv run train.py > run.log 2>&1` (background).
  Pick an idle H20 via `nvidia-smi` (util ~0%, mem <700MiB); the shared node intermittently saturates
  with another user's Protenix jobs and the budget is wall-clock-dt-gated (infra-errors). If a clean
  window is unavailable, poll/relaunch until uncontended (idle-GPU gating).
- **Resources**: single NVIDIA H20 (either index — identical hardware); fixed `TIME_BUDGET_S=300`.
- **Estimated runtime**: ~390–420s wall (≈6.5–7 min), same shape as baseline (possibly slightly more
  epochs if dt drops).
- **Log output**: stdout+stderr → `run.log` at project root (sole source of truth). Per-step lines use
  `\r`; extract dt via `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms"`.
- **Monitoring**: background Monitor that fires on the final summary block / traceback / process exit;
  tail run.log for the summary.

## Abort Criteria
- Any Python traceback / non-zero exit, or NaN/inf in `loss:` → kill, mark failed.
- **GPU contention** (sustained dt band well above the steady value / epoch count trending far below
  ~88): infra workaround — kill and relaunch on an idle window (NOT a research failure).
- Total wall-clock of a single committed run approaching 10 min (600s) → kill, treat as failure.

## Verification Protocol

### Verification Procedure
Run after the committed clean run completes. Baseline = 96.22 (from `exp-index.sh baseline`).

1. **Primary metric clears the bar** (NECESSARY):
   - Command: `grep -aE "^best_test_acc:" run.log`
   - Pass iff `best_test_acc >= 96.32`. Else no-improvement.
2. **Clean completion within budget** (NECESSARY):
   - Command: `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^num_params:" run.log`
   - Pass iff summary block present, `total_seconds < 600`, exit 0.
3. **No hard-constraint violations** (NECESSARY):
   - `git diff --name-only` = train.py only; eval-line count == `num_epochs:` (≤1 eval/epoch); no new
     deps; seed 42; prepare.py/eval untouched; **num_params 4,299,866 (UNCHANGED — model not modified)**.
   - **Fairness gate**: confirm the committed run was uncontended (steady dt band, no sustained
     contention); a contention-shortened run is invalid and must be re-run.
   - Timeout per command: 30s. Overall run timeout: 600s wall.

### Informational Metrics (Optional) — the diagnostic for this experiment
- `num_epochs:` & steady dt — vs baseline ~91 ep / ~8ms. THE key signal: did cuDNN cut dt and add epochs?
- `final_test_loss:` — vs baseline 0.195 (convergence check: lower/equal ⇒ converging; higher ⇒
  under-trained).
- `peak_vram_mb:` — expect ≈ baseline (benchmark mode may pick a higher-memory algo; should stay well
  under H20 capacity).

## Expected Outcome / Decision
- **If `best_test_acc >= 96.32`** on a clean run: improvement — commit, merge to `autoresearch/dev`, PR.
- **If dt dropped & more epochs but acc within-noise/below**: no-improvement — net is EPOCH-SATURATED at
  this recipe (high-value finding: closes the throughput→epochs direction; stop chasing epochs).
- **If dt unchanged (no-op)**: no-improvement — torch.compile already conv-optimal; closes the
  cuDNN-autotune sub-lever (does NOT resolve convergence-bound vs saturated — that would need a working
  dt reducer, e.g. the deprioritized max-autotune / CPU-cutout variants).
