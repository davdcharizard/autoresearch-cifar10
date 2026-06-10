# Plan EXP-011: Mixup (mild α=0.2) GPU-vectorized, stacked on Cutout + compile enabler
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md

## Milestones

### Milestone 1: Code changes implemented and parse-clean
- [ ] Edit `train.py` only: add `MIXUP_ALPHA = 0.2`; in the training loop, AFTER the existing `cutout_batch` line,
      add per-batch Mixup (sample scalar λ~Beta(α,α), permute the batch, lerp inputs, keep permuted targets); change
      the loss to the Mixup convex combination of two label-smoothed cross-entropies; add
      `compiled_model = torch.compile(model, mode="reduce-overhead")` and route the training forward through it;
      keep eval eager (`evaluator.evaluate(model, device)` UNCHANGED).
- [ ] `python -c "import ast; ast.parse(open('train.py').read())"` passes; `uv run ruff check train.py` passes.
- [ ] Sanity: param count prints **4,299,866** (UNCHANGED — Mixup adds no params); `git diff --name-only
      autoresearch/dev` = only `train.py`; eval line still eager `model`; `grep manual_seed train.py` → 42.

### Milestone 2: Run launched and confirmed training
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background.
- [ ] `run.log` shows `Device: cuda`, `num_params 4,299,866`, clean compile (no graph breaks/recompile spam),
      steady-state dt printed (expect ~9ms ≈ compiled-k4; Mixup is near-free per step), loss decreasing, no NaN.
      Note: Mixup makes the *training* loss higher/noisier (mixed targets) — this is EXPECTED, not divergence;
      judge health by the per-epoch eval accuracy climbing, not the train loss absolute level.

### Milestone 3: Run completes within budget and summary emitted
- [ ] `run.log` contains the `best_test_acc:` summary; `total_seconds` < 600.
- [ ] Record `num_epochs` (expect ~85–89, compiled) and `final_test_loss` — KEY signals: Mixup should LOWER test
      loss if it's regularizing well; a high final_test_loss + still-rising eval ⇒ under-converged (needs budget).

### Milestone 4: Verification verdict
- [ ] Apply the Verification Protocol. PASS all three ⇒ improvement; Cond-2 fail (best_test_acc < 96.10) ⇒
      no-improvement; constraint breach ⇒ invalid; crash/empty summary ⇒ crash.

## Code Changes
All changes confined to `train.py` (only editable file; `prepare.py` hook-protected). The full EXP-003 recipe stays
FIXED (k=4, Cutout(16), PEAK_LR 0.2 / 5% warmup cosine, Nesterov, label smoothing 0.1, batch 128, WD 1e-4, bf16,
channels_last, seed 42). Cutout is RETAINED — Mixup stacks on top.

- **train.py — hyperparameter**: add `MIXUP_ALPHA = 0.2` to the hyperparameter block. *Why mild*: Beta(0.2,0.2) is
  U-shaped (most samples nearly pure, occasional strong mix) → regularizes without drastically slowing convergence,
  threading the fixed-budget constraint that makes aggressive Mixup risky.

- **train.py — Mixup in the training loop** (after the existing `inputs = cutout_batch(inputs, CUTOUT_SIZE)` line):
  ```python
  lam = float(torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA).sample())
  perm = torch.randperm(inputs.size(0), device=inputs.device)
  inputs = lam * inputs + (1.0 - lam) * inputs[perm]
  targets_b = targets[perm]
  ```
  Then the loss becomes the Mixup convex combination (replacing the single cross_entropy):
  ```python
  loss = lam * F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING) \
       + (1.0 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=LABEL_SMOOTHING)
  ```
  *Why*: standard per-batch Mixup (Zhang et al. 2018). Interpolation-based regularization — a DIFFERENT mechanism
  than Cutout's occlusion — that improves generalization at fixed capacity. GPU-vectorized (one Beta scalar, one
  `randperm`, one lerp, one extra CE on logits): no per-sample CPU `.item()` sync → does NOT throttle the dataloader
  (avoids the EXP-002 trap). *Seed-consistency*: Beta uses the CPU default generator (seeded by `torch.manual_seed(42)`),
  `randperm(device=cuda)` uses the CUDA generator (seeded by `torch.cuda.manual_seed(42)`) → fully reproducible.

- **train.py — compile + eval split** (validated EXP-007/008/010 pattern): after model on device + num_params
  printed, add `compiled_model = torch.compile(model, mode="reduce-overhead")`; training forward
  `outputs = compiled_model(inputs)`; eval on eager `model` (UNCHANGED). *Why*: Mixup slows convergence; compile
  buys ~89 epochs (vs 77) to converge. **Compile sees only the forward** — the Mixup mixing + varying-λ loss are in
  the eager loop, so the per-step λ poses no CUDA-graph/recompile risk. EXP-007: compiled-k4 = 95.92 ≈ baseline
  (null standalone effect), so any gain over ~96.0 is attributable to Mixup.

**Risks/edge cases**: (a) under-convergence — even mild Mixup slows fitting; if the run is epoch-starved the result
could be a soft regression (diagnosable via num_epochs + a high, still-falling final_test_loss); (b) train loss
will read higher/noisier (mixed targets) — EXPECTED, judge by eval acc; (c) compile cost ~20s charged to budget
(accounted); (d) param count UNCHANGED confirms no architecture change.

## Configuration Changes
- MIXUP_ALPHA: (new) `0.2` — mild Beta concentration for per-batch Mixup
- Augmentation: Mixup (per-batch λ) stacked AFTER Cutout(16) in the training loop; loss = Mixup convex combo of two
  label-smoothed CEs
- Execution: training forward via `torch.compile(model, mode="reduce-overhead")`; eval on eager `model`
- ALL else UNCHANGED: WIDTH_MULT 4, NUM_BLOCKS 3, Cutout(16), PEAK_LR 0.2, WARMUP_FRAC 0.05, WD 1e-4, label
  smoothing 0.1, batch 128, Nesterov, bf16, channels_last, cosine, MAX_STEPS 10_000_000, seed 42, eval frozen.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background via Bash run_in_background).
- Resources: 1× NVIDIA H20 (GPU 0; GPU 1 free). VRAM ≈ EXP-007 (~500 MB of 98 GB).
- Estimated runtime: ~300s training (incl. ~20s one-time compile) + ~60–90s startup/eval ≈ 6–8 min. Expect
  num_epochs ~85–89.
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^final_test_loss:|^num_params:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` → kill, crash. (Note: a higher/noisier *train* loss from mixed targets is NOT divergence — only
  NaN/inf or a clearly exploding trend after warmup counts.)
- Python traceback in `run.log` (Beta/randperm shape error, compile failure, empty `best_test_acc:`) → crash;
  `tail -n 50 run.log`.
- Recompile spam / repeated multi-second stalls cutting throughput → record; if it pushes wall-clock toward the
  10-min limit, treat as failure.
- No new output in `run.log` for > 3 min while training (allow the one-time ~20s compile) → kill (hang).
- Total wall-clock > 10 min → kill, failure.
- num_epochs collapsing well below ~70 → NOT an abort, but record (unexpected, since Mixup is near-free per step).

## Verification Protocol

### Verification Procedure
Run from project root after completion. Baseline = **96.00** (`exp-index.sh baseline`); success bar = **96.10**.

1. **Clean completion within budget** (necessary cond 2): `grep -aE "^best_test_acc:|^total_seconds:" run.log`;
   `tail -n 50 run.log` for tracebacks. PASS if `best_test_acc:` present, `total_seconds < 600`, no traceback.
   Timeout 10 min. FAIL → crash.
2. **Metric improvement** (necessary cond 1): parse `best_test_acc`. PASS if `best_test_acc >= 96.10`
   (= baseline 96.00 + 0.1). Else → no-improvement. Stop here on fail.
3. **No constraint violations** (necessary cond 3): `git diff --name-only autoresearch/dev` = only `train.py`;
   no `pyproject.toml`/`uv.lock` diff (Beta/randperm/compile = core torch, no new dep); eval-line count
   (`grep -c "eval ep" run.log`) == num_epochs (eval once/epoch); seed unchanged (`grep manual_seed train.py` → 42);
   num_params == 4,299,866. PASS if all hold, else → invalid. All necessary conditions must PASS; stop at first
   failure.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~85–89. KEY for interpreting a
  near-96.0 result (under-converged vs saturated).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — vs EXP-003's 0.204 / compiled-k4's 0.208. Mixup should
  LOWER this if regularizing well; a higher value signals under-convergence.
- num_params: `grep -aE "^num_params:" run.log` — expect 4,299,866 (confirms Mixup is parameter-free).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`.
