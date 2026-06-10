# Plan EXP-008: Squeeze-Excitation blocks on k=4 (+ torch.compile enabler)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md

## Milestones

### Milestone 1: Code changes implemented and parse-clean
- [ ] Edit `train.py` only: add an `SEModule`, call it in `BasicBlock.forward` (scale conv2/bn2 output before the
      residual add), add `SE_REDUCTION=16`, compile the training model (reduce-overhead), train through compiled
      model, eval the eager model
- [x] `python -c "import ast; ast.parse(open('train.py').read())"` passes; `uv run ruff check train.py` passes
- [x] Sanity: param count = 4,333,550 (≈ +0.8% vs 4,299,866); only train.py changed (+26/-1); eval still eager `model`
- [x] (SE compiled throughput already smoke-tested in planning: 12.8ms/step → ~60 epochs; eager 18ms; compiles clean)

### Milestone 2: Run launched and confirmed training
- [x] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] `run.log` shows `Device: cuda`, `num_params 4,333,550`, steady-state dt=**9ms** (BETTER than the smoke
      test's 12.8ms → tracking ~80 epochs, not 60), ~13,900 img/s, loss decreasing 1.24, no NaN/recompile

### Milestone 3: Run completes within budget and summary emitted
- [x] `run.log` contains `best_test_acc:` summary (95.86%); `total_seconds` < 600 (399.8)
- [x] `num_epochs`=82 — dt was 9ms (better than smoke test) so NOT epoch-starved; SE got a fair, well-trained test

### Milestone 4: Verification verdict
- [ ] FAIL: `best_test_acc` 95.86 < 96.10 (and < 96.00 baseline) → no-improvement. SE ≈ compiled-k4 (95.92), no gain.
- [~] skipped (aborted after metric failure)

## Code Changes
All changes confined to `train.py` (only editable file; `prepare.py` hook-protected). Width stays k=4 and the
full recipe (bf16, channels_last, cosine, Nesterov, label smoothing, batch 128, WD 1e-4, PEAK_LR 0.2, Cutout(16),
seed 42) is FIXED. Two coupled changes: SE (the accuracy intervention) + `torch.compile` (the throughput enabler
that keeps SE's epoch count viable). Per EXP-007, compiled-k4 alone = 95.92 ≈ baseline (no standalone accuracy
effect), so any gain over ~96.0 is attributable to SE.

- **train.py — `SEModule`** (new class, module level):
  ```python
  class SEModule(nn.Module):
      def __init__(self, channels, reduction):
          super().__init__()
          hidden = max(channels // reduction, 4)
          self.fc1 = nn.Linear(channels, hidden)
          self.fc2 = nn.Linear(hidden, channels)
      def forward(self, x):
          w = F.adaptive_avg_pool2d(x, 1).flatten(1)
          w = F.relu(self.fc1(w))
          w = torch.sigmoid(self.fc2(w))
          return x * w.view(w.size(0), w.size(1), 1, 1)
  ```
  *Why*: standard SE channel recalibration (Hu et al. 2018) — learned per-channel gating, a new accuracy axis at
  fixed width. `max(.,4)` floors the bottleneck for the 64-channel stage. Gets kaiming init via `_weights_init`
  (matches `nn.Linear`); fc2's sigmoid-gate starting near 0.5 is fine.

- **train.py — `BasicBlock`**: construct `self.se = SEModule(out_channels, SE_REDUCTION)`; in `forward`, apply SE
  to the post-bn2 output **before** the residual add:
  `out = self.bn2(self.conv2(out)); out = self.se(out); out += self.shortcut(x); return F.relu(out)`.
  *Why this placement*: the canonical SE-ResNet location (recalibrate the residual branch before merging).

- **train.py — hyperparameter**: add `SE_REDUCTION = 16` to the hyperparameter block (standard SE ratio).

- **train.py — compile + eval split** (identical pattern to EXP-007, the validated enabler): after the model is
  on device and `num_params` printed, add `compiled_model = torch.compile(model, mode="reduce-overhead")`; change
  the training forward to `outputs = compiled_model(inputs)`; keep eval on the eager `model`
  (`evaluator.evaluate(model, device)` — UNCHANGED). *Why*: SE roughly doubles eager step time (launch-bound; 9→18ms
  in the smoke test); compile recovers it to ~12.8ms so the run keeps ~60 epochs instead of ~40 — giving SE a
  fair-ish test rather than the EXP-002/EXP-004 epoch-starvation trap.

## Configuration Changes
- SE_REDUCTION: (new) `16` — SE bottleneck ratio (channels→channels/16, floored at 4)
- Architecture: each BasicBlock gains an SE module on its residual branch (+~33.7k params, +0.8%)
- Execution: training forward via `torch.compile(mode="reduce-overhead")`; eval on eager model
- ALL else UNCHANGED: WIDTH_MULT 4, Cutout(16), PEAK_LR 0.2, WD 1e-4, label smoothing 0.1, batch 128, bf16,
  channels_last, Nesterov, cosine schedule, MAX_STEPS 10_000_000, seed 42, eval frozen

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background)
- Resources: 1× NVIDIA H20 (GPU 0; GPU 1 free). VRAM ≈ EXP-007 (~500 MB; SE adds negligible activation memory).
- Estimated runtime: ~300s training (incl. ~20s one-time compile charged to budget) + ~60–80s startup/eval ≈
  6–8 min. Expect num_epochs ~58–65 (SE cost cuts epochs even with compile).
- Log output: stdout/stderr → `run.log`. Extract via
  `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:|^final_test_loss:" run.log`.
- Tool skill: none (local).

## Abort Criteria
- Loss `NaN`/`inf` or diverging after warmup → kill, treat as crash.
- Python traceback in `run.log` (SE shape error, compile failure, empty `best_test_acc:`) → crash; `tail -n 50 run.log`.
- Recompile spam / repeated multi-second stalls mid-training sharply cutting throughput → record; if it pushes
  wall-clock toward the limit, treat as failure.
- No new output in `run.log` for > 3 min while training (allow for one-time compile) → kill (hang).
- Total wall-clock > 10 min → kill, failure.
- num_epochs collapses well below ~55 (toward EXP-004's 35) → not an abort, but record: SE cost worse than the
  smoke test, benefit likely epoch-masked (note for analysis).

## Verification Protocol

### Verification Procedure
Run from project root after completion. Baseline = **96.00** (`exp-index.sh baseline`); success bar = **96.10**.

1. **Clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:" run.log`; `tail -n 50 run.log`
   for tracebacks. PASS if `best_test_acc:` present, `total_seconds < 600`, no traceback. Timeout 10 min.
2. **Metric improvement**: parse `best_test_acc`. PASS if `best_test_acc >= 96.10`, else → no-improvement.
3. **No constraint violations**: `git diff --name-only autoresearch/dev` = only `train.py`; no pyproject/uv.lock
   diff (SE = `nn` layers, compile = core torch → no new deps); eval-line count == num_epochs (eval once/epoch);
   seed unchanged (`grep manual_seed train.py` → 42). PASS if all hold, else → invalid.
   All necessary conditions must PASS; stop at first failure.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — **KEY confound signal**: expect ~58–65.
  Interpreting the verdict requires this — a no-improvement near 96.0 with ~60 epochs means SE compensated for the
  epoch loss (per-epoch value) rather than SE being useless.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — vs EXP-003's 0.204 (does SE reduce loss?).
- num_params: `grep -aE "^num_params:" run.log` — expect ~4,333,550 (confirms SE added, +0.8%).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`.
