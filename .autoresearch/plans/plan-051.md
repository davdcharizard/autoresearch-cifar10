# Plan EXP-051: LayerScale — learnable per-channel residual-branch scaling
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-051.md

Baseline = **96.22%** (EXP-012, 6c417a4); bar = baseline + 0.1 = **96.32%**. Cleanest remaining throughput-free sub-lever: add a learnable per-channel scale (init 0.1) on each BasicBlock's residual branch (CaiT LayerScale). Tests whether a learnable residual-magnitude DOF helps generalization at fixed capacity. Low EV (EXP-026 zero-init-γ was null — "needs depth"), but confounder-free and closes residual-scaling in its modern form.

## Milestones

### Milestone 1: Code change implemented and smoke-tested
- [ ] In `BasicBlock.__init__`, add `self.layer_scale = nn.Parameter(torch.full((out_channels, 1, 1), 0.1))`.
- [ ] In `BasicBlock.forward`, multiply the residual branch by the scale before the add: `out = out * self.layer_scale` immediately before `out += self.shortcut(x)`.
- [ ] Smoke check: `python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --name-only` = `train.py` only.
- [ ] Smoke check (uv run): model instantiates; num_params == 4,299,866 + 1,344 (LayerScale scalars: 3×64 + 3×128 + 3×256) = **4,301,210**; confirm 9 `layer_scale` params each init 0.1.

### Milestone 2: Experiment running and throughput-neutral
- [ ] Launch `uv run train.py > run.log 2>&1` on the idle GPU; confirm run.log is written.
- [ ] Early signal: dt steady ~8ms (LayerScale is a fused elementwise multiply — must NOT raise dt or break the CUDA graph), ep1 test_acc in the normal range, no NaN.

### Milestone 3: Run completes and is verified
- [ ] Run prints the summary block; `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, dt distribution, `peak_vram_mb`; compare to bar 96.32; confirm ~91 ep (throughput-neutral).

## Code Changes
- **train.py** — `BasicBlock` only (two edits):
  1. `__init__`: add `self.layer_scale = nn.Parameter(torch.full((out_channels, 1, 1), 0.1))` (per-channel, broadcasts over N,H,W; channels_last-safe).
  2. `forward`: change
     ```python
     out = self.bn2(self.conv2(out))
     out += self.shortcut(x)
     ```
     to
     ```python
     out = self.bn2(self.conv2(out))
     out = out * self.layer_scale          # LayerScale (CaiT): learnable per-channel residual scale
     out += self.shortcut(x)
     ```
  - **Why this tests the hypothesis**: scaling each residual branch by a learnable per-channel γ (init 0.1) gives the net a magnitude DOF on the residual contribution — down-weighting branches early (identity-dominated start) and learning per-channel residual strength. Throughput-free → a clean, confound-free test of the residual-scaling DOF in its modern form.
  - **Risks / edge cases**: (a) init 0.1 down-scales all residual branches at start — on a shallow 9-block net this should recover quickly (0.1 is the CaiT default for ≤18 blocks; not the tiny 1e-4 that would over-suppress); (b) the `(C,1,1)` Parameter multiply is a static elementwise op → no CUDA-graph break, dt should stay 8ms (watch as early signal); (c) `_weights_init` only touches Conv2d/Linear weights via `isinstance`, so `layer_scale` keeps its 0.1 init (not re-initialized) — confirm in smoke test.

## Configuration Changes
- New: 9 LayerScale parameters (per-channel, init 0.1), +1,344 params total → 4,301,210.
- Unchanged: width k=4, depth, BATCH_SIZE 128, PEAK_LR 0.2, WARMUP_FRAC 0.05, MOMENTUM 0.9, WEIGHT_DECAY 1e-4, LABEL_SMOOTHING 0.1, CUTOUT_SIZE 16, Nesterov SGD, time-fraction cosine schedule, seed 42, compile reduce-overhead, TrivialAugment + Cutout.
- Note: WD applies to `layer_scale` (it is a model parameter in the single SGD group) — standard; 1e-4 WD on a 0.1-init scalar is negligible pull.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background (`run_in_background: true`).
- Resources: single NVIDIA H20. Shared node GPUs 0/1; check `nvidia-smi`, launch on idle GPU.
- Estimated runtime: ~300s training + ~2s startup + eval ≈ 400s wall (< 600s).
- Log output: stdout/stderr → `run.log` in project root.
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf or diverging.
- dt rises above ~8ms and stays there (graph break / unfused multiply) → kill and diagnose (would confound the throughput-neutral premise).
- No output / log not advancing > 3 min after launch.
- Total wall-clock approaching 600s without a summary → kill.

## Verification Protocol

### Verification Procedure
Run after completion; stop at the first failed necessary condition.

1. **Baseline**: `bash .../exp-index.sh baseline experiment-indices/improve-cifar10-test-accuracy.tsv` → 96.22, bar **96.32**.
2. **Necessary condition 1 — `best_test_acc >= 96.32`**: `grep -aE "^best_test_acc:" run.log` → parse float. PASS iff `>= 96.32`; else no-improvement. (Absent ⇒ crash → `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:|^num_params:" run.log` → summary printed, `total_seconds < 600`, `num_params == 4,301,210` (baseline + 1,344 LayerScale scalars). No NaN/traceback.
4. **Necessary condition 3 — no hard-constraint violations**: `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps; seed 42 unchanged; no seed hacking.
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.22: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — confirm ~91 ep (throughput-neutral, no epoch confound).
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect steady 8ms.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare to baseline 0.195 (does LayerScale move loss even if not top-1?).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`.
