# Plan EXP-057: Full-coverage faithful GPU AugMix (multi-chain + Beta clean-mix)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-057.md

Baseline = **96.45%** (EXP-054, commit 86161d9); bar = **96.55%**. EXP-056 validated the GPU-augmentation throughput unlock (cheap, full-coverage feasible) but its naive harsh single-stack policy regressed (95.39). This loop restores AugMix's two defining properties — multi-chain diversity + clean-image convex mixing — GPU-side at FULL coverage (the EXP-054 recipe was wall-forced to a 50% subset; full coverage is the untried beneficial direction).

## Milestones

### Milestone 1: Code implemented and smoke-tested
- [ ] In train.py, replace EXP-054's CPU AugMix with a GPU `gpu_augmix(x)` (full implementation in Code Changes below): W=3 independently-augmented affine+photometric chains, Dirichlet(1) mix across chains, Beta(1,1)=Uniform clean-mix with the original. Add a `_aug_chain(x)` helper. Keep `cutout_batch` after it; CPU pipeline → crop+flip+ToTensor+Normalize (remove `RandomApply([AugMix()])`).
- [ ] Wire `inputs = gpu_augmix(inputs)` into the train loop before `cutout_batch`.
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --name-only` = train.py only; no new imports.
- [ ] Smoke: GPU harness — (128,3,32,32) channels_last cuda tensor through `gpu_augmix`: output shape (128,3,32,32), finite, float32, channels_last; measure ms/batch (expect ~1.5-2.5ms, 3 grid_samples). Confirm a clean-mix bound: output should be closer to the input than EXP-056's single harsh stack (sanity: mean abs diff from clean is moderate). num_params still 4,299,866.

### Milestone 2: Running and EPOCH/CONTENTION feasibility confirmed
- [ ] **Pre-launch idle-GPU check**: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader` + `--query-compute-apps=pid,used_memory`; launch on a GPU with util ~0% and mem <700MiB (EXP-056 Run 1 lesson: contention invalidates the run).
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background). Confirm run.log writes.
- [ ] **Early gate** (~60-90s wall, post-compile): two-point window (Δsteps/Δwall via ps etimes + log steps). Compute (a) mean dt (Σdt/steps), (b) **wall/Σdt ratio** (contention tell — must be ~1.3-2×, NOT ~10× as in EXP-056 Run 1), (c) projected epochs = (300 / mean_dt_s)/390. **ABORT if: dt>~11ms (epochs<~76) → W=2 fallback; OR wall/Σdt ≫ 2.5 → contention, relaunch on a clean GPU.**
- [ ] Early signal: ep1 test_acc normal (~45-48%), no NaN, dt steady.

### Milestone 3: Completes and verified
- [ ] Summary prints; `total_seconds < 600`.
- [ ] Extract best_test_acc, num_epochs, num_steps, dt dist, total_seconds, peak_vram_mb, final_test_loss; compare best_test_acc to bar **96.55**.
- [ ] Remove `run.log`.

## Code Changes
- **train.py — new helpers** (after `cutout_batch`). Exact implementation:
  ```python
  def _aug_chain(x):
      """One independently-augmented version of the batch: per-sample random affine
      (rotation/shear/scale) via grid_sample + photometric (brightness/contrast).
      Gentle magnitudes (EXP-053). Operates on the normalized channels_last batch."""
      b = x.size(0)
      dev = x.device
      r = lambda: (torch.rand(b, device=dev) * 2 - 1)
      ang = r() * (12.0 * math.pi / 180.0)
      shx, shy = r() * 0.1, r() * 0.1
      sx, sy = 1 + r() * 0.1, 1 + r() * 0.1
      cos, sin = ang.cos(), ang.sin()
      o = torch.zeros(b, device=dev)
      l = torch.ones(b, device=dev)
      R = torch.stack([cos, -sin, sin, cos], -1).view(b, 2, 2)
      Sh = torch.stack([l, shx, shy, l], -1).view(b, 2, 2)
      S = torch.stack([sx, o, o, sy], -1).view(b, 2, 2)
      M = R @ Sh @ S
      theta = torch.cat([M, torch.zeros(b, 2, 1, device=dev)], dim=-1)
      grid = F.affine_grid(theta, x.shape, align_corners=False)
      a = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
      bright = (r() * 0.1).view(b, 1, 1, 1)
      contrast = (1 + r() * 0.15).view(b, 1, 1, 1)
      mean = a.mean(dim=(1, 2, 3), keepdim=True)
      return (a - mean) * contrast + mean + bright

  def gpu_augmix(x, width=3):
      """Faithful AugMix (Hendrycks ICLR 2020) GPU-side, full-coverage: mix `width`
      independently-augmented chains with per-image Dirichlet(1) weights, then convex-mix
      that blend with the ORIGINAL clean image with a per-image Beta(1,1)=Uniform weight.
      The clean-mix BOUNDS the per-image shift (the property EXP-056's harsh stack lacked).
      Replicates the EXP-054 winner (w3 AugMix) at 100% coverage (was wall-forced to 50%)."""
      b = x.size(0)
      dev = x.device
      # Dirichlet(1,...,1) weights via normalized Exp(1) (= -log U); shape (width,b)
      gs = [(-(torch.rand(b, device=dev).clamp_min(1e-6)).log()) for _ in range(width)]
      tot = sum(gs)
      mix = torch.zeros_like(x)
      for i in range(width):
          mix = mix + (gs[i] / tot).view(b, 1, 1, 1) * _aug_chain(x)
      m = torch.rand(b, device=dev).view(b, 1, 1, 1)  # Beta(1,1) = Uniform
      out = m * x + (1 - m) * mix
      return out.to(memory_format=torch.channels_last)
  ```
  (`math`, `F`, `torch` already imported.)
- **train.py — train loop** (~line 232): insert `inputs = gpu_augmix(inputs)` immediately before `inputs = cutout_batch(inputs, CUTOUT_SIZE)`.
- **train.py — train_tf**: remove `transforms.RandomApply([transforms.AugMix()], p=0.5)`; update the comment block.
  - **Why this tests the hypothesis**: restores AugMix's two defining shift-bounding properties (multi-chain mix + clean convex-mix) that EXP-056 omitted, at full coverage. Tests whether the proven 96.45 recipe at 100% coverage (vs the wall-forced 50% subset) clears the bar.
  - **Risks / edge cases**: (a) **epoch wall** — 3 grid_samples; gated (dt>11ms → W=2). (b) **full coverage may over-regularize even clean-mixed** → near-noise null/mild regression vs 96.45; fallback is the subset variant (brainstorm candidate 3). (c) mixing is linear → valid on normalized data. (d) runs eager (outside torch.compile, like cutout_batch). (e) contention → idle-GPU check + gate.

### Contingency (only if Milestone-2 gate trips)
- dt>~11ms (epoch wall): re-run with `width=2` (2 chains, ~1ms cheaper). If still tight, width=2 + drop photometric from `_aug_chain`.
- Contention (wall/Σdt ≫ 2.5): TaskStop, relaunch on a verified-idle GPU.

## Configuration Changes
- Augmentation: CPU `RandomApply([AugMix() w3], p=0.5)` REMOVED → GPU `gpu_augmix` (W=3 chains, Dirichlet mix, Beta clean-mix, full-coverage) in the train loop before Cutout. CPU → crop+flip+ToTensor+Normalize. No model/optimizer/schedule/seed/batch/compile changes. num_params unchanged.
- Magnitudes per chain: rotation ±12°, shear ±0.1, scale [0.9,1.1], brightness ±0.1, contrast [0.85,1.15] (gentle; AugMix's mixing softens further).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background. MUST use `uv run`.
- Resources: single NVIDIA H20. Shared node GPUs 0/1 — **verify idle before launch** (EXP-056 contention lesson); pick util ~0%, mem <700MiB.
- Estimated runtime: ~300s Σdt; wall ~400-440s (dt-bound, ~10ms; +~2ms over 8ms baseline for 3 chains). Target < 600s.
- Log output: stdout/stderr → `run.log`.
- Tool skill: none (local run).

## Abort Criteria
- **dt > ~11ms** at Milestone-2 (epochs < ~76) → W=2 fallback.
- **wall/Σdt ≫ 2.5** early → GPU contention → TaskStop, relaunch on a clean idle GPU (NOT a code failure).
- Loss NaN/inf or diverging — check ep1.
- Total wall approaching ~595s without a summary → kill. [Unlikely — dt-bound.]
- No output / log not advancing > 3 min after launch.

## Verification Protocol

### Verification Procedure
Run after completion; stop at the first failed necessary condition.

1. **Baseline**: `bash /SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/improve-cifar10-test-accuracy.tsv` → 96.45, bar **96.55**.
2. **Necessary condition 1 — `best_test_acc >= 96.55`**: `grep -aE "^best_test_acc:" run.log` → parse float. PASS iff `>= 96.55`; else no-improvement. (Absent ⇒ crash → `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:|^num_params:" run.log` → summary printed, **`total_seconds < 600`**, `num_params == 4,299,866`. No NaN/traceback (`grep -ic "nan\|traceback" run.log` → 0).
4. **Necessary condition 3 — no hard-constraint violations**: `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch; no new deps (affine_grid/grid_sample core torch); seed 42 unchanged; ran on an uncontended GPU (fair dt-budget).
5. Remove `run.log`.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.45: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~78-82 (3-chain dt premium).
- total_seconds: `grep -aE "^total_seconds:" run.log` — expect ~400-440s.
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect ~10ms.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare to EXP-054's 0.1968 (should be MUCH lower than EXP-056's 0.224 if the clean-mix works as intended).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`.
