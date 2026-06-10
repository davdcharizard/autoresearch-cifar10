# Plan EXP-059: GPU faithful AugMix at the proven p=0.5 coverage (W=3)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-059.md

Baseline = **96.45%** (EXP-054, commit 86161d9); bar = **96.55%**. EXP-056/057 validated the GPU-aug infra but ran at 100% coverage (both regressed: harsh-stack 95.39, faithful 95.64). ~50% coverage is the PROVEN interior optimum (EXP-054 CPU AugMix p=0.5 = 96.45; EXP-055 <50% and EXP-057 100% both worse). This loop runs the one untried same-family variant: faithful GPU AugMix at the proven 50% coverage, delivering continuous-magnitude affine chains (potentially richer than torchvision's discrete ops) at the proven operating point.

**Failed-approaches justification (per skill requirement)**: GPU-aug at FULL coverage is a recorded failure (EXP-056/057). This differs materially: it applies the SAME faithful gpu_augmix to only ~50% of each batch — the exact coverage that is the proven optimum on CPU (EXP-054) and was never run on the GPU path. EXP-057's analysis explicitly flagged "GPU-AugMix-on-50%-subset" as the one remaining untried variant. The compute is run on only the ~50% subset (not full-batch-then-masked) so dt stays ~9.5ms (avoiding EXP-057 W=3's 11ms epoch wall).

## Milestones

### Milestone 1: Code implemented and smoke-tested
- [ ] In train.py, add `_aug_chain(x)` and `gpu_augmix(x, width=3)` helpers after `cutout_batch` (exact code in Code Changes below — recovered verbatim from plan-057).
- [ ] In the train loop, BEFORE `inputs = cutout_batch(...)`, apply gpu_augmix to a random ~50% subset of the batch (per-sample index mask; compute aug only on the subset for the dt saving).
- [ ] Remove the CPU `transforms.RandomApply([transforms.AugMix()], p=0.5)` line from `train_tf` (aug moves to GPU); CPU → crop+flip+ToTensor+Normalize.
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --name-only` = train.py only; no new imports (math/F/torch already imported).
- [ ] Smoke: GPU harness — a (128,3,32,32) channels_last cuda batch through the subset-apply path returns (128,3,32,32) finite float32 channels_last; gpu_augmix on 64 samples runs; measure ms/batch (expect ~1.5ms aug on the subset). num_params unchanged = 4,299,866.

### Milestone 2: Running and dt/epoch + contention feasibility confirmed
- [ ] **Pre-launch idle-GPU check**: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader` + `--query-compute-apps=pid,used_memory`; launch on a GPU with util ~0% and mem <700MiB.
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background). Confirm run.log writes.
- [ ] **Early gate** (~60-90s wall, post-compile): two-point window. Compute (a) steady mean dt, (b) **wall/Σdt ratio** (~1.3-2× expected, NOT ~10×), (c) projected epochs = (300/mean_dt_s)/390. **ABORT if: dt > ~10.5ms (epochs < ~75) → W=2 fallback; OR wall/Σdt ≫ 2.5 → contention, relaunch on a clean GPU.**
- [ ] Early signal: ep1 test_acc normal (~45-48%), no NaN, dt steady.

### Milestone 3: Completes and verified
- [ ] Summary prints; `total_seconds < 600`.
- [ ] Extract best_test_acc, num_epochs, num_steps, dt dist, total_seconds, peak_vram_mb, final_test_loss; compare best_test_acc to bar **96.55**.
- [ ] Remove `run.log`.

## Code Changes
- **train.py — new helpers** (after `cutout_batch`, verbatim from plan-057):
  ```python
  def _aug_chain(x):
      """One independently-augmented version of the batch: per-sample random affine
      (rotation/shear/scale) via grid_sample + photometric (brightness/contrast)."""
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
      """Faithful AugMix GPU-side: mix `width` Dirichlet(1)-weighted chains, then
      Beta(1,1)=Uniform convex-mix with the ORIGINAL clean image (shift-bounding)."""
      b = x.size(0)
      dev = x.device
      gs = [(-(torch.rand(b, device=dev).clamp_min(1e-6)).log()) for _ in range(width)]
      tot = sum(gs)
      mix = torch.zeros_like(x)
      for i in range(width):
          mix = mix + (gs[i] / tot).view(b, 1, 1, 1) * _aug_chain(x)
      m = torch.rand(b, device=dev).view(b, 1, 1, 1)
      out = m * x + (1 - m) * mix
      return out.to(memory_format=torch.channels_last)
  ```
- **train.py — train loop** (immediately before `inputs = cutout_batch(inputs, CUTOUT_SIZE)`), apply to a random ~50% subset (compute aug ONLY on the subset for the dt saving):
  ```python
  # GPU faithful AugMix on a random ~50% subset (proven coverage, EXP-054; train only, EXP-059)
  b = inputs.size(0)
  idx = torch.randperm(b, device=inputs.device)[: b // 2]
  inputs[idx] = gpu_augmix(inputs[idx], width=3)
  ```
- **train.py — train_tf**: remove `transforms.RandomApply([transforms.AugMix()], p=0.5)`; update the comment to describe the CPU→GPU 50%-subset move.
  - **Why this tests the hypothesis**: reproduces the EXP-054 winner's proven 50% coverage via the validated GPU path with continuous-affine chains; tests whether the GPU representation matches or beats CPU torchvision AugMix at the proven operating point.
  - **Risks / edge cases**: (a) **dt** — W=3 on a 64-sample subset ≈ +1.5ms → dt ~9.5ms → ~81 ep; gated (abort dt>10.5ms → W=2). (b) `inputs[idx] = ...` advanced-index assignment runs EAGER (outside torch.compile, like cutout_batch) — fine; the compiled forward receives the full channels_last `inputs`. (c) continuous-affine chains may be harsher than discrete ops → mild regression (graceful). (d) contention → idle-GPU check + gate.

## Configuration Changes
- Augmentation: CPU `RandomApply([AugMix() w3], p=0.5)` REMOVED → GPU `gpu_augmix(W=3)` on a random ~50% of each batch, before Cutout. CPU → crop+flip+ToTensor+Normalize. No model/optimizer/schedule/seed/batch/compile changes. num_params unchanged (4,299,866).
- Magnitudes per chain (from EXP-057): rotation ±12°, shear ±0.1, scale [0.9,1.1], brightness ±0.1, contrast [0.85,1.15].

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background. MUST use `uv run`.
- Resources: single NVIDIA H20. Shared node GPUs 0/1 — **verify idle before launch**; pick util ~0%, mem <700MiB.
- Estimated runtime: ~300s Σdt; wall ~400-440s (dt ~9.5ms). Target < 600s.
- Log output: stdout/stderr → `run.log`.
- Tool skill: none (local run).

## Abort Criteria
- **dt > ~10.5ms** at Milestone-2 (epochs < ~75) → W=2 fallback (W=2 on 50% ≈ +1ms → dt ~9ms).
- **wall/Σdt ≫ 2.5** early → GPU contention → TaskStop, relaunch on a clean idle GPU.
- Loss NaN/inf or diverging — check ep1.
- Total wall approaching ~595s without a summary → kill. [Unlikely — dt-bound.]
- No output / log not advancing > 3 min after launch.

## Verification Protocol

### Verification Procedure
Run after completion; stop at the first failed necessary condition.

1. **Baseline**: `bash /SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/improve-cifar10-test-accuracy.tsv` → 96.45, bar **96.55**.
2. **Necessary condition 1 — `best_test_acc >= 96.55`**: `grep -aE "^best_test_acc:" run.log` → parse float. PASS iff `>= 96.55`; else no-improvement. (Absent ⇒ crash → `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:|^num_params:" run.log` → summary printed, **`total_seconds < 600`**, `num_params == 4,299,866`. No NaN/traceback (`grep -ic "nan\|traceback" run.log` → 0).
4. **Necessary condition 3 — no hard-constraint violations**: `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch; no new deps (affine_grid/grid_sample core torch); seed 42 unchanged; ran on an uncontended GPU.
5. Remove `run.log`.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.45: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~80-84.
- total_seconds: `grep -aE "^total_seconds:" run.log` — expect ~400-440s.
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect ~9-10ms.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — vs EXP-054 0.1968; GPU-50% should be CLOSER to 0.1968 than EXP-057's 0.2115 (100%) if coverage is the fix.
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`.
