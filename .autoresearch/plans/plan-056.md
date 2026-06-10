# Plan EXP-056: GPU-batched diverse augmentation (affine + photometric), full-coverage — the throughput unlock

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-056.md

Baseline = **96.45%** (EXP-054, commit 86161d9); bar = baseline + 0.1 = **96.55%**. The CPU-side augmentation-diversity lever is mapped at its wall-limited frontier (w3/p=0.5). This loop moves augmentation OFF the starved 8-worker CPU dataloader and ONTO the idle GPU (which is ~6-7ms/step idle while the CPU starves), delivering full-coverage geometric+photometric diversity. Direct precedent: EXP-003 moved Cutout CPU→GPU and won +0.58pp.

## Key budget mechanics (drives the design)
- dt timer wraps from `t0` (after dataloader yields) through `torch.cuda.synchronize()` (train.py:226-249) — so GPU aug added in the loop **counts toward Σdt** → costs epochs. This is the epoch-wall risk (the recurring killer: EXP-002/004/038). Mitigation: keep the GPU op cheap (sub-ms affine+photometric on a 128×3×32×32 tensor) and gate dt/epochs early.
- Removing the CPU `RandomApply([AugMix()])` lightens the dataloader from ~12ms/batch (AugMix starvation) to ~4-5ms/batch (crop+flip baseline) → the wall stops being dataloader-bound and becomes dt-bound (~dt + sync). Expected wall comfortably < 600s; the binding constraint flips to epochs (Σdt budget).
- Expected: dt 8ms → ~8.5-9.5ms; epochs 91 → ~82-86; wall ~350-420s.

## Milestones

### Milestone 1: Code implemented and smoke-tested
- [ ] Add a `gpu_augment(x)` function near `cutout_batch` (train.py): vectorized, seeded, no per-sample CPU `.item()` syncs — per-image random AFFINE (rotation ±12°, shear ±0.1, anisotropic scale ∈ [0.9,1.1]) via `F.affine_grid`+`F.grid_sample` (bilinear, padding_mode='reflection', align_corners=False) + PHOTOMETRIC (brightness add ∈ ±0.1, contrast scale ∈ [0.85,1.15] around per-image mean). Operates on the normalized float32 channels_last batch; returns channels_last.
- [ ] Wire into the train loop (train.py ~line 231): apply `gpu_augment` to `inputs` BEFORE `cutout_batch` (geometric/photometric first, occlusion last — mirrors the old CPU-AugMix→GPU-Cutout order).
- [ ] In `train_tf`, REMOVE the `transforms.RandomApply([transforms.AugMix()], p=0.5)` line (lighten CPU to RandomCrop+Flip+ToTensor+Normalize). Update comments.
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --name-only` = train.py only; no new imports (F.affine_grid/grid_sample are core torch — confirmed torch 2.9.1).
- [ ] Smoke: short Python harness — build a (128,3,32,32) channels_last cuda tensor, run `gpu_augment` then `cutout_batch`: output shape (128,3,32,32), finite (no NaN/inf), dtype float32. Confirm `num_params` still 4,299,866 (aug-only change, model untouched).

### Milestone 2: Experiment running and EPOCH/WALL feasibility confirmed
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background) on an idle GPU (check nvidia-smi). Confirm run.log writes.
- [ ] **Early dt/epoch gate** (after ~60-90s wall, post torch.compile): from run.log read steady-state dt and the time-budget %. Project epochs ≈ (TIME_BUDGET 300s / mean_dt_s) / 390 batches-per-epoch. **If mean dt > ~11ms (→ projected epochs < ~76, under-training territory) → ABORT (TaskStop), record GPU-aug too costly, go to contingency.** Also project wall (steady-state ms/step × est-steps + startup); expect < 600s comfortably.
- [ ] Early signal: ep1 test_acc normal (~45-48%), no NaN, dt steady.

### Milestone 3: Run completes and is verified
- [ ] Summary prints; `total_seconds < 600`.
- [ ] Extract best_test_acc, num_epochs, num_steps, dt dist, total_seconds, peak_vram_mb, final_test_loss; compare best_test_acc to bar **96.55**.
- [ ] Remove `run.log` before the next experiment.

## Code Changes
- **train.py — new function** (after `cutout_batch`, ~line 58). Exact implementation:
  ```python
  def gpu_augment(x):
      """Vectorized GPU augmentation on the normalized channels_last batch (train only):
      per-sample random affine (rotation/shear/anisotropic-scale) via grid_sample +
      photometric (brightness/contrast). Full-coverage, gentle magnitudes (AugMix severity
      is interior-optimal, EXP-053). Seeded torch RNG, no per-sample CPU sync (cf. EXP-002).
      Moves the diversity lever off the wall-limited CPU dataloader onto the idle GPU."""
      b = x.size(0)
      dev = x.device
      r = lambda: (torch.rand(b, device=dev) * 2 - 1)  # per-image U(-1,1)
      ang = r() * (12.0 * math.pi / 180.0)             # rotation ±12°
      shx, shy = r() * 0.1, r() * 0.1                  # shear ±0.1
      sx, sy = 1 + r() * 0.1, 1 + r() * 0.1            # scale ∈ [0.9,1.1]
      cos, sin = ang.cos(), ang.sin()
      o, l = torch.zeros(b, device=dev), torch.ones(b, device=dev)
      R = torch.stack([cos, -sin, sin, cos], -1).view(b, 2, 2)
      Sh = torch.stack([l, shx, shy, l], -1).view(b, 2, 2)
      S = torch.stack([sx, o, o, sy], -1).view(b, 2, 2)
      M = R @ Sh @ S                                   # (b,2,2)
      theta = torch.cat([M, torch.zeros(b, 2, 1, device=dev)], dim=-1)  # (b,2,3), no translation
      grid = F.affine_grid(theta, x.shape, align_corners=False)
      x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
      # photometric (correct on std=(1,1,1) normalized data): brightness=additive, contrast=scale-about-mean
      bright = (r() * 0.1).view(b, 1, 1, 1)
      contrast = (1 + r() * 0.15).view(b, 1, 1, 1)
      mean = x.mean(dim=(1, 2, 3), keepdim=True)
      x = (x - mean) * contrast + mean + bright
      return x.to(memory_format=torch.channels_last)
  ```
  (`math` is already imported. `F` and `torch` imported.)
- **train.py — train loop** (~line 231): insert `inputs = gpu_augment(inputs)` immediately before the existing `inputs = cutout_batch(inputs, CUTOUT_SIZE)`.
- **train.py — train_tf**: delete the `transforms.RandomApply([transforms.AugMix()], p=0.5)` line; update the comment block to describe the CPU→GPU augmentation move.
  - **Why this tests the hypothesis**: delivers the proven diversity lever (geometric rotate/shear/scale — NONE of which the current crop+flip+Cutout pipeline covers — plus photometric brightness/contrast) to 100% of images, computed on the idle GPU instead of the starved CPU dataloader. Tests whether full-coverage GPU diversity ≥ the 50%-subset CPU AugMix net of a modest epoch cost.
  - **Risks / edge cases**: (a) **epoch wall** — gated at Milestone 2 (abort if dt>~11ms). (b) grid_sample on channels_last — converted back to channels_last on return; runs eager (outside torch.compile, like cutout_batch) so no compile interaction. (c) photometric semantics on normalized data — brightness(additive)/contrast(scale-about-mean) are mathematically valid post-normalization; magnitudes gentle. (d) reflection padding avoids border artifacts. (e) diversity mismatch — affine+brightness/contrast is a SUBSET of AugMix's op pool (no posterize/solarize/equalize); a within-noise null still validates the GPU-aug path.

### Contingency (only if Milestone-2 gate trips on epochs)
- If dt > ~11ms (grid_sample heavier than expected): drop the photometric block (keep affine only) and/or apply `gpu_augment` to a random subset of steps. Re-launch once. If still epoch-starved, record GPU-aug-too-costly and proceed to analysis with the finding.

## Configuration Changes
- Augmentation: CPU `RandomApply([AugMix() w3], p=0.5)` REMOVED → GPU `gpu_augment` (affine+photometric, full-coverage) ADDED in the train loop before Cutout. CPU pipeline now crop+flip+ToTensor+Normalize. No model/optimizer/schedule/seed/batch/compile changes. num_params unchanged (4,299,866).
- Magnitudes: rotation ±12°, shear ±0.1, scale [0.9,1.1], brightness ±0.1, contrast [0.85,1.15] — gentle (EXP-053: aug magnitude is interior-optimal, so keep mild; the lever is coverage/diversity, here delivered at full coverage).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background. MUST use `uv run` (bare python lacks torchvision).
- Resources: single NVIDIA H20. Shared node GPUs 0/1; check `nvidia-smi`, launch on idle GPU.
- Estimated runtime: ~300s Σdt budget; wall ~350-420s (dt-bound now; CPU no longer starves). Target < 600s.
- Log output: stdout/stderr → `run.log` in project root.
- Tool skill: none (local run).

## Abort Criteria
- **Mean dt > ~11ms** at the Milestone-2 gate (→ projected epochs < ~76, under-training) → abort, go to contingency.
- Loss NaN/inf or diverging (grid_sample/photometric bug) — check ep1.
- Total wall-clock approaching ~595s without a summary → kill (constraint breach = failure). [Unlikely — wall is dt-bound here.]
- No output / log not advancing > 3 min after launch.

## Verification Protocol

### Verification Procedure
Run after completion; stop at the first failed necessary condition.

1. **Baseline**: `bash /SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/improve-cifar10-test-accuracy.tsv` → 96.45, bar **96.55**.
2. **Necessary condition 1 — `best_test_acc >= 96.55`**: `grep -aE "^best_test_acc:" run.log` → parse float. PASS iff `>= 96.55`; else no-improvement. (Absent ⇒ crash → `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:|^num_params:" run.log` → summary printed, **`total_seconds < 600`**, `num_params == 4,299,866`. No NaN/traceback (`grep -ic "nan\|traceback" run.log` → 0).
4. **Necessary condition 3 — no hard-constraint violations**: `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps (affine_grid/grid_sample core torch); seed 42 unchanged; no seed hacking.
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.45: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — **KEY**: confirms the epoch cost of GPU aug (expect ~82-86; compare to baseline 91).
- total_seconds (wall): `grep -aE "^total_seconds:" run.log` — expect ~350-420s (dt-bound).
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect ~8.5-9.5ms (vs 8ms baseline); the GPU-aug dt premium.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare to EXP-054's 0.1968.
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` — grid_sample grid adds a little; expect modest rise from ~454 MB.
