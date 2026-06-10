# Plan EXP-024: BlurPool / anti-aliased downsampling (Zhang 2019)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] Add a parameter-free `BlurPool2d(nn.Module)` to train.py (fixed 3×3 binomial depthwise kernel as a registered buffer; `F.conv2d(x, filt, stride=stride, padding=1, groups=channels)`).
- [ ] Rewire `BasicBlock` for the downsample case (`stride != 1`): make `conv1` stride-1; add `self.blur = BlurPool2d(out_channels, stride=2)`; in `forward`, apply blur after `relu(bn1(conv1(x)))` before `conv2`. For the projection shortcut, prepend `BlurPool2d(in_channels, stride=2)` before the 1×1 (now stride-1) conv. Non-downsample blocks (`stride==1`) UNCHANGED.
- [ ] `uv run ruff check train.py` clean; `python -c "import ast; ast.parse(open('train.py').read())"` parses.
- [ ] `git diff --name-only` shows only `train.py`.

### Milestone 2: Experiment runs and is confirmed healthy
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background). Confirm clean startup: `params: 4,299,866` (UNCHANGED — blur kernels are buffers, not params), clean compile, no traceback, no NaN, no shape-mismatch.
- [ ] **Record realized epoch count and dt** — the KEY confound check. Compare num_epochs to baseline 91 and dt to baseline 8ms.

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, prints summary block, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `final_test_loss`, `num_epochs`, `total_seconds`, `peak_vram_mb`, `dt` from run.log.

## Code Changes
- **train.py** (the ONLY file modified):
  1. Add `BlurPool2d` module (after imports / near `cutout_batch`):
     ```python
     class BlurPool2d(nn.Module):
         """Anti-aliased downsampling (Zhang 2019): fixed binomial blur then stride-2 subsample. Param-free, depthwise."""
         def __init__(self, channels, stride=2):
             super().__init__()
             self.stride, self.channels = stride, channels
             a = torch.tensor([1.0, 2.0, 1.0])
             k = a[:, None] * a[None, :]
             k = k / k.sum()
             self.register_buffer("filt", k[None, None].repeat(channels, 1, 1, 1))
         def forward(self, x):
             return F.conv2d(x, self.filt, stride=self.stride, padding=1, groups=self.channels)
     ```
  2. `BasicBlock.__init__`: `conv1` stride becomes 1 (was `stride`); add `self.blur = BlurPool2d(out_channels, stride=2) if stride != 1 else None`; shortcut for `stride != 1` becomes `nn.Sequential(BlurPool2d(in_channels, stride=2), nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False), nn.BatchNorm2d(out_channels))` (1×1 now stride-1). The `stride==1 & in!=out` shortcut (none exist in this net, but keep correct) stays a 1×1 stride-1; `stride==1 & in==out` stays Identity.
  3. `BasicBlock.forward`: `out = F.relu(self.bn1(self.conv1(x)))`; `if self.blur is not None: out = self.blur(out)`; `out = self.bn2(self.conv2(out))`; `out += self.shortcut(x)`; `return F.relu(out)`.
  - **Why this tests the hypothesis**: anti-aliases both stride-2 downsample sites (layer2, layer3) per Zhang 2019, restoring shift-invariance to improve generalization — a convergence-neutral mechanism (no params, no stochastic penalty) matching the diagnosed binding constraint.
  - **Risks/edge cases**: (a) **Compute/epoch confound** — conv1 now runs at the pre-subsample (higher) resolution → ~4× FLOPs at the two heaviest convs; may cut epochs below 91 (EXP-015 / capacity pattern). k=4 is launch-bound so it MAY be absorbed — MUST measure epoch count. (b) Spatial alignment verified: 3×3 stride-2 pad-1 gives 32→16→8→4, matching the strided-conv output, so conv-path and shortcut stay aligned (no shape error). (c) `register_buffer` keeps params at 4,299,866. (d) torch.compile must trace the depthwise fixed-kernel conv — standard op, no graph break expected; if compile errors, that's a code-error retry.

## Configuration Changes
- No hyperparameter changes. Structural change only. Unchanged: full EXP-012 recipe (k=4 4.3M params, PEAK_LR 0.2
  cosine-to-0, batch 128, Nesterov, WD 1e-4, LS 0.1, TrivialAugment + Cutout(16), torch.compile, bf16, channels_last, seed 42).

## Execution Environment
- Method: local — `cd <project-root> && CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background launch.
- Resources: 1× NVIDIA H20 (98GB); VRAM may rise modestly (higher-res conv1 activations) but well within budget.
- Estimated runtime: ~390–430s total wall-clock; well under 600s.
- Log output: all stdout/stderr → `run.log` (source of truth).
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf or diverging.
- Traceback / crash at startup (shape mismatch, compile graph break) — fix code error, counts as one retry.
- No new log output for > 3 minutes (silent hang).
- `params` ≠ 4,299,866 at startup (would mean blur kernels leaked into parameters — bug to fix).
- total wall-clock approaching 600s — kill and treat as failure.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22%**; success bar = **96.32%** (+0.1pp).

1. **Baseline**: `bash "/SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh" baseline "/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/.autoresearch/experiment-indices/improve-cifar10-test-accuracy.tsv"` → confirm 96.22.
2. **Cond 1 — primary metric clears bar**: `grep -aE "^best_test_acc:" run.log` → PASS iff `best_test_acc >= 96.32`.
3. **Cond 2 — clean completion within budget**: `best_test_acc` and `total_seconds` present; `grep -ac "Traceback" run.log` == 0; `total_seconds < 600`.
4. **Cond 3 — no constraint violations**: `git diff --name-only` = train.py only; `num_params` == 4,299,866; eval-count (`grep -ac "eval ep" run.log`) == `num_epochs`; no new deps; seed 42 intact.
5. Compare and render verdict. Empty `best_test_acc` ⇒ crashed (`tail -n 50 run.log`).
6. **Confound attribution (mandatory):** record `num_epochs` and `dt`. If `num_epochs` is materially below 91 (≲85), flag the result as compute-confounded (anti-aliasing's added FLOPs cost epochs, à la EXP-015) — a regression is then NOT a clean test of anti-aliasing's merit; note this in the analysis.
7. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — **KEY confound diagnostic** (vs baseline 91).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — vs baseline 0.195 (same LS, so comparable here).
- dt (ms/step): from the step log lines — vs baseline ~8ms (rising dt = compute cost realized).
