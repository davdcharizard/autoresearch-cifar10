# Plan EXP-047: Ghost BatchNorm — implicit regularization via small-sub-batch statistics (dt-safe)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-047.md

## Milestones

### Milestone 1: GhostBatchNorm2d implemented and dt/CUDA-graph-safe by construction
- [ ] Add a `GhostBatchNorm2d(nn.BatchNorm2d)` subclass to `train.py` (above `BasicBlock`) with `num_splits` arg.
- [ ] Training forward: split ONLY the outer batch dim via `x.view(s, N//s, C, H, W)` (valid view on channels_last — never the channel-fold trick), normalize each group by its own biased mean/var over `(group,H,W)`, apply affine, reshape back to `(N,C,H,W)`. Update `running_mean`/`running_var` IN-PLACE (`.mul_().add_()`) from FULL-batch stats (no buffer reassignment).
- [ ] Eval forward: delegate to standard `F.batch_norm(..., training=False, ...)` (byte-identical to `nn.BatchNorm2d`) so the frozen eval path is unchanged.
- [ ] Swap the 4 BN construction sites to `GhostBatchNorm2d(..., num_splits=GHOST_SPLITS)`: stem `bn1`, `BasicBlock.bn1`, `BasicBlock.bn2`, downsample BN. Add `GHOST_SPLITS = 4` near the hyperparameters.
- [ ] `python -c "import ast; ast.parse(open('train.py').read())"` parses; confirm `_weights_init` still only inits Conv/Linear (GhostBN inherits BN default init), seed/optimizer/schedule/aug all unchanged, `mode="reduce-overhead"` unchanged.

### Milestone 2: Run launched on idle GPU and confirmed healthy
- [ ] `nvidia-smi` → pick an idle GPU (util ~0%, mem <700MiB); both 0/1 idle at plan time.
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` in background.
- [ ] Confirm within ~90s: banner `ResNet-20 | params: 4,299,866` (param count UNCHANGED — GhostBN adds no params), compile completes, loss falling normally (ep1 ~45%, not anomalous), no traceback.

### Milestone 3: dt / epoch verified — THE CRITICAL GATE (de-risks the EXP-042 confound)
- [ ] Extract dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`.
- [ ] Confirm dt steady **~8ms** (≤9ms) and `num_epochs` ≈ baseline ~91. A dt rise (≥~11ms) means the GhostBN reshape/reductions broke the CUDA graph or forced a layout copy → the run is THROUGHPUT-CONFOUNDED (not a clean regularization test) → discard and treat as a code/impl issue to fix (see Abort Criteria), NOT a research no-improvement.

### Milestone 4: Accuracy verified against baseline
- [ ] Extract `best_test_acc`; compare to bar 96.32 (baseline 96.22 + 0.1).

## Code Changes
- **train.py — new `GhostBatchNorm2d` class** (inserted above `BasicBlock`, ~L64):
  ```python
  class GhostBatchNorm2d(nn.BatchNorm2d):
      """BatchNorm computing statistics over `num_splits` disjoint ghost sub-batches
      (Hoffer et al. 2017): noisier per-ghost stats act as an implicit regularizer.
      dt-/CUDA-graph-safe: static shapes (BATCH_SIZE fixed, drop_last=True), the view
      splits ONLY the outer batch dim (valid on channels_last — the channel-fold trick
      is NOT), and running stats update IN-PLACE (no buffer reallocation, cf. EXP-031).
      Eval path is byte-identical to nn.BatchNorm2d (population running stats)."""
      def __init__(self, num_features, num_splits, **kw):
          super().__init__(num_features, **kw)
          self.num_splits = num_splits

      def forward(self, x):
          if self.training:
              N, C, H, W = x.shape
              s = self.num_splits
              xv = x.view(s, N // s, C, H, W)                      # valid view on channels_last
              mean = xv.mean(dim=(1, 3, 4), keepdim=True)          # (s,1,C,1,1) per-ghost mean
              var = xv.var(dim=(1, 3, 4), unbiased=False, keepdim=True)
              xn = (xv - mean) * torch.rsqrt(var + self.eps)
              xn = xn.view(N, C, H, W)
              out = xn * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
              # in-place running-stat update from FULL-batch stats (clean population estimate for eval)
              with torch.no_grad():
                  fmean = x.mean(dim=(0, 2, 3))
                  fvar = x.var(dim=(0, 2, 3), unbiased=True)
                  self.running_mean.mul_(1 - self.momentum).add_(self.momentum * fmean)
                  self.running_var.mul_(1 - self.momentum).add_(self.momentum * fvar)
              return out
          return F.batch_norm(x, self.running_mean, self.running_var,
                              self.weight, self.bias, False, self.momentum, self.eps)
  ```
  - **Why it tests the hypothesis**: per-ghost (size-32) statistics inject normalization noise = an implicit regularizer on the one untouched axis (normalization), throughput-neutral so it dodges the epoch wall. If normalization-noise regularization is a live lever on this generalization-bound net, top-1 rises; else a clean no-improvement closes the axis.
  - **Risks/edge cases**: (a) `N // s` must be integer — 128/4=32 ✓ (every train batch is exactly 128 via `drop_last`); eval uses the else-branch (no split constraint). (b) channels_last: splitting the outer dim is a valid view; the channel-fold trick would force an NCHW copy — explicitly avoided. (c) dt: the manual reductions (per-ghost + full-batch stats) are slightly more work than one cudnn BN; the 3×3 convs dominate so dt should stay ~8ms, but Milestone 3 is a hard gate. (d) eval byte-identical to BN → frozen `Eval.evaluate()` path unchanged.
- **train.py — swap 4 BN sites** to `GhostBatchNorm2d(<channels>, num_splits=GHOST_SPLITS)`: `self.bn1=nn.BatchNorm2d(16)` (stem, L103), `BasicBlock.bn1`/`bn2` (L71/L75), downsample `nn.BatchNorm2d(out_channels)` (L83). Add `GHOST_SPLITS = 4` to the hyperparameter block.
- **NO other changes** — optimizer (SGD+Nesterov), schedule (time-fraction cosine, peak 0.2), aug (TA+Cutout), label smoothing, WD, seed 42, batch 128, `mode="reduce-overhead"` all UNCHANGED.

## Configuration Changes
- `GHOST_SPLITS`: (new) → `4` (ghost size 128/4 = 32; a mild split — Hoffer's regime is large-batch, so at batch 128 a moderate ghost is the sensible first probe, avoiding over-noisy stats).
- No recipe/hyperparameter changes otherwise.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background; harness re-invokes on completion.
- Resources: single idle NVIDIA H20 (shared node, idx 0/1 both idle at plan time). VRAM trivial (≈baseline). Fixed 300s training budget.
- Estimated runtime: ~6-7 min wall (compile ~5-15s startup + 300s training + per-epoch evals). Must be < 10 min.
- Log output: `run.log` in project root. dt lines use `\r` — extract via `tr '\r' '\n'`.
- Tool skill: none (local).

## Abort Criteria
- Loss diverges (NaN/inf) or fails to fall below ~1.0 in the first few epochs.
- Traceback / shape error (esp. a `view`-size/stride error from the channels_last reshape, or a `num_splits` divisibility error) → fix the implementation (code error, single retry) — NOT a research failure.
- **dt steady-state ≥~11ms while the GPU is idle** → GhostBN broke the CUDA graph / forced a layout copy (EXP-042 class). This is an IMPLEMENTATION confound, not a clean result: treat as a code issue (attempt a dt-safe fix within scope, e.g. derive full-batch stats from the group stats to drop the second reduction pass), one retry; if still slow, set Outcome failed and report the dt-confound (do NOT report a confounded accuracy number as no-improvement).
- GPU contention mid-run (dt ≫ 8ms with a foreign process co-resident, per infra-errors.md) → discard as contention-confounded, rerun on idle GPU.
- No `dt:`/epoch-eval output after ~120s (silent hang).
- Total wall-clock approaches 10 min without summary.

## Verification Protocol

### Verification Procedure
Baseline (from experiment index) = **96.22%**; bar = **96.32%** (baseline + 0.1).

1. **Run completes cleanly within budget** — `grep -aE "^best_test_acc:|^training_seconds:|^total_seconds:|^num_epochs:|^num_steps:|^peak_vram_mb:|^num_params:" run.log`. Pass: `best_test_acc` present/non-empty, `total_seconds` < 600, `training_seconds` ≈ 300, `num_params` = 4,299,866 (unchanged — GhostBN adds no params). Empty `best_test_acc` ⇒ crash (`tail -n 50 run.log`). Run timeout: 600s wall.
2. **Throughput-neutrality gate (de-confound)** — dt distribution via `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`; confirm steady ~8ms (≤9ms) and `num_epochs` ≈ ~91. If dt rose, the accuracy comparison is confounded (see Abort Criteria) — do not render a research verdict on a slow run.
3. **Primary necessary condition** — `grep -aE "^best_test_acc:" run.log`. Pass iff `best_test_acc ≥ 96.32`.
4. **No hard-constraint violations** — `git diff --name-only` = `train.py` only; `prepare.py`/eval untouched; `evaluate()` once/epoch (loop structure unchanged); no new deps (GhostBN uses only torch); seed 42 unchanged; no seed hacking (GhostBN is a deterministic architectural change, adds no RNG draws).
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — confirms throughput-neutrality (target ~91 ep).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` (expect ≈ baseline ~491).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — does GhostBN move loss (polish) even if top-1 is flat?
