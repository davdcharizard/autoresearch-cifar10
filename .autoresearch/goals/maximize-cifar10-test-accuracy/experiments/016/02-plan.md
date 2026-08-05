# Plan EXP-016: Ghost Batch Normalization (regularizing activation-statistic noise)
- **Created**: 2026-06-30

Chosen idea: `01-brainstorm.md` § Chosen Idea + `proposals/idea-01.md`. Reviewed: `01-idea-review.md`.
Baseline (`04-results.tsv`): **96.38** (EXP-008). Bar: best GBN cell ≥ **96.48** (+0.1pp) AND clearly above the SAME-SESSION control (stored 96.38 too weak at the ~0.1pp noise floor).

## Hypothesis (testable)
GhostBatchNorm at ghost_size 64–128 (vs full-batch 512) injects activation-statistic noise — a regularization mechanism orthogonal to the saturated input-aug/weight-decay/label-smoothing/loss-geometry axes — and, composed with the existing weight-EMA, raises `best_test_acc` to ≥96.48 over the same-session control at matched ~150 epochs, ep25 within ~0.5pp of control, fully annealed. If every GBN cell ties at healthy epochs/ep25 → BN-noise regularization is redundant with the existing stack; the ceiling is not regularization-mechanism-movable.

## Milestones

### Milestone 1: Implement GhostBatchNorm2d + correctness smokes
- [ ] Add `import os`; `GHOST_SIZE = int(os.environ.get("GHOST_SIZE", "512"))` (≥batch or ≤0 = standard BN; default 512 = baseline).
- [ ] Add `GhostBatchNorm2d(nn.BatchNorm2d)` module (see Code Changes). Use it in `conv_bn` in place of `nn.BatchNorm2d(c_out)` → `GhostBatchNorm2d(c_out, ghost_size=GHOST_SIZE)` (7 BN sites).
- [ ] Print `ghost_size` in the summary block.
- [ ] **Smoke A — g=512 bypass equivalence (review #2)**: `GHOST_SIZE=512` (and `=0`) GhostBatchNorm2d numerically equals `nn.BatchNorm2d` (max abs diff < 1e-4) on a fixed (512,C,8,8) channels_last bf16 input, in BOTH train (output + running_mean/var after the call) and eval mode.
- [ ] **Smoke B — GHOST PATH math (review #2,#6, the critical one)**: with g=128 on a fixed (512,C,8,8) fp32 input, assert: (i) each ghost group's normalized output (pre-affine, weight=1/bias=0) has per-(group,channel) mean≈0, var≈1; (ii) the manually-updated `running_mean/running_var` EXACTLY match what `nn.BatchNorm2d` would store from the SAME full batch (compare to a reference nn.BatchNorm2d.forward on the same input — full-batch moments + unbiased running-var convention); (iii) output is finite and the reshape grouping uses the batch axis (sanity: g=N gives standard BN, g=N/2 gives 2 groups).
- [ ] **Smoke C — EMA buffer correctness (review #3)**: run 5 train steps with g=128, call `ema_model.update_parameters(model)` each; assert `ema_model.module` BN `running_mean/var` equal the EMA-average of the raw model's (full-batch) running stats (within fp tol), and have shape [C] (not C·splits) so `use_buffers=True` averaging is sound.
- [ ] **Gradient smoke**: one fwd+bwd through the full model with `GHOST_SIZE=128`; assert all conv/BN/γ/β params receive finite grads, loss finite, no NaN.
- [ ] Verify scope (review #8): `git status --short` shows ONLY train.py modified (no untracked files); `git diff --quiet -- prepare.py` (byte-unchanged).

### Milestone 2: Throughput pre-check + mitigation (predict num_epochs cost; review #5)
- [ ] INLINE probe (`uv run python -c`, NO file): build the model, time ~100 fwd+bwd steps at static (512,3,32,32) channels_last/bf16 for GHOST_SIZE=512 vs 128, print img/s for each. The custom path replaces fused cuDNN BN with fp32 reductions/broadcasts/affine across 7 sites — measure the real cost (review #5).
- [ ] Pass criterion: GBN-128 img/s within ~8% of standard (predicts num_epochs ≥ ~140, comparable to the ~149 baseline). 
- [ ] **MITIGATION if >15% slowdown (review #5 — do NOT just caveat):** switch the per-ghost normalization to a FUSED path — `F.batch_norm(xf.reshape(g, C*G, H, W), None, None, None, None, True, 0.0, eps).reshape(N,C,H,W)` (uses cuDNN's fused BN over the ghost-folded view, no running-stat tracking) while KEEPING the manual full-batch running-stat update + C-sized buffers (so eval/EMA stay clean). Re-run smokes A/B/C, re-probe. If STILL >15% after the fused path, apply GBN to only the LATER BN sites (layer2/layer3, where regularization matters most) to cut cost; record the partial-application as a deviation. Only if all mitigations fail do the cells run with an explicit under-anneal confound flag.
- [ ] `nvidia-smi` GPU-1 idle check before the official run (infra-errors EXP-010/014; foreign PID 1723342 intermittent).

### Milestone 3: Run the 3-cell same-session set
- [ ] c0 `GHOST_SIZE=512` → `run_c0.log` (standard-BN control)
- [ ] cA `GHOST_SIZE=128` → `run_cA.log`
- [ ] cB `GHOST_SIZE=64` → `run_cB.log`
- [ ] Each: `CUDA_VISIBLE_DEVICES=1 GHOST_SIZE=... timeout 600 uv run train.py > run_<cell>.log 2>&1`, `nvidia-smi` → `gpu_<cell>.log` before each AND a background `nvidia-smi` sample ~halfway through each cell (review #9 — catch mid-run foreign jobs) appended to `gpu_<cell>.log`.
- [ ] All cells num_epochs in band + total_seconds < 600; equal contention (cross-check num_epochs across cells — a lone low-epoch cell signals contention → re-run full set).

### Milestone 4: Verdict
- [ ] Extract best_test_acc, final_test_acc, num_epochs, total_seconds, ep25 for all cells.
- [ ] PRIMARY (review #1,#10): best GBN cell ≥ **96.48** AND > same-session c0 by a CLEAR margin (>0.1pp, i.e. above the noise floor). A cell that clears 96.48 but is within ~0.1pp of c0 is NOT a clean win.
- [ ] CONFIRMATION for a hairline win (review #10): if the best GBN cell clears both gates but by <~0.15pp over c0, RE-RUN the winning cell + a fresh same-session c0 once more; require the win to replicate before recording `improvement`. (A single max(cA,cB) hairline is otherwise measurement-fishing.)
- [ ] Under-fit/instability diagnosis: ep25 vs c0; best-vs-final trend; num_epochs (read the M2 throughput caveat).
- [ ] ON A WIN: bake winning GHOST_SIZE as the train.py default so bare `uv run train.py` reproduces it.

## Code Changes
- **train.py** (ONLY editable file):
  - `import os`; `GHOST_SIZE = int(os.environ.get("GHOST_SIZE", "512"))`.
  - NEW module (placed near `conv_bn`):
    ```python
    class GhostBatchNorm2d(nn.BatchNorm2d):
        """BN that normalizes over ghost sub-batches of size `ghost_size` during
        training (regularizing activation noise), while keeping CLEAN eval running
        stats updated from the FULL-batch moments — so the AveragedModel(use_buffers)
        EMA averages correct stats and eval is identical to nn.BatchNorm2d.
        ghost_size >= batch (e.g. 512) or non-divisible -> exact standard BN."""
        def __init__(self, num_features, ghost_size=512, **kw):
            super().__init__(num_features, **kw)
            self.ghost_size = ghost_size
        def forward(self, x):
            N = x.shape[0]
            if (not self.training) or self.ghost_size <= 0 or self.ghost_size >= N or N % self.ghost_size != 0:
                return super().forward(x)          # eval / disabled (<=0) / full-batch / g>=N == standard BN
            C, H, W = x.shape[1:]; g = self.ghost_size; G = N // g
            xf = x.float()
            xg = xf.reshape(G, g, C, H, W)
            mean = xg.mean(dim=(1, 3, 4), keepdim=True)
            var = xg.var(dim=(1, 3, 4), keepdim=True, unbiased=False)
            xn = ((xg - mean) * torch.rsqrt(var + self.eps)).reshape(N, C, H, W)
            out = xn * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
            with torch.no_grad():               # clean running-stat update from FULL-batch moments
                m = self.momentum
                self.running_mean.mul_(1 - m).add_(m * xf.mean(dim=(0, 2, 3)))
                self.running_var.mul_(1 - m).add_(m * xf.var(dim=(0, 2, 3), unbiased=True))
                self.num_batches_tracked.add_(1)
            return out.to(x.dtype)
    ```
  - `conv_bn`: `nn.BatchNorm2d(c_out)` → `GhostBatchNorm2d(c_out, ghost_size=GHOST_SIZE)`.
  - Summary: `print(f"ghost_size:       {GHOST_SIZE}")`.
  - Why it tests the hypothesis: single-variable swap of the normalization's TRAIN-time statistic granularity; eval path, optimizer, LR, aug, seed all unchanged. The g=512 default reproduces the exact EXP-008 baseline (regression guard).
  - Eval-stat note (review #4 — no overclaim): updating each layer's running stats from FULL-batch moments removes that layer's per-ghost stat NOISE, but the standard BN train/eval gap remains (and is mildly amplified because downstream layers see ghost-normalized upstream activations during training). This is inherent to GBN, not a bug; eval runs the whole net in eval mode (all running stats), consistent end-to-end. Smoke C checks raw/EMA buffer sanity; the per-epoch EMA eval (already in the loop) is the empirical check that eval stats are healthy.
  - Risks/edges: (a) `reshape` on channels_last fp32 may copy → throughput cost (M2 measures). (b) running var uses unbiased=True to match nn.BatchNorm2d's running-var convention; per-ghost normalization uses biased var (matches BN's batch-norm convention). (c) bf16: stats in fp32 via `xf=x.float()`. (d) drop_last=True ⇒ N=512 always ⇒ G=512/g integer for g∈{64,128}. (e) EMA `use_buffers=True` averages the C-sized running_mean/var — unchanged shapes, so compatible.

## Configuration Changes
- GHOST_SIZE: (new) 512 (=standard BN, default/control) | 128 (4 ghosts) | 64 (8 ghosts). Avoid 32 first (over-reg/instability risk at 150ep; only if ep25 healthy).
- No change to: model topology, optimizer, LR schedule, EMA, TTA, batch 512, aug (Cutout12+RandomErasing), whitening, seed 42.

## Execution Environment
- Method: local; each cell a separate `train.py` process. `CUDA_VISIBLE_DEVICES=1 GHOST_SIZE=<g> timeout 600 uv run train.py > run_<cell>.log 2>&1`.
- Resources: single GPU (H20) on **GPU 1** (`CUDA_VISIBLE_DEVICES=1`, GPU 0 busy). VRAM ~1.6GB.
- Estimated runtime: ~450–520s wall/cell × 3 ≈ 25 min + smokes. (GBN may add modest wall via the reshape; M2 quantifies.)
- Log output: experiments/016/run_c0/cA/cB.log; gpu_<cell>.log.
- Tool skill: none.

## Abort Criteria
- Equivalence smoke fails (g=512 ≠ nn.BatchNorm2d > 1e-3) → implementation bug; fix before running.
- Any cell diverges (test_acc stuck ~10–20% mid-training; ghost too small destabilizes BN) → record, abort that cell.
- num_epochs < ~130 for a GBN cell → heavy under-anneal from GBN cost; note as confound (still record).
- Foreign GPU-1 job appears mid-cell (nvidia-smi) → contention; mark `_contended`, re-run full set when idle.
- Any cell wall ≥ 600s (exit 124) or crash (empty `best_test_acc:`) → infra failure; `tail -50`.

## Verification Protocol

### Verification Procedure
1. Baseline: `exp-index.sh baseline ...` → 96.38; bar = 96.48. Same-session c0 = noise control.
2. Run all 3 cells (Milestone 3).
3. Extract per cell: `grep "^best_test_acc:\|^final_test_acc:\|^num_epochs:\|^total_seconds:\|^ghost_size:" run_<cell>.log`. Empty best ⇒ crash → `tail -50`.
4. ep25 + late trend: `grep "eval ep  25" run_<cell>.log`; last ~5 `eval ep` lines.
5. **Necessary conditions (goal file)**:
   - (a) Completes, within budget, valid best_test_acc, wall < 600s. FAIL → no-improvement/crash.
   - (b) Best GBN cell ≥ **96.48** AND > same-session c0 by a clear margin (>0.1pp). Hairline win (<~0.15pp over c0) requires the M4 confirmation re-run to replicate. FAIL → no-improvement.
   - (c) Integrity (review #8): `git status --short` shows only train.py modified (no untracked code); `git diff --quiet -- prepare.py` (byte-unchanged); ≤1 eval/epoch; seed 42; Smokes A/B/C passed (BN math/eval/EMA correct). FAIL → invalid.
6. Same-session validity: all cells num_epochs in band (note any GBN epoch cost from M2), no contention, wall < 600.
7. ON A WIN: bake winning GHOST_SIZE as default; re-confirm bare `uv run train.py` reproduces within noise before commit.
8. Cleanup: logs stay in experiments/016/; no run.log in repo root.

### Informational Metrics (Optional)
- peak_vram_mb, num_epochs, training_seconds, total_seconds, num_params (unchanged — GBN adds no params): `grep` from logs.
- ep25 test_acc per cell — over-regularization/instability diagnostic.
- GBN throughput cost: img/s GBN-128 vs standard from the M2 probe (predicts epoch impact).
