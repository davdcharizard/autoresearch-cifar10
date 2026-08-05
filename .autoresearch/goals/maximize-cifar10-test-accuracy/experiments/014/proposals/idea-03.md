# Proposal 014-03: Ghost BatchNorm — per-sub-batch normalization noise as a mechanistically-distinct regularizer (the proven DavidNet trick we never adopted)

## Summary
Replace the 10 `nn.BatchNorm2d` layers in `conv_bn` with a hand-written `GhostBatchNorm2d` that, during training, normalizes each forward over independent "ghost" sub-batches of the 512-image mini-batch (try 4 ghosts of 128 — the DavidNet default — and 8 of 64) instead of over all 512. Smaller per-statistic sample → noisier per-ghost mean/var → an extra regularizing perturbation. This is the one MAJOR component of the canonical cifar10-fast/DavidNet recipe this codebase NEVER adopted, and it injects noise in a space (BN activation statistics) orthogonal to every regularizer already tried (input-aug, weight-decay, label-smoothing, loss-geometry). It is throughput-free (a batch-dim reshape, no extra convs). Same-session standard-BN baseline cell is mandatory; pre-register num_epochs ~150.

## Mechanism / Reasoning
Standard `nn.BatchNorm2d` at batch 512 normalizes each channel over all 512×H×W elements, giving a low-variance estimate of the batch mean/var. Ghost BN partitions the 512 images into `g` ghosts and normalizes each ghost by ITS OWN mean/var. With g=4, each statistic is estimated from 128×H×W elements → the per-example normalization is perturbed by sampling noise in the ghost's composition. Each example sees a slightly different, stochastically-rescaled activation distribution every step — a multiplicative/additive noise on the normalized activations that does NOT exist with full-batch BN. Hoffer et al. ("Train longer, generalize better", 2017) introduced this precisely as the regularizer that recovers the generalization gap lost when training at large batch; "Four Things Everyone Should Know to Improve BatchNorm" (arXiv:1906.03548) and "Ghost Noise for Regularizing DNNs" (arXiv:2305.17205) formalize it as injected statistic-noise and report gains under exactly the super-convergence/one-cycle CIFAR regime we run.

The honest crux against the regularization-bound diagnosis: EXP-011 (CutMix), EXP-012 (WD-shaping, LS), EXP-013 (SAM) all tied or lost, and project-insights warns the throughput-free regularization sub-levers "saturate one after another." Ghost BN's claim to break the pattern is that every prior lever perturbed a DIFFERENT space — input pixels (Cutout/RE/CutMix), weights (WD), targets (LS), loss geometry (SAM) — whereas Ghost BN perturbs the BN forward statistics, a channel none of them touched, AND it is the proven recipe component genuinely missing here (not a 2nd-of-a-class). That makes it the most defensible remaining throughput-free probe. The downside is equally honest: the net is already heavily regularized (Cutout12 + RandomErasing + LS0.2 + EMA), and like CutMix, Ghost BN may simply re-regularize a net at its ceiling — depressing early convergence with no annealed-ceiling gain. I estimate this is more-likely-than-not to tie, but it is the highest-information remaining throughput-free test because it closes a real recipe gap.

## Concrete implementation sketch (train.py-specific, pure-torch)
All edits in `train.py`. No new deps — `F.batch_norm` is the only primitive needed.

1. New module, placed above `conv_bn` (line 101):
   ```python
   class GhostBatchNorm2d(nn.BatchNorm2d):
       def __init__(self, c, num_splits=4, **kw):
           super().__init__(c, **kw)
           self.num_splits = num_splits
       def forward(self, x):
           if self.training:
               N, C, H, W = x.shape
               g = self.num_splits
               if N % g != 0:            # safety; drop_last=512 => always divisible
                   g = 1
               # view batch as g independent "batches" stacked on the channel dim
               xr = x.view(N // g, g * C, H, W)
               rm = self.running_mean.repeat(g)
               rv = self.running_var.repeat(g)
               out = F.batch_norm(xr, rm, rv, None, None,
                                  True, self.momentum, self.eps)
               # fold the g per-ghost running-stat updates back into the buffers
               with torch.no_grad():
                   self.running_mean.copy_(rm.view(g, C).mean(0))
                   self.running_var.copy_(rv.view(g, C).mean(0))
               out = out.view(N, C, H, W)
               return out * self.weight[None, :, None, None] + self.bias[None, :, None, None]
           else:
               return F.batch_norm(x, self.running_mean, self.running_var,
                                   self.weight, self.bias, False, self.momentum, self.eps)
   ```
   This is the classic cifar10-fast trick (view `[N, C, H, W] → [N/g, g*C, H, W]`, one fused `F.batch_norm` call, NO Python loop over ghosts → throughput-free). Passing `None` for weight/bias inside the reshaped call avoids per-ghost broadcasting of affine params over the merged `g*C` axis; affine is reapplied after the un-view with the true `[C]` params. `F.batch_norm(..., training=True)` computes per-ghost stats AND writes momentum-EMA updates into the temporary `rm/rv` (length `g*C`); we average the `g` updates back into the real `[C]` buffers so eval running stats stay calibrated.

2. `conv_bn` (line 102): replace `nn.BatchNorm2d(c_out)` with `GhostBatchNorm2d(c_out, num_splits=NUM_SPLITS)`. Add module-level `NUM_SPLITS = int(os.environ.get("NUM_SPLITS","1"))` (1 = standard BN baseline; 4 and 8 = ghost cells). Add `import os`.

3. Running-stat momentum: there are now `g` stat estimates folded per step but we average them (not `g` sequential EMA updates), so the effective momentum per step is unchanged vs standard BN — no momentum retune needed. (The default `momentum=0.1` per-step EMA still applies once per step.) This is the #1 correctness choice and it keeps eval BN well-calibrated: `prepare.py` `Eval` runs `model.eval()` at batch 256, so eval uses `running_mean/var` exclusively — those buffers must reflect the true full-data statistics, which averaging the ghost estimates preserves (mean of unbiased per-ghost means = full-batch mean).

4. Interaction with `AveragedModel(use_buffers=True)` (line 255): EMA averages `running_mean/running_var` into the eval model. Because step 3 keeps the per-step buffer update equivalent to standard BN (one averaged update/step), the EMA'd buffers are calibrated exactly as today. No change to the EMA path. `GhostBatchNorm2d` subclasses `nn.BatchNorm2d`, so `use_buffers` discovers its buffers normally.

5. channels_last/bf16: the `view` ops require contiguous-in-the-viewed-layout tensors. channels_last makes `[N,C,H,W]` non-contiguous in default strides, so `x.view(N//g, g*C, H, W)` may error. Mitigation: use `x.reshape(...)` (copies if needed) OR `x.contiguous(memory_format=torch.contiguous_format)` before the view and restore after. SMOKE-TEST this first — it is the top implementation trap. If reshape forces a copy each BN, verify img/s stays ~26k (the throughput gate).

## Expected effect (quantified)
Literature reports ~0.5–1.0pp from Ghost BN at matched epochs on under-regularized nets. Here the net is already strongly regularized and near its ceiling, so I down-weight that: realistic range is −0.2pp (over-regularization, CutMix-like early-convergence depression with no ceiling gain) to +0.2pp (the missing recipe component closes a real gap). Modal outcome: tie within the ~0.1pp noise floor. g=4 (mild) is likeliest positive; g=8 (stronger noise) risks over-regularization. Watch the ep25 trajectory: a CutMix-like ep25 drop with no tail recovery is the over-regularization signature (EXP-011).

## Risks
- **Over-regularization / tie** (most likely): the net is at its regularization ceiling; added BN-noise re-regularizes redundantly, depressing early convergence like CutMix (EXP-011) with no annealed gain. This is the modal outcome.
- **Running-stat miscalibration breaking eval BN** (#1 correctness trap): eval uses running stats only; if the buffer-update fold is wrong (e.g. forgetting to average the `g` updates, or letting `F.batch_norm` write `g*C` stats that never reach the `[C]` buffers), eval BN is mis-scaled and accuracy collapses. Mitigated by the explicit `.mean(0)` fold; verify with a 1-epoch eval-acc sanity smoke.
- **Throughput regression from the reshape copy** under channels_last (EXP-005 showed layout/shape cost is real). If `reshape` copies each of 10 BN layers per step, epochs could drop <150 → under-anneal confound. num_epochs is the pre-registered gate (≥140 valid); if it drops, the impl isn't throughput-free and the result is confounded.
- **bf16 view/affine dtype mismatch**: `self.weight`/`bias` are fp32 under autocast; the manual affine `out * weight + bias` must broadcast-multiply in the autocast region cleanly. Smoke-test a forward+backward for dtype errors.

## Verification approach
Same-session multi-cell run, GPU 1 (`CUDA_VISIBLE_DEVICES=1`), one process, under `timeout 600`: cell-0 `NUM_SPLITS=1` (standard-BN baseline, isolates the ~0.1pp noise floor), cell-A `NUM_SPLITS=4` (DavidNet default), cell-B `NUM_SPLITS=8`. Pre-register and print num_epochs + img/s per cell (the throughput-free gate — must stay ~150 / ~26k). Before timed runs: a smoke that (i) forward+backward runs under bf16/channels_last without view/dtype errors, (ii) 1-epoch eval acc is sane (not collapsed → running stats calibrated). Win = a ghost cell beats stored 96.38 by ≥0.1pp AND beats same-session cell-0 by >0.1pp. Honest pre-commitment: if both ghost cells land within ±0.1pp of cell-0, record as a clean null closing the Ghost-BN recipe gap — do not re-test other ghost sizes.

## Effort
Medium. The module is ~20 lines but the correctness traps (buffer-update fold, channels_last reshape, bf16 affine) demand careful smoke-testing before the timed 3-cell run. One experiment loop.

## Sources
- Hoffer et al., "Train longer, generalize better: closing the generalization gap in large batch training" (NeurIPS 2017) — introduces Ghost BN as the large-batch generalization regularizer.
- "Four Things Everyone Should Know to Improve BatchNorm" (arXiv:1906.03548) — Ghost BN as statistic-noise regularization, throughput-free.
- "A New Look at Ghost Normalization" (arXiv:2007.08554); "Ghost Noise for Regularizing Deep Neural Networks" (arXiv:2305.17205) — ~1% gains, helps under super-convergence on CIFAR-10.
- `knowledge/references/fast-cifar10-recipes.md` — DavidNet/cifar10-fast splits batch 512 into ghost batches (the `[N/g, g*C, H, W]` reshape trick); davidcpage/cifar10-fast.
- `train.py`: `conv_bn` (101-106, 10 BN layers), batch 512 / `drop_last=True` (219-228), `AveragedModel(use_buffers=True)` EMA of BN buffers (255-257), bf16 autocast + channels_last (230, 294-301), eval via `prepare.py` `Eval` at `model.eval()`/batch 256 (running-stats-only path).
- 03-experiment-learnings / project-insights: regularization-bound ceiling, throughput-free sub-lever saturation (EXP-011/012/013), ~0.1pp noise floor, num_epochs-first under-anneal gate.
