# Plan EXP-017: Throughput-free BN-affine noise (cheap GhostBN surrogate, layer3-first)
- **Created**: 2026-06-30

Chosen idea: `01-brainstorm.md` § Chosen Idea + `proposals/idea-01.md`. Review: `01-idea-review.md` (Codex pick, 8/10). Baseline (`04-results.tsv`): **96.38** (EXP-008). Bar: best noise cell ≥ **96.48** (+0.1pp) AND > same-session c0 by a CLEAR >0.1pp margin (stored 96.38 too weak at the ~0.1–0.2pp floor; EXP-016's c0 drew low at 96.14).

## Hypothesis (testable)
EXP-016 showed layer3 GhostBN beat its same-session control +0.24pp (96.38 vs 96.14) at 16 FEWER epochs — the first positive regularization signal on this goal — capped only because ghosting breaks the fused channels_last BN kernel (~50% slower → under-anneal). GhostBN's regularization is algebraically a per-(sample,channel) jitter of the normalization (scale σ/σ̃ + shift (μ−μ̃)/σ̃). `NoisyBN` injects that jitter as ONE elementwise op on the fused-BN output → fused kernel preserved → ~150 epochs (no under-anneal). Prediction: at the calibrated ghost-equivalent σ, applied at the layer3 sites (the EXP-016 signal site), best_test_acc reaches ≥96.48 and clears the same-session c0 by >0.1pp at matched ~149 epochs (confirmed by a mandatory replication run). NULL (softened per plan-review #6): if all cells tie at healthy epochs/ep25, THIS throughput-free activation-noise form does not move the ceiling — which does NOT fully close the GhostBN axis, because the surrogate drops GhostBN's group-shared, data-dependent stat-error structure (only the magnitude is matched). A tie therefore demotes (not eliminates) the BN-noise hypothesis: the definitive closer would be the faithful compile-funded layer3 GhostBN (EXP-014 throughput recipe), and otherwise the next loop pivots to an inductive-bias change (BlurPool/SE). A confirmed WIN is a genuine improvement attributable to throughput-free BN/activation noise.

## Milestones

### Milestone 1: Implement NoisyBN + calibrate σ + correctness smokes
- [ ] Add `BN_NOISE = float(os.environ.get("BN_NOISE", "0.0"))` and `BN_NOISE_MIN_CH = int(os.environ.get("BN_NOISE_MIN_CH", "0"))` (`import os` already added in EXP-016? NO — train.py reverted to baseline; ADD `import os`).
- [ ] Add `NoisyBN(nn.BatchNorm2d)` (see Code Changes); use it in `conv_bn` in place of `nn.BatchNorm2d(c_out)` → `NoisyBN(c_out, noise=BN_NOISE, min_ch=BN_NOISE_MIN_CH)`.
- [ ] Print `bn_noise` and `bn_noise_min_ch` in the summary block.
- [ ] **Smoke A — σ=0 / eval bypass equivalence**: with `BN_NOISE=0` (and with `BN_NOISE=0.1` but in `.eval()` mode), `NoisyBN` output + running_mean/var EXACTLY equal `nn.BatchNorm2d` (max abs diff 0.0) on a fixed (512,C,8,8) channels_last bf16 input — noise is train-only and σ=0 is the exact EXP-008 baseline (regression guard).
- [ ] **Smoke B — noise statistics**: with `BN_NOISE=0.1, min_ch=0` in train mode, assert: output finite; mean over many draws of `(NoisyBN(x)−BN(x))` ≈ 0 (unbiased); the perturbation acts on the NORMALIZED part — at channels where β≈0 the mul term ≈ σ·|y|, and crucially a channel's β is recoverable unchanged (set a known β, verify the noise-free expectation of NoisyBN equals BN, i.e. β not jittered); per-(sample,channel) granularity (constant across spatial within a sample/channel); running_mean/var identical to a reference nn.BatchNorm2d on the same input (noise never touches buffers).
- [ ] **Smoke C — per-layer EMA buffer cleanliness (+ network-shift caveat)**: 5 train steps with `BN_NOISE=0.1`, `ema_model.update_parameters` each; assert `ema_model.module` BN running_mean/var equal the EMA-average of the raw model's running stats, shape [C]. NOTE (plan-review #2): this checks PER-LAYER buffer math only; the network-level train/eval stat shift (downstream BN sees noised activations in training) is INHERENT to activation noise (as in GhostBN/dropout) and is NOT asserted away — the empirical check is the per-epoch EMA eval curve in M3 (healthy ep25, best≈final).
- [ ] **Gradient smoke**: one fwd+bwd through the full model with `BN_NOISE=0.1`; assert finite loss + all grads finite for params with `requires_grad=True` ONLY (EXCLUDE the frozen whitening conv, plan-review #7).
- [ ] **CALIBRATION smoke (review #1,#3)**: on REAL CIFAR batches (average over ≥4 batches; init model — stat structure is architecture-determined, treat as ballpark), at EACH layer3 BN INPUT (3 sites, C=512: conv_bn(256→512)@8×8 + Residual's two conv_bn(512→512)@4×4), split the batch into ghost groups of g=128 (4 groups) and measure across (group,channel): `σ_add* = std of (μ_ghost−μ_full)/σ_full` and `σ_mul* = std of (σ_full/σ_ghost − 1)` (CORRECTED form — the normalization rescales by σ_full/σ_ghost). Report per-site + mean of each. σ_add* and σ_mul* should be similar magnitude (~0.01–0.03); set `σ_cal ≈ max(σ_add*, σ_mul*)` (so the single scalar covers both). If they differ >2×, note it and consider per-component noise (NoisyBN supports `mul`/`add` toggles). Expected ~0.01–0.03 → confirms the brainstorm's 0.1/0.2 was too strong.
- [ ] Verify scope: `git status --short` shows ONLY train.py modified (no untracked files); `git diff --quiet -- prepare.py`.

### Milestone 2: Throughput pre-check (confirm epoch-neutrality; review #2)
- [ ] INLINE probe (`uv run python -c`, NO file): time ~100 fwd+bwd steps at static (512,3,32,32) channels_last/bf16 for `BN_NOISE=0` vs `BN_NOISE=0.1, min_ch=512` (layer3) vs `BN_NOISE=0.1, min_ch=0` (all-site). Print img/s for each.
- [ ] Pass criterion: layer3 noise img/s within ~5% of standard (predicts num_epochs ≥ ~142). RNG draws are [512,C,1,1] (small, broadcast over spatial) + 1–2 full-size elementwise ops/site; expect near-zero slowdown. MITIGATION if >8% (plan-review #4): (a) layer3-only already minimizes to 3 sites; (b) drop the additive term (`add=False`, mul-only) to halve the elementwise ops; (c) if STILL >8%, the cell runs with an explicit under-anneal confound flag (but this is very unlikely for pure elementwise noise). Re-probe after any mitigation.
- [ ] `nvidia-smi` GPU-1 idle check before the official run (infra-errors: foreign PID intermittent; log before EVERY cell).

### Milestone 3: Run the 3-cell same-session set
- [ ] c0 `BN_NOISE=0` → `run_c0.log` (standard-BN control).
- [ ] cA `BN_NOISE=<σ_cal> BN_NOISE_MIN_CH=512` → `run_cA.log` (layer3, calibrated ghost-equivalent — PRIMARY, the faithful epoch-neutral EXP-016 reproduction).
- [ ] cB `BN_NOISE=<~2.5×σ_cal> BN_NOISE_MIN_CH=512` → `run_cB.log` (layer3, stronger — maps the noise-strength response; checks whether more-than-ghost noise helps).
- [ ] σ_cal from M1 calibration (fallback if calibration unavailable: cA=0.03, cB=0.08). Record the chosen σ values in 03-execute.md before running.
- [ ] Each: `CUDA_VISIBLE_DEVICES=1 BN_NOISE=... BN_NOISE_MIN_CH=... timeout 600 uv run train.py > run_<cell>.log 2>&1`; `nvidia-smi` → `gpu_<cell>.log` before each + a mid-run sample appended (catch mid-session contention, infra-errors EXP-014).
- [ ] All cells num_epochs in band (~142–155) + total_seconds < 600; cross-check num_epochs across cells (a lone low-epoch cell = contention → re-run full set when GPU 1 idle).

### Milestone 4: Verdict
- [ ] Extract best_test_acc, final_test_acc, num_epochs, total_seconds, ep25 for all cells.
- [ ] PRIMARY (BOTH gates, review #7): best noise cell ≥ **96.48** AND > same-session c0 by a CLEAR margin (>0.1pp). A cell clearing 96.48 but within ~0.1pp of c0 is NOT a clean win.
- [ ] **MANDATORY CONFIRMATION (plan-review #5)**: this goal has a ~0.2pp noise floor and 10 prior no-improvements → a single-run win is too vulnerable to a low-c0 / high-cell draw (EXP-016's c0 drew low by 0.24pp). For ANY apparent win (best cell clears both gates), RE-RUN the winning cell + a fresh same-session c0 once more; record `improvement` ONLY if the win replicates (best cell ≥96.48 AND > the new c0 by >0.1pp in BOTH runs). A win in run 1 that evaporates in run 2 → no-improvement.
- [ ] Under-fit/over-reg diagnosis: ep25 vs c0 (σ too high → ep25 drops); best-vs-final (annealed?); num_epochs (throughput-free check).
- [ ] ON A WIN: bake winning BN_NOISE / BN_NOISE_MIN_CH as the train.py defaults so bare `uv run train.py` reproduces it.

## Code Changes
- **train.py** (ONLY editable file):
  - `import os` (top, with other stdlib imports).
  - `BN_NOISE = float(os.environ.get("BN_NOISE", "0.0"))`; `BN_NOISE_MIN_CH = int(os.environ.get("BN_NOISE_MIN_CH", "0"))` in the hyperparameter block (default 0.0 → exact EXP-008 baseline).
  - NEW module (placed near `conv_bn`):
    ```python
    class NoisyBN(nn.BatchNorm2d):
        """Standard fused BatchNorm2d, plus TRAIN-only per-(sample,channel) Gaussian
        jitter of the NORMALIZED activation (NOT the affine shift beta) — a
        throughput-free surrogate for GhostBN's regularizing statistic noise (EXP-016)
        that keeps the fused channels_last kernel (no under-anneal). Division-free
        reconstruction: gamma*x_hat = y - beta, so perturbing x_hat by
        (1+s*e_mul)+s*e_add gives y' = y + (y-beta)*s*e_mul + gamma*s*e_add — beta is
        untouched (faithful to GhostBN, which perturbs only the normalization).
        noise<=0 or num_features<min_ch -> exact standard BN. NOTE: this is a SURROGATE
        — the noise is independent per-sample Gaussian, NOT GhostBN's group-shared,
        data-dependent stat error; magnitude is matched via the M1 calibration, the
        correlation structure is not."""
        def __init__(self, num_features, noise=0.0, min_ch=0, mul=True, add=True, **kw):
            super().__init__(num_features, **kw)
            self.noise = float(noise)
            self.active = self.noise > 0.0 and num_features >= min_ch
            self.mul = mul; self.add = add
        def forward(self, x):
            y = super().forward(x)                 # fused full-batch BN (+affine); eval-safe, buffers updated noise-free
            if self.training and self.active:
                C = y.shape[1]
                shp = (y.shape[0], C, 1, 1)
                beta = self.bias.view(1, C, 1, 1)
                if self.mul:                       # perturb normalized scale: (y-beta) = gamma*x_hat
                    y = y + (y - beta) * (self.noise * torch.randn(shp, device=y.device, dtype=y.dtype))
                if self.add:                       # perturb normalized shift, gamma-scaled
                    gamma = self.weight.view(1, C, 1, 1)
                    y = y + (self.noise * gamma) * torch.randn(shp, device=y.device, dtype=y.dtype)
            return y
    ```
  - `conv_bn`: `nn.BatchNorm2d(c_out)` → `NoisyBN(c_out, noise=BN_NOISE, min_ch=BN_NOISE_MIN_CH)`.
  - Summary: `print(f"bn_noise:         {BN_NOISE}")` and `print(f"bn_noise_min_ch:  {BN_NOISE_MIN_CH}")`.
  - Why it tests the hypothesis: a single-variable, throughput-free injection of EXP-016-magnitude normalization noise; eval path, optimizer, LR, aug, seed, EMA all unchanged. `BN_NOISE=0` reproduces the exact EXP-008 baseline (regression guard, Smoke A).
  - Noise form (design note, addressing plan-review #1): noise perturbs ONLY the normalized activation x̂ (β untouched) via the division-free `y + (y−β)·σε_mul + γ·σε_add` form — faithful to GhostBN's "perturb the normalization, not the affine" structure. It remains a SURROGATE in two ways the M1 calibration does NOT fix: (i) independent per-(sample,channel) Gaussian vs GhostBN's per-(group,channel) SHARED, data-dependent error; (ii) the inherent downstream train/eval stat shift (downstream BN sees noised activations in training; see Smoke C note). These are acknowledged, not claimed away — so a TIE constrains "this activation-noise form" not the entire GhostBN axis (plan-review #6).
  - Risks/edges: (a) σ too large → over-regularize/under-fit (mitigated: σ calibrated to ghost-equivalent ~0.01–0.03, cB only ~2.5×). (b) RNG in the autograd graph — gradient smoke checks finiteness (trainable params only). (c) bf16: draw noise in y.dtype — fine, a regularizer not a precision-critical stat. (d) EMA `use_buffers=True` averages C-sized running_mean/var; each NoisyBN updates ITS buffers noise-free (super().forward, before noise) so per-layer buffers are clean, but the NETWORK-level train/eval stat shift (downstream layers) is inherent to any activation-noise method (same as GhostBN/dropout) — the per-epoch EMA eval curve (best≈final, healthy ep25) is the empirical check, NOT a claim of network-level identity to nn.BatchNorm2d.

## Configuration Changes
- BN_NOISE: (new) 0.0 (=standard BN, default/control) | σ_cal (cA, calibrated layer3 ghost-equivalent, ~0.01–0.03) | ~2.5×σ_cal (cB).
- BN_NOISE_MIN_CH: (new) 0 (all sites) | 512 (layer3-only — cA/cB use 512, the proven EXP-016 site).
- No change to: model topology, optimizer, LR schedule, EMA, TTA, batch 512, aug (Cutout12+RandomErasing), whitening, seed 42.

## Execution Environment
- Method: local; each cell a separate `train.py` process. `CUDA_VISIBLE_DEVICES=1 BN_NOISE=<σ> BN_NOISE_MIN_CH=<m> timeout 600 uv run train.py > run_<cell>.log 2>&1`.
- Resources: single GPU (H20) on **GPU 1** (`CUDA_VISIBLE_DEVICES=1`, GPU 0 busy). VRAM ~1.6GB.
- Estimated runtime: ~450s wall/cell × 3 ≈ 23 min + smokes. (Noise is throughput-free → no wall inflation beyond RNG.)
- Log output: experiments/017/run_c0/cA/cB.log; gpu_<cell>.log.
- Tool skill: none.

## Abort Criteria
- σ=0 ≠ nn.BatchNorm2d (Smoke A diff > 0) → implementation bug; fix before running.
- Any cell diverges (test_acc stuck ~10–20% mid-training; σ too large destabilizes) → record, abort that cell, lower σ.
- num_epochs < ~142 for a noise cell → unexpected throughput cost (RNG); note confound (still record), prefer layer3-only.
- Foreign GPU-1 job appears mid-cell (nvidia-smi) → contention; mark `_contended`, re-run full set when idle (infra-errors EXP-010/014).
- Any cell wall ≥ 600s (exit 124) or crash (empty `best_test_acc:`) → infra failure; `tail -50`.

## Verification Protocol

### Verification Procedure
1. Baseline: `exp-index.sh baseline ...` → 96.38; bar = 96.48. Same-session c0 = noise control.
2. Run all 3 cells (Milestone 3).
3. Extract per cell: `grep "^best_test_acc:\|^final_test_acc:\|^num_epochs:\|^total_seconds:\|^bn_noise:\|^bn_noise_min_ch:" run_<cell>.log`. Empty best ⇒ crash → `tail -50`.
4. ep25 + late trend: `grep "eval ep  25 " run_<cell>.log`; last ~5 `eval ep` lines.
5. **Necessary conditions (goal file)**:
   - (a) Completes, within budget, valid best_test_acc, wall < 600s. FAIL → no-improvement/crash.
   - (b) Best noise cell ≥ **96.48** AND > same-session c0 by a clear margin (>0.1pp), REPLICATED across the mandatory M4 confirmation run (winning cell + fresh c0 re-run; win must hold in both). FAIL or non-replication → no-improvement.
   - (c) Integrity: `git status --short` shows only train.py modified; `git diff --quiet -- prepare.py` (byte-unchanged); ≤1 eval/epoch; seed 42; Smokes A/B/C passed (σ=0≡baseline, noise unbiased, eval/EMA buffers clean). FAIL → invalid.
6. Same-session validity: all cells num_epochs in band (~142–155), no contention, wall < 600.
7. ON A WIN: bake winning BN_NOISE/BN_NOISE_MIN_CH as defaults; re-confirm bare `uv run train.py` reproduces within noise before commit.
8. Cleanup: logs stay in experiments/017/; no run.log in repo root.

### Informational Metrics (Optional)
- peak_vram_mb, num_epochs, training_seconds, total_seconds, num_params (unchanged — NoisyBN adds no params): `grep` from logs.
- ep25 test_acc per cell — over-regularization/under-fit diagnostic (σ too high → ep25 drops vs c0).
- Calibrated σ_add*/σ_mul* per layer3 site (M1) — the GhostBN(g=128)-equivalent noise EXP-016 injected; interpret cA's σ against it.
