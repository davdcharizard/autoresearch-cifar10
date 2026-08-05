# Plan EXP-018: BlurPool anti-aliased downsampling (MaxBlurPool) at layer1/2/3

- **Created**: 2026-06-30

## Summary & Hypothesis
Replace the naive `nn.MaxPool2d(2)` subsampling at layer1/2/3 with **MaxBlurPool** (dense max at stride 1 → fixed binomial blur, depthwise, stride 2) to restore approximate shift-equivariance. Per Zhang 2019 (ICML, arXiv:1904.11486), anti-aliased downsampling acts as effective regularization that *raises clean accuracy* — the first **architectural-inductive-bias** lever on this goal, after capacity/epochs/optimizer/all-regularization axes saturated (12 straight no-improvements; generalization-ceiling diagnosis, project-insights High EXP-014). **Hypothesis**: a blurpool cell reaches best_test_acc ≥ baseline+0.1pp (≥96.48) AND beats the same-session standard-MaxPool control c0 by a clear >0.1pp margin at near-full epochs (≥~135), replicated on a confirmation re-run. A tie at healthy epochs/ep25 demotes the downsampling inductive bias as ceiling-moving on this small-image, strongly-augmented net at 300s.

Baseline (from `04-results.tsv`): **96.38** (EXP-008, commit 07c3760), metric best_test_acc (%), higher is better.

## Milestones

### Milestone 1: Implementation + correctness/throughput smokes (no official run yet)
- [ ] Add `import os`; add env reads `BLUR_KSIZE` (int, default 0 = baseline `nn.MaxPool2d`) and `BLUR_LAYERS` (str, default "123").
- [ ] Add `BlurPool2d`, `MaxBlurPool2d`, and `make_pool(channels, layer_id)` helper to `train.py` (before `ResNet9`).
- [ ] Wire `make_pool(128,1)/(256,2)/(512,3)` into `layer1/2/3` in place of `nn.MaxPool2d(2)`; leave the final `nn.MaxPool2d(4)` head UNCHANGED (reviewer concern #3).
- [ ] Add `blur_ksize` / `blur_layers` to the summary prints.
- [ ] **Smoke A (regression)**: `BLUR_KSIZE=0` → model uses `nn.MaxPool2d(2)` exactly; intermediate spatial sizes 16/8/4 and logits [N,10]; num_params == 7,784,627 (unchanged). PASS required.
- [ ] **Smoke B (shape parity)**: `BLUR_KSIZE∈{2,3,5}` → intermediate spatial sizes after layer1/2/3 are EXACTLY 16/8/4 (match baseline) for a dummy [2,3,32,32]; logits [2,10]. PASS required.
- [ ] **Smoke C (kernel correctness)**: each blur buffer sums to 1.0 (DC-preserving); buffer shape [C,1,k,k]; buffer is NOT in `model.parameters()` (no optimizer entry, no grad). PASS required.
- [ ] **Smoke D (train backward + memory-format)**: one fwd+bwd under `autocast(bf16)` + `channels_last` with `BLUR_KSIZE=3` → finite loss, finite grads on all `requires_grad=True` params (whitening conv excluded), blur buffer `.grad is None`. PASS required.
- [ ] **Smoke F (eval / flip-TTA / EMA coverage)** (reviewer concern #6): with `BLUR_KSIZE=3`, build the model + an `AveragedModel(use_buffers=True)` EMA wrapper. (i) `model.eval()` in **native fp32, NO autocast** → finite logits [N,10] for both an input and its `.flip(-1)` mirror (covers the flip-TTA eval path); (ii) after one `ema.update_parameters(model)`, assert every EMA blur buffer equals the raw model's blur buffer within 1e-6 (constant buffer invariance, validates the `train.py:75-78`-class EMA claim); (iii) eval logits are finite under both channels_last and contiguous. PASS required.
- [ ] **Smoke E (M2 throughput probe — FULL train step)** (reviewer concern #3): time img/s over ~200 **complete training steps** that mirror `train.py`'s loop exactly — DataLoader batch → `.to(channels_last)` → `autocast(bf16)` forward → CE loss → `loss.backward()` → `optimizer.step()` → `torch.cuda.synchronize()` (and the EMA `update_parameters` once past warmup) — NOT forward-only, so the BlurPool backward + `F.pad` overhead is counted. Run for c0 (ksize0), cA (ksize3,"123"), cB (ksize2,"123") at the real (512,3,32,32) shape; predict num_epochs = round(c0_epochs × c0_imgps/cell_imgps using the clean ~149 c0 anchor). Record. **GATE: if predicted cA epochs < 135, set cA `BLUR_LAYERS=12` (drop the small-spatial layer3 blur) and re-probe**; the official cells use whatever config clears the ≥135 prediction.

### Milestone 2: Official same-session 3-cell run on GPU 1
Log dir shorthand `D = .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/018` (all paths from project root; reviewer concern #7 — single consistent location).
- [ ] Confirm GPU 1 idle via `nvidia-smi` (log to `$D/gpu_c0.log`) — abort/wait if a foreign multi-GB job is present (infra-errors: mid-session contention confounds).
- [ ] For EACH cell, start a **background nvidia-smi sampler** (e.g. `nvidia-smi --query-gpu=utilization.gpu,memory.used -i 1 --format=csv -l 5 > $D/gpu_<cell>.log &`) for the cell's duration and stop it after, so mid-run contention is captured (reviewer concern #4); cross-check the cell's printed img/s for mid-run step-time drift.
- [ ] c0 (control): `CUDA_VISIBLE_DEVICES=1 BLUR_KSIZE=0 timeout 600 uv run train.py > $D/run_c0.log 2>&1`
- [ ] cA (PRIMARY): `CUDA_VISIBLE_DEVICES=1 BLUR_KSIZE=3 BLUR_LAYERS=<123 or 12 per M1 gate> timeout 600 uv run train.py > $D/run_cA.log 2>&1`
- [ ] cB: `CUDA_VISIBLE_DEVICES=1 BLUR_KSIZE=2 BLUR_LAYERS=<same as cA> timeout 600 uv run train.py > $D/run_cB.log 2>&1`
- [ ] Record best_test_acc, num_epochs, training_seconds, peak_vram_mb, ep25 for each cell.

### Milestone 3: Verdict + (conditional) confirmation re-run
- [ ] Compare best blurpool cell vs same-session c0 and the 96.48 bar.
- [ ] **If any apparent win** (cell ≥96.48 AND >c0+0.1pp): MANDATORY confirmation re-run of c0 + the winning cell in a fresh same-session pair; require the win to replicate (low-c0-draw lesson, EXP-016/017). Only then → improvement.
- [ ] Verify equal conditions: num_epochs within the clean ~135–154 band, all cells equally (un)contended (gpu_*.log), prepare.py byte-unchanged.

## Code Changes
- **train.py** (sole editable file):
  - **Imports/config** (top): `import os`; `BLUR_KSIZE = int(os.environ.get("BLUR_KSIZE", "0"))`; `BLUR_LAYERS = os.environ.get("BLUR_LAYERS", "123")`. Default `BLUR_KSIZE=0` ⇒ exact EXP-008 baseline (regression-safe).
  - **New modules** (before `ResNet9`, after `conv_bn`/`Residual`/`GatedResidual`):
    ```python
    class BlurPool2d(nn.Module):
        """Fixed binomial low-pass blur, depthwise, stride 2 (Zhang 2019 anti-aliasing)."""
        def __init__(self, channels, ksize):
            super().__init__()
            coeffs = {2: [1., 1.], 3: [1., 2., 1.], 5: [1., 4., 6., 4., 1.]}[ksize]
            k = torch.tensor(coeffs)
            k2 = k[:, None] * k[None, :]
            k2 = k2 / k2.sum()                                   # DC-preserving (sum=1)
            self.register_buffer("kernel", k2[None, None].repeat(channels, 1, 1, 1))  # [C,1,k,k]
            l = (ksize - 1) // 2
            self.pad = (l, ksize - 1 - l, l, ksize - 1 - l)      # symmetric for odd, [0,1] for k=2
            self.channels = channels
        def forward(self, x):
            x = F.pad(x, self.pad, mode="reflect")
            return F.conv2d(x, self.kernel, stride=2, groups=self.channels)  # dtype via autocast

    class MaxBlurPool2d(nn.Module):
        """Anti-aliased drop-in for nn.MaxPool2d(2): dense max (stride 1) then blur-subsample."""
        def __init__(self, channels, ksize):
            super().__init__()
            self.mp = nn.MaxPool2d(2, stride=1)
            self.blur = BlurPool2d(channels, ksize)
        def forward(self, x):
            return self.blur(self.mp(x))

    def make_pool(channels, layer_id):
        if BLUR_KSIZE > 0 and str(layer_id) in BLUR_LAYERS:
            return MaxBlurPool2d(channels, BLUR_KSIZE)
        return nn.MaxPool2d(2)
    ```
  - **ResNet9.__init__**: `nn.MaxPool2d(2)` → `make_pool(128, 1)` / `make_pool(256, 2)` / `make_pool(512, 3)` in layer1/2/3. Final `self.pool = nn.MaxPool2d(4)` UNCHANGED.
  - **Summary prints**: add `print(f"blur_ksize: {BLUR_KSIZE} | blur_layers: {BLUR_LAYERS}")`.
  - **Why**: this is the minimal change that swaps the aliasing subsampling for an anti-aliased one, toggled by env so c0 reproduces the exact baseline and a single code path serves all cells. The blur is a fixed buffer (0 trainable params, excluded from optimizer/grad automatically), so the comparison isolates the inductive-bias change, not capacity.
  - **Risks/edge cases**:
    - *Throughput* (#1 failure mode): 3 depthwise blurs + `F.pad` at spatial 32/16/8. Standard convs → NO fused-BN-kernel break (unlike GhostBN EXP-016). Gated by M1 Smoke E (≥135 ep) with a layer1/2-only fallback. Per EXP-014, ±10ep near 150 ≈ 0.02–0.03pp (below bar) → small cost is non-confounding and works *against* the blur cells (a >0.1pp win despite fewer epochs is strong).
    - *dtype*: buffer registered fp32 (like every other conv weight); under `autocast(bf16)` `F.conv2d` casts both args — NO manual `.to(x.dtype)` per forward (reviewer concern #2). In eval (native dtype) input & buffer are both fp32 → match. Mirrors how the existing fp32 conv weights are handled.
    - *reflect pad < input dim*: smallest map is layer3 maxpool-stride1 output 7×7; max pad is ksize=5→2 (<7) ✓.
    - *EMA*: `AveragedModel(use_buffers=True)` averages the constant kernel buffer → stays constant (no drift). flip-TTA valid (symmetric blur commutes with horizontal flip up to the reflect pad).
    - *channels_last*: 4D buffer [C,1,k,k] takes channels_last with `model.to(memory_format=...)`; depthwise conv preserves format.

## Configuration Changes
- `BLUR_KSIZE`: 0 → {0 (c0), 3 (cA), 2 (cB)} (env; 0 = baseline MaxPool, 3 = triangle filter [1,2,1] canonical default, 2 = rect [1,1] lightest). Rationale: ksize=3 is Zhang's default best; ksize=2 (Rect-2) is the lightest filter Zhang shows still helps — maps the smoothing-strength response and is the cheapest (throughput safety).
- `BLUR_LAYERS`: "123" → "123" (default) or "12" (M1 gate fallback if ksize3/all-layers predicts <135 ep). Rationale: layer1/2 are the large-spatial sites where aliasing is most severe; dropping the layer3 (8×8) blur is the cheapest throughput recovery and keeps the dominant anti-aliasing effect.
- No other hyperparameters change (PEAK_LR 0.4, WD 5e-4, LS 0.2, EMA 0.998, etc. all held) — single-variable test of the downsampling operator.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=1 [env] timeout 600 uv run train.py > run_cX.log 2>&1` from project root, run sequentially in background (each ~7–8 min wall).
- Resources: single NVIDIA H20, **GPU 1** (GPU 0 busy — hard constraint); ~1.6 GB VRAM (blur adds none).
- Estimated runtime: ~3 cells × ~7.5 min ≈ 23 min + smokes (~3 min) + possible confirmation pair (~15 min).
- Log output: per-cell `experiments/018/run_c0.log`/`run_cA.log`/`run_cB.log`; `gpu_c0.log`/`gpu_cA.log`/`gpu_cB.log` (nvidia-smi before each cell). Metrics via `grep "^best_test_acc:\|^num_epochs:\|^training_seconds:\|^peak_vram_mb:\|blur_ksize:" run_cX.log` and the `eval ep 25` line for ep25.
- Tool skill: none (local run).

## Abort Criteria
- Foreign multi-GB job on GPU 1 at cell start OR appearing mid-cell (background `gpu_*.log` sampler, reviewer concern #4) → discard the affected same-session set and re-run the FULL set when GPU 1 is idle (infra-errors: same-session controls only hold if ALL cells equally contended).
- Any cell's `num_epochs` drops below **135** (under-anneal / unflagged contention) → discard that same-session set; if GPU 1 was idle (genuine throughput cost), restrict cA/cB to `BLUR_LAYERS=12` and re-run the full set; if contended, re-run as-is when idle.
- NaN/inf loss or eval acc stuck ~10% (random) for >25 epochs → kill, inspect (shape/padding bug); do not retry blindly.
- Smoke A/B/C/D/F failure → fix code before any official run (no run on a failing smoke).
- Wall-clock > 600s on any cell → `timeout` kills it = failure for that cell.

## Verification Protocol

### Verification Procedure
Baseline = **96.38** (from `exp-index.sh baseline`). `D` = the experiments/018 log dir (above). Necessary conditions (goal file), evaluated in order; STOP at first failure — EXCEPT the integrity cross-checks in (a0), which run REGARDLESS of the accuracy outcome (reviewer concern #1: catch metric gaming even on a "win").

0. **(a0) Metric integrity / anti-gaming (run ALWAYS, before trusting any number)** (reviewer concern #1): for each cell, assert (i) **exactly one** `^best_test_acc:` summary line; (ii) the number of `eval ep` lines == `num_epochs` (≤1 eval/epoch — no extra evals); (iii) the summary `best_test_acc` **equals `max(...)` of the per-epoch `test_acc` values** parsed from the `eval ep ... test_acc:` lines (`train.py:349-356` vs the summary at `:372-382`) — a summary inflated above the realized per-epoch max ⇒ `invalid`; (iv) `git diff --quiet -- prepare.py` (eval harness byte-unchanged) and `git status --porcelain` shows only `train.py` modified; (v) seed 42 unchanged; (vi) `num_params` == 7,784,627 (blur is a buffer, not a param). Any failure here ⇒ `invalid` (overrides any apparent accuracy win).
1. **(a) Completes within budget, valid metric, wall < 600s, fully annealed**: for each cell, `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:\|^num_epochs:" $D/run_cX.log` returns numeric values; `training_seconds ≈ 300`; exit code ≠ 124 (timeout); **`num_epochs ≥ 135` for EVERY cell** (reviewer concern #2 — this is now a HARD pass gate, not informational; a 130–134-epoch cell fails here and the set is re-run/layer-restricted per Abort Criteria). PASS = all cells valid, < 600s wall, and ≥135 epochs.
2. **(b) Improvement over baseline AND > same-session c0 by a clear margin, replicated**: let `M = max(cA, cB)` best_test_acc, achieved by the "winning cell". PASS-candidate requires `M ≥ 96.48` AND `M > c0 + 0.10`. If met → **MANDATORY confirmation re-run**: a fresh same-session pair (c0' + the winning cell config), both ≥135 epochs and equally uncontended. Require the winning cell to again satisfy `≥96.48 AND > c0' + 0.10`. **Borderline rule (reviewer concern #5)**: if EITHER paired delta (original `M−c0` or confirmation `M'−c0'`) is `< 0.15pp`, run a THIRD c0''/winner pair and require the mean paired delta across the three pairs to be `≥ 0.12pp` AND every winner draw `≥ 96.48`. Only a replicated, non-hairline win = `improvement`. Any failure (M<96.48, or M−c0≤0.10, or confirmation/third-pair not replicated) → `no-improvement`; STOP (do not need (c) — (a0) already ran).
3. **(c) (subsumed)**: integrity is enforced in (a0) above and runs regardless of outcome.

Equal-conditions check (informational, supports the verdict): num_epochs for all cells within the clean ~135–154 band and consistent with the M2 (full-step) throughput prediction; the background `gpu_*.log` shows GPU 1 idle (0–3%) for the FULL duration of every cell (no mid-run foreign job).

### Informational Metrics (Optional)
(All from `$D/run_cX.log`, the experiments/018 log dir.)
- peak_vram_mb: `grep "^peak_vram_mb:" $D/run_cX.log` — expect ~1635 MB all cells (blur adds no params).
- num_epochs / training_seconds / num_steps: `grep "^num_epochs:\|^training_seconds:\|^num_steps:" $D/run_cX.log` — confirms full-budget use and the throughput cost vs c0.
- num_params: `grep "^num_params:" $D/run_cX.log` — expect 7,784,627 unchanged (blur = buffer).
- ep25 test_acc: the `eval ep  25` line per cell — under-fit/over-smoothing watch (within ~0.5pp of c0; fully annealed best≈final).
- M2 throughput img/s (c0 vs cA vs cB): from Smoke E (full train-step) — grounds the num_epochs prediction.
