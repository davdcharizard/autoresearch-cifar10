# Plan EXP-005: Second ReZero-gated residual block in layer3 (capacity-via-depth probe)
- **Created**: 2026-06-28

## Summary & Baseline
Append one `GatedResidual(512)` (ReZero, α=0 → identity at init) to `layer3`, operating at 4×4 resolution. Channel-preserving (512→512), no new random bracket convs (unlike widening), and needs no LR retune. 10→12 learnable convs. `PEAK_LR` held at 0.4 (clean single-variable capacity test). Whitening + EMA + flip-TTA byte-identical. Current baseline (from `04-results.tsv`): **96.00%**; improvement bar **≥96.10%** (+0.1pp — hard, at the airbench96 ceiling).

> **Init-equivalence caveat (plan-review #1):** the new block's *output* is exact identity at init (α=0), so it does not disrupt the learned dynamics — but the net is NOT bit-identical to EXP-004's init. `self.apply(self._weights_init)` walks modules in registration order; the new block's 2 convs consume extra kaiming RNG draws *before* `fc`, so `fc.weight` (and module-order-later inits) get different random values than EXP-004. This is a **noise-level perturbation** (a different random init of one `Linear(512,10)`), not a confound for the capacity test (the architecture delta is the single variable; the verdict is the metric, not bit-equivalence). The early-trajectory "matches EXP-004" check is therefore a *sanity* signal (identity block ⇒ no gross early disruption), interpreted with this caveat — not a claim of bit-identical curves.

## Milestones

### Milestone 1: Code change + correctness/gradient smoke
- [ ] Edit `train.py:151` — append `GatedResidual(512)` to `self.layer3`: `nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512), GatedResidual(512))`. (`GatedResidual` class already exists from EXP-004, lines 119–137; reused verbatim, no new class.)
- [ ] `uv run python -c "import py_compile; py_compile.compile('train.py', doraise=True)"` → exits 0.
- [ ] Smoke (`PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 uv run python`, in-process, NO training run): build `ResNet9()`, and let `blk = model.layer3[3]` (the new gated block), assert:
  - (a) `type(blk).__name__ == "GatedResidual"` and **identity at init** — for `h=torch.randn(2,512,4,4)`, `torch.allclose(blk(h), h, atol=1e-6)` (α=0).
  - (b) **gradient flows to the new gate (dead-block guard)** — one forward+backward through the full net on `torch.randn(2,3,32,32)`; assert `blk.alpha.grad is not None and blk.alpha.grad.abs().item() > 0`.
  - (b2) **branch trains after α moves (plan-review #5 — α.grad≠0 is necessary but not sufficient)** — build a tiny SGD optimizer over `model.parameters()` (filtered by requires_grad, exactly as `main()` does), run **2 steps** on random data; assert after step 1 that `blk.alpha` has moved off 0 (`blk.alpha.detach().abs().item() > 0`), and after step 2's backward that a branch conv weight receives nonzero gradient (`blk.c2[0].weight.grad.abs().sum().item() > 0`) — proving the branch convs train once the gate opens (they get zero grad at α=0 on step 1, nonzero once α≠0). Also assert `blk.alpha` is in the optimizer's param set.
  - (c) full `_forward_once` returns `[2,10]`, finite; feature map entering `self.pool` is `512×4×4` (chain intact).
  - (d) `learnable` params == **12,503,810** (EXP-004's 7,783,169 + 4,720,641); total == **12,505,268**.
  - (e) `whiten.weight.requires_grad == False`.

### Milestone 2: Official run + throughput/early-trajectory gate
- [ ] Launch the official run (background, GPU 1) capturing to `run.log`.
- [ ] Within the first ~30s, read the `step ... img/s` print: confirm projected epochs **≥ ~115** (a 4×4 block should cost *less* throughput than EXP-004's 8×8 block which gave 142 ep; the time-based schedule guarantees the anneal completes regardless, but <110 epochs is the under-annealing red flag — record for analysis, do not auto-abort).
- [ ] Early-convergence check vs EXP-004 (identity init should NOT disrupt early epochs): `eval ep 1` within noise of EXP-004's 58.70% (alarm if < ~54%), `ep 10` near 85.2% (alarm if < ~80%). A large early deficit would mean the identity-init assumption is violated → flag.
- [ ] Confirm no NaN/divergence; `wall` stays well under 600s.

### Milestone 3: Completion + verification
- [ ] Run completes with a full summary (`best_test_acc:` present), `total_seconds` < 600.
- [ ] Run the Verification Protocol below; record results in `03-execute.md`.

## Code Changes
- **`train.py` (line 151)** — change
  `self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))`
  → `self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512), GatedResidual(512))`.
  *Why*: adds 2 learnable conv layers (10→12 convs) at 4×4 resolution — extends EXP-004's validated ReZero capacity lever. Channel- and spatial-preserving (512→512, 4×4→4×4), so `pool`(MaxPool2d(4) → 512×1×1), `fc`(Linear(512,10)), and whiten/prep/layer1/layer2 are all untouched. The `GatedResidual` is ReZero-gated (`x + α·c2(c1(x))`, α init 0), so the new block is **exact identity at init** → the net starts bit-equivalent to the proven 96.00% net and earns capacity gradually as α ramps off zero (live gradient path ∂L/∂α=⟨grad_out,branch_out⟩≠0). Risk: more params, fewer epochs (under-annealing — see Abort); diminishing returns of a 2nd capacity block.

No other edits. `Residual`/`GatedResidual` classes, `_forward_once`/`forward` (TTA logic), `compute_whitening_weight`, optimizer (`requires_grad`-filtered; the new α and convs auto-join the SGD group), EMA construction (`AveragedModel` deep-copies the deeper net), schedule, all HP constants, seed 42 — byte-identical.

## Configuration Changes
- None. `PEAK_LR=0.4`, `PCT_START=0.15`, `EMA_*`, `TTA_START_FRAC=0.8`, `WEIGHT_DECAY=5e-4`, `LABEL_SMOOTHING=0.2`, `BATCH_SIZE=512` unchanged. (Deliberate: ReZero's gradual ramp removes the stability rationale for an LR retune → clean single-variable capacity test at the validated LR. The new α gets WD 5e-4 — negligible on a gradient-driven scalar.)

## Execution Environment
- **Method**: local, `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` (GPU 1 — GPU 0 in use, hard constraint).
- **Resources**: single NVIDIA H20, GPU 1. Expected peak VRAM ~1.7–1.9 GB (the 4×4 block's activations are the smallest per-stage tensor; non-binding vs 98 GB).
- **Estimated runtime**: ~7.5 min wall (300s training + eval + startup), under the 600s kill. Expected ~120–135 epochs (a 4×4/512 block should be cheaper than EXP-004's 8×8 block → fewer-epoch loss than EXP-004's 174→142).
- **Log output**: `run.log` in project root (redirected; grep it, do not flood context). Launch in background, monitor early epochs, block on completion.
- **Tool skill**: none (local run).

## Abort Criteria
- **NaN/Inf or diverging loss** (loss print `nan`/`inf` or climbing past progress≈0.15) → kill, research failure (no retry).
- **No log output / no per-epoch `eval` line after ~120s** → kill, infra failure (retry ≤2).
- **Wall approaching 600s before completion** → `timeout 600` enforces; exit 124 = failure.
- **Throughput collapse** (projected epochs < ~110 from early img/s) → deliberately **NOT an auto-abort**, and this is the considered resolution of the brainstorm's "mandatory ≥110 gate" (plan-review #3): a *completed* run yields the actual epoch count AND the full trajectory, which is strictly more informative than aborting — and the time-based schedule still anneals LR→0 regardless. BUT the throughput premise of this idea (4×4 block is cheaper than EXP-004's 8×8 → more annealing budget) is then **falsified**, so the analysis MUST attribute any resulting no-improvement to **under-annealing / failed-throughput-premise**, distinguishable via the trajectory (mid-training lead present but tail truncated) — NOT to "capacity saturated." If cuDNN picks a poor 4×4/512 kernel and throughput is *worse* than EXP-004's 8×8 block, that is itself the finding (and the fallback is a 2nd layer2/8×8 block, per the brainstorm).
- **Dead-block guard** — Milestone 1 smoke (b) (`alpha.grad ≠ 0` on the new layer3 block) MUST pass before the official run; if it fails, the block is untrainable and the run is meaningless.
- Note: the official run IS the experiment — the M1 smoke is a fast in-process correctness check (no training); throughput/early gates are read live from the official run's first epochs.

## Verification Protocol

### Verification Procedure
Baseline = **96.00%** (from `exp-index.sh baseline` on `04-results.tsv`; bar = 96.10%). The **authoritative wall guard is the run process's exit code** (captured at launch): exit 0 = clean; exit 124 = `timeout` wall-kill = crash (plan-review #7 — `total_seconds` is measured inside `main()` and slightly under-reports true process wall, so the exit code is the real 600s guard, with `total_seconds` as a corroborating informational read). Evaluate the three goal necessary conditions in order (stop at first failure):

1. **Clean run within wall guard** — the run process exited **0** (not 124); `grep -c "^best_test_acc:" run.log` returns **1**. Corroborate: `awk -F: '/^total_seconds:/{gsub(/ /,"",$2); print ($2+0 < 600) ? "PASS "$2 : "FAIL "$2}' run.log` prints `PASS`. Empty `best_test_acc` ⇒ `tail -n 50 run.log`, crash.
   - Metrics read: `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:\|^num_epochs:\|^peak_vram_mb:\|^num_params:" run.log`.
2. **Full training budget + scope intact** — `awk -F: '/^training_seconds:/{gsub(/ /,"",$2); print ($2+0 >= 295) ? "PASS "$2 : "FAIL "$2}' run.log` prints `PASS`. `git diff --quiet -- prepare.py && git diff --quiet autoresearch/maximize-cifar10-test-accuracy-dev -- prepare.py` (both exit 0). **Diff-content check is AUTHORITATIVE for scope/leakage (plan-review #4/#6):** `git diff --stat autoresearch/maximize-cifar10-test-accuracy-dev` shows only `train.py`; `git diff autoresearch/maximize-cifar10-test-accuracy-dev -- train.py` is **read in full** and must contain ONLY the single one-token `layer3` append (`, GatedResidual(512)`) — nothing in `_forward_once`/`forward`/the training loop/HP constants/eval/seed/data-loading. (`.autoresearch/` is gitignored, so writing `03-execute.md` does not affect this check — resolves the sequencing note.) Because the diff is provably this one line, no leakage/eval/seed code can have been introduced; the greps below are secondary corroboration only. Timeout 30s.
3. **Improvement ≥ +0.1pp + genuineness** — extract `BEST=$(grep "^best_test_acc:" run.log | grep -oP '[0-9]+\.[0-9]+')`; **PASS iff `awk "BEGIN{exit !($BEST >= 96.10)}"` (i.e. BEST ≥ 96.10)**. Genuineness: `MAXEP=$(grep "eval ep" run.log | grep -oP 'best:\s*\K[0-9]+\.[0-9]+' | sort -rn | head -1)` must equal `$BEST` (both bare floats, normalized — the summary's trailing `%` is stripped by the `[0-9]+\.[0-9]+` extraction). Reward-hack/leakage corroboration (authoritative check is step-2 diff): `grep -c "evaluator.evaluate(" train.py` == 1; `grep -c "manual_seed" train.py` == 3 (two global seed-42 + local whitening `Generator(0)`); `grep -nE "train\s*=\s*False|test_set|testset|\.targets|evaluator\.loader" train.py` → none; the CIFAR10 call is train split (`grep -n "CIFAR10" train.py` shows the `train=True` call, verified in the step-2 diff to be unchanged).
   - Verdict: BEST ≥96.10 + all above ⇒ **improvement**; valid run BEST <96.10 ⇒ **no-improvement**; scope/seed/eval violation ⇒ **invalid**; crash/timeout (exit 124) ⇒ **crash**.

### Informational Metrics (Optional)
- `peak_vram_mb`: `grep "^peak_vram_mb:" run.log` — expect ~1.7–1.9 GB.
- `training_seconds` / `num_epochs` / `num_steps`: `grep "^training_seconds:\|^num_epochs:\|^num_steps:" run.log` — confirm full budget + epoch count (expect ~120–135 vs EXP-004's 142; fewer epochs is the capacity/throughput trade — note whether the 4×4 block cost less throughput than EXP-004's 8×8 block, per the idea's hypothesis).
- `num_params`: `grep "^num_params:" run.log` — expect **12,505,268** total (EXP-004's 7,784,627 + 4,720,641); learnable 12,503,810 (smoke check (d)).
- Early/mid trajectory: `grep "eval ep" run.log | sed -n '1p;10p;25p;50p'` — for the mechanism trace (identity init ⇒ ep1/ep10 match EXP-004 within noise; ep25–50 should LEAD EXP-004's 92.63%/94.00% if the block adds usable capacity).
