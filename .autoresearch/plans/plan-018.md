# Plan EXP-018: CutMix (regional label-mixing aug), GPU-vectorized per batch, on the TA+Cutout recipe
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] Add `CUTMIX_PROB = 0.5` and `CUTMIX_ALPHA = 1.0` constants near the other hyperparameters.
- [ ] Add a `cutmix_batch(x)` helper (GPU paste from a shuffled batch copy; returns `x, perm, lam`).
- [ ] In the training loop: after the existing Cutout line, with prob `CUTMIX_PROB` apply CutMix; compute the
      two-term soft-target loss on CutMix batches, plain CE otherwise.
- [ ] `uv run ruff check train.py` clean; `git diff` = train.py only (constants + helper + loop/loss).

### Milestone 2: Experiment running and early signal confirmed
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Confirm clean startup: `params: 4,299,866` (UNCHANGED — aug-only change), clean compile, no traceback, no NaN.
- [ ] Read steady-state `dt`/`img/s` (step ~400–500) — expect ~8ms/step (~84–91 epochs; CutMix is a cheap GPU op,
      throughput-neutral → fair same-budget test). Train loss will read higher than baseline (soft targets) — EXPECTED.

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, `final_test_loss`, `num_steps`, `num_params`, `peak_vram_mb`.

## Code Changes
- **train.py** (the ONLY editable file):
  1. **Constants** (near L28): add
     ```
     CUTMIX_PROB = 0.5    # fraction of batches that receive CutMix (Yun et al. 2019)
     CUTMIX_ALPHA = 1.0   # Beta(alpha,alpha); alpha=1.0 == Uniform(0,1) lambda
     ```
  2. **Helper** (after `cutout_batch`, reusing its style; `math` is already imported):
     ```
     def cutmix_batch(x):
         """CutMix (Yun et al. 2019, arXiv:1905.04899): paste a random rectangular box from a
         shuffled copy of the batch; the label mix weight lam is the kept-area fraction. Train-only,
         on-GPU (one randperm + one slice-paste), no per-sample CPU sync — throughput-neutral."""
         b, _, h, w = x.shape
         lam = float(torch.rand(1).item())          # Beta(1,1) == U(0,1) for alpha=1.0
         perm = torch.randperm(b, device=x.device)
         cut_rat = math.sqrt(1.0 - lam)
         cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
         cx = int(torch.randint(w, (1,)).item())
         cy = int(torch.randint(h, (1,)).item())
         x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
         y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
         x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
         lam = 1.0 - (x2 - x1) * (y2 - y1) / (w * h)  # area-corrected kept fraction
         return x, perm, lam
     ```
  3. **Training loop** (replace the Cutout line + loss block). After `inputs = cutout_batch(inputs, CUTOUT_SIZE)`:
     ```
     do_cutmix = float(torch.rand(1).item()) < CUTMIX_PROB
     if do_cutmix:
         inputs, perm, lam = cutmix_batch(inputs)
     ...
     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
         outputs = compiled_model(inputs)
         if do_cutmix:
             loss = lam * F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING) \
                    + (1.0 - lam) * F.cross_entropy(outputs, targets[perm], label_smoothing=LABEL_SMOOTHING)
         else:
             loss = F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING)
     ```
  - **Why this tests the hypothesis**: CutMix adds a strong *regional label-mixing* regularizer (real local features
    pasted across images) on top of TA (photometric) + Cutout (occlusion), targeting the residual generalization gap
    at fixed capacity. The `do_cutmix` coin-flip `.item()` and the box-coord `.item()`s are CPU-tensor reads (no GPU
    sync), and the loop already does `loss.item()` + `torch.cuda.synchronize()` per step, so no new throughput cost.
  - **Risks/edge cases**: (a) lam≈1 → empty box (cut_w/h=0) → paste is a no-op, lam recomputes to 1.0 → plain CE
    (safe, no crash); lam≈0 → near-full replace → loss≈CE(targets[perm]) (safe). (b) Label-mixing family matches the
    failed weak Mixup (EXP-011) + CutMix wants long schedules → may underfit at ~84–91 ep → graceful no-improvement.
    (c) Stacking with Cutout may over-regularize. (d) compile: `compiled_model` sees unchanged input SHAPE (128,3,32,32)
    every step regardless of CutMix → no recompile / CUDA-graph issue; the loss branch is eager.

## Configuration Changes
- Add `CUTMIX_PROB = 0.5`, `CUTMIX_ALPHA = 1.0` (standard CutMix; α=1.0 ⇒ λ~Uniform). All else inherited from the
  EXP-012 baseline (k=4, batch 128, peak LR 0.2 — now confirmed optimal EXP-016/017, WARMUP 0.05, Nesterov, WD 1e-4,
  LS 0.1, Cutout(16), TrivialAugment, torch.compile, seed 42, commit 6c417a4). Cutout is KEPT (validated recipe).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background via Bash `run_in_background`).
- Resources: single NVIDIA H20 (GPU 0); ~0.5 GB VRAM; 8 dataloader workers.
- Estimated runtime: ~300s training + ~10–20s startup + ~10–15s compile ≈ 330–360s wall-clock (< 600s budget).
- Log output: all stdout/stderr → `run.log`. Metrics via
  `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^final_test_loss:|^num_steps:|^num_params:|^peak_vram_mb:" run.log`.
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf or sustained divergence → kill. (Note: train loss reading HIGHER than baseline is EXPECTED with
  soft-target mixing — not a divergence; only NaN/inf or a monotonic climb with no eval recovery is an abort.)
- Traceback / process exit ≠ 0 at startup (e.g., indexing/shape error in `cutmix_batch`) → kill, fix, single retry.
- `num_params` ≠ 4,299,866 → unexpected for an aug-only change → kill, investigate.
- Throughput collapse: if `dt` ≫ ~10ms steady-state (CutMix should be ~free) → note a throughput confound for analysis.
- Early-accuracy sanity: if `test_acc` < ~90% past ~50% of budget (≈ ep 45) → suspect over-regularization/underfit →
  note for analysis (let it finish unless NaN/stall).
- No log progress for > ~120s after startup → kill (silent hang).

## Verification Protocol

### Verification Procedure
Baseline = **96.22** (`exp-index.sh baseline`); success bar = **96.32** (+0.1pp per goal). After the run completes:

1. **Cond 1 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:" run.log` returns a
   value AND `total_seconds < 600`, AND `grep -ac "Traceback" run.log` == 0. Pass = all hold. (Timeout: 600s.)
2. **Cond 2 — primary metric clears bar**: parse `best_test_acc`; PASS iff `best_test_acc >= 96.32`. FAIL →
   verdict no-improvement. (Decisive condition.)
3. **Cond 3 — no constraint violations** (only if Cond 2 passes): `git diff --name-only` lists ONLY `train.py`;
   seed 42 intact; eval-line count == `num_epochs` (eval once/epoch — CutMix is train-only, eval path untouched);
   `num_params` == 4,299,866 (unchanged); no new deps (uses torch/math only).

### Informational Metrics (Optional)
- `num_epochs`: `grep -a "^num_epochs:" run.log` — fairness check (expect ~84–91; throughput-neutral).
- `final_test_loss`: `grep -a "^final_test_loss:" run.log` — NOTE: with CutMix the model trains on soft targets so
  test loss may rise even if acc holds/improves; corroborate via ACC, not loss (cf. Mixup EXP-011 loss artifact).
- `peak_vram_mb`: `grep -a "^peak_vram_mb:" run.log` — expect ~454 MB (a shuffled-batch copy is tiny).
- `img/s` & `dt`: step ~400–500 — confirm ~8ms/step (rule out a throughput confound).
