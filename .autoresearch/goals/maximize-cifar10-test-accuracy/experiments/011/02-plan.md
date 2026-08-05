# Plan EXP-011: CutMix data-mixing regularization
- **Created**: 2026-06-29

<!-- Goal/metric/constraints: goals/maximize-cifar10-test-accuracy/01-definition.md. Baseline 96.38 (EXP-008, commit 07c3760), bar ≥96.48. Chosen idea + hypothesis: experiments/011/01-brainstorm.md (§ Chosen Idea); full proposal: experiments/011/proposals/idea-01.md; review refinements: experiments/011/01-idea-review.md. -->

## Summary

Add **CutMix** (Yun et al., arXiv:1905.04899) to the training step as a throughput-free regularizer complementary to the existing single-image occlusion aug (Cutout12 + RandomErasing). With per-batch probability `CUTMIX_P`, paste an area-`(1-λ)` box from a batch permutation into each image and train on the area-corrected two-term loss `λ·CE(out,y) + (1-λ)·CE(out,y_perm)`; disable CutMix in the final 15% so the low-LR tail (where EMA averages and accuracy is set) trains on clean images. Targets the converged diagnosis: the net is **regularization-bound with a ~4× epoch surplus**, and throughput-free regularization is the only >noise lever (EXP-008, +0.38pp). Two review-mandated refinements are baked in: **(1)** box center / λ / apply-coin are drawn on **CPU** (no CUDA `.item()` sync inside the timed step → genuinely throughput-free); **(2)** `LABEL_SMOOTHING` is env-overridable so the **LS interaction** is tested as a 2-cell decision (LS 0.2 → LS 0.1) rather than silently shipping a possibly-over-softened CutMix.

## Milestones

### Milestone 1: Code implemented + smoke-verified (no full run yet)
- [ ] Add `import os` (and confirm no other new imports needed; pure torch).
- [ ] Add constants: `CUTMIX_ALPHA=1.0`, `CUTMIX_P=float(os.environ.get("CUTMIX_P","0.5"))`, `CUTMIX_OFF_TAIL_FRAC=0.85`; make `LABEL_SMOOTHING=float(os.environ.get("LABEL_SMOOTHING","0.2"))`.
- [ ] Add module-level `cutmix_batch(inputs, alpha)` helper (CPU-drawn box/λ; on-device `randperm` for indexing only).
- [ ] Edit the training step (`train.py:299-303`) to branch on `use_cutmix` and compute the two-term mixed loss; everything else (EMA update, LR schedule, eval gating, logging) untouched.
- [ ] **Smoke (off-budget, no eval harness)**: `CUDA_VISIBLE_DEVICES=1 uv run python -c "..."` importing `cutmix_batch`, calling it on a dummy `[512,3,32,32]` CUDA tensor. Assert only directly-checkable invariants (review concern #7 — the helper does not return box coords, so "differs only in box" is not testable): output shape `==[512,3,32,32]`, `0.0<=lam<=1.0`, `perm.shape==(512,)` and `perm.dtype` long. Run it in a loop over ~50 draws and assert that at least one draw with `lam<1.0` yields `(mixed!=inputs).any()` (a non-empty paste occurs) and that the changed-pixel fraction is `<= (1-lam)+1/32` (paste confined to ≈ the area-implied box, not the whole image). Confirms correctness without consuming the 300s budget.
- [ ] `python -c "import ast; ast.parse(open('train.py').read())"` parses clean; `git status --porcelain` shows only `M train.py`.

### Milestone 2: Cell-1 full run (CutMix @ LS 0.2) — the clean single-variable test
- [ ] Confirm GPU 1 is uncontended (`nvidia-smi`): no large foreign job on device 1 (per infra-errors EXP-010 throughput confound). If contended, wait/retry — do NOT judge absolute accuracy against the 96.38 baseline under contention.
- [ ] Run `CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1` (defaults: CUTMIX_P=0.5, LS=0.2).
- [ ] Extract: `grep "^best_test_acc:\|^num_epochs:\|^training_seconds:\|^total_seconds:\|^peak_vram_mb:\|^num_params:" run.log`; scan the per-epoch trajectory (ep25/50/75/100/tail).
- [ ] **Throughput guard (single source of truth, used identically in Abort + Verification)**: clean baseline comparison requires `num_epochs ≥ 142` (EXP-008 ran ~150). Bands: **≥142** = clean, comparable to 96.38; **135–141** = borderline mild contention → prefer a re-run when GPU 1 is fully free before accepting any absolute comparison; **<135** at an otherwise-free GPU → suspect a residual CUDA sync from the CutMix draws → fix code before trusting; **<110** → contention confound → abort/redo (do not record). Also sanity-check `cutmix_applied ≈ 0.5×(0.85×steps)` (≈42% of steps at p=0.5 with tail-disable) — a far-lower rate signals the empty-box/coin path is misbehaving.

### Milestone 3: LS decision (cell-2 only if warranted) + verdict
- [ ] Apply the verdict logic (see Verification Procedure): if cell-1 clears the bar (≥96.48, num_epochs ≥142) → that's the result. If cell-1 is flat/below-bar OR reads under-fit (ep25 < ~91.5 vs EXP-008's 92.31), run **cell-2**: `LABEL_SMOOTHING=0.1 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_ls01.log 2>&1` (tests the over-softening hypothesis).
- [ ] (Optional de-risk, only if cell-1 reads clearly over-augmented — ep25 depressed AND below baseline) run `CUTMIX_P=0.25 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_p025.log 2>&1`.
- [ ] **Thin-winner confirmation (anti-false-positive, review concern #8/#11)**: because running multiple cells and picking the max raises multiple-comparison risk near the ~0.1pp noise floor, any winning cell whose `best_test_acc` lands in the thin band **[96.48, 96.55)** gets **one confirmation re-run**; it must clear 96.48 on BOTH runs to count (the re-run varies via epoch-count jitter — it does NOT re-roll the seed, so this is legitimate, not seed-hacking). A clear win **≥96.55** needs no confirmation. cell-1 (defaults) is the primary config; cell-2/p-cells are pre-registered fallbacks, not an open sweep.
- [ ] **Bake-and-confirm (review concern #1)**: if a NON-default cell wins, set its `LABEL_SMOOTHING`/`CUTMIX_P` as the static default in `train.py` (remove the env fallback) and **re-run once with NO env overrides** to confirm the committed file reproduces the reported `best_test_acc` (within epoch-jitter). The committed `train.py` must be exactly what produced the recorded metric. If cell-1 (defaults) wins, no bake needed.
- [ ] **Preserve logs for audit (review concern #12)**: copy the deciding `run*.log`(s) to `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/011/` (gitignored, durable) BEFORE removing `run*.log` from the project root. The analyze phase reads trajectory/throughput/config from the archived copies.

## Code Changes

All edits in `train.py` only (sole editable file). Verified against the current code (training step `train.py:279-304`, criterion `train.py:251`, hyperparameter block `train.py:19-31`).

- **`train.py` (imports, ~line 1-2)**: add `import os` (used for env-overridable constants).

- **`train.py` (hyperparameter block, ~line 19-31)**: add CutMix constants and make two knobs env-overridable for the 2-cell decision without further file edits (within scope — only train.py changes):
  ```python
  LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.2"))  # was: 0.2 literal
  CUTMIX_ALPHA = 1.0            # Beta(α,α); α=1 → uniform λ (CutMix-paper default)
  CUTMIX_P = float(os.environ.get("CUTMIX_P", "0.5"))   # per-batch apply probability
  CUTMIX_OFF_TAIL_FRAC = 0.85   # disable CutMix once progress ≥ this (clean low-LR tail for EMA)
  ```
  *Why*: tests the chosen idea; env-overridability lets cell-2 (LS 0.1) and the optional p=0.25 de-risk run without editing the tracked file mid-experiment. Defaults reproduce the proposal's primary config.

- **`train.py` (new module-level helper, near the Cutout class ~line 42-61)**: add `cutmix_batch`. **CPU draws for the box center, λ, and apply-coin** (review must-fix: a `.item()` on a CUDA tensor forces a sync inside the timed step); only `torch.randperm` is on-device because it is used purely for advanced indexing (no host sync):
  ```python
  def cutmix_batch(inputs, alpha):
      """Region-mix a batch with a permutation of itself. Returns (mixed, perm, lam)
      with lam = AREA-CORRECTED weight on the ORIGINAL targets. Box/λ drawn on CPU
      (no CUDA sync); perm on-device (indexing only)."""
      n, _, h, w = inputs.shape
      perm = torch.randperm(n, device=inputs.device)          # device tensor, indexing only
      lam = float(torch.distributions.Beta(alpha, alpha).sample())  # CPU scalar
      r = (1.0 - lam) ** 0.5
      cut_h, cut_w = int(h * r), int(w * r)
      cy = int(torch.randint(h, (1,)).item())                 # CPU RNG → no CUDA sync
      cx = int(torch.randint(w, (1,)).item())
      y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
      x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
      mixed = inputs.clone()
      mixed[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
      lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)             # exact pasted-area fraction
      return mixed, perm, lam
  ```
  *Edge cases*: when the box is empty (λ≈1 → cut_h/cut_w=0) the slice is a no-op and lam→1.0 (pure clean loss) — correct. Edge-clamped boxes are area-corrected by the lam recompute. RNG draws are deterministic under the existing `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)` (no seed hacking; just consumes the seeded streams).

- **`train.py` (training step, replace `train.py:299-303`)**: branch on a CPU-drawn apply-coin; disable in the tail; two-term loss reusing the SAME `criterion` (so label_smoothing composes linearly with the area-mixed targets). Exactly **one forward + one backward** per step (the criterion is just evaluated twice on one `outputs`). Also **count realized CutMix applications** (review concern #4/#6 — empty boxes at λ≈1 and identity-permutation rows silently reduce the effective rate; making it observable closes the bookkeeping hole):
  ```python
  optimizer.zero_grad(set_to_none=True)
  use_cutmix = (progress < CUTMIX_OFF_TAIL_FRAC) and (CUTMIX_P > 0.0) \
               and (float(torch.rand(1).item()) < CUTMIX_P)   # CPU coin, no CUDA sync
  if use_cutmix:
      mixed, perm, lam = cutmix_batch(inputs, CUTMIX_ALPHA)
      if lam < 1.0:                       # non-empty box actually pasted
          cutmix_applied += 1
      with torch.autocast("cuda", dtype=torch.bfloat16):
          outputs = model(mixed)
          loss = lam * criterion(outputs, targets) + (1.0 - lam) * criterion(outputs, targets[perm])
  else:
      with torch.autocast("cuda", dtype=torch.bfloat16):
          outputs = model(inputs)
          loss = criterion(outputs, targets)
  loss.backward()
  ```
  Initialize `cutmix_applied = 0` near the other loop counters (`train.py:269-273`). *Why this tests the hypothesis*: adds the region-mixing + soft-label regularizer on top of the proven recipe with zero GPU-step cost, so any accuracy change is attributable to CutMix (not to lost epochs). `loss.item()` logging (`train.py:317`) still works (loss is a scalar tensor). `progress` is already computed at `train.py:286`. (Empty-box rate is ~0.1% at α=1 since λ>0.999 is needed to round a 32px side to 0 — negligible, but now measured rather than assumed; the `lam<1.0` recompute already makes an empty box a correct pure-clean loss.)

- **`train.py` (final summary block, ~line 372-382)**: append self-describing config + realized-rate prints so each `run.log` fully identifies the config that produced it (review concern #1/#2 — env-overridable knobs must be echoed):
  ```python
  print(f"label_smoothing:  {LABEL_SMOOTHING}")
  print(f"cutmix_alpha:     {CUTMIX_ALPHA}")
  print(f"cutmix_p:         {CUTMIX_P}")
  print(f"cutmix_off_tail:  {CUTMIX_OFF_TAIL_FRAC}")
  print(f"cutmix_applied:   {cutmix_applied}/{step} ({100*cutmix_applied/max(step,1):.1f}%)")
  ```

**Untouched** (asserts clean attribution): architecture/`ResNet9`, whitening, EMA wiring, LR schedule, TTA gate, optimizer, batch size, `prepare.py`. `num_params` must stay `7,784,627`.

## Configuration Changes
- `LABEL_SMOOTHING`: 0.2 → **0.2 (cell-1, default)**, with **0.1 (cell-2)** via env if cell-1 under-fits/misses (review-mandated LS-interaction test).
- New `CUTMIX_ALPHA = 1.0`, `CUTMIX_P = 0.5` (→ 0.25 only as optional de-risk), `CUTMIX_OFF_TAIL_FRAC = 0.85`.
- Rationale: α=1.0 and p=0.5 are the CutMix-paper / standard settings, deliberately conservative given strong occlusion aug already present; tail-disable matches the "accuracy lands in the low-LR tail" + EMA-denoise patterns. See `proposals/idea-01.md` and `knowledge/references/mixing-augmentation.md`.

## Execution Environment
- **Method**: local, `CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1` from project root.
- **Resources**: single NVIDIA H20, **GPU 1** (mandatory — GPU 0 in use; user constraint). VRAM ~1.6 GB (soft); CutMix `.clone()` adds one transient `[512,3,32,32]` tensor (~a few MB) — negligible.
- **Estimated runtime**: ~445–460s wall per cell (300s training + eval/startup). 1 cell if cell-1 wins; up to 2–3 cells (≈25 min total) including the LS-0.1 companion / optional p=0.25.
- **Log output**: stdout+stderr → `run.log` (cell-1), `run_ls01.log` (cell-2). Per-epoch `eval ep N | test_acc | best` lines + final `---` summary block are the source of truth.
- **Tool skill**: none (local run).

## Abort Criteria
- **Divergence**: smoothed train loss → NaN/inf, or test_acc collapses toward ~10% and stays (cf. EXP-009 Muon LR-too-high signature) — kill, inspect (CutMix should not destabilize SGD; would indicate a loss-composition bug).
- **Throughput confound**: `num_epochs < 110` (GPU-1 contention, per infra-errors EXP-010) → result not comparable to baseline; kill/redo when GPU 1 is free rather than recording a confounded number. (See the single-source-of-truth throughput-guard bands in Milestone 2: ≥142 clean / 135–141 borderline re-run / <135 fix-code / <110 abort.)
- **Wall-clock**: any run exceeding ~600s wall (the 10-min kill) → treat as failure; investigate (e.g., an accidental CUDA sync ballooning step time).
- **Smoke failure** (Milestone 1): if `cutmix_batch` assertions fail, fix before any full run.

## Verification Protocol

### Verification Procedure
Baseline = **96.38** (`exp-index.sh baseline` on `goals/maximize-cifar10-test-accuracy/04-results.tsv`); bar = **96.48** (+0.10pp). Run conditions in order; stop at the first necessary-condition failure.

1. **NC1 — completes in budget, valid metric, ≤10 min** (timeout 600s):
   `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:" run.log`. PASS iff a numeric `best_test_acc` is printed, `training_seconds ≈ 300` (full budget used), exit code 0, and `total_seconds < 600`. Empty grep ⇒ crash → read `tail -n 50 run.log`.
2. **NC2 — beats baseline by ≥0.10pp, clearly above the ~0.1pp noise floor**: PASS iff `best_test_acc ≥ 96.48` at `num_epochs ≥ 142` (clean throughput). Anti-bookkeeping: confirm the summary `best_test_acc` equals the max per-epoch `best:` in the trajectory (`grep "eval ep" run.log` → max test_acc == summary). A single-run delta of +0.05–0.09pp does NOT pass (noise-floor protocol). Evaluate cell-1 first; if it fails NC2, run cell-2 (LS 0.1) and evaluate NC1–NC3 on it; the best cell that satisfies all NCs is the result. **Multiple-comparison guard**: a winner in the thin band [96.48, 96.55) must clear 96.48 on a confirmation re-run (Milestone 3); ≥96.55 is decisive without it. If a non-default cell is the winner it must be baked-and-confirmed so the committed `train.py` reproduces the metric.
3. **NC3 — genuine/in-scope**: `git status --porcelain` shows only `M train.py`; `git diff --quiet -- prepare.py` (byte-unchanged); `grep "^num_params:" run.log` == `7,784,627`; seeds `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)` intact (`grep "manual_seed" train.py`). **≤1 eval/epoch**: the evaluator is called exactly once, at the single existing site inside the per-epoch loop (`train.py:349`) — this experiment does NOT touch the eval path, so the per-epoch count is unchanged by construction; confirm by reading the diff of the eval region (the `grep -c "evaluator.evaluate" train.py == 1` static check is a supporting signal, not the sole proof — verify the call remains once-per-epoch in the loop body, not added inside the step loop).
4. **Throughput attribution guard** (not an NC, but gates trust): `grep "^num_epochs:" run.log` ∈ [142,155] at uncontended GPU. `< 135` ⇒ investigate a residual CUDA sync (the must-fix) before accepting; `< 110` ⇒ contention confound (abort criteria).

Verdict: all NCs pass on some cell → **improvement**; valid runs but no cell clears NC2 → **no-improvement**; scope/integrity breach → **invalid**; no valid metric → **crash**.

### Informational Metrics (Optional)
- `peak_vram_mb`: `grep "^peak_vram_mb:" run.log` — VRAM headroom (expect ~1.6 GB).
- `num_epochs` / `num_steps` / `training_seconds`: `grep "^num_epochs:\|^num_steps:\|^training_seconds:" run.log` — confirms full budget + throughput band.
- `num_params`: `grep "^num_params:" run.log` — model-size invariant (7,784,627).
- ep25 test_acc (under-fit detector): first few `eval ep` lines — compare to EXP-008's ~92.31.
