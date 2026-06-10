# Plan EXP-014: RandAugment(2, 9) replacing TrivialAugmentWide (keep Cutout(16) + compile)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] Replace `transforms.TrivialAugmentWide()` → `transforms.RandAugment()` in `train_tf` (defaults num_ops=2,
      magnitude=9 — the standard CIFAR setting). Update the adjacent comment.
- [ ] `uv run ruff check train.py` clean; `git diff` shows ONLY the augmentation line + comment changed (Cutout(16),
      compile, all else untouched from the EXP-012 baseline).

### Milestone 2: Experiment running and early signal confirmed
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Confirm clean startup: `params: 4,299,866` (UNCHANGED), no traceback, no NaN, compile completes.
- [ ] Read steady-state `dt`/`img/s` from step ~400–500 — expect ~8–9ms/step (≈ EXP-012; RA is CPU PIL ops like TA,
      no GPU sync). A large throughput drop would signal RA is heavier than TA (epoch-budget risk).

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, `final_test_loss`, `num_steps`, `num_params` from run.log.

## Code Changes
- **train.py** (the ONLY editable file): in the `train_tf` Compose, replace `transforms.TrivialAugmentWide()` with
  `transforms.RandAugment()` (torchvision defaults: num_ops=2, magnitude=9, num_magnitude_bins=31). Update the inline
  comment to reference RandAugment(2,9)/EXP-014.
  - **Why this tests the hypothesis**: RandAugment(2,9) applies TWO random ops per image (vs TA's single op) at the
    standard CIFAR magnitude — a direct probe of the demonstrated-live "more augmentation strength helps" axis
    (EXP-012 gained adding TA; EXP-013 lost reducing aug). RA+Cutout is the canonical strong CIFAR-WRN recipe.
  - **Risks/edge cases**: (a) lit says TA≈RA → may null within ~0.2pp noise of 96.22 (the +0.1 bar → ≥96.32 is
    demanding); (b) 2 ops could over-augment → mild underfit at the 300s budget (watch final_test_loss > 0.195 + acc
    drop). No param/scope/throughput risk expected (RA is a PIL transform, no GPU sync). Corroborate any gain with
    loss + late-eval cluster, not a lone best epoch.

## Configuration Changes
- Train auto-augmentation: `TrivialAugmentWide()` → `RandAugment()` (num_ops=2, magnitude=9 — torchvision CIFAR
  defaults; rationale: brainstorm-014, RandAugment paper arXiv:1909.13719, canonical RA+Cutout CIFAR-WRN recipe).
- No other change: k=4, batch 128, peak LR 0.2 cosine, Nesterov, WD 1e-4, LS 0.1, seed 42, Cutout(16),
  torch.compile(reduce-overhead) — all inherited from the EXP-012 baseline (commit 6c417a4).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background via Bash `run_in_background`).
- Resources: single NVIDIA H20 (GPU 0); ~0.5 GB VRAM; 8 dataloader workers.
- Estimated runtime: ~300s training + ~10–20s startup + ~10–15s compile ≈ 330–360s wall-clock (< 600s budget).
- Log output: all stdout/stderr → `run.log`. Metrics via
  `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^final_test_loss:|^num_steps:|^num_params:" run.log`.
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf or divergence → kill.
- Traceback / process exit ≠ 0 at startup (e.g., RandAugment arg error) → kill, fix, single retry.
- `num_params` ≠ 4,299,866 at startup → unexpected (this change must not touch params) → kill, investigate.
- No log progress for > ~120s after startup → kill (silent hang / dataloader stall — would suggest RA is far heavier
  than TA on CPU).
- NOTE: a moderately lower epoch count from RA's 2-op CPU cost is informative, not an abort trigger — let the run
  complete and record epochs; only a *severe* stall (no progress) aborts.

## Verification Protocol

### Verification Procedure
Baseline = **96.22** (`exp-index.sh baseline`); success bar = **96.32** (+0.1pp per goal). After the run completes:

1. **Cond 1 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:" run.log` returns a value
   AND `total_seconds < 600`, AND `grep -ac "Traceback" run.log` == 0. Pass = all hold. (Timeout: 600s.)
2. **Cond 2 — primary metric clears bar**: parse `best_test_acc`; PASS iff `best_test_acc >= 96.32` (baseline 96.22
   + 0.1). FAIL → verdict no-improvement. (Decisive condition.)
3. **Cond 3 — no constraint violations** (only if Cond 2 passes): `git diff --name-only` lists ONLY `train.py`;
   `num_params == 4,299,866` (unchanged); seed 42 intact; eval-line count == `num_epochs` (eval once/epoch).

### Informational Metrics (Optional)
- `num_epochs`: `grep -a "^num_epochs:" run.log` — fairness check (expect ~88–91 ⇒ fair converged test; a much lower
  count would indicate RA's 2-op CPU cost throttled throughput).
- `final_test_loss`: `grep -a "^final_test_loss:" run.log` — key corroborator. < 0.195 ⇒ RA improved the fit over TA;
  > 0.195 with acc↓ ⇒ RA(2,9) over-augmented at this budget.
- `img/s` & `dt`: from step ~400–500 — confirm ~8–9ms/step (≈ EXP-012) to rule out a throughput confound.
