# Plan EXP-012: Recipe-scalar refresh — weight-decay shaping + label-smoothing retune
- **Created**: 2026-06-29

<!-- Goal/metric/constraints: goals/maximize-cifar10-test-accuracy/01-definition.md. Baseline 96.38 (EXP-008, commit 07c3760), bar ≥96.48. Chosen idea + hypothesis + cell plan: experiments/012/01-brainstorm.md (§ Chosen Idea, § Review); reused full proposal: experiments/011/proposals/idea-02.md; review: experiments/012/01-idea-review.md. -->

## Summary

Refresh the stale SGD regularization scalars on a DIFFERENT axis than the now-saturated input-space aug (EXP-011): **weight-decay shaping** (headline) — split the optimizer so decay (5e-4) hits only conv/fc weight matrices and is removed (wd=0) for BN γ/β and the ReZero α — plus a **label-smoothing retune** (0.2→0.1). Run as a 4-cell same-session design (review-mandated, to beat the ~0.1pp throughput-jitter noise floor) with `WD_SHAPING` and `LABEL_SMOOTHING` env-toggled so no file edit happens between cells; **instrument the ReZero α** (print final raw + EMA α) to verify WD-shaping actually lets the capacity gate grow. Throughput-free → cannot under-anneal.

Cells (all else byte-identical to EXP-008):
| cell | WD_SHAPING | LS | role |
|------|-----------|----|----|
| 0 | off | 0.2 | same-session baseline (reproduces EXP-008) |
| A | **on** | 0.2 | **headline** — isolates WD-shaping |
| B | **on** | **0.1** | bundle (shaping + LS retune) |
| C | off | **0.1** | isolates LS (clean, CutMix-free — EXP-011's LS-0.1 was CutMix-confounded) |

## Milestones

### Milestone 1: Code implemented + smoke-verified
- [ ] Add `import os`; make `LABEL_SMOOTHING=float(os.environ.get("LABEL_SMOOTHING","0.2"))` and add `WD_SHAPING=os.environ.get("WD_SHAPING","0")=="1"` (default OFF → unmodified env reproduces baseline training behavior; cell-0 confirms empirically).
- [ ] Replace the optimizer construction with a branch: `WD_SHAPING` true → two param groups (decay = params with `ndim>=2`; no-decay wd=0 = params with `ndim<=1`); false → the original single-group `optim.SGD([requires_grad params], wd=WEIGHT_DECAY)` (byte-equivalent baseline). Add a cheap assert that the no-decay group is small (BN γ/β + α only, <5% of learnable params).
- [ ] Add summary prints: `wd_shaping`, `label_smoothing`, `no_decay_params` count, and **`rezero_alpha`** (raw `model.layer2[2].alpha.item()`) + **`rezero_alpha_ema`** (`ema_model.module.layer2[2].alpha.item()`).
- [ ] **Smoke (off-budget)**: `CUDA_VISIBLE_DEVICES=1 uv run python -c "..."` importing `ResNet9` from train.py, instantiating it. Build the EXPECTED no-decay id-set robustly via modules: `expected = {id(p) for m_ in net.modules() if isinstance(m_, nn.BatchNorm2d) for p in (m_.weight, m_.bias)} | {id(net.layer2[2].alpha)}`. Replicate the partition (`no_decay=[p for p in net.parameters() if p.requires_grad and p.ndim<=1]`, `decay=... ndim>=2`). Assert `{id(p) for p in no_decay} == expected` (EXACT — no conv/fc leaks in, no BN/α missing); assert `id(net.whiten.weight)` is in neither group (requires_grad False); assert all `decay` params have `ndim>=2`. Build `optim.SGD([{...wd 5e-4},{...wd 0}], ...)` and assert `len(param_groups)==2`, weight_decays [5e-4, 0.0]. Confirms the invocation path too (EXP-011 verified `uv run train.py` works in this env). (The in-`main` exact-count assert re-checks at runtime.)
- [ ] `python -c "import ast; ast.parse(open('train.py').read())"` clean; `git status --porcelain` only `M train.py`; `git diff --quiet -- prepare.py`.

### Milestone 2: Run the 4 cells (same session, GPU 1 free)
- [ ] Confirm GPU 1 uncontended (`nvidia-smi`). Run all four back-to-back so they share host throughput:
  - cell-0: `WD_SHAPING=0 LABEL_SMOOTHING=0.2 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_c0.log 2>&1`
  - cell-A: `WD_SHAPING=1 LABEL_SMOOTHING=0.2 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_cA.log 2>&1`
  - cell-B: `WD_SHAPING=1 LABEL_SMOOTHING=0.1 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_cB.log 2>&1`
  - cell-C: `WD_SHAPING=0 LABEL_SMOOTHING=0.1 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_cC.log 2>&1`
- [ ] For each: `grep "^best_test_acc:\|^num_epochs:\|^training_seconds:\|^total_seconds:\|^wd_shaping:\|^label_smoothing:\|^no_decay_params:\|^rezero_alpha:\|^rezero_alpha_ema:\|^num_params:" run_c*.log`.
- [ ] **Throughput guard**: every cell `num_epochs ≥ 142` (clean, comparable). Bands (single source of truth, used in Abort + Verification): **≥142** clean; **135–141** mild contention → re-run when fully free before accepting; **<110** abort/redo. If host load drifts across cells, the cross-cell ranking still holds (all equally slowed) but absolute-vs-stored-baseline needs ≥142.
- [ ] **Sequential-drift check**: cells run sequentially (c0→cA→cB→cC) on the dedicated free GPU 1. Record each cell's `num_epochs`; if they spread by >5 epochs across cells, host load drifted mid-session → the same-session comparison is weakened, re-run the affected cell(s) adjacent to cell-0. (On an idle H20 thermal/load drift is minimal; this is a guard, not an expectation.)

### Milestone 3: Decision + verdict
- [ ] Build the table (best_test_acc, num_epochs, rezero_alpha, rezero_alpha_ema per cell). Headline read: **cell-A vs cell-0** (WD-shaping effect at matched session); **mechanism check** (informational, not an NC): WD-shaping fired if cell-A's `rezero_alpha` exceeds cell-0's by ≥10% relative (loose corroboration — raw vs EMA α may differ via tail dynamics; a small/zero α-delta means any effect came from BN-γ/β decoupling instead, which is still a valid finding). LS read: cell-C vs cell-0. Bundle: cell-B vs cell-0.
- [ ] **Win** = some cell with `best_test_acc ≥ 96.48` (stored-baseline NC2) AND > same-session cell-0 by ≥0.10pp, at `num_epochs ≥142`. **Thin-winner confirmation**: a winner in [96.48, 96.55) gets one confirmation re-run **of BOTH the winning cell AND cell-0 back-to-back** (the ≥0.10 gap over same-session cell-0 must hold on the confirmation pair; seed unchanged, epoch-jitter varies both → legitimate, controlled).
- [ ] **Bake-and-confirm**: if a non-default cell wins, set its `WD_SHAPING`/`LABEL_SMOOTHING` as the static default in `train.py`. Note the env-toggle reads the SAME constants the static default sets (only the `os.environ.get` default differs), so the no-env committed run is behavior-equivalent to the winning cell; the confirmation re-run verifies this and that the committed file (no env) reproduces the metric within epoch-jitter.
- [ ] **Preserve logs**: copy deciding `run_c*.log` to `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/012/` BEFORE removing from root. (`.autoresearch/` is gitignored — this does NOT affect `git status --porcelain`/NC3, which stays "only `M train.py`".)

## Code Changes

All in `train.py` only. Verified against current code: optimizer at the `optim.SGD([p for p in model.parameters() if p.requires_grad], ...)` block; `learnable_params` computed just above; model `layer2 = Sequential(conv_bn(128,256), MaxPool2d(2), GatedResidual(256))` so the ReZero gate is `model.layer2[2].alpha` (shape [1]); EMA via `ema_model.module`. Param inventory (from idea-02 proposal, re-verified): every conv and the fc are `bias=False` ⇒ the ONLY `ndim<=1` learnable params are BN γ/β (1-D each) + the ReZero α (shape [1]); frozen whitening conv has `requires_grad=False` (excluded). So `p.ndim<=1` is an exact, robust decay/no-decay split.

- **`train.py` (imports)**: add `import os`.
- **`train.py` (hyperparameter block)**: `LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.2"))`; add `WD_SHAPING = os.environ.get("WD_SHAPING", "0") == "1"` (default OFF so an unmodified invocation = baseline). *Why env*: enables the 4-cell same-session design without editing the tracked file between runs (only train.py changes; in scope).
- **`train.py` (optimizer construction)**: branch —
  ```python
  if WD_SHAPING:
      decay, no_decay = [], []
      for p in model.parameters():
          if not p.requires_grad:
              continue
          (no_decay if p.ndim <= 1 else decay).append(p)
      # EXACT guard: no_decay must be precisely the BN γ/β + the ReZero α (count = 2·#BN + 1),
      # all ndim<=1; catches any stray param silently entering the no-decay group.
      n_bn = sum(1 for mod in model.modules() if isinstance(mod, nn.BatchNorm2d))
      assert len(no_decay) == 2 * n_bn + 1 and all(p.ndim <= 1 for p in no_decay), \
          (len(no_decay), 2 * n_bn + 1)
      optimizer = optim.SGD(
          [{"params": decay, "weight_decay": WEIGHT_DECAY},
           {"params": no_decay, "weight_decay": 0.0}],
          lr=PEAK_LR, momentum=MOMENTUM, nesterov=True,
      )
      no_decay_count = sum(x.numel() for x in no_decay)
  else:
      optimizer = optim.SGD(
          [p for p in model.parameters() if p.requires_grad],
          lr=PEAK_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True,
      )
      no_decay_count = 0
  ```
  *Why this tests the hypothesis*: removes the decay restoring-force from BN γ/β (effective-LR effect) and the ReZero α (lets the capacity gate grow), a weight-space regularization-allocation change distinct from the saturated input aug. The LR-schedule loop `for g in optimizer.param_groups: g["lr"]=lr` already writes LR to every group — no change. *Edge case*: the `WD_SHAPING=0` branch uses the exact current optimizer call, so cell-0 is **training-behavior-equivalent** to EXP-008 — the only file-level diffs are the new `import os`, env parsing, and extra summary prints, none of which touch the training computation or RNG. cell-0 is the same-session control that confirms this empirically against the stored 96.38.
- **`train.py` (summary block)**: add self-describing + mechanism prints:
  ```python
  print(f"wd_shaping:       {WD_SHAPING}")
  print(f"label_smoothing:  {LABEL_SMOOTHING}")
  print(f"no_decay_params:  {no_decay_count}")
  print(f"rezero_alpha:     {model.layer2[2].alpha.item():.4f}")
  print(f"rezero_alpha_ema: {ema_model.module.layer2[2].alpha.item():.4f}")
  ```

**Untouched**: architecture, whitening, EMA wiring, LR schedule, TTA, augmentation, batch size, seeds, `prepare.py`. `num_params` stays 7,784,627.

## Configuration Changes
- `WD_SHAPING`: new flag (off baseline; on for cells A/B). Removes wd from BN γ/β + ReZero α only; conv/fc wd held at 5e-4.
- `LABEL_SMOOTHING`: 0.2 (cells 0/A) vs 0.1 (cells B/C).
- `PEAK_LR` held at 0.4 throughout (avoid confounding the WD-shaping effective-LR effect with an LR change; clean attribution). Rationale: idea-02 proposal §"Effective-LR side effect"; conv-weight wd unchanged so the conv effective-LR drift is unchanged vs baseline (the γ/β decoupling is second-order on conv dynamics).

## Execution Environment
- **Method**: local, `[env] CUDA_VISIBLE_DEVICES=1 uv run train.py > run_c?.log 2>&1` ×4, back-to-back.
- **Resources**: single H20, **GPU 1** (mandatory; GPU 0 in use). VRAM ~1.6 GB (param-group split adds nothing).
- **Estimated runtime**: ~445–460s wall/cell → ~30 min for 4 cells (+ any confirmation/bake re-run).
- **Log output**: per-cell `run_c0/cA/cB/cC.log`; per-epoch eval lines + final `---` summary are the source of truth.
- **Tool skill**: none (local).

## Abort Criteria
- **Divergence**: smoothed train loss → NaN/inf or test_acc collapse (kill; WD-shaping should not destabilize — would indicate a param-group bug). 
- **Throughput confound**: any cell `num_epochs < 110` (GPU-1 contention, infra-errors EXP-010) → not comparable; redo when free.
- **Wall**: any run >600s (10-min kill) → failure, investigate.
- **Smoke failure** (M1): optimizer group/param-count assertions fail → fix before full runs.

## Verification Protocol

### Verification Procedure
Baseline = **96.38** (`exp-index.sh baseline`); bar = **96.48** (+0.10pp). Conditions in order; stop at first NC failure.

1. **NC1 — completes in budget, valid metric, ≤10 min** (timeout 600s/cell): for the deciding cell(s), `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:"` → numeric best_test_acc printed, `training_seconds≈300`, exit 0, `total_seconds<600`. Empty grep ⇒ crash (`tail -n 50`).
2. **NC2 — beats baseline by ≥0.10pp, clearly above noise (≥96.48)**: PASS iff the best cell's `best_test_acc ≥ 96.48` at `num_epochs ≥142` AND exceeds same-session **cell-0** by ≥0.10pp (noise-robustness). Anti-bookkeeping (exact parser): `grep "eval ep" run_cX.log | sed -E 's/.*test_acc: ([0-9.]+)%.*/\1/' | sort -rn | head -1` must equal the summary `best_test_acc` for that cell (compares max per-epoch test_acc to the printed best). +0.05–0.09pp does NOT pass (noise floor). Thin-winner [96.48,96.55) → confirmation re-run of {winner, cell-0} required.
3. **NC3 — genuine/in-scope**: `git status --porcelain` only `M train.py`; `git diff --quiet -- prepare.py`; `num_params` 7,784,627; seeds `manual_seed(42)`/`cuda.manual_seed(42)` intact; ≤1 eval/epoch (eval path untouched).

Verdict: a cell passes all NCs → **improvement** (bake-and-confirm it); all cells valid but none clears NC2 → **no-improvement**; scope/integrity breach → invalid; no valid metric → crash.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:"` per cell (~1.6 GB).
- num_epochs / num_steps / training_seconds: throughput band confirmation (~142–150).
- num_params: 7,784,627 invariant.
- **rezero_alpha / rezero_alpha_ema**: `grep "^rezero_alpha"` — mechanism check; expect cell-A/B α > cell-0/C α if WD-decoupling let the gate grow.
- no_decay_params: confirms the shaping group is small (BN γ/β + α only).
