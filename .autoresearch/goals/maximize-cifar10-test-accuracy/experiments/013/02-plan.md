# Plan EXP-013: Tail-only Sharpness-Aware Minimization (SAM)

- **Created**: 2026-06-29

<!-- Goal/metric/constraints: goals/maximize-cifar10-test-accuracy/01-definition.md. Baseline 96.38 (EXP-008, commit 07c3760), bar ≥96.48. Chosen idea + hypothesis + cell plan: experiments/013/01-brainstorm.md (§ Chosen Idea, § Review); full proposal: experiments/013/proposals/idea-04.md; idea-review fixes: experiments/013/01-idea-review.md §1–5; plan-review fixes: experiments/013/02-plan-review.md (1–6). SAM reference: knowledge/references/sam-sharpness-aware.md. -->

## Summary

Hand-implement **Sharpness-Aware Minimization** (Foret et al., ICLR 2021) in `train.py` — no new deps.
A SAM step does two forward-backward passes: 1st at `w` for the gradient, an **ascent** to the
worst-case neighbor `w + e_w` (`e_w = ρ·g/‖g‖`, ρ=0.05), a 2nd fwd-bwd at the perturbed point, then
**restore** `w` and `optimizer.step()` with the perturbed gradient → biases SGD toward FLAT minima (a
loss-GEOMETRY mechanism orthogonal to every saturated regularizer). To control the 2× cost (which
would otherwise halve epochs → under-anneal), apply SAM **only in the low-LR tail** where this recipe's
accuracy concentrates (EXP-001) and basin-selection is meaningful; plain SGD before.

**Gate (idea-review §1)**: `progress` is the elapsed-budget fraction, so "final 35%" = `progress >=
SAM_START_FRAC` with **SAM_START_FRAC=0.65** (NOT 0.35, which would run SAM for 65% of training → ~101
ep under-anneal). Epochs ≈ `150·(1 − f/2)`, f = SAM-active fraction: f=0.35 → ~124 ep; f=0.25 (start
0.75) → ~133 ep. Both above the ~110 under-anneal cliff.

Same-session multi-cell design (env-toggled, EXP-012 pattern — no file edit between cells), `SAM_RHO`
(0.0 = off → baseline) and `SAM_START_FRAC`:

| cell | SAM_RHO | SAM_START_FRAC | role | epoch target |
|------|---------|----------------|------|--------------|
| 0 | 0.0 | — | same-session baseline (reproduces EXP-008) | ~150 |
| A | 0.05 | **0.65** | **headline** — SAM in final 35% | ~124 |
| B | 0.05 | **0.75** | lighter SAM (final 25%), safer on epochs / built-in fallback | ~133 |

## Milestones

### Milestone 1: Code implemented + smoke-verified
- [ ] Add `import os`. Add hyperparameters (after line 30): `SAM_RHO = float(os.environ.get("SAM_RHO", "0.0"))` (0.0 = SAM OFF → unmodified invocation reproduces baseline) and `SAM_START_FRAC = float(os.environ.get("SAM_START_FRAC", "0.65"))` (only active when SAM_RHO>0).
- [ ] Add two module-level BN helpers `_bn_freeze_stats` / `_bn_restore_stats` that toggle `track_running_stats` (NOT momentum) — see Code Changes. With `track_running_stats=False` on the perturbed pass, BN uses batch stats (correct SAM behavior) and updates NO running buffers: `running_mean`, `running_var`, AND `num_batches_tracked` are all left untouched (plan-review §3 — momentum=0 would still increment num_batches_tracked). This matters because the EMA averages BN buffers (`use_buffers=True`).
- [ ] Add a module-level `sam_step(model, optimizer, criterion, inputs, targets, sam_params, rho)` doing the full two-pass + restore + `optimizer.step()` (see Code Changes), so the SAME code path is exercised by both the training loop AND the smoke (plan-review §2).
- [ ] Before the training loop (after `ema_started = False`, ~line 258): `sam_params = [p for p in model.parameters() if p.requires_grad]` (excludes frozen whitening, matches optimizer) and `sam_step_count = 0`.
- [ ] Replace the training step (lines 299–304) with the gate: `use_sam = SAM_RHO > 0.0 and progress >= SAM_START_FRAC`; if False → the existing plain step (byte-identical) with `report_loss = loss`; if True → `report_loss = sam_step(...)`, `sam_step_count += 1`.
- [ ] Change logging line (317) `train_loss_f = loss.item()` → `train_loss_f = report_loss.item()` (unperturbed 1st-pass loss in SAM steps; one `.item()`, no extra sync).
- [ ] Add summary prints: `sam_rho`, `sam_start_frac`, `sam_steps`, `sam_step_frac` (sam_step_count/step).
- [ ] **Smoke (off-budget, exercises the REAL `sam_step`)**: see the snippet in Verification Procedure. Asserts: (1) `sam_step` returns a finite loss and runs end-to-end (2nd backward + restore + optimizer.step); (2) at least one param CHANGED vs a pre-step snapshot (optimizer actually stepped); (3) `num_batches_tracked` increased by EXACTLY 1 across one `sam_step` (the perturbed pass did NOT touch BN buffers); (4) all BN `track_running_stats` restored to True, no leftover `_sam_saved_trs`; (5) an isolated ascent→restore returns params to within `atol=1e-5` (restore is float-approximate, residual ≪ the SGD update — plan-review §1; NOT bit-exact, so `allclose` not `equal`); (6) gate logic (`0.0`→never, `(0.05,0.65)`→iff progress≥0.65).
- [ ] `python -c "import ast; ast.parse(open('train.py').read())"` clean; `git status --porcelain` only `M train.py`; `git diff --quiet -- prepare.py`.

### Milestone 2: Run the 3 cells (same session, GPU 1 free)
- [ ] Confirm GPU 1 uncontended (`nvidia-smi`). Run back-to-back so they share host throughput:
  - cell-0: `SAM_RHO=0.0 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_c0.log 2>&1`
  - cell-A: `SAM_RHO=0.05 SAM_START_FRAC=0.65 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_cA.log 2>&1`
  - cell-B: `SAM_RHO=0.05 SAM_START_FRAC=0.75 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_cB.log 2>&1`
- [ ] For each: `grep "^best_test_acc:\|^final_test_acc:\|^num_epochs:\|^training_seconds:\|^total_seconds:\|^sam_rho:\|^sam_start_frac:\|^sam_steps:\|^sam_step_frac:\|^num_params:\|^peak_vram_mb:" run_c*.log`.
- [ ] **Epoch bands (contiguous; SAM legitimately costs epochs — distinct from throughput-free changes):**
  - cell-0 baseline: **≥142** expected (throughput-free); <142 ⇒ host contention → re-run when free.
  - SAM cells A/B: **≥110 = valid read** (the intended SAM cost); **<110 → under-anneal ABORT**, re-run that cell at a larger SAM_START_FRAC (0.80, lighter SAM). No undefined zone — 110 is the single boundary.
  - Record num_epochs for every cell.
- [ ] **Under-anneal DIAGNOSTIC (recorded, not a hard gate — plan-review §6):** from each SAM cell's `eval ep` trajectory, note whether the best epoch falls in the final ~2 epochs with test_acc still monotonically rising over the last ~5 epochs (the EXP-007 "still-climbing" signature). If so, flag the cell under-anneal-suspect in the analysis and prefer the lighter cell-B / a start-0.80 re-run for the verdict. This informs interpretation; it does NOT by itself pass/fail NC2 (rounded best==final is too brittle to gate on).
- [ ] **Stability check:** scan each SAM cell's `eval ep` lines from the SAM onset for any test_acc collapse / NaN train loss → bf16 ascent instability (Abort).

### Milestone 3: Decision + verdict
- [ ] Build the table (best_test_acc, final_test_acc, num_epochs, sam_steps per cell). Headline read: **best SAM cell (A or B) vs same-session cell-0**.
- [ ] **Win** = a SAM cell with `best_test_acc ≥ 96.48` AND > same-session cell-0 by ≥0.10pp at `num_epochs ≥ 110`, **AND it survives a MANDATORY confirmation re-run** (plan-review §4: because the verdict takes the best of cells A/B + a baseline = multiple looks, every SAM win — not only thin ones — must be confirmed). Confirmation = re-run {the winning cell, cell-0} back-to-back in one session; the win holds iff the winning cell still clears 96.48 AND still beats the confirmation cell-0 by ≥0.10pp. Seed unchanged; epoch-jitter varies both cells legitimately. If the win does not reproduce → no-improvement (it was a noise/multiple-looks artifact).
- [ ] **Bake-and-confirm**: if a SAM cell wins and the confirmation holds, set its `SAM_RHO`/`SAM_START_FRAC` as the static `os.environ.get` defaults in train.py; the env-toggle reads the SAME constants, so the no-env committed run is behavior-equivalent to the winning cell (the confirmation re-run, run with those env values, already verified the metric).
- [ ] **Preserve logs**: copy deciding `run_c*.log` to `experiments/013/` BEFORE removing from root (`.autoresearch/` is gitignored — does NOT affect `git status --porcelain`/NC3).

## Code Changes

All in `train.py` only. Verified against current code: training step at lines 299–304; `progress` at line 286 (reused by the SAM gate — no new timing call); `loss.item()` logging at line 317; EMA update at 306–310 (after the step, sees post-step weights — unchanged); 10 `nn.BatchNorm2d`; frozen `whiten` conv `requires_grad=False` (excluded from `sam_params`). Params are fp32 (autocast casts only inside the forward), so `p.grad` and the grad-norm are fp32 — the ascent is full-precision (idea-review §5, bf16-instability mitigation).

- **`train.py` (imports)**: add `import os`.
- **`train.py` (BN helpers, module level)** — toggle `track_running_stats` so the perturbed pass updates NO BN buffers at all (plan-review §3):
  ```python
  def _bn_freeze_stats(m):
      if isinstance(m, nn.modules.batchnorm._BatchNorm):
          m._sam_saved_trs = m.track_running_stats
          m.track_running_stats = False   # batch stats for norm; NO update to running_mean/var/num_batches_tracked

  def _bn_restore_stats(m):
      if isinstance(m, nn.modules.batchnorm._BatchNorm) and hasattr(m, "_sam_saved_trs"):
          m.track_running_stats = m._sam_saved_trs
          del m._sam_saved_trs
  ```
  (With `training=True, track_running_stats=False`, BN's forward skips `num_batches_tracked.add_(1)` and passes `running_mean=None` → uses batch statistics and leaves all running buffers untouched. Restored in a `finally` so an exception/NaN can't leave a BN in the frozen state.)
- **`train.py` (SAM step, module level)** — one function called by BOTH the loop and the smoke (plan-review §2):
  ```python
  def sam_step(model, optimizer, criterion, inputs, targets, sam_params, rho):
      # 1st pass at w (BN running stats update HERE, normally)
      optimizer.zero_grad(set_to_none=True)
      with torch.autocast("cuda", dtype=torch.bfloat16):
          loss = criterion(model(inputs), targets)
      loss.backward()
      report_loss = loss.detach()  # unperturbed loss for comparable logging

      # ascent: e_w = rho * g / (||g|| + eps) on fp32 master params
      e_ws = []  # local per-step state (no persistent dict -> no stale-restore leak, idea-review §2)
      with torch.no_grad():
          grad_norm = torch.norm(
              torch.stack([p.grad.detach().norm(2) for p in sam_params if p.grad is not None]), 2
          )
          scale = rho / (grad_norm + 1e-12)
          for p in sam_params:
              if p.grad is None:
                  continue
              e_w = p.grad * scale
              p.add_(e_w)
              e_ws.append((p, e_w))

      # 2nd pass at the perturbed point; BN buffers frozen (try/finally restores, idea-review §3)
      optimizer.zero_grad(set_to_none=True)
      model.apply(_bn_freeze_stats)
      try:
          with torch.autocast("cuda", dtype=torch.bfloat16):
              loss = criterion(model(inputs), targets)
          loss.backward()
      finally:
          model.apply(_bn_restore_stats)

      # restore original weights (float-approximate; residual << SGD step), then the real step
      with torch.no_grad():
          for p, e_w in e_ws:
              p.sub_(e_w)
      optimizer.step()  # Nesterov/momentum/wd applied to the perturbed grad at the restored w
      return report_loss
  ```
- **`train.py` (pre-loop setup, after `ema_started = False` ~line 258)**: `sam_params = [p for p in model.parameters() if p.requires_grad]`; `sam_step_count = 0`.
- **`train.py` (training step — replace lines 299–304)**:
  ```python
  use_sam = SAM_RHO > 0.0 and progress >= SAM_START_FRAC
  if not use_sam:
      optimizer.zero_grad(set_to_none=True)
      with torch.autocast("cuda", dtype=torch.bfloat16):
          outputs = model(inputs)
          loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
      report_loss = loss
  else:
      report_loss = sam_step(model, optimizer, criterion, inputs, targets, sam_params, SAM_RHO)
      sam_step_count += 1
  ```
  *Why this tests the hypothesis*: the perturbed-point gradient steers SGD toward flat minima — a generalization mechanism distinct from every saturated regularizer. The LR-schedule block (286–292) and EMA update (306–310) are unchanged — `optimizer.step()` restored `w`, so EMA sees correct post-step weights. The `SAM_RHO=0.0` path is the exact current step → cell-0 is training-behavior-equivalent to EXP-008.
- **`train.py` (logging line 317)**: `train_loss_f = loss.item()` → `train_loss_f = report_loss.item()`.
- **`train.py` (summary block, after `num_params`)**:
  ```python
  print(f"sam_rho:          {SAM_RHO}")
  print(f"sam_start_frac:   {SAM_START_FRAC}")
  print(f"sam_steps:        {sam_step_count}")
  print(f"sam_step_frac:    {sam_step_count / max(step, 1):.3f}")
  ```

**Untouched**: architecture, whitening, EMA wiring, LR schedule, TTA, augmentation, batch size, seeds, `prepare.py`. `num_params` stays 7,784,627.

## Configuration Changes
- `SAM_RHO`: 0.0 (cell-0 baseline) vs 0.05 (cells A/B). ρ=0.05 = canonical CIFAR-10 default (Foret et al.; knowledge/references/sam-sharpness-aware.md).
- `SAM_START_FRAC`: 0.65 (cell-A, final 35%) vs 0.75 (cell-B, final 25%). Brackets the SAM-active fraction — the dominant (under-anneal) axis, more informative than a ρ sweep for the first SAM test.
- All else held at EXP-008 values (PEAK_LR 0.4, wd 5e-4, LS 0.2, EMA 0.998, PCT_START 0.15, batch 512).

## Execution Environment
- **Method**: local, `[env] CUDA_VISIBLE_DEVICES=1 uv run train.py > run_c?.log 2>&1` ×3, back-to-back, same session.
- **Resources**: single H20, **GPU 1** (mandatory; GPU 0 in use). VRAM ~1.6 GB + ~31 MB e_w buffers.
- **Estimated runtime**: ~445s wall/cell (training_seconds capped at 300; SAM's 2× tail compute shows up as FEWER epochs, not more wall) → ~25 min for 3 cells (+ mandatory confirmation pair ~15 min on a win). 10-min/cell wall cap respected.
- **Log output**: per-cell `run_c0/cA/cB.log`; per-epoch `eval ep` lines + final `---` summary are the source of truth.
- **Tool skill**: none (local).

## Abort Criteria
- **Divergence / bf16 instability**: smoothed train loss → NaN/inf, or a test_acc collapse in the SAM tail (after progress≥SAM_START_FRAC) → kill the cell; ascent-step instability (check the fp32-norm path) — a correctness failure, not a SAM verdict.
- **Under-anneal**: any SAM cell `num_epochs < 110` → not comparable; re-run that cell at SAM_START_FRAC=0.80. cell-0 `num_epochs < 142` → host contention (infra-errors EXP-010), re-run when free.
- **Wall**: any run >600s (10-min kill) → failure (SAM should not inflate wall — training_seconds is budget-capped).
- **Smoke failure** (M1): any assertion fails → fix before full runs.

## Verification Protocol

### Verification Procedure
Baseline = **96.38** (`exp-index.sh baseline`); bar = **96.48** (+0.10pp). Conditions in order; stop at first NC failure.

**Smoke snippet (M1, run FIRST — exercises the real `sam_step`):**
```bash
CUDA_VISIBLE_DEVICES=1 uv run python -c "
import torch, torch.nn as nn
import train as T
from train import ResNet9
net = ResNet9(10).cuda().to(memory_format=torch.channels_last); net.train()
opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
crit = nn.CrossEntropyLoss(label_smoothing=0.2)
sam_params = [p for p in net.parameters() if p.requires_grad]
x = torch.randn(64,3,32,32, device='cuda').to(memory_format=torch.channels_last)
y = torch.randint(0,10,(64,), device='cuda')
bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
# warm the BN counters once so num_batches_tracked is defined/nonzero
with torch.autocast('cuda', dtype=torch.bfloat16): crit(net(x), y).backward()
opt.zero_grad(set_to_none=True)
nbt_before = int(bns[0].num_batches_tracked.item())
snap = [p.detach().clone() for p in sam_params]
rl = T.sam_step(net, opt, crit, x, y, sam_params, 0.05)            # the REAL step
assert torch.isfinite(rl), rl
assert any(not torch.equal(p.detach(), s) for p,s in zip(sam_params, snap)), 'optimizer did not step'
assert int(bns[0].num_batches_tracked.item()) == nbt_before + 1, 'perturbed pass touched BN buffers'  # exactly +1
assert all(m.track_running_stats for m in bns) and not any(hasattr(m,'_sam_saved_trs') for m in bns), 'BN flag not restored'
# isolated ascent->restore is float-APPROXIMATE (not bit-exact)
o2 = [p.detach().clone() for p in sam_params]
with torch.no_grad():
    gn = torch.norm(torch.stack([p.grad.norm(2) for p in sam_params if p.grad is not None]),2)  # grads still present from sam_step's 2nd pass
    sc = 0.05/(gn+1e-12); assert torch.isfinite(gn) and gn>0 and torch.isfinite(sc)
    ews=[(p, p.grad*sc) for p in sam_params if p.grad is not None]
    for p,e in ews: p.add_(e)
    for p,e in ews: p.sub_(e)
for p,o in zip(sam_params, o2): assert torch.allclose(p.detach(), o, atol=1e-5), 'restore not within tol'
def gate(rho,sf,prog): return rho>0.0 and prog>=sf
assert gate(0.0,0.65,0.99) is False and gate(0.05,0.65,0.64) is False and gate(0.05,0.65,0.65) is True
print('SMOKE OK | n_bn=%d sam_params=%d gn=%.3f'%(len(bns), len(sam_params), gn.item()))
"
```

1. **NC1 — completes in budget, valid metric, ≤10 min** (timeout 600s/cell): for the deciding cell(s), `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:"` → numeric best_test_acc, `training_seconds≈300`, exit 0, `total_seconds<600`. Empty grep ⇒ crash (`tail -n 50`).
2. **NC2 — beats baseline by ≥0.10pp, clearly above noise (≥96.48)**: PASS iff the best SAM cell's `best_test_acc ≥ 96.48` AND exceeds same-session **cell-0** by ≥0.10pp at `num_epochs ≥ 110`, **AND the win reproduces on a mandatory confirmation re-run** of {winning cell, cell-0} back-to-back (still ≥96.48 AND still ≥0.10pp over the confirmation cell-0). The confirmation is required for ANY SAM win because the verdict takes the best of cells A/B (multiple looks; plan-review §4). Anti-bookkeeping (exact parser): `grep "eval ep" run_cX.log | sed -E 's/.*test_acc: ([0-9.]+)%.*/\1/' | sort -rn | head -1` must equal the summary `best_test_acc` for that cell. +0.05–0.09pp does NOT pass (noise floor). Under-anneal diagnostic (Milestone 2) is recorded for interpretation but is not a separate pass/fail gate.
3. **NC3 — genuine/in-scope**: `git status --porcelain` only `M train.py`; `git diff --quiet -- prepare.py`; `num_params` 7,784,627 (all cells); seeds `manual_seed(42)`/`cuda.manual_seed(42)` intact; ≤1 eval/epoch (eval path untouched).

Verdict: a SAM cell passes all NCs (incl. confirmation) → **improvement** (bake-and-confirm); all cells valid but none clears NC2 (or a win fails confirmation) → **no-improvement**; a SAM cell <110 ep / under-anneal-suspect and not re-runnable to a clean read → that cell is under-anneal-confounded (report as such — method unproven, not discredited); scope/integrity breach → invalid; no valid metric → crash.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:"` per cell (~1.6 GB).
- num_epochs / training_seconds: throughput + SAM-cost band (cell-0 ~150; SAM cells ~124/~133 target).
- num_params: 7,784,627 invariant.
- sam_steps / sam_step_frac: confirms SAM fired in the intended tail fraction (cell-A ~0.35 of steps, cell-B ~0.25).
