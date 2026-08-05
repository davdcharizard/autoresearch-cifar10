# Plan EXP-007: Widen layer2 (8×8 stage) 256→384
- **Created**: 2026-06-28

## Milestones

### Milestone 1: Code change implemented + local checks pass
- [ ] Edit `ResNet9.__init__` in `train.py`: layer2 `conv_bn(128,256)`→`conv_bn(128,384)` and `GatedResidual(256)`→`GatedResidual(384)`; layer3 `conv_bn(256,512)`→`conv_bn(384,512)`. No other line changes.
- [ ] `uv run python -c "import py_compile; py_compile.compile('train.py', doraise=True)"` → clean compile.
- [ ] `git diff ae31206 -- train.py` confined to those two `__init__` lines (layer2 + layer3); `git diff --name-only ae31206 -- ` lists ONLY `train.py`. No change to any HP constant, schedule, optimizer, augmentation, training loop, `forward`, `pool`, or `fc`. (Diff base is the **baseline commit `ae31206`**, not the mutable branch name, so the scope check is stable.)
- [ ] In-process smoke (CPU ok): build `ResNet9`, load whitening, forward `[8,3,32,32]` train-mode and eval-mode → assert output `[8,10]` finite; assert internal shapes flow (layer2 emits 384ch at 8×8, layer3 emits 512ch at 4×4, pool→512, fc OK); **print and record `num_params`** — expected **9,997,235** total (verified by hand-computation; +2,212,608 over EXP-004's 7,784,627); confirm `fc` stays `Linear(512,10)`.
- [ ] Confirm the GatedResidual(384) is still identity-init: the gate is `model.layer2[2].alpha` (after the edit `layer2 = Sequential(conv_bn, MaxPool2d, GatedResidual)` → index **2**, NOT 3) — assert it == 0 and `allclose(model.layer2[2](h), h)` for a random `h` of shape `[2,384,8,8]`.

### Milestone 2: Official run completes within wall guard
- [ ] Remove any stale sentinel first (`rm -f run.log run_exit.txt`), then launch the EXACT command that writes the exit code immediately after `timeout`: `( timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'; echo "RUN_EXIT=$?" > run_exit.txt ) &` — this guarantees `run_exit.txt` is created with THIS run's exit code (not a stale one).
- [ ] `run.log` has exactly one `^best_test_acc:` line; `run_exit.txt` shows `RUN_EXIT=0` (not 124).

### Milestone 3: Verification
- [ ] Run the Verification Procedure (C1→C2→C3, stop at first failure). Record `num_epochs` and `img/s` as first-class diagnostic evidence (under-annealing is the predicted failure mode).

## Code Changes
- **`train.py` — `ResNet9.__init__` (lines 150-151)**:
  - `self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), GatedResidual(256))` → `nn.Sequential(conv_bn(128, 384), nn.MaxPool2d(2), GatedResidual(384))`
  - `self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))` → `nn.Sequential(conv_bn(384, 512), nn.MaxPool2d(2), Residual(512))`
  - **Why this tests the hypothesis**: widens the capacity-binding 8×8 stage (Diagnosis §1) by 1.5× at the full-speed kernel stage where EXP-004 measured +0.13pp, toward airbench96's documented 384 middle width. The layer3 stem input must ripple 256→384 to match layer2's new output; layer3 OUTPUT stays 512 so `pool`/`fc=Linear(512,10)` are untouched.
  - **Why no other change**: PEAK_LR=0.4 held for single-variable cleanliness. Caveat (per plan review): the GatedResidual(384) branch is identity-init (α=0, inert), but the **widened main-path convs `layer2[0]` (128→384) and `layer3[0]` (384→512) are NOT identity-preserving** — so holding LR is a deliberate attribution choice, not a guarantee the optimal LR is unchanged. This admits a third failure mode (LR/optimization mismatch with the wider net) alongside under-annealing and capacity-saturation; the analyze phase distinguishes them via the trajectory (early-epoch fit + num_epochs), and an LR retune is a possible follow-up if accuracy is flat with adequate epochs.
  - **Known noise-floor confound (unavoidable, acknowledged)**: changing the architecture shifts the CPU RNG stream consumed by `self.apply` (kaiming init), so downstream same-shaped modules (`layer3`'s `Residual(512)`, `fc`) get different initial weights, and the `DataLoader` shuffle/worker seeds (derived from global RNG state at `iter()`) also shift → a slightly different data stream. This is inherent to ANY architecture change here (EXP-004/005 had it too) and is part of the ~0.1pp noise floor. It means a sub-0.1pp result is not cleanly attributable; only a clearly-greater-than-noise gain registers — which is exactly why this idea was chosen for its larger expected headroom.
  - **Risks/edge cases**: (a) the main failure mode is throughput→epoch loss — `layer2[0]` convolves at **16×16** (before its MaxPool) and is widened 256→384, plus the two GatedResidual convs at 8×8 (2.25×) and the layer3 stem at 8×8 (1.5×); all three add compute, so under-annealing the low-LR tail is the predicted way it fails, observable via `num_epochs`/`img/s`. (b) Shapes: only layer2 out / layer3 in change; all other shapes invariant (verified in smoke). (c) `GatedResidual(384)` reuses the proven class — identity-init preserved.

## Configuration Changes
- None beyond the architecture widths above. PEAK_LR, schedule, EMA, whitening, augmentation, batch size, TTA, optimizer all held at EXP-004 values. This is a single-variable capacity change (with the LR-attribution caveat noted above).

## Execution Environment
- Method: local, single process. `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` from project root.
- Resources: 1× NVIDIA H20, **GPU 1** (`CUDA_VISIBLE_DEVICES=1` — GPU 0 in use, hard constraint). VRAM: the wider middle adds activations/params but stays a small fraction of 98 GB (EXP-004 used 1.6 GB; expect ~1.8–2.2 GB).
- Estimated runtime: ~430–490s wall. Training 300s fixed; epochs projected ~115–130 (down from EXP-004's 142–150 due to the wider layer2). Fewer epochs → fewer evals, partially offsetting the slower steps.
- Log output: `run.log` (redirected). Summary parsed for best_test_acc, training_seconds, total_seconds, num_epochs, num_steps, peak_vram_mb, num_params; step lines for img/s.
- Tool skill: none (local run).

## Abort Criteria
- `timeout` exit 124 (wall ≥ 600s) → wall-clock failure. Unlikely (fewer epochs than baseline), but if hit, it would indicate the wider net is far slower than projected; treat as failed run.
- Python traceback / non-zero non-124 exit (e.g. a shape error from a missed ripple) → code error; fix per execute rules (one retry).
- Divergence (NaN/inf loss) in step logs → would indicate the width change destabilized training (kaiming+BN should prevent this); abort and inspect. If early loss is clearly worse than EXP-004's trajectory and not recovering, that is the under-annealing signal — let the run finish (it informs the 320 fallback) rather than aborting, since the run is the measurement.
- **Pre-registered fallback (NOT an abort — a follow-up):** if best < 96.10 AND `num_epochs` ≤ **110** (concrete under-annealing cutoff; EXP-004 baseline 142–150), the next experiment is the same widen at **256→320** to recover epochs. If best < 96.10 but `num_epochs` ≥ ~125 (epochs adequate, accuracy flat) → capacity-saturated at this scale OR LR mismatch → pivot to depth/other stage or an LR retune. Recorded so the analyze phase has the contingency.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`): **96.00%**, commit ae31206. Improvement bar = baseline + 0.1 = **96.10%**. Run conditions in order; STOP at the first failure.

**C1 — Clean run within the wall guard** (necessary):
- Authoritative: `RUN_EXIT` from `run_exit.txt` must be `0` (not `124`; the `timeout` exit is the real wall guard — it measures full process wall including import + module-level `Eval()` which precede the printed `total_seconds`).
- `grep -c '^best_test_acc:' run.log` == 1.
- Secondary: `awk '/^total_seconds:/{print ($2<600)?"PASS":"FAIL"}' run.log` == PASS.
- Pass = RUN_EXIT==0 AND one summary line AND printed total_seconds<600.

**C2 — Full training budget + scope/integrity intact** (necessary):
- Full budget: `awk '/^training_seconds:/{print ($2>=295)?"PASS":"FAIL"}' run.log` == PASS.
- Frozen eval harness: `git diff --quiet ae31206 -- prepare.py && echo PASS || echo FAIL` == PASS (diff base = baseline commit `ae31206`; catches staged+unstaged).
- Scope (tracked): `git diff --name-only ae31206 -- | sort` lists exactly `train.py`.
- **Scope (untracked)**: `git status --porcelain` shows only ` M train.py` plus the run artifacts `run.log`/`run_exit.txt` — NO untracked importable files (e.g. a stray `sitecustomize.py`/`*.py` in cwd that could shadow modules). Any other untracked `.py` is a red flag → investigate before accepting.
- Change confinement: `git diff ae31206 -- train.py` shows changes ONLY on the two `__init__` width lines (layer2, layer3) — no HP/schedule/optimizer/augmentation/loop/`forward` edit. PEAK_LR still 0.4: `awk '/^PEAK_LR/{print $3}' train.py` == `0.4` (tolerates the trailing comment). Authoritative manual diff read.
- Param cross-check: `grep '^num_params:' run.log` equals **`9,997,235`** exactly (comma-formatted total, matching the hand-computed and Milestone-1 smoke value) — confirms the widen took effect and nothing else changed shape.
- Pass = budget PASS AND prepare.py unchanged AND tracked scope==train.py only AND no stray untracked .py AND diff confined to the two width lines AND num_params==9,997,235.

**C3 — Improvement ≥ +0.1pp + genuineness** (necessary):
- `BEST=$(awk '/^best_test_acc:/{print $2+0}' run.log)`; PASS iff `awk -v b=$BEST 'BEGIN{print (b>=96.10)?"PASS":"FAIL"}'` == PASS.
- Metric-genuineness cross-check: summary `best_test_acc` == max per-epoch `best:` — `awk -F'best: ' '/eval ep/{split($2,a,"%"); if(a[1]+0>m)m=a[1]+0} END{printf "%.2f\n", m}' run.log` equals the summary value.
- Genuineness: exactly one `evaluator.evaluate(` site (`grep -c 'evaluator.evaluate(' train.py` == 1 → ≤1 eval/epoch); seeds unchanged (`grep -n 'manual_seed' train.py` shows `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)`); the gain comes from the architecture width change only.
- Pass = `BEST>=96.10` AND genuineness → **improvement**. `BEST<96.10` → **no-improvement** (then read `num_epochs` to diagnose under-annealing vs capacity-saturation per the brainstorm falsifier).

### Informational Metrics (Optional)
- **num_epochs / num_steps / img/s** (FIRST-CLASS this experiment, per the idea review): `grep '^num_epochs:\|^num_steps:' run.log`; img/s from step lines (`grep 'img/s' run.log | tail -n 3`). Used to attribute any shortfall to under-annealing (epochs ≪115) vs capacity-saturation (epochs adequate, accuracy flat). EXP-004 ref: 142–150 epochs at ~26k img/s.
- peak_vram_mb: `grep '^peak_vram_mb:' run.log` — expect ~1.8–2.2 GB (soft constraint, ample headroom).
- num_params: `grep '^num_params:' run.log` — the new widened count (≈10.0M), vs EXP-004's 7,784,627.
- total_seconds: `grep '^total_seconds:' run.log` — realized wall vs the ~430–490s estimate.
