# Plan EXP-060: CutMix substituted for RandomErasing at matched dose (p=0.5, α=1.0)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-060.md

Current baseline (exp-index.sh baseline): **96.71 @ 1990397** → success bar **≥ 96.81**. Family stats (EXP-027): mean ≈ 96.57, σ ≈ 0.16, family band [96.41, 96.73]. Family signatures: dt 22.0–22.8ms, 138–140 ep, 13,400–13,515 steps, params 4,286,026, startup ~12–15s, total ~480–510s.

**Single variable**: occlusion regularizer TYPE — noise-fill (RandomErasing, CPU, per-image) → signal-fill with area-proportional labels (CutMix, GPU, per-batch) at the SAME application dose p=0.5. Regularizer count stays 3 (LS + TA + occlusion). Heat, noise (batch/momentum), schedule, numerics (same compiled graph — mixing is eager, outside `torch.compile`), and dose all held.

**Implementation note (refinement vs brainstorm)**: the brainstorm proposed compile-static masks; in fact only `model` is compiled in train.py — the input mixing and the loss are eager code outside the graph, so the standard slice-assignment CutMix is compile-inert by construction. No dual-branch warmup, no dynamic-shape exposure (single batch shape; EXP-059's dynamic=False lesson does not arise). The charged-region rule instead becomes: the mixing path must be SYNC-FREE (all box math in Python ints/floats from CPU RNG; no `.item()`/`.cpu()` on CUDA tensors before the loss).

## Milestones

### Milestone 1: Code changes + CPU sanity
- [ ] train.py edited per Code Changes below (4 hunks)
- [ ] `/tmp/exp060_sanity.py` passes on CPU (`CUDA_VISIBLE_DEVICES="" uv run python /tmp/exp060_sanity.py`):
  (a) params == 4,286,026; (b) CutMix unit test — forced-apply: mixed region equals permuted source at the box, untouched elsewhere, lam_adj ∈ (0,1) equals 1 − box_area/1024, mixed loss finite and backward()s; forced-skip: byte-identical to plain CE path; (c) RandomErasing absent from train_tf (assert no RandomErasing instance in transforms); (d) 3-step smoke at lr 0.01, losses decreasing

### Milestone 2: GPU probe (charged-step pricing, ~2 min)
- [ ] `/tmp/exp060_gpu_probe.py` on GPU 0 — gate: zero compute apps AND 1-min load < 40 (probe gate stricter than launcher's 60 per EXP-059); compile `torch.compile(model)` exactly as train.py, 3-iter warmup, then time 40 steps of the FULL charged path with CutMix forced ON every step (worst-case pricing: randperm + slice-assign + dual CE)
- [ ] **Pre-registered probe branches** (P = ms/step, family band [21.5, 23.5]):
  - P ≤ 23.5 → launch (toll ≤ ~1ms worst-case at p=0.5 → effective ≤ ~0.5ms ≈ −3 ep ≈ −0.04pp, acceptable)
  - 23.5 < P ≤ 24.5 → launch, but shift step/epoch bands down accordingly and record the priced toll (≈ −0.07pp at p=0.5 duty cycle) in the exp-log
  - P > 24.5 → implementation bug presumed (CutMix's ops are provably ~0.2ms-class; a sync in the hot path is the likely culprit) — fix and re-probe, ≤ 2 retries; if irreducibly > 24.5, record cost-closure reasoning in the exp-log and do NOT launch (deferral law: the toll would exceed any absorbed-prior residual)

### Milestone 3: Gated launch + watchdog
- [ ] Copy `/tmp/exp046_composite.sh` → `/tmp/exp060_composite.sh` (standard single-phase launcher, unchanged thresholds: dual gates apps==0 & load<60 poll 30s×240; GATE_KILL D0>26; contention kill 4 consecutive windows > max(26, D0×1.25); STARTUP_KILL tick 12; NaN/divergence guards; WALL_CAP)
- [ ] Launch via Bash run_in_background + until-grep watcher task + TaskOutput(block=true, timeout=600000); re-block on timeout (gate waits can exceed 1h)
- [ ] GATE_DECISION D0 within [21.5, 23.5+probe-shift]; no kill markers

### Milestone 4: Completion + verification
- [ ] RC=0; summary block present in run.log
- [ ] Integrity ledger (pre-condition for Condition 1): steps ∈ [13,100, 13,600] (shift down by probe toll × duty cycle if M2 branch 2 fired); epochs ∈ [133, 142]; params 4,286,026; training_seconds 300.0; total ≤ 600; evals ≤ epochs; no NaN; **ep1 tripwire RELAXED to ≥ 25 for this experiment** (mixed-label supervision may legitimately slow the first-epoch read vs family ~34–39; the divergence guard still protects the pathological case) — if ep1 ∈ [25, 30), note it as a regularizer-shift observation, not contamination
- [ ] Verification per protocol below; verdict per pre-registered branches

## Code Changes

All in `train.py` (the only editable file), 4 hunks:

1. **Constants** (after LABEL_SMOOTHING): add
   ```python
   CUTMIX_P = 0.5      # same application dose as the RandomErasing it replaces
   CUTMIX_ALPHA = 1.0  # Beta(1,1) = Uniform — canonical CIFAR setting
   ```

2. **Transforms**: remove the `transforms.RandomErasing(...)` entry from `train_tf` (keep RandomCrop, RandomHorizontalFlip, TrivialAugmentWide, ToTensor, Normalize). This also slightly REDUCES CPU loader work (helps the ~3% loader margin, EXP-013).

3. **Charged loop** — after `targets = targets.to(...)`, before the LR update, insert the sync-free eager CutMix block:
   ```python
   use_cutmix = torch.rand(()).item() < CUTMIX_P          # CPU RNG (seeded), no GPU sync
   if use_cutmix:
       lam = torch.rand(()).item()                        # Beta(1,1) == Uniform(0,1)
       r = math.sqrt(1.0 - lam)
       cut_w, cut_h = int(32 * r), int(32 * r)
       cx, cy = int(torch.randint(0, 32, (1,))), int(torch.randint(0, 32, (1,)))   # CPU RNG
       x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, 32)
       y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, 32)
       perm = torch.randperm(inputs.size(0), device=device)
       if x2 > x1 and y2 > y1:
           inputs[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
       lam_adj = 1.0 - ((x2 - x1) * (y2 - y1)) / (32.0 * 32.0)
       targets_b = targets[perm]
   ```
   All box math is Python ints/floats from CPU RNG — zero GPU→CPU syncs added; GPU work is one randperm, one strided copy, one index-gather of targets.

4. **Loss** (inside the existing autocast block, replacing the single CE):
   ```python
   outputs = model(inputs)
   if use_cutmix:
       loss = lam_adj * F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING) \
            + (1.0 - lam_adj) * F.cross_entropy(outputs, targets_b, label_smoothing=LABEL_SMOOTHING)
   else:
       loss = F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING)
   ```
   Both CEs share the same logits (one forward). The compiled `model(inputs)` call is identical in both branches — no graph variants, no recompiles. NOTE: the printed `smooth_train_loss` becomes a mix of plain and mixed-CE values; its LEVEL will read higher than family (mixed targets) — this is expected and must not be mistaken for a regression (trajectory judged by eval lines, not train CE).

   Compile warmup (existing 3-iter loop) stays unchanged — it exercises `model` + plain CE; the mixed branch adds no compiled code.

No other changes. Eval path, timer semantics, schedule, optimizer, batch size untouched.

## Configuration Changes
- RandomErasing(p=0.5, scale 0.02–0.4, random fill): REMOVED (the variable under test)
- CUTMIX_P: — → 0.5 (dose-matched to the removed RE; reg-dose closure respected)
- CUTMIX_ALPHA: — → 1.0 (canonical CIFAR value, λ ~ Uniform; arXiv 1905.04899)

## Execution Environment
- Method: local, GPU 0 only, via `/tmp/exp060_composite.sh` (gated launcher + inline watchdog), training command inside: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (no tee)
- Resources: GPU 0 (wait for free per goal constraint — dual gate apps==0 AND load<60), ~3.2GB VRAM, 180-core host (run needs ~10)
- Estimated runtime: ~480–510s total (300s charged); gate wait potentially much longer if GPU 0 busy
- Log output: `run.log` in project root (deleted after analysis per goal procedure); launcher telemetry on the background task's stdout (`/tmp/exp060_composite_run1.log` if redirected)
- Tool skill: none (local). Launch pattern: Bash run_in_background → until-grep watcher → TaskOutput(block=true, timeout=600000), re-block on timeout.

## Abort Criteria
- GATE_KILL: D0 (median of first 3 watchdog windows) > 26ms → projected epochs make the bar arithmetically implausible
- CONTENTION_KILL: 4 consecutive 15s windows > max(26, D0×1.25) ms
- STARTUP_KILL: no step lines by tick 12 (180s)
- NaN/divergence: any `loss: nan/inf`, or eval accuracy < 15% after 5% progress
- WALL_CAP: still running after the 44-tick watchdog window (~660s) — kill; >600s total = failure per goal
- Post-hoc step-ledger gate (EXP-058): num_steps below band ⇒ contaminated — relaunch byte-identically (≤2), never analyze

## Verification Protocol

### Verification Procedure

Step 0 — **Integrity pre-condition** (gates Condition 1; from goal-learnings Protocol Findings):
`RC=0`; no kill markers in launcher log; D0 and all windows within thresholds; `grep -c "^  eval ep" run.log` ≤ num_epochs; `num_params: 4,286,026`; `training_seconds: 300.0`; `total_seconds ≤ 600`; steps/epochs within the (possibly probe-shifted) bands [13,100, 13,600] / [133, 142]; no NaN lines; ep1 ≥ 25 (relaxed tripwire, see M4). If the ledger fails ⇒ contamination ⇒ relaunch byte-identically (this is branch (v), not a verdict).

Condition 1 — **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1):
`tr '\r' '\n' < run.log | grep "^best_test_acc:"` — numeric compare. Timeout: none (file read). First-failure-stop applies.

Condition 2 — **completes within budget**: `total_seconds ≤ 600` from the summary block.

Condition 3 — **validation at most once per epoch**: eval-line count ≤ num_epochs (structural: one eval per epoch loop).

**Pre-registered decision branches** (all terminal):
- (i) best ≥ 96.81 → near-bar single draw on an absorption-prior candidate: run ONE byte-identical replicate (EXP-052 protocol) and decide on the PAIR MEAN ≥ 96.81; mean ≥ bar → improvement (commit); mean < bar → no-improvement (single-draw was the σ-tail)
- (ii) best ∈ [96.41, 96.73] → no-improvement; absorption law extends to augmentation TYPE — the regularization axis is closed in type AND dose; measured-ceiling conclusion maximally strengthened (this was the highest-published-prior untested construction)
- (iii) best ∈ (96.73, 96.81) → no-improvement (noise-band high read; never single-draw promoted)
- (iv) best < 96.41 → no-improvement; mixed-label supervision is over-pressure even at constant dose — EXP-009's mechanism extends from stacking to substitution; type axis closed from below
- (v) integrity failure → infra; relaunch byte-identically (≤2 attempts), never analyze a contaminated read

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ≈ family ~3,100MB + randperm/copy scratch, < 3,300)
- num_epochs: `grep "^num_epochs:"` (expect 136–140 at ≤ +0.3ms toll)
- num_params: `grep "^num_params:"` (must be exactly 4,286,026 — architecture untouched)
