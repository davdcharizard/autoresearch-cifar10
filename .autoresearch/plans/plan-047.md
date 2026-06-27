# Plan EXP-047: Multi-scale decision head — fc over concat[GAP(stage2), GAP(stage3)]
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-047.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): baseline mean ≈ 96.57, σ ≈ 0.16.

Projection: +1,280 params (fc 256→384 inputs; params 4,286,026 → **4,287,306**, +0.03%); one extra `adaptive_avg_pool2d` on (B,128,16,16) + a 384-d concat is sub-0.1ms → expected dt ≈ 22.4–23.0ms, ~136–139 epochs — baseline-identical signatures. Pure decision-layer information-routing change; recipe constants untouched.

## Milestones

### Milestone 1: Code change implemented and passing CPU sanity
- [x] `train.py` `ResNet.__init__`: `self.fc = nn.Linear(w2 + w3, num_classes)` (replacing `nn.Linear(w3, num_classes)`)
- [x] `train.py` `ResNet.forward`: capture `out2 = self.layer2(out)`; `out3 = self.layer3(out2)`; pool BOTH (`F.adaptive_avg_pool2d(out2, 1)`, `F.adaptive_avg_pool2d(out3, 1)`), flatten, `torch.cat([h2, h3], dim=1)` → `self.fc`. Nothing else in the file touched (BasicBlock, recipe constants, loop all baseline).
- [x] CPU sanity (note: smoke criterion passed at lr 0.01/6 steps; the plan's lr-0.05 2-step variant overshot on random labels — wiring checks all clean) (CUDA_VISIBLE_DEVICES="", `sys.path.insert(0, <project root>)`):
  - param count == **4,287,306** exactly (= 4,286,026 + 128×10)
  - `model.fc.in_features == 384`
  - forward (4,3,32,32) → (4,10) finite
  - semantic check: full `model(x)` equals manual `fc(cat([GAP(layer2_out).flatten, GAP(layer3_out).flatten]))` recomputation (routing wired correctly)
  - 2-step train smoke, finite decreasing loss, and `fc.weight.grad[:, :128]` nonzero (stage-2 path receives gradient)

### Milestone 2: Gated launch on GPU 0 confirmed running
- [x] Reuse `/tmp/exp046_composite.sh` AS-IS (verified present); launched in background (task bwn2ey94a)
- [x] Confirm GATE_DECISION / step prints appearing (GATE_DECISION D0=22.7ms projected_epochs=136)

### Milestone 3: Run resolved — full completion OR pre-registered gate/abort branch
- [ ] If GATE_KILL (D0 median > 26ms): branch (iv) — unexpected GAP/concat mispricing, verdict `invalid`, record kernel datum, NO relaunch
- [ ] If CONTENTION_KILL/STARTUP_KILL: relaunch byte-identically once dual gates re-clear (max 2, infra)
- [x] If completed: rc=0 + summary block present (best 96.15, 138 ep, dt 21.7–22.8ms — branch (iii) dilution)

### Milestone 4: Verification executed (first-failure-stop) and exp-log updated
- [x] Integrity pre-condition (PASS, pristine), then necessary conditions in order (cond 1 FAIL: 96.15 < 96.81); results in exp-log-047.md

## Code Changes
- **train.py** (only file; ~4 lines in `ResNet`): the final linear classifier currently sees only GAP(stage3) (256-d) — the network's eval-time information bottleneck. Concatenate GAP(stage2) (128-d, higher spatial resolution mid-level summary) so the classifier reads 384-d. Why this tests the hypothesis: it is the minimal change that alters WHAT information reaches the decision layer while leaving all representations, costs, and training dynamics untouched. Risks: (a) kernel mispricing — gate covers, very unlikely (stock ops); (b) mild deep-supervision side effect (direct smooth gradient into stage 2 via fc) — small by construction (one linear layer's gradient vs the full residual stack's); (c) channels_last + cat on flattened 2-d tensors — no format hazard (flatten produces contiguous 2-d). Kaiming init and selective WD apply to the new fc exactly as to the old (ndim>1 rule unchanged).

## Configuration Changes
- None. All recipe constants, schedule, optimizer, augmentation identical to baseline (certified EXP-007…036).

## Execution Environment
- Method: local, composite background script `/tmp/exp046_composite.sh` reused verbatim (validated this session; identical thresholds apply since expected dt is baseline-band):
  1. Dual launch gates (GPU-0 zero compute apps AND 1-min load < 60), poll 30s × 240 (infra-errors EXP-032/011)
  2. `rm -f run.log`; `uv run train.py > run.log 2>&1 &` (no tee)
  3. Watchdog 44 × 15s, windowed dt from pct deltas (≥200-step windows only): GATE_KILL D0 > 26ms; CONTENTION_KILL 4 consecutive > max(D0×1.25, 26); STARTUP_KILL tick 12; NaN guard; divergence (eval < 15% after ep5); WALL_CAP tick 44 (~660s)
  4. On exit: rc, summary greps, last-8 evals
- Resources: GPU 0 only (H20); VRAM ~1.6GB; CIFAR-10 cached in `data/`
- Estimated runtime: ~485s clean (startup ~19–25s [new graph → inductor compile], 300s charged, ~139 uncharged evals); GATE_KILL branch ~90s
- Log output: `<project root>/run.log` + composite stdout (background task output file)
- Tool skill: none (local)

## Abort Criteria
- GATE_KILL D0 > 26ms → pre-registered branch (iv): verdict `invalid`, metric NaN, record kernel datum; do NOT relaunch or rework this loop
- CONTENTION_KILL / STARTUP_KILL → infra: relaunch byte-identically when gates clear (max 2, then Outcome failed)
- NaN loss or eval < 15% after epoch 5 → research failure, no retry
- Wall ≥ ~660s → kill, failure (goal cap 600s)

## Verification Protocol

### Verification Procedure
First-failure-stop. **Integrity pre-condition** (guards false readings): pristine profile — ≥200-step windows mean ≤ ~24ms, none > 27 (off-rung); num_epochs within 130–142 (projection ~138; below ~125 implies contention or hidden kernel cost — cross-check pct deltas); printed params == **4,287,306**; training_seconds == 300.0; eval lines ≤ num_epochs. Contention-caused integrity failure → rerun (infra), not a verdict.

1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1pp): `grep "^best_test_acc:" run.log` (timeout 10s; empty ⇒ crash ⇒ `tail -n 50 run.log`). Pass ≥ 96.81. **Replicate branch**: read ∈ [96.70, 96.80] with pristine profile → run the pre-registered replicate pair (two byte-identical gated runs; improvement only if mean of runs ≥ 96.81). Fail < 96.70 (or replicate mean < 96.81).
2. **Within budget**: composite rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600 (timeout 10s)
3. **Eval cadence**: `grep -c "eval ep" run.log` ≤ num_epochs value (timeout 10s)

Cleanup per goal Procedure: delete `run.log` at loop end (analyze housekeeping).

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- num_epochs: `grep "^num_epochs:" run.log` (vs ~138 — the GAP/concat dt datum)
- num_params: `grep "^num_params:" run.log` (expect 4,287,306)
- final_test_loss: `grep "^final_test_loss:" run.log` (family ~0.185 ⇒ inert-null branch; below ⇒ routed information real)
