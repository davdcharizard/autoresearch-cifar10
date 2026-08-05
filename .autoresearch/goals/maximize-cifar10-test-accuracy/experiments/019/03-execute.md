# EXP-019: Squeeze-Excitation channel attention (layer2+layer3 residual branches)

## Execution

Overall Status & Info:
- **Created**: 2026-06-30
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-019
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (all 5 runs executed cleanly with valid, trustworthy metrics; verification primary necessary condition NOT met → no-improvement, rendered in analyze)

## Implementation Notes

### Summary
Implemented SE channel attention per the plan, all in `train.py` (the only editable file). Added `import os` and two env reads `SE_RATIO`(=16) / `SE_LAYERS`(="" baseline). Added `class SE(nn.Module)`: GAP `x.mean((2,3),keepdim=True)` → 1×1 conv `c→cr` (cr=max(8,c//r)) → ReLU → 1×1 conv `cr→c` → `2*sigmoid` gate → channel rescale, with `fc2` zero-init in `__init__`. Threaded `use_se`/`se_ratio` into `Residual` and `GatedResidual` (SE applied after `c2`, before the residual/α add). Made `se_layers`/`se_ratio` constructor args of `ResNet9` (defaulting to the env globals) so in-process smokes build variants without env juggling, and wired digits {1,2,3}→layer1/2/3. The load-bearing identity-init is a post-`self.apply()` loop that re-zeros every `SE.fc2` weight+bias (kaiming from `apply` clobbers the in-`__init__` zero). Milestone 1 (4 smokes) and Milestone 2 (throughput probe) both passed; smoke/probe scripts live in `/tmp` (outside the repo) so the working tree shows only `train.py`.

### Surprises & Discoveries
- The per-block GAP did NOT introduce a CUDA sync stall: `x.mean((2,3))` is an async reduction (no `.item()`/host transfer), so the throughput probe measured cA at 0.972× and cB at 0.946× baseline — both throughput-neutral and well clear of the num_epochs≥135 under-anneal gate. This was the single biggest execution risk flagged in the SE reference; it did not materialize.
- Model-level identity-init verified exactly: every SE block returns its input bit-for-bit at init (`torch.equal(se(x), x)`), confirming `2*sigmoid(0)=1.0` after the post-`apply` re-zero — the validated recipe is bit-unperturbed at init even in the un-ReZero'd `Residual(128)`/`Residual(512)` blocks.

### Decisions
- **cB = `SE_LAYERS="123"` (placement breadth), NOT `SE_RATIO=8`** (per plan): SE params are already negligible (<0.6%), so adding layer1 SE is a more informative second operating point than doubling the bottleneck. cB is DIAGNOSTIC-ONLY — the verdict is keyed on cA (the chosen layer2+3 hypothesis) vs c0, to avoid placement-search on the test metric (plan-review P0).

## Experimental Adjustments
- **Ran a confirmation pair {c0b, cAb} despite cA not clearing 96.48 on session 1**: session 1 drew LOW (c0 96.11, ~0.27pp under the stored baseline 96.38), yet cA beat its same-session control by +0.28pp (96.39 vs 96.11) — a real signal above the ~0.1pp noise floor masked by the host draw (the symmetric case to the EXP-016/017 low-c0 lesson). The confirmation pair tests whether the +0.28 same-session delta replicates AND whether a normal host draw lets cA clear 96.48. Seed is fixed (42) — this measures documented host-draw variance, not seed-hacking. (ref: Run 2 — c0 96.11/149ep, cA 96.39/144ep)

## Run Log

### Run 1 — Throughput probe (Milestone 2, advisory)

Metadata:
- **Job ID**: local (foreground), GPU 1
- **Log file(s)**: stdout (not persisted); results inline below
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-30
- **Ended**: 2026-06-30

Description:
- Warm full-train-step (fwd+bwd+opt, autocast bf16, channels_last, sync each step) img/s for c0/cA/cB to pre-check the num_epochs≥135 gate before committing 3 full runs.

Observations:
- GPU 1 idle before probe: `1, 3843 MiB, 0 %` (no foreign job).
- c0 (no SE) 26,771 img/s (1.000×); cA(23) 26,009 img/s (0.972×, est ~146 ep); cB(123) 25,334 img/s (0.946×, est ~142 ep). Both SE configs OK vs the ≥135 gate. (source: /tmp/exp019_probe.py stdout)

Key Metrics:
- throughput cA/c0: 0.972× → est_epochs ~146 (source: probe stdout)
- throughput cB/c0: 0.946× → est_epochs ~142 (source: probe stdout)

### Run 2 — Official same-session cells c0/cA/cB (Milestone 3)

Metadata:
- **Job ID**: local background, GPU 1
- **Log file(s)**: run_c0.log / run_cA.log / run_cB.log (project root, deleted after recording)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-30 09:45
- **Ended**: 2026-06-30 10:09

Description:
- Three full 300s-budget training runs in one session on GPU 1: c0 (`SE_LAYERS=""`, control), cA (`SE_LAYERS="23"`, primary/verdict-bearing), cB (`SE_LAYERS="123"`, diagnostic). A background `nvidia-smi -l 5` sampler logs GPU-1 occupancy for the whole session (mid-run contention guard). Expect c0 ≈ 96.38 ±noise at ~150 ep; testing whether SE lifts cA ≥ 96.48 AND > c0 by >0.1pp at num_epochs ≥ 135.

Observations:
- Clean session, no contention: GPU-1 mem 3843→6040 MiB (all our process), GPU-0 util always 0 (no foreign spillover). All cells equally uncontended. (source: /tmp/exp019_smi.log)
- All `num_epochs ≥ 135` gate passed: c0 149, cA 144, cB 139. SE throughput cost matches the probe (cA ~0.97×, cB ~0.95×). (source: run_{c0,cA,cB}.log)
- Anti-gaming integrity: summary `best_test_acc` == per-epoch max test_acc for all three (96.11/96.39/96.21). Only train.py modified. (source: run logs + git status)
- cA fully annealed (best 96.39 hit before end, final 96.24 — peaked-then-settled, not under-anneal). cB likewise (best 96.21, final 96.19).
- DIAGNOSTIC: cB (all-3 SE) 96.21 < cA (layer2+3) 96.39 → adding layer1 SE HURTS vs layer2+3; the chosen layer2+3 placement is the better operating point. cB is informational only, does not bear on the verdict.

Key Metrics:
- c0 (no SE): best_test_acc 96.11% @ 149 ep, 7,784,627 params, peak_vram 1635 MB (source: run_c0.log)
- cA (SE 23): best_test_acc 96.39% @ 144 ep, 7,826,403 params (+41,776), peak_vram 1635 MB → **+0.28pp vs c0** (source: run_cA.log)
- cB (SE 123): best_test_acc 96.21% @ 139 ep, 7,828,587 params (+43,960) → +0.10pp vs c0 (source: run_cB.log)
- ep25 health (proxy): cA/cB early eval lines run ABOVE c0 (SE init not depressing early convergence) — e.g. ep1 cA 61.78% / cB 65.52% vs c0 56.25%.

### Run 3 — Confirmation pair {c0b, cAb} (Milestone 4)

Metadata:
- **Job ID**: local background, GPU 1 (bd4uqngdu)
- **Log file(s)**: run_c0b.log / run_cAb.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-30 10:09
- **Ended**: 2026-06-30 10:24

Description:
- Second same-session pair to test replication of the cA +0.28pp same-session delta and whether a normal host draw lets cA clear the absolute 96.48 bar. Fresh c0b (`SE_LAYERS=""`) + cAb (`SE_LAYERS="23"`), background nvidia-smi sampler (/tmp/exp019_smi2.log). Improvement requires cAb ≥ 96.48 AND cAb − c0b > 0.1pp.

Observations:
- Clean pair, no contention: GPU-1 mem 3843→6016 MiB (all ours), GPU-0 util always 0. num_epochs c0b 150 / cAb 144 (≥135 gate ✓, matching session 1). best==per-epoch-max for both (96.29/96.31). (source: /tmp/exp019_smi2.log, run_{c0b,cAb}.log)
- DECISIVE: the session-1 +0.28pp same-session delta did NOT replicate. cAb − c0b = +0.02pp (within the ~0.1pp noise floor) → SE ties the control once the host-draw noise is controlled. Session 1's c0 (96.11) was a low draw; with a normal c0b (96.29), SE shows no real edge.
- cA absolute never clears 96.48: 96.39 (s1) and 96.31 (s2), both ≈ the stored baseline 96.38, not +0.1pp above it.

Key Metrics:
- c0b (no SE): best_test_acc 96.29% @ 150 ep (source: run_c0b.log)
- cAb (SE 23): best_test_acc 96.31% @ 144 ep → **+0.02pp vs c0b** (source: run_cAb.log)
- Two-pair same-session deltas: +0.28 (s1) and +0.02 (s2) → mean ≈ +0.15 but NON-replicating; absolute cA ∈ {96.39, 96.31} < 96.48 in both.

## Verification Results

### Conditions Checked

1. **Completion + budget (necessary)** — PASS. All cells produced a valid `best_test_acc`, `training_seconds`=300.0, `total_seconds` ≤ 449.7s (< 600s wall cap). (source: run_{c0,cA,cB,c0b,cAb}.log)
2. **Under-anneal gate num_epochs≥135 + equal contention (necessary for trustworthy comparison)** — PASS. Epochs: c0 149, cA 144, cB 139, c0b 150, cAb 144 (all ≥135, all in the clean band). Background nvidia-smi: no foreign job either session (GPU-1 mem all ours ≤6040 MiB, GPU-0 util always 0). (source: /tmp/exp019_smi.log, /tmp/exp019_smi2.log)
3. **Primary metric — cA ≥ 96.48 AND cA − c0 > 0.1pp, replicated (necessary)** — **FAIL**. cA absolute = 96.39 (s1) / 96.31 (s2), both < 96.48 (never clears baseline+0.1pp). Same-session delta did NOT replicate: +0.28pp (s1) → +0.02pp (s2). The confirmation pair shows SE ties the control once host-draw noise is controlled. **Experiment is a no-improvement.**
4. **Confirmation re-run** — run (see condition 3); the apparent s1 win did not replicate.
5. **ep25 sanity** — PASS (not load-bearing given the metric fail): SE cells' early-epoch accuracy ran AT/ABOVE c0 (identity-init did not depress early convergence).
6. **Integrity / anti-gaming (ALWAYS run)** — PASS. `git status --porcelain` shows only train.py; prepare.py byte-unchanged; ≤1 eval/epoch (loop unchanged); seed 42; summary `best_test_acc` == per-epoch max for all 5 runs; c0/c0b have zero SE modules.

### Informational Metrics
- peak_vram_mb: ~1635 MB all cells (SE adds <1 MB; VRAM soft constraint comfortably met). (source: run logs)
- num_params: c0 7,784,627 (baseline); cA/cAb 7,826,403 (+41,776, +0.54%); cB 7,828,587 (+43,960). (source: run logs)
- num_epochs: c0 149 / cA 144 / cB 139 / c0b 150 / cAb 144 — SE throughput cost ~0.97× (cA), matching the probe; no under-anneal. (source: run logs)
- Diagnostic placement: cB (all-3 SE) 96.21 < cA (layer2+3) 96.39 → layer1 SE is net-negative; layer2+3 is the better placement (vindicates the chosen hypothesis), but still ties the control overall.

## Errors & Dead Ends

## Human Notes

>
