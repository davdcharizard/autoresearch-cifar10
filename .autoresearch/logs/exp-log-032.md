# Experiment Log EXP-032: SWA tail — freeze cosine at 85%, equal-average iterates, eval BN-re-estimated SWA model

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-032.md
- **Plan**: plans/plan-032.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-032
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
All five plan-032 edits applied on `autoresearch/exp-032` (cut from `autoresearch/dev` @ 1990397): (1) `from torch.optim.swa_utils import AveragedModel, update_bn` (verified in torch 2.9.1); (2) `SWA_START_FRAC = 0.85`; (3) `swa_model = AveragedModel(base_model)` in uncharged startup; (4) one in-step branch freezing `lr_now` at `lr_at(SWA_START_FRAC)` ≈ 0.030 once progress ≥ 0.85 (charged, trivial); (5) per-epoch eval block branches on the same threshold — tail epochs do `update_parameters(base_model)` → `update_bn(train_loader, swa_model, device)` → `evaluator.evaluate(swa_model, device)`, pre-tail epochs evaluate `base_model` unchanged. Still exactly one eval per epoch; eval print format untouched. Sanity: AST OK; diff = 16 insertions / 1 deletion at exactly the 5 sites.

### Surprises & Discoveries
- `AveragedModel` defaults to `use_buffers=False`: the SWA copy's BN buffers stay at their deepcopy values and are NEVER averaged — which is correct for this design because `update_bn` resets and re-estimates them from scratch (momentum=None cumulative pass over the full augmented loader) before every tail eval. Without update_bn the SWA model would carry stale/init stats — the EXP-029 failure signature; the watchdog's SWA-bug gate (<92 after pct 87) guards this.
- `AveragedModel.update_parameters` on the first call (n_averaged=0) COPIES the current weights, so the first SWA eval is the raw iterate with re-estimated BN — a clean isolation of the BN-re-estimation effect at n=1.

### Decisions
- `swa_model` created before compile warmup (weights don't matter — first update_parameters overwrites them); keeps the training-loop body free of construction logic.
- update_bn runs the FULL 97-batch loader pass (canonical) rather than a subset — ~2s/call × ~21 calls ≈ +42s wall, inside the 600s envelope (projected ~535s); thinning to every-2nd-epoch is the documented Run-2 lever if wall pressure appears.
- 8 loader workers kept (baseline): dt 22.4ms is loader-fed at this pace; avoids stacking a second change.

## Run Log

### Run 1
- **Description**: Full run on GPU 0. Training is byte-identical to baseline until 85% of the 300s charged budget (~ep 118); from there the LR freezes at the cosine's value (~0.030, canonical constant SWA tail) and each epoch's single eval scores the equal-weight iterate average with BN re-estimated on the augmented loader. Expected: dt 22.4ms / ~139 epochs unchanged; pre-SWA evals identical-in-distribution to baseline family; first SWA eval ≈ family level; SWA trail then climbs ABOVE the family as snapshots accumulate (paper-scale +0.2–0.6). Hypothesis passes if best ≥ 96.81 from a tail eval with a plateau-level shift. Wall ~535s.
- **Job ID**: local background composite, Claude task b97wxm44v
- **Log file**: run.log
- **WandB**: N/A
- **Status**: killed (CONTENTION_KILL at tick 38, pct 98.6, rc=143)
- **Started**: 2026-06-10 (GPU-0 pre-check + launch + watchdog incl. SWA-bug gate, one composite)
- **Ended**: 2026-06-10
- **Observations**: Pristine 21.6–23.0ms windows through pct 91.2 — INCLUDING ~6 SWA-tail epochs after the 85% switch (ticks 29–32), so update_bn/SWA machinery is NOT the slowdown source. From tick 33 charged dt ballooned to 72–102ms with the step counter near-frozen; post-kill check: GPU 0 EMPTY but host 1-min load average 241 (vs ~11–17 at launch) — foreign CPU job starved H2D/host threads (charged). SWA trail before contamination: first SWA eval ep118 95.66, then 96.03/96.12 → 96.27 by ep127 with test_loss falling monotonically (0.2097→0.1778) — machinery working, no EXP-029 signature (SWA-bug gate never fired). Run lost ~10 epochs to contention and died ~4 epochs short; not analyzable as a research result.
- **Key Metrics**: best-at-kill 96.27 (ep 127, SWA eval); 128 evals; steps ~12,450+ at 296s charged (clean projection ~13,400)

## Experimental Adjustments
- **Run 2 launch gate extended**: the GPU-0 zero-compute-apps pre-check missed this contamination class (CPU-side). Run 2's waiter requires BOTH zero GPU-0 compute apps AND host 1-min load average < 60 (180-core box; the run needs ~10 cores), polling every 30s up to 60 min. Training command and code byte-identical to Run 1.

### Run 2
- **Description**: Byte-identical rerun of Run 1 (contention rerun per protocol). Launch gated on GPU-0 free AND 1-min load < 60; watchdog identical (contention 4×>27ms, SWA-bug gate <92 past pct 87, startup/NaN/wall guards), per-tick load logging added for diagnosis.
- **Job ID**: local background composite, Claude task bobq7q8yw
- **Log file**: run.log
- **WandB**: N/A
- **Status**: killed (CONTENTION_KILL at tick 14, pct 27.8, rc=143)
- **Started**: 2026-06-10
- **Ended**: 2026-06-10
- **Observations**: Clean 22.0–23.3ms through tick 10, then 57/108/108/114ms windows with step crawl — but host load was FALLING (43→15) throughout, ruling out CPU contamination. Post-kill `nvidia-smi`: foreign PID 1624123 resident on GPU 0 with 5944 MiB — GPU time-slicing, the EXP-011 signature. Kill at pct 27.8, far before SWA start — definitively environment, not the experiment code. Run 1's load-241 spike was plausibly the same foreign job's CPU phase.
- **Key Metrics**: n/a (killed at 28% of budget; 30 evals, trajectory family-normal to that point)

### Run 3
- **Description**: Byte-identical rerun. Launch gate now requires ALL of: zero GPU-0 compute apps, host 1-min load < 60 — polling every 30s for up to 2 hours (foreign PID 1624123 must exit; per standing directive we wait for GPU 0, never use GPU 1). Watchdog unchanged.
- **Job ID**: local background composite, Claude task bxujemiie, PID 1427318
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed (rc=0)
- **Started**: 2026-06-10 (after 51 gate polls ≈ 25 min — foreign GPU PID exited, load settled to 7)
- **Ended**: 2026-06-10 (556s wall incl. composite overhead)
- **Observations**: Fully clean end-to-end: 268 windows 21.0–23.3ms (mean 22.3, 0 slow), load 7–17 throughout, zero contention. SWA boundary: last raw eval ep118 94.92 → first SWA eval ep119 95.63 (+0.71 at n=1 snapshot: BN re-estimation + average≡iterate). SWA trail then climbed monotonically with test_loss falling every epoch to 0.1756 — BETTER than the baseline family's final ~0.185 — but accuracy plateaued at 96.5–96.6, still creeping (+0.02/ep) at cutoff. best 96.60 = final eval.
- **Key Metrics**: best_test_acc 96.60 | final 96.60 | final_test_loss 0.1756 | training_seconds 300.0 | total_seconds 533.3 | startup 12.4s | VRAM 1844.2MB (SWA copy +231MB) | epochs 139 | steps 13,475 | params 4,286,026 | ~21 SWA snapshots (ep119–139)

## Errors & Dead Ends

### 2026-06-10 — Run 1 CONTENTION_KILL at pct 98.6 (CPU-side contamination)
- Error: watchdog killed PID 1371258 after 4 windows >27ms (72/90/102/90ms), step counter near-frozen from tick 33
- Root cause: foreign CPU job — host load average spiked from ~11–17 to 241 on the 180-core box; GPU 0 had NO foreign compute apps. Charged dt includes H2D + host-side loop; CPU starvation inflates it. NOT self-inflicted: the SWA tail ran 6 epochs at clean 21.6–23.0ms before the spike.
- Source: task b97wxm44v output ticks 29–38; post-kill `nvidia-smi` (0 procs, 0% util) + `uptime` (load 241.49)
- Do NOT retry: launching while 1-min load > ~60 — GPU-free is not a sufficient pre-check; gate on load average too.

## Verification Results

### Conditions Checked
1. **best_test_acc ≥ 96.81 (bar = baseline 96.71 + 0.1)** — **FAIL**. `grep "^best_test_acc:" run.log` → 96.60%. −0.21 vs bar; −0.11 vs recorded baseline; +0.03 vs baseline mean (96.57) — within noise, no detectable true effect.
   - Pre-condition (profile): **PASS** — 268 windows, mean 22.3ms, slow>27: 0; num_epochs 139 (exact clean projection); training_seconds 300.0; params 4,286,026; eval_lines 139 = epochs. Run 3 uncontaminated and trustworthy (Runs 1–2 were confirmed foreign contention — GPU PID 1624123 / host load 241 — and were rerun per protocol).
2. **Completes within budget** — not evaluated (first-failure-stop). Informationally: rc=0, total_seconds 533.3 ≤ 600 ✓.
3. **Validation ≤ once/epoch** — not evaluated (first-failure-stop). Informationally: 139 evals = 139 epochs ✓.

**Informational**: first-SWA-eval jump +0.71 over the last raw eval (ep118 94.92 → ep119 95.63; isolates BN re-estimation at n=1 — positive, not the EXP-029 damage); SWA test_loss strictly better than baseline family (0.1756 vs ~0.185 final) while accuracy capped at the baseline MEAN — the same improved-loss/no-acc-gain signature as EXP-011's EMA; trail still climbing +0.02/ep at cutoff (~21 snapshots; paper SWA uses 30–80+); update_bn wall cost ≈ +40s as projected (533.3 vs ~493 baseline).

## Human Notes
(autopilot — none)
