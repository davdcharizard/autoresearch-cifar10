# Experiment Log EXP-031: Progressive resizing 24→32 (in-step GPU downsample, switch at 50% of budget)

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-031.md
- **Plan**: plans/plan-031.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-031
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
All four plan-031 edits applied to `train.py` on `autoresearch/exp-031` (cut from `autoresearch/dev` @ 1990397): (1) constants `LOW_RES = 24`, `RES_SWITCH_FRAC = 0.5` after LABEL_SMOOTHING; (2) `torch.compile(model, dynamic=False)` for static per-shape graphs; (3) dual-shape startup warmup — after the existing 3 iterations at 32×32, `warm_x24 = F.interpolate(warm_x, size=LOW_RES, mode="bilinear")` plus 3 more forward/backward iterations, so both inductor graphs and cudnn.benchmark tunings land in startup (no optimizer.step → weights unchanged); (4) timed-step branch immediately after the lr-assignment loop: `if progress < RES_SWITCH_FRAC: inputs = F.interpolate(inputs, size=LOW_RES, mode="bilinear")` — charged inside the timed region (after `t0`, before autocast). Everything else — transforms, loader, schedule, optimizer, model, eval — byte-identical to baseline. Sanity: AST parse OK; `git diff` = 16 insertions / 2 deletions at exactly the 4 sites; phase 2 (progress ≥ 0.5) executes the unmodified baseline path.

### Surprises & Discoveries
- None at implementation time. The warmup block's `del` list needed `warm_x24` added (anticipated by the plan's "same pattern" wording).

### Decisions
- Warmup order: 32×32 graph first (matching baseline), then 24×24 — preserves the baseline's startup signature for the full-res graph; the 24-graph compile lands second. Expected startup grows ~+3s.
- `F.interpolate` called without `align_corners`/`antialias` args (defaults): bilinear, align_corners=False — matches the MosaicML composer default and is the cheapest kernel variant.

## Run Log

### Run 1
- **Description**: Full budget-matched run of progressive resizing on GPU 0. First 50% of the 300s charged budget trains at 24×24 (in-step bilinear downsample, cost charged), second 50% at native 32×32 — numerically the baseline regime. Expected: phase-1 dt ≈ 14–17ms (FLOPs ×0.5625 vs 22.4ms baseline), total epochs 160–185 (vs 139), wall ~460–490s. Phase-1 evals will read LOW (train/test resolution mismatch) — cosmetic under the max-statistic; the post-switch trajectory should rejoin the baseline family within ~10 epochs. Hypothesis passes if best_test_acc ≥ 96.81 with epochs ≥155 and a plateau-level shift (final-7 median ≥ 96.6).
- **Job ID**: local background composite, Claude task bsqo3zrxt
- **Log file**: run.log
- **WandB**: N/A
- **Status**: killed (WALL_CAP_KILL at 601s, rc=143)
- **Started**: 2026-06-10 (GPU-0 pre-check + launch + phase-aware watchdog in one composite)
- **Ended**: 2026-06-10
- **Observations**: Mechanism on target — phase-1 windows 13.3–14.3ms (predicted 14–17), switch landed at pct≈50/step≈11,000 (~ep 113) with NO switch stall, phase-2 windows 22.0–22.7ms (baseline-identical numerics), zero contention, loss curve healthy. Killed by the 600s wall cap at 81% charged progress, ep 159, best 95.24 still climbing (anneal incomplete — hypothesis unmeasured). Uncharged wall (evals ~146s + loader stalls ~120s) was the overrun source; see Errors & Dead Ends.
- **Key Metrics**: phase-1 dt ≈ 13.7ms; phase-2 dt ≈ 22.4ms; ep at switch ≈ 113; best-at-kill 95.24 (ep 159, 81%)

### Run 2
- **Description**: Identical hypothesis and charged training path; two uncharged wall-side fixes (phase-1 eval thinning to every 3rd epoch; 16 loader workers — see Experimental Adjustments). Expected: same dt signatures (P1 ~13.7ms, P2 ~22.4ms), total epochs ~182, eval lines ~107, wall ~490s. Verification per plan with eval_lines ≤ num_epochs.
- **Job ID**: local background composite, Claude task b4mr9j2hr, PID 1316468
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed (rc=0)
- **Started**: 2026-06-10
- **Ended**: 2026-06-10 (480s wall incl. composite overhead)
- **Observations**: GPU 0 free at poll 1. Clean end-to-end: P1 windows 13.1–13.7ms, switch at ~ep113/pct≈50 with no stall (transition window 17.5ms = blend), P2 windows 22.0–22.8ms — baseline-identical numerics signature. Post-switch adaptation fast: eval ep114 dipped to 80.49 (mid-switch), ep115 already 86.92, then a steady full-res climb. Final plateau: ep179–185 = {96.69, 96.62, 96.55, 96.64, 96.64, 96.63, 96.67}, final-7 median 96.64, best 96.69 (ep 179). Eval thinning + 16 workers delivered the wall fix exactly as projected: total 457.0s (vs 601+ killed Run 1), 109 eval lines, startup 17.5s.
- **Key Metrics**: best_test_acc 96.69 | final 96.67 | final_test_loss 0.1846 | training_seconds 300.0 | total_seconds 457.0 | startup 17.5s | VRAM 1613.0MB | epochs 185 | steps 17,866 | params 4,286,026 | P1 mean 13.5ms / P2 mean 22.3ms (0 slow windows in either)

## Experimental Adjustments
- **After Run 1 WALL_CAP_KILL**: two wall-side (uncharged) fixes for Run 2, neither touching the charged training path, hypothesis, or eval semantics:
  1. **Phase-1 eval thinning** — eval only every 3rd epoch while `total_training_time/TIME_BUDGET_S < RES_SWITCH_FRAC`; phase 2 keeps one eval per epoch. Phase-1 evals are cosmetic (24px-train/32px-eval mismatch, never the max — Run 1 best always arrived late) and cost ~1.3s uncharged wall each (~113 phase-1 epochs ≈ 146s). Constraint is "validation at MOST once per epoch" — evaluating less often is compliant. Saves ~98s.
  2. **num_workers 8→16** (`2 * NUM_WORKERS` in train.py's DataLoader) — at phase-1 dt ~13.5ms the GPU outpaces the 8-worker CPU augmentation pipeline; ~120s of uncharged loader stalls observed in Run 1 phase 1. Box has 180 cores at load ~11–17. Worker count does not change batch order (sampler-driven) and only perturbs per-worker augmentation RNG streams — same noise class as a reshuffle, within baseline replicate σ. Saves ~90s.
  - Run 2 wall projection: ~18 startup + 300 charged + ~127 evals×1.3s (~165s with ~38 phase-1 + ~95 phase-2... recompute: ~38 + ~69 = 107 evals ≈ 139s) + ~30s residual stalls ≈ **~490s**, comfortably under the 600s cap.
  - Verification note: plan's "eval_lines = num_epochs" expectation becomes "eval_lines ≤ num_epochs" (condition 3 unchanged); num_epochs ≥150 gate unchanged.

## Errors & Dead Ends

### 2026-06-10 — Run 1 WALL_CAP_KILL at 601s (81% of charged budget, ep 159)
- Error: watchdog killed PID 1253784 at 601s wall; rc=143; no summary block; best at kill 95.24 (still climbing, LR anneal incomplete)
- Root cause: plan's ~460–490s wall estimate ignored that UNCHARGED wall costs scale with epoch count: ~113 phase-1 epochs × ~1.3s eval ≈ 146s + ~120s loader stalls (GPU at 13.5ms/step outpaces 8-worker CPU augmentation; stalls uncharged under the t0-after-yield accounting but real wall). Projected uncapped total ≈ 680s.
- Source: task bsqo3zrxt output (ticks 3–39: phase-1 win 13.3–14.3ms, phase-2 win 22.0–22.7ms, zero contention); run.log (159 eval lines)
- Do NOT retry: an unmodified relaunch — the overrun is deterministic arithmetic, not contention. Mechanism itself worked exactly as hypothesized (dt and phase-2 numerics on target).

## Verification Results

### Conditions Checked
1. **best_test_acc ≥ 96.81 (bar = baseline 96.71 + 0.1)** — **FAIL**. `grep "^best_test_acc:" run.log` → 96.69%. Miss by 0.12pp vs bar; −0.02 vs recorded baseline; +0.12 vs baseline mean (96.57) = within 1σ (0.16) — no detectable true effect.
   - Pre-condition (per-segment profile): **PASS** — P1: 221 win, mean 13.5ms, slow>22: 0 | P2: 135 win, mean 22.3ms (≤24 numerics guard), slow>27: 0; num_epochs 185 ≥ 150 (throughput delivered). Run uncontaminated and trustworthy — no rerun; the miss is a measured research result.
2. **Completes within budget** — not evaluated (first-failure-stop). Informationally: rc=0, total_seconds 457.0 ≤ 600 ✓.
3. **Validation ≤ once/epoch** — not evaluated (first-failure-stop). Informationally: 109 eval lines ≤ 185 epochs ✓ (phase-1 thinning to every 3rd epoch).

**Informational**: phase-1 dt 13.5ms at 24px on H20 (resolution cost datum: ×0.60 of 22.4ms for ×0.5625 FLOPs — near-ideal scaling); epochs 185 (+46 vs 139); post-switch rejoin: eval dipped to 80.49 at the switch (ep114), recovered to 86.92 within 1 epoch and climbed monotonically — adaptation cost minimal and fast; final-7 median 96.64 vs baseline-family plateau ≈96.6 — plateau LEVEL unchanged. The +46 epoch surplus converted to ≈0: low-res phase-1 epochs are NOT baseline-epoch-equivalents; the toll consumed the entire arithmetic gain.

## Human Notes
(autopilot — none)
