# Experiment Log EXP-066: Progressive resolution scheduling (24×24 → 32×32)

## Execution
- **Created**: 2026-06-09
- **Brainstorm**: brainstorm/brainstorm-066.md
- **Plan**: plans/plan-066.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-066
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan Milestone 1 with two edits to train.py. (1) Added constants `LOW_RES = 24` and `RESIZE_FRAC = 0.5` after `CUTOUT_SIZE` (L29-30). (2) In the training loop, replaced the unconditional `inputs = cutout_batch(inputs, CUTOUT_SIZE)` with a resolution-scheduled block: for the first 50% of the time budget, `F.interpolate` downscales the float32 input to 24×24 (then `.contiguous(memory_format=channels_last)` to keep the layout the compiled graph expects) and Cutout is scaled to `round(16·24/32)=12` px; after frac≥0.5 the input stays 32×32 with full Cutout and a one-time `>> FULL-RES PHASE` marker prints. Added `full_res_announced = False` to the pre-loop counters (L222). Resize sits between the `targets.to()` line and the autocast forward — OUTSIDE the compiled forward and OUTSIDE autocast (operates on float32), so reduce-overhead captures one stable CUDA-graph per phase-shape. AST OK; `git diff --name-only` == train.py only (23 insertions, 1 deletion).

### Surprises & Discoveries
None during implementation. Confirmed `cutout_batch` reads `h,w` from `x.shape`, so it auto-adapts to the resized tensor — passing a scaled `size` keeps the hole at the same 50%-side fraction with no change to the helper. The architecture needs no change: global-avg-pool before the FC head accepts the 24→12→6 feature map (vs 32→16→8) unchanged.

### Decisions
No deviations from plan. Resize placed before Cutout (so Cutout operates at the phase resolution with a proportionally-scaled hole) rather than after (which would shrink a full-res hole on downscale). `align_corners=False` (standard for bilinear image resize). Kept the resize on float32 (before autocast) to avoid a bf16 interpolate. The single 24→32 transition at frac 0.5 yields exactly one one-time CUDA-graph recapture (a single slow step, negligible vs Σdt) — the EXP-042 cudagraph rule is respected because the input shape is constant within each phase (no data-dependent branch inside the compiled forward).

## Run Log

### Run 1
- **Description**: Progressive resolution schedule on the EXP-054 AugMix-p0.5 best — first 50% of the 300s budget at 24×24 (0.5625× conv FLOPs, proportional Cutout-12), then 32×32 to finish. Tests whether cheaper early-phase steps buy net-new effective epochs (the proven EXP-003 throughput→epochs win mechanism) on an epoch-starved net, lifting best_test_acc to ≥96.55. Key diagnostics: num_epochs (>91 confirms the 24×24 phase realized a speedup) and the dt distribution (expect a low-dt 24×24 cluster + an ~8ms 32×32 cluster; a 14-16ms cluster would mean a cudagraph break). Launching on an idle GPU.
- **Job ID**: (local, background bash)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09
- **Observations (Gate A, 24×24 phase)**: dt = **6ms** (vs 8ms 32×32 baseline) → the resolution reduction realized a ~25% per-step speedup AND cudagraph did NOT break (6ms is lower, not the 14-16ms EXP-042 break signature). GPU 1 at 40% util, GPU 0 idle (uncontended). Loss descending, eval climbing. The "buy epochs" mechanism was firing in the low-res phase as hypothesized.
- **Key Metrics**: best_test_acc **95.82%** (best ep89; −0.63pp vs baseline 96.45, ≪ 96.55 bar) | final_test_loss 0.2055 (> EXP-054's 0.1968) | **training_seconds 300.0 (compute budget respected exactly)** | **total_seconds 617.2 (WALL BREACH: >600 by 17.2s — the most severe to date, partly change-caused)** | num_epochs **89 (FEWER than baseline 91)** | num_steps 34438 | num_params 4,299,866 | peak_vram_mb 511.7. **dt: TWO regimes — ~450 samples @6ms (24×24 phase, 59 ep) + ~235 samples @10ms (32×32 tail, 30 ep)**; the full-res marker fired ep59/frac0.500. 0 NaN/error.
- **CRITICAL FINDING — the mechanism backfired on throughput**: the 24×24 phase ran at 6ms (saved 2ms) BUT the 32×32 tail ran at **10ms, not the 8ms baseline** (+2ms). Two input shapes under `torch.compile(reduce-overhead)` produce TWO CUDA-graphs; the multi-graph state + per-step `F.interpolate` raised the full-res dt by 25%, exactly CANCELLING the low-res saving → net **89 epochs < 91 baseline**. So progressive resizing did NOT buy net epochs here — it cost them. Compounded with low-res early training under-learning fine detail that the throughput-starved 30-epoch tail couldn't re-sharpen, best_test_acc regressed −0.63pp.

## Experimental Adjustments
(none)

## Errors & Dead Ends
(none)

## Verification Results

### Conditions Checked
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **95.82%** < 96.55. **FAILED** decisively (−0.63pp vs baseline 96.45). (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: **WALL BREACH** — total_seconds 617.2 > 600 (by 17.2s, the most severe to date). training_seconds 300.0 (gated compute budget respected exactly), num_params 4,299,866 ✓, 0 NaN/error ✓, summary printed ✓. Unlike EXP-065's 2.5s base-variance overrun, this 17.2s breach is **partly change-caused**: the 2-cudagraph reduce-overhead state + per-step `F.interpolate` raised both the compile/recapture wall AND the full-res dt (8→10ms).
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only ✓; no new deps (F.interpolate/torch already imported) ✓; seed 42 unchanged ✓; evaluate() once/epoch (loop structure unchanged) ✓; uncontended (GPU 0 idle throughout) ✓.

**Verdict deliberation (no-improvement vs invalid)**: total_seconds 617.2 > 600 breaches the wall hard constraint (goal file: "Any run exceeding 10 minutes total must be... treated as a failure"), and this breach is partly change-caused — a stronger case for `invalid` than EXP-061/065. Chose **no-improvement**, consistent with the EXP-061 (604.6s, change-caused eval overhead) and EXP-065 (602.5s) precedent, because: (a) necessary condition 1 (accuracy) fails FIRST and DECISIVELY (95.82 ≪ 96.55) on a fully trustworthy value — the idea lost ON ITS MERITS, independent of wall; (b) the actively-gated Σdt=300s compute budget was respected EXACTLY (training_seconds 300.0, fair training); (c) recording the real metric (−0.63pp) is more informative to future loops than NaN; (d) the breach does not make the accuracy untrustworthy. The wall breach is documented prominently as an INDEPENDENT closure reason (progressive resizing is wall-INFEASIBLE on this recipe regardless of accuracy) and strengthened into infra-errors.

**Verdict**: no-improvement — valid training run (Σdt=300s respected) that decisively missed the accuracy bar (95.82 < 96.55, −0.63pp). Progressive resolution scheduling backfired: the 2-cudagraph multi-shape compile raised the 32×32-tail dt 8→10ms, cancelling the 24×24 saving → 89 ep < 91 baseline (NO net epochs bought), and the low-res early training under-learned fine detail the throughput-starved tail couldn't recover. Wall breach noted (617.2 > 600, most severe to date, partly change-caused).
