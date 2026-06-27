# EXP-046: Anti-aliased shortcut — avg-pool the identity path at stage transitions

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-046.md
- **Plan**: plans/plan-046.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-046
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification condition 1 failed — best_test_acc 96.65 < 96.81; pre-registered branch (ii) absorbed null)

## Implementation Notes

### Summary

One-line logic change in `train.py` `BasicBlock.forward` (M1): the pad shortcut's strided slice `shortcut[:, :, ::self.stride, ::self.stride]` was replaced with `if self.stride != 1: shortcut = F.avg_pool2d(shortcut, self.stride)`; the channel zero-pad line is unchanged and nothing else in the file was touched. Affects exactly two forward sites (layer2[0], layer3[0]). CPU sanity (`/tmp/exp046_sanity.py`, CUDA_VISIBLE_DEVICES="") passed all checks: param count exactly 4,286,026 (zero-param change confirmed); need_pad True only at layer2[0]/layer3[0] with stride 2 and pad_channels = out/2; forward (4,3,32,32)→(4,10) finite; semantic check — on a constant-valued input the avg-pooled shortcut equals the old sliced shortcut, on a random input they differ while shapes match (anti-aliasing active); a stride-1 block's forward is bit-identical to the manual identity-shortcut computation; 2-step train smoke with finite decreasing loss (3.1037 → 1.4894). M2: `/tmp/exp046_composite.sh` built from the validated `/tmp/exp045_composite.sh` via sed (thresholds 31→26: GATE_KILL comparison, kill message, contention floor), diff-confirmed (4 lines changed — the `> 31ms (projected` pattern appears in both the header comment and the kill message), chmod +x.

### Surprises & Discoveries

- None at implementation time. The change applied exactly as planned; the guard `if self.stride != 1` is needed because need_pad is also True for stride-1 channel-increase blocks in principle, though in this architecture need_pad only fires at the two stride-2 sites (asserted in sanity).

### Decisions

- Kept the existing `if self.need_pad:` structure and nested the stride guard inside it, rather than restructuring the forward — minimal diff, preserves the EXP-020-validated zero-pad path byte-for-byte.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (pending — composite script launches train.py in background)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log (+ composite stdout via task output)
- **WandB**: N/A
- **Status**: completed (rc=0, PROC_EXITED tick 33)
- **Started**: 2026-06-10 22:13
- **Ended**: 2026-06-10 22:21 (total_seconds 485.3)

Description:
- Single gated run of the anti-aliased-shortcut variant on GPU 0 via `/tmp/exp046_composite.sh`: dual launch gates (zero GPU-0 compute apps AND 1-min load < 60, poll 30s×240), then `uv run train.py > run.log 2>&1` with a 44×15s watchdog. Windowed dt from pct deltas; GATE_KILL if D0 (median of first 3 ≥200-step windows) > 26ms; CONTENTION_KILL on 4 consecutive windows > max(D0×1.25, 26); STARTUP_KILL tick 12; NaN/divergence/WALL_CAP guards. Expected: baseline-identical signatures (dt ≈ 22.4–23.0ms, ~136–139 epochs, params 4,286,026) with best_test_acc resolving one of the pre-registered branches — (i) ≥ 96.81 improvement (replicate pair if 96.70–96.80), (ii) 96.42–96.72 at family test_loss = absorbed null closing the downsample-quality class, (iii) GATE_KILL = avg_pool2d mispricing, invalid.

Observations:
- Dual launch gates cleared at poll 1 (apps=0, load=6); GATE_DECISION D0=22.5ms, projected_epochs=137, contention_thresh=28.1ms (source: composite stdout, task bfxm7rkd6). avg_pool2d shortcut prices at ~zero — dt squarely in the expected 22.4–23.0 band, branch (iii) eliminated.
- Pristine run end to end: all 30 ≥200-step windows in 22.0–22.7ms, slow_streak never >0, PROC_EXITED tick 33, rc=0 (source: composite stdout ticks 3–32).
- Plateau identical to baseline family: last 8 evals 96.51–96.65, test_loss 0.1848–0.1880 — no level shift from anti-aliasing (source: composite stdout LAST 8 EVALS).

Key Metrics:
- best_test_acc: 96.65% @ ep 133 (source: run.log summary; LAST 8 EVALS)
- final_test_loss: 0.1880 (family ~0.185 → absorbed-null branch)
- num_epochs: 139; num_steps: 13,428; startup_seconds: 19.2; peak_vram_mb: 1613.0; training_seconds: 300.0; total_seconds: 485.3; num_params: 4,286,026

## Verification Results

### Conditions Checked

- **Integrity pre-condition** — PASS. All 30 ≥200-step windows 22.0–22.7ms (mean ≈22.4 ≤ 24, none > 27); num_epochs 139 ∈ [130,142]; printed params == 4,286,026; training_seconds == 300.0; eval lines 139 ≤ num_epochs 139. (source: composite stdout ticks 3–32; run.log summary)
- **Condition 1: best_test_acc ≥ 96.81** — FAIL. Read 96.65 (< 96.81 and below the 96.70 replicate-band floor, so no replicate pair per plan). (source: `grep "^best_test_acc:" run.log` → 96.65%)
- **Condition 2: within budget** — skipped — aborted after prior failure (informationally: rc=0, total_seconds 485.3 ≤ 600).
- **Condition 3: eval cadence** — skipped — aborted after prior failure (informationally: 139 evals ≤ 139 epochs).

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
