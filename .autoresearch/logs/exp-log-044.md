# EXP-044: Within-cliff asymmetric widening — stage widths 64/160/256 (dt-gated)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-044.md
- **Plan**: plans/plan-044.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-044
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (pre-registered GATE_KILL branch (iii) — the dt screen resolved decisively; no relaunch per plan)

## Implementation Notes

### Summary
Exactly per plan Milestone 1, three edits to `train.py` only: (1) constant `WIDTH_MULT = 4` replaced by `STAGE_WIDTHS = (64, 160, 256)` with a comment recording the baseline widths; (2) `ResNet.__init__` signature changed from `width_mult=1` to `widths=(16, 32, 64)` and the width derivation `16/32/64 * width_mult` replaced by tuple unpack; (3) construction call passes `STAGE_WIDTHS` and the model print line now shows `64/160/256 asymmetric`. No other line touched — blocks, init, transforms, loader, optimizer, schedule, compile/warmup, timed loop, and once-per-epoch eval cadence are byte-identical to baseline @ 1990397. CPU sanity suite (`/tmp/exp044_sanity.py`, run with `CUDA_VISIBLE_DEVICES=""`) passed all four checks: param count exactly 4,793,290 (matches the analytic delta +506,880 conv / +384 BN over baseline 4,286,026); forward (4,3,32,32)→(4,10) finite; pad shortcuts assert 96 padded channels at BOTH transitions (160−64 and 256−160) with identity shortcuts everywhere else; 2-step train smoke loss 3.91→2.21.

### Surprises & Discoveries
- None at implementation time. `BasicBlock`'s pad-shortcut logic handled the asymmetric widths with zero modification, as the plan predicted. A pleasing coincidence: both stage transitions now pad exactly 96 channels.

### Decisions
- Kept the `widths` tuple as a default-argument signature (`widths=(16, 32, 64)`) rather than requiring it, mirroring the old `width_mult=1` default style — smallest-diff choice, no behavioral difference (the call site always passes `STAGE_WIDTHS`).

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background task bdtr3bk2k (/tmp/exp044_composite.sh; watchdog stdout at /tmp/claude-0/.../tasks/bdtr3bk2k.output)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: GATE_KILLed (pre-registered dt screen, exit 47)
- **Started**: 2026-06-10 21:48:56 (gates clear on poll 1: apps=0, load=4)
- **Ended**: 2026-06-10 ~21:50:15 (killed at watchdog tick 5, ~80s into run)

Description:
- Single gated run of the 64/160/256 asymmetric-width net under the unchanged certified recipe. Dense-law projection: dt ≈ 24.8ms → ~125 epochs; the composite launcher GATE_KILLs in ~90s if the first-3-window median exceeds 28ms (kernel mispricing branch). Expected outcomes per pre-registered branches: (i) best ≥ 96.81 → first capacity gain at converged epochs; (ii) plateau ≈ 96.3–96.5 with family test_loss → capacity level-saturated at 4×, gap closed; (iii) GATE_KILL → new kernel-pricing law for 32-aligned-not-64-aligned widths.

Observations:
- Launch via /tmp/exp044_composite.sh (exp042 D0-median pattern; sole diff: contention threshold = max(D0×1.25, 28ms)). Gate-decision watcher armed (task buxaa7ddr); composite task bdtr3bk2k notifies on completion.
- Gates clear immediately (poll 1: apps=0, load=4 — pristine host). Startup normal; params printed **4,793,290** as predicted; 97 batches/epoch (source: run.log header lines).
- **GATE_KILL at tick 5**: three ~350-step windows measured **32.4 / 31.7 / 32.6 ms** → D0 median 32.4ms > 28ms gate; projected ~95 epochs (source: task bdtr3bk2k output). Printed dt agrees (32ms uniform, img/s ~16,050 vs baseline ~23,000) and host stayed clean (load 8.5, zero GPU-0 apps post-kill) — this is true kernel cost, NOT contention.
- Dense-law prediction was 24.8ms (+2.4ms for +18% FLOPs); measured +10.0ms over baseline ≈ 4× the law. The 160-channel stage-2 (32-aligned but NOT 64-aligned) misprices — consistent with reinterpreting EXP-005's 80/160/320 collapse as alignment+cliff, and with EXP-034's 48/96/192 fallback (27.4ms, also non-64-aligned widths) running above its dense-law projection.
- Training was healthy pre-kill: loss falling cleanly, eval ep1–10 trail 29.97 → 77.48 (no NaN, no divergence) — the kill is purely the throughput screen (source: run.log eval lines).

Key Metrics:
- D0 windowed dt: 32.4ms (windows 32.4/31.7/32.6 over steps ~300–1000) → projected ~95 epochs = starvation regime (source: bdtr3bk2k output, run.log step prints)
- num_params: 4,793,290 (printed, matches analytic) (source: run.log L2)
- best_test_acc: N/A — killed at ~11% progress per pre-registered screen; metric NaN per plan branch (iii)

Key Metrics:
- (pending)

## Verification Results

### Conditions Checked

- **Integrity pre-condition / dt gate**: FAILED BY DESIGN — D0 = 32.4ms > 28ms with a provably clean host (gates clear at launch, load 8.5 and zero GPU-0 apps at decision, three consistent ~350-step windows). This is the pre-registered branch (iii) from brainstorm-044/plan-044: kernel mispricing, not contention — rerun would re-measure the same kernels, so NO relaunch.
- **Condition 1 (best_test_acc ≥ 96.81)**: skipped — not evaluable; run killed at ~11% progress per the screen. Metric NaN per plan.
- **Conditions 2–3**: skipped — aborted after pre-condition branch.
- **Verdict basis (for analyze)**: `invalid` with metric NaN per plan-044 Abort Criteria — "GATE_KILL: ... verdict `invalid`, metric NaN, idea closed on hardware grounds".

### Informational Metrics

- Not collected (conditions did not pass). Hardware datum recorded instead: 64/160/256 prices at 32.4ms (+10.0ms vs baseline at +18% FLOPs; dense law predicted +2.4ms).

## Errors & Dead Ends

### 2026-06-10 — GATE_KILL: 160-channel stage misprices ~4× off the dense law
- Error: `GATE_KILL: D0=32.4ms > 28ms (projected 95 epochs < ~111)` — exit 47 from /tmp/exp044_composite.sh
- Root cause: non-64-aligned channel width. 160 = 5×32 keeps every layer ≤256 (no EXP-040 cliff) yet the stage-2 kernels (160×160, 64→160, 160→256) cost +10ms vs baseline where +18% FLOPs predicts +2.4ms — the inductor/cuDNN kernel selection on H20+compile+channels_last+bf16 needs 64-multiples, not just 32-multiples. Retro-consistent: EXP-005 (80/160/320) and EXP-034's 48/96/192 fallback (27.4ms) both contained non-64-multiples and both ran above dense-law projections.
- Source: bdtr3bk2k output (gate windows 32.4/31.7/32.6ms); run.log step prints (32ms uniform, img/s 16k, clean host)
- Do NOT retry: any stage width that is not a multiple of 64 on this stack (only 64/128/192/256 are within both the cliff AND alignment constraints — and 192 is unverified-but-suggested-slow by EXP-034's 27.4ms fallback; dt-gate it if ever used). The only strictly-within-laws asymmetric candidates left are permutations of {64,128,192?,256} — i.e., 64/192/256 (gate first) — or nothing.

## Human Notes

> (none — autopilot)
