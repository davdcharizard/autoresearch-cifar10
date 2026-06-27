# EXP-066: Kernel-size corner — 5x5 stem (launchable) + 5x5 stage-3 (probe-only) behind the internal-control GPU probe

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-066.md
- **Plan**: plans/plan-066.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-066
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Milestone 1: `/tmp/exp066_sanity.py` builds both variants by patching the imported train.py
ResNet — S replaces the stem conv with 5x5/padding=2 (+3,072 params), T replaces all 6 stage-3
convs with 5x5/padding=2 (stride preserved on the transition conv). All assertions passed:
exact param counts (S 4,289,098; T 10,053,194), spatial dims preserved (stem 32x32, stage-3
8x8), forward/backward clean, 5 SGD steps at lr 0.01 reduce loss on both variants. Milestone 2:
`/tmp/exp066_gpu_probe.py` (EXP-064 internal-control pattern) times B/S/T compiled charged
steps (default compile, channels_last, bf16, 3-iter warmup, 40 timed steps incl. optimizer.step,
median of last 30) in one session; P_norm = 22.4×P/B. Probe launched behind clean gates.
train.py is NOT modified unless the pre-registered launch branch (P_norm_S ≤ 22.9) fires.

### Surprises & Discoveries
- 5x5 stride-2 padding=2 preserves the expected 16→8 spatial map exactly ((16+4−5)//2+1 = 8),
  so the T variant needs no other geometry changes.
- None otherwise yet; probe pending.

### Decisions
- Stage-3 (T) is probe-only per the plan's pre-run arithmetic: even fast-path dense-law pricing
  (~30.3ms → ~101 epochs → −0.53 starvation) makes its launch inequality unsatisfiable; it is
  timed purely as the second kernel-size pricing datum.
- Probe steps include optimizer.step (unlike the train.py warmup) to mirror the full charged
  step; identical procedure for all three nets keeps the internal-control ratio valid.

## Experimental Adjustments

## Run Log

### Run 1 (GPU probe — uncharged)

Metadata:
- **Job ID**: background task bx21972m3 (local)
- **Log file(s)**: /tmp/exp066_probe_out.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-11 (gates: GPU 0 zero compute apps; host load 7.53)
- **Ended**: 2026-06-11 (exit 0)

Description:
- Internal-control GPU probe pricing the kernel-size corner: B (baseline control, session-valid
  iff 21.5–23.5ms), S (stem-5x5), T (stage3-5x5). Pre-registered branch: P_norm_S ≤ 22.9 →
  implement stem-5x5 in train.py and run the full experiment; P_norm_S > 22.9 → NO-LAUNCH,
  kernel-size corner closes on cost at zero charged seconds (verdict invalid/NaN). T is
  probe-only cartography. Expected: T ≥ ~28ms; S uncertain (fast path vs slow tier is exactly
  the unknown being measured).

Observations:
- Session VALID: B = 22.18ms ∈ [21.5, 23.5]; load 10.11 → 9.86 across the session
  (source: /tmp/exp066_probe_out.log L1-5).
- S(stem5x5) = 22.44ms → **P_norm_S = 22.66 ≤ 22.9 → LAUNCH branch fires** (toll +0.26ms ≈
  −1–2 epochs ≈ −0.02pp, negligible) (source: /tmp/exp066_probe_out.log L3,L6).
- T(stage3-5x5) = 30.72ms → P_norm_T = 31.02 — within 2.4% of the fast-path dense-law
  prediction (30.3ms for 1.59× FLOPs). NEW PRICING DATUM: square 5x5 kernels stay ON the
  dense fast path (unlike 1D/odd shapes, EXP-044/045); kernel SIZE prices at FLOPs, not at a
  slow tier. Stage-3 5x5 remains unlaunchable on starvation arithmetic as pre-registered
  (source: /tmp/exp066_probe_out.log L4,L7).

Key Metrics:
- B=22.18 S=22.44 T=30.72 (medians of last 30 of 40 steps); P_norm_S=22.66, P_norm_T=31.02
  (source: /tmp/exp066_probe_out.log L8).

### Run 2 (full stem-5x5 run — charged)

Metadata:
- **Job ID**: background task bypsic286 (composite /tmp/exp066_composite.sh)
- **Log file(s)**: run.log (project root); composite telemetry /tmp/exp066_composite_run1.log
- **WandB**: N/A
- **Status**: completed (rc=0, pristine — no watchdog events)
- **Started**: 2026-06-11 11:39
- **Ended**: 2026-06-11 ~11:48

Description:
- Full 300s-budget run of train.py with the single stem change (conv1 3x3→5x5, padding 2,
  +3,072 params → 4,289,098). Everything else byte-identical to baseline. Composite enforces
  dual launch gates, D0 gate (>26ms kill), contention/NaN/divergence/startup kills, WALL_CAP.
  Expected: family signatures (dt ~22.4–22.7ms, ~137–139 epochs, steps 13.1–13.6k), plateau
  inside the family band 96.41–96.73 (absorption-law null); a read ≥ 96.81 triggers the
  pre-registered replicate-pair protocol.

Observations:
- Pristine telemetry: GATES_CLEAR poll 1 (load 8), D0 = 22.7ms (= probe P_norm_S 22.66),
  projected 136 epochs, all 30 post-gate windows 22.0–23.3ms, slow_streak never > 0
  (source: /tmp/exp066_composite_run1.log L1-34).
- Result is BELOW the family band: best 96.14 < floor 96.41 (mean − 2.7σ) — a REAL structural
  negative (EXP-030/047 class), not an absorbed null. test_loss 0.1929 vs family ~0.185
  (source: /tmp/exp066_composite_run1.log L37-39).
- Plateau still mildly creeping at cutoff: last 8 evals 96.05→96.14, best AT the final epoch —
  the still-organizing signature (EXP-030 rhyme): the 75-weight stem filters appear to drag
  early-feature organization through the whole run (source: /tmp/exp066_composite_run1.log
  L48-55).

Key Metrics:
- best_test_acc: 96.14% @ ep137 (source: /tmp/exp066_composite_run1.log L37)
- training_seconds 300.0; total_seconds 480.5; startup 10.7; peak_vram 1613.0MB; num_epochs
  137; num_steps 13,266; num_params 4,289,098 (source: /tmp/exp066_composite_run1.log L40-46)

## Verification Results

### Conditions Checked

- **Integrity pre-condition (gates Condition 1)**: PASS — num_steps 13,266 ∈ [13,100, 13,600];
  num_params 4,289,098 (exact expected); dt windows 22.0–23.3ms family-band with zero slow
  streaks; epochs 137 ∈ [134, 141]; no contamination signature; D0 matches the probe reading
  (source: /tmp/exp066_composite_run1.log L3-33, L44-46). The result is a trustworthy measured
  read of the stem-5x5 intervention.
- **Condition 1 — best_test_acc ≥ bar 96.81 (baseline 96.71 + 0.1, via exp-index baseline)**:
  FAIL — 96.14 < 96.81. Pre-registered branch (iii): best < 96.41 (family floor) →
  no-improvement with structural-negative reading (96.14 = mean − 2.7σ). Replicate-pair
  protocol (branch i) not triggered (source: /tmp/exp066_composite_run1.log L37).
- **Condition 2 — run completes ≤ 600s total**: not evaluated for verdict (aborted after first
  failure per protocol); informationally PASS — total_seconds 480.5, rc=0.
- **Condition 3 — validation at most once per epoch**: not evaluated for verdict;
  informationally PASS — single `evaluator.evaluate` call per epoch loop, diff is 1 line in
  `ResNet.__init__` (`git diff` confirms no loop changes).

### Informational Metrics

- peak_vram_mb: 1613.0 (= family — the +3,072-param stem is VRAM-invisible)
- num_epochs: 137 (family band)
- num_params: 4,289,098 (as designed)
- Probe pricing data (the corner's cost map): B=22.18, S=22.44, T=30.72;
  P_norm_S=22.66, P_norm_T=31.02 — square 5x5 kernels are FAST-PATH (dense-law FLOPs-priced;
  T within 2.4% of the 30.3ms prediction), unlike 1D/odd shapes (EXP-044/045)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
