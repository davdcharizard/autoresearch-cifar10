# EXP-047: Multi-scale decision head — fc over concat[GAP(stage2), GAP(stage3)]

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-047.md
- **Plan**: plans/plan-047.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-047
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification condition 1 failed — best_test_acc 96.15 < 96.81; pre-registered branch (iii) dilution, below mean band)

## Implementation Notes

### Summary

Two edits in `train.py` `ResNet` (M1): `self.fc = nn.Linear(w2 + w3, num_classes)` and the forward now pools BOTH stage outputs — `h2 = GAP(layer2_out).view(B,-1)` (128-d), `h3 = GAP(layer3_out).view(B,-1)` (256-d) — and classifies from `fc(cat([h2, h3], dim=1))` (384-d). BasicBlock, recipe constants, and the training loop untouched. CPU sanity: params exactly 4,287,306 (= baseline 4,286,026 + 128×10); fc.in_features == 384; forward (4,3,32,32)→(4,10) finite; manual-recomputation equivalence confirms routing is wired exactly as designed (h2 dims 128, h3 dims 256); stage-2 fc columns receive nonzero gradient; train smoke at lr 0.01 over 6 steps decreases 3.1052 → 0.2402. M2: `/tmp/exp046_composite.sh` verified present and reused verbatim (gate 26ms / contention floor 26 already correct for baseline-band dt).

### Surprises & Discoveries

- The plan's original smoke criterion (2 steps at lr 0.05 on 16 random-labeled samples) failed with loss 3.219 → 3.413 — pure overshoot, not a wiring bug: all structural checks passed and the same model at lr 0.01 decreases monotonically over 6 steps. The 384-d Kaiming-init fc has larger initial logit variance than the 256-d baseline head, making the high-lr toy smoke flakier.

### Decisions

- Replaced the flaky lr-0.05/2-step smoke with lr-0.01/6-step monotone-decrease as the trainability check (recorded in plan M1 tick). No design deviation from the plan otherwise.
- Pooled tensors are flattened BEFORE concat (cat on 2-d tensors) to avoid any channels_last format edge case on 4-d concat.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (pending — composite launches train.py in background)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log (+ composite stdout via task output)
- **WandB**: N/A
- **Status**: completed (rc=0, PROC_EXITED tick 35)
- **Started**: 2026-06-10 22:33
- **Ended**: 2026-06-10 22:42 (total_seconds 520.0)

Description:
- Single gated run of the multi-scale-head variant on GPU 0 via `/tmp/exp046_composite.sh` (reused verbatim): dual launch gates, then `uv run train.py > run.log 2>&1` with the 44×15s watchdog (GATE_KILL D0 > 26ms; CONTENTION/STARTUP/NaN/divergence/WALL_CAP guards). Expected: baseline-identical signatures (dt ≈ 22.4–23.0ms, ~138 epochs, params 4,287,306) resolving one of the pre-registered branches — (i) ≥ 96.81 improvement (replicate pair if 96.70–96.80), (ii) mean band 96.42–96.72 at family test_loss = routing class closed inert, (iii) ≤ 96.42 = dilution (closed from below), (iv) GATE_KILL = invalid.

Observations:
- Dual launch gates cleared; GATE_DECISION D0=22.7ms, projected_epochs=136, contention_thresh=28.4ms (source: composite stdout, task bwn2ey94a). GAP(stage2)+concat prices at ~zero — dt in the expected 22.4–23.0 band, branch (iv) eliminated. Run proceeding to full completion.

- Pristine run end to end: all 32 ≥200-step windows in 21.7–22.8ms, slow_streak never >0, PROC_EXITED tick 35, rc=0 (source: composite stdout ticks 3–34).
- Plateau LEVEL clearly BELOW family: last 8 evals 96.09–96.15 (family 96.5–96.7), test_loss 0.1905–0.1939 vs family ~0.185 (source: composite stdout LAST 8 EVALS). This is pre-registered branch (iii): mid-level stage-2 features dilute the decision head — a real negative effect (mean read ≈ −0.42 ≈ −2.6σ vs baseline-recipe mean), not noise.

Key Metrics:
- best_test_acc: 96.15% @ ep 132 (source: run.log summary; LAST 8 EVALS)
- final_test_loss: 0.1917 (above family ~0.185)
- num_epochs: 138; num_steps: 13,329; startup_seconds: 22.8; peak_vram_mb: 1639.5; training_seconds: 300.0; total_seconds: 520.0; num_params: 4,287,306

## Verification Results

### Conditions Checked

- **Integrity pre-condition** — PASS. All 32 ≥200-step windows 21.7–22.8ms (mean ≈22.5 ≤ 24, none > 27); num_epochs 138 ∈ [130,142]; printed params == 4,287,306; training_seconds == 300.0; eval lines 138 ≤ num_epochs 138. (source: composite stdout; run.log summary)
- **Condition 1: best_test_acc ≥ 96.81** — FAIL. Read 96.15 (< 96.81, far below the replicate band). (source: `grep "^best_test_acc:" run.log` → 96.15%)
- **Condition 2: within budget** — skipped — aborted after prior failure (informationally: rc=0, total_seconds 520.0 ≤ 600).
- **Condition 3: eval cadence** — skipped — aborted after prior failure (informationally: 138 evals ≤ 138 epochs).

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
