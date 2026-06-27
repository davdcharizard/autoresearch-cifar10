# EXP-061: Stage-1-heavy depth reallocation [3,3,3] → [4,3,2] at equal FLOPs

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-061.md
- **Plan**: plans/plan-061.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-061
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Three hunks to train.py exactly as planned: (1) `NUM_BLOCKS = 3` → `STAGE_BLOCKS = (4, 3, 2)` with the EXP-017 mirror rationale in a comment; (2) `ResNet.__init__` now unpacks per-stage counts `b1, b2, b3 = stage_blocks` into the three `_make_layer` calls (`_make_layer` and `BasicBlock` unchanged — they were already per-stage capable); (3) call site `ResNet(STAGE_BLOCKS, ...)` and banner print updated (`ResNet-20 (4x wide, stages (4, 3, 2))`). Recipe (transforms, loss, optimizer, schedule, warmup, timed loop, eval) byte-identical to baseline. CPU sanity ALL PASS: params exactly 3,179,338; structure lens 4/3/2 with widths 64/128/256 and stride-2 only at the two stage transitions; fwd/bwd shape (16, 10); smoke decreasing.

### Surprises & Discoveries
- None at implementation time — `_make_layer` already accepted a per-stage count, so the change is as minimal as the plan predicted (the uniform `num_blocks` was simply passed three times before).

### Decisions
- Banner prints the stage tuple to make the variant self-identifying in run.log (depth arithmetic `2*sum+2` still reads 20).

## Experimental Adjustments

## Run Log

### Run 0 (GPU probe — uncharged, pre-launch)

Metadata:
- **Job ID**: foreground Bash (probe ran clean first attempt)
- **Log file(s)**: probe stdout (recorded below)
- **Status**: completed

Description:
- dt pricing of the (4,3,2) shape per plan M2. Per-block law (EXP-034: ~2.5ms/block width-independent) predicts family-band P ≈ 21.5–23.5; stage-1 blocks carry 4× the activations of stage-3 blocks so a modest rise is plausible. Branches: P ≤ 23.5 launch (bands from formula steps ∈ [300000/(P+1.5), 300000/(P+0.1)]); (23.5, 26] launch with priced deferral toll; > 26 no launch — cost-closure (EXP-040/042 precedent).

Observations:
- Gate clean (apps=0, load 9.7 < 40); probe asserted params 3,179,338 in vivo. **P = 22.51ms ∈ [21.5, 23.5] → LAUNCH branch.** The EXP-034 per-block law holds for stage-heavy reallocation: (4,3,2) is dt-NEUTRAL despite stage-1 blocks carrying 4× the activations — block count, not activation volume, prices the step. **Probe-revised bands (recorded pre-launch)**: run dt ≈ 23.0–23.5 (probe + 0.5–1.0 offset); D0 ∈ [22.2, 23.8]; steps ∈ [12,495, 13,269] (= 300000/(P+1.5) … 300000/(P+0.1)); epochs ∈ [126, 138] (steps/97.65 ±2).

Key Metrics:
- P = 22.51ms (load 9.7) — source: probe stdout above

### Run 1

Metadata:
- **Job ID**: background task b85sa0wqs (composite), b3orotrr0 (watcher)
- **Log file(s)**: run.log (training); /tmp/exp061_composite_run1.log (gate/watchdog)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-11 09:28
- **Ended**: 2026-06-11 (RC=0 at watchdog tick 33, ~490s)

Description:
- Single gated run of the (4,3,2) reallocation via /tmp/exp061_composite.sh (single-phase launcher, neutralized tail threshold; dual gates; GATE_KILL D0>26; contention 4×>max(26, D0×1.25); NaN/divergence guards; WALL_CAP). Expected: params 3,179,338 (THE integrity assert), steps/epochs per probe-revised bands, ~460–510s total. ep1 tripwire relaxed to ≥25 (architecture change); contamination judged by step ledger. Decision per plan-061 branches: (i) ≥96.81 → byte-identical replicate, pair MEAN decides; (ii) [96.41, 96.73] → allocation curve flat-topped at uniform, axis closed bidirectionally; (iii) (96.73, 96.81) → no-improvement; (iv) <96.41 → stage-3 block COUNT load-bearing, axis closed from below; (v) infra → relaunch ≤2.

Observations:
- PRISTINE run. Gates poll 1 (apps=0, load 9); GATE_DECISION D0=23.3 ∈ [22.2, 23.8] (probe 22.51 + 0.8 offset — historical +0.5–1.0 holds); windows ticks 5–32 read 22.3–23.3ms, slow_streak 0 throughout; PROC_EXITED tick 33, RC=0. (source: /tmp/exp061_composite_run1.log)
- Ledger ON probe-revised bands: 12,986 steps ∈ [12,495, 13,269]; 134 epochs ∈ [126, 138]; **params 3,179,338 exactly (the architecture-landed assert)**; startup 9.3s; 300.0s charged; 479.2s total; VRAM 1,799MB (predicted 1.8–2.2GB — more 32×32 activations). (source: run.log summary)
- Trajectory clean but DEPRESSED at the plateau: ep1 37.97 (family-normal — no early-heat penalty from the reallocation), smooth climb, converged-FLAT plateau 96.19–96.39 over the last 8 evals at FAMILY test_loss (0.190 vs ~0.185–0.19) — a level deficit, not starvation (anneal complete at 134 ep) and not a basin-quality signature (test_loss family-equal). (source: run.log eval lines)

Key Metrics:
- best_test_acc: 96.39 @ ep131 (source: run.log summary; branch (iv) boundary — 0.02 BELOW the family floor 96.41, = mean−1.1σ)
- final_test_acc: 96.28 / final_test_loss: 0.1900 @ ep134 (source: run.log summary)
- training_seconds: 300.0 | total_seconds: 479.2 | startup: 9.3 | num_epochs: 134 | num_steps: 12,986 | num_params: 3,179,338 | peak_vram_mb: 1,799.4 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Integrity pre-condition** (plan-061 M4 / step 0): PASS — RC=0; no kill markers; D0 23.3 ∈ [22.2, 23.8]; all windows ≤ 23.3 < 29.1 thresh; steps 12,986 ∈ [12,495, 13,269]; epochs 134 ∈ [126, 138]; `num_params: 3,179,338` exact; `training_seconds: 300.0`; `total_seconds: 479.2` ≤ 600; 134 evals = 134 epochs; ep1 37.97 ≥ 25 (clears the standard 30 too); zero NaN. (source: run.log + /tmp/exp061_composite_run1.log)
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: FAIL — best_test_acc 96.39. Pre-registered branch (iv): < 96.41 → stage-3 block COUNT is load-bearing (its added params were not, per EXP-017); the allocation axis closes from below on the stage-1-heavy side. Boundary caveat for analysis: the read is only 0.02 under the floor (mean−1.1σ), so the honest strength of the claim is "no gain, mild-deficit-consistent" — but under EITHER reading (low family tail or true small deficit) the favorable-slope hypothesis is REFUTED and uniform remains the allocation optimum. First-failure-stop: no escalation.
- **Condition 2 — completes within budget**: PASS informationally (479.2 ≤ 600).
- **Condition 3 — validation ≤ once/epoch**: PASS informationally (134/134, structural).

### Informational Metrics

- peak_vram_mb: 1,799.4 | num_epochs: 134 | num_params: 3,179,338 (informational; Condition 1 failed)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
