# Experiment Log EXP-038

## Execution
- **Created**: 2026-06-09
- **Brainstorm**: brainstorm/brainstorm-038.md
- **Plan**: plans/plan-038.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-038
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Single edit to `train.py` L100-101: replaced the uniform width derivation
(`k = width_mult; w1,w2,w3 = 16k,32k,64k` → {64,128,256}) with explicit fat-head widths
`w1, w2, w3 = 44, 128, 320`. Milestone 1 complete: AST OK; model instantiates with
`fc.in_features == 320` and **params 5,915,922** (vs baseline 4,299,866); `git diff
--name-only` = train.py only. The `_make_layer`/`BasicBlock` projection shortcuts and
`fc=Linear(w3,10)` + adaptive avg-pool adapt to the new widths with no other change.

### Surprises & Discoveries
Params rose +37.6% (4.30M→5.92M) — larger than a naive read of "+44 channels" suggests,
because stage3 holds the bulk of parameters (w3² conv weights × multiple convs) and 256→320
is a 1.56× param scaling there, only partly offset by stage1's 64→44 (0.47×) shrink on its
smaller param share. This is EXACTLY the fat-head premise: stage3 params are cheap in FLOPs
(8×8 spatial), so a large PARAM increase is ~FLOP-neutral. The compute-neutrality claim rests
on FLOPs (epochs), not params — verified via the realized epoch count at runtime, not params.

### Decisions
Hardcoded explicit widths rather than introducing a per-stage multiplier vector, to keep the
diff minimal (one line). `width_mult`/`WIDTH_MULT` left in the signature/call but unused
(harmless — no lint gate).

**Run-1 → Run-2 width adjustment (compute-neutrality fix, NOT a hypothesis change):** Run 1 used
{44,128,320} and showed dt ~36–38ms (≈4.7× baseline 8ms) in ep1. Two confounds were present:
(a) two external Protenix `batch_inference.py` jobs (other user) were saturating the shared H20
(97% util) during the launch → GPU compute contention; (b) w1=44 is not a multiple of 8, so the
expensive 32×32 stage1 convs fall off the tensor-core/channels_last fast path. Both inflate
wall-clock without changing FLOPs — exactly the EXP-015 "FLOPs-neutral ≠ wall-clock-neutral"
trap. Run 1 was killed before completion (would have fit ~20 ep → confounded regression, zero
signal on the capacity-placement hypothesis). For Run 2 the widths were changed to the
tensor-core-aligned, FLOP-matched **{48,128,304}** (48=16×3, 304=16×19, both multiples of 16;
≈−0.9% FLOPs, +32 net channels in stage3) and relaunched once GPU 0 was confirmed free (3 MiB,
0% util). The fat-head HYPOTHESIS is unchanged; only the width instantiation was made
alignment-safe so the test is genuinely compute-neutral. dt is re-checked in Run 2's first ~50
steps before trusting neutrality.

## Run Log

### Run 1
- **Description**: Full 300s-budget training run of the fat-head {44,128,320} variant on a
  single H20. Tests whether reallocating capacity from the spatially-expensive stage1 to the
  spatially-cheap discriminative stage3 at ~constant FLOPs lifts best_test_acc above the bar
  96.32 (baseline 96.22) WITHOUT the epoch-wall regression of uniform widening. Expected:
  compute-neutral (~88–91 ep, dt ~8ms); key check = realized epoch count stays ≈baseline.
- **Job ID / PID**: (local background, killed)
- **Log file**: run.log (overwritten by Run 2)
- **WandB**: n/a
- **Status**: killed — dt ~36–38ms (≈4.7× baseline) from GPU contention (external Protenix jobs)
  + w1=44 channel misalignment; would have under-trained to ~20 ep. No valid metrics.
- **Started**: 2026-06-09
- **Observations**: ep1 dt 36–38ms, params 5,915,922; eval ep1 19.55%. Killed at ~step 350.
- **Key Metrics**: none (aborted).

### Run 2
- **Description**: Re-run of the fat-head reallocation with tensor-core-aligned, FLOP-matched
  widths {48,128,304} on the now-free GPU 0. Same hypothesis as Run 1 (capacity placement at
  constant FLOPs), with widths made alignment-safe so the test is genuinely compute-neutral.
  Expected: dt ~8ms, ~88–91 ep; tests best_test_acc vs bar 96.32. Key early check: dt in first
  ~50 steps must be ~8–9ms, else the reallocation is not wall-clock-neutral.
- **Job ID / PID**: (local background, via /tmp/exp038_launcher2.sh on confirmed-idle GPU 0)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0) — ACCEPTED clean run (launcher v2, attempt 1; dt uniform 10–11ms,
  no contention spikes; num_epochs 73 ≥ 64 threshold).
- **Started**: 2026-06-09
- **Observations**: CLEAN uncontended run — dt distribution 446×10ms + 114×11ms + 1×12ms (NO
  24ms contention spikes, unlike the discarded contended attempts). The fat-head's true
  uncontended dt is ~10.5ms = **+31% vs baseline's 8ms** despite ≈−0.9% FLOPs → only 73 epochs
  fit the 300s budget (vs baseline ~91). test_acc climbed normally but plateaued under-trained.
- **Key Metrics**: best_test_acc 95.47% | final_test_acc 95.40% | final_test_loss 0.2082 |
  num_epochs 73 | num_steps 28087 | total_seconds 390.4 | peak_vram_mb 425.0 | params 5,468,154.
  Δ vs baseline 96.22 = **−0.75pp**.

## Experimental Adjustments
- **Run 1 → Run 2**: widths {44,128,320} → {48,128,304} (tensor-core-aligned, multiples of 16) to
  remove the channel-misalignment dt penalty; relaunched on confirmed-free GPU 0. See Decisions.
- **Shared-node GPU contention (Run 2 attempts)**: both H20s are under heavy, volatile external
  load (other user's Protenix `batch_inference.py` cycling jobs onto GPU 0 and 1). Every manual
  launch got co-located with a foreign 5–6 GB process within seconds → dt ~24ms (3× the 8ms
  baseline) → confounded (the budget is wall-clock-dt-gated). Switched to an automated fair-run
  launcher (/tmp/exp038_launcher.sh): polls for an idle GPU, launches, measures dt over the first
  ~45s, and COMMITS only if min dt < 13ms (else kills + retries). This guarantees the committed
  run starts uncontended; mid-run contention (if any) is caught via the final epoch count
  (expect ~88–91 ep; a materially lower count ⇒ re-run). params {48,128,304} = 5,468,154.

## Errors & Dead Ends

### 2026-06-09 — Fat-head dt blowup (channel misalignment + shared-GPU contention)
- Error: ep1 dt ~36–38ms vs baseline 8ms (≈4.7×) with widths {44,128,320}.
- Root cause: (a) external Protenix `batch_inference.py` jobs (user ykzhang) saturating the
  shared H20 (97% util) during launch → compute contention; (b) w1=44 not a multiple of 8 →
  stage1 32×32 convs fall off the tensor-core/channels_last fast cuDNN path.
- Source: run.log (Run 1) L5 step lines; nvidia-smi compute-apps showed 13GB+5GB external use.
- Do NOT retry: never pick a stage width that is not a multiple of 8 (prefer 16) under
  channels_last + torch.compile; verify GPU is free (nvidia-smi util ~0%) before launching a
  dt-gated run on this shared node, and re-check dt empirically in the first ~50 steps.

## Verification Results

### Conditions Checked
1. **Primary metric clears the bar** (`best_test_acc >= 96.32`): **FAILED** — 95.47%
   (−0.75pp vs baseline 96.22). Source: run.log `best_test_acc: 95.47%`.
2. **Clean completion within budget**: PASSED — summary printed, total_seconds 390.4 (< 600),
   exit 0, training_seconds 300.0. (Recorded for completeness; verdict set by condition 1.)
3. **No hard-constraint violations**: PASSED — `git diff --name-only` = train.py only; eval-line
   count 73 == num_epochs 73 (≤1 eval/epoch); no new deps; seed 42; prepare.py/eval untouched.
   (num_params 5,468,154 changed — expected, capacity is not constrained.)

Outcome: clean uncontended run, valid result, primary necessary condition failed → **no-improvement**.
The fat-head reallocation is FLOP-neutral but NOT wall-clock-neutral (memory-bound wider stage3) →
epoch wall → under-train → regress.

### Informational Metrics
- peak_vram_mb: 425.0 (≈ baseline; the wider stage3 at 8×8 adds little activation memory).
- num_epochs / num_steps: **73 / 28087** — the decisive datum: clean dt ~10.5ms (+31% vs 8ms)
  → ~20% fewer epochs than baseline ~91 despite ≈−0.9% FLOPs (FLOP-neutral ≠ wall-clock-neutral,
  EXP-015 pattern; wider 304-ch stage3 is memory-bandwidth-bound).
- final_test_loss: 0.2082 (vs baseline 0.195) — elevated → genuine under-training, not converged.

## Human Notes
(none — autopilot)
