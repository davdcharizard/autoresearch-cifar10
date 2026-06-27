# EXP-012: BATCH_SIZE 1024 + PEAK_LR 0.8 (linear scaling — last throughput lever)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md
- **Plan**: plans/plan-012.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-012
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification necessary condition 2 failed — best_test_acc 96.66 < 96.81; clean valid run, research no-improvement)

## Implementation Notes

### Summary

Exactly per plan-012 Milestone 1: two constants in train.py on branch `autoresearch/exp-012` — `BATCH_SIZE` 512→1024 and `PEAK_LR` 0.4→0.8 with the comment updated to the new linear-scaling arithmetic (`0.1 x (1024/128)`). Nothing else touched: the compile-warmup synthetic batch, DataLoader, steps/epoch (97→48 via drop_last), and the time-keyed LR schedule all key off the constants. `py_compile` passed; `git diff --stat` confirms a 2-line modification in a single file.

### Surprises & Discoveries

- None at implementation time — the recipe was designed constants-first, so the lever is genuinely two tokens.

### Decisions

- Launch immediately into the observed fully-idle window (both GPUs 0 compute apps, 0% util) rather than staging further checks — EXP-011 showed clean windows on this shared node are transient; the contention protocol (throughput detector + post-run epoch sanity) covers us if the window closes mid-run.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background task bm70wed4e (local, GPU 0 via CUDA_VISIBLE_DEVICES=0)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: completed (clean — valid hypothesis test)
- **Started**: 2026-06-10 07:32
- **Ended**: 2026-06-10 07:42 (exit 0, 560.8s total)

Description:
- Single run of the baseline recipe with batch and peak LR doubled together (Goyal linear scaling — preserves gradient-noise scale, distinguishing this from EXP-010's fixed-batch LR raise). Tests whether the H20's utilization headroom at batch 512 (22ms/step) lets a 1024-batch step cost < 2x, converting to more epochs in the fixed 300s — the only mechanism with a 100% conversion record. Expected: 48 steps/epoch, dt 36–42ms at step 100 (kill only if projected epochs < 65), ~148–173 epochs, mid-schedule trajectory tracking baseline (not EXP-010's 3pp depression), VRAM ~3.2GB, total ≤ ~520s, best_test_acc ≥ 96.81 if the hypothesis holds.

Observations:
- Batches per epoch: 48 as computed (source: run.log header)
- dt-gate at steps 350–500: dt 41–42ms steady, img/s ~24.7k vs baseline 22.9k (+8%) → projected epochs = 300/(48×0.0415) ≈ 150 — above the 142 no-gain band, far above the 65 kill floor. GATE PASSED. Window cross-check: step 500 at 7.0% ⇒ 42ms cumulative — print and pct agree, no contention. (source: run.log step lines 350–500)
- Epoch-1 eval 31.62% (sane for raw-weights eval at LR 0.8 warmup; cf. baseline ~35%, EXP-010's 34.4% — no instability signature); epoch-5 56.65% climbing normally. (source: run.log eval lines)
- No NaN, no SLOW events so far.

- THROUGHPUT HYPOTHESIS HELD: 151 epochs / 7228 steps, cum dt 41.5ms steady end-to-end (windows 36–42ms; SLOW detector zero events; 151 vs 150 projected = contention sanity PASSED). +12 epochs over baseline's 139. (source: run.log pct windows; monitor bhxorltu2 empty)
- ACCURACY DID NOT FOLLOW: mid-schedule ran FAR below baseline (ep 20: 69.5 vs ~88; ep 60: 83.2 vs ~92; ep 100: 91.1 vs ~96) with bouncy evals through the hot phase — linear scaling did NOT preserve the optimization trajectory at this scale (BN statistics quality and/or the 0.8 peak break the equivalence). The cosine tail recovered spectacularly (+5.5pp over the last 50 epochs) but landed at 96.66, −0.05pp short of baseline — the large-batch deficit ate the +12-epoch gain almost exactly. (source: run.log eval lines ep 20/60/100/140–151)
- Tail shape: converged plateau 96.53–96.66 over ep 146–151, final 96.63 ≈ best 96.66 — NOT starvation; more epochs would not have closed it. (source: run.log eval tail)
- total_seconds 560.8 — inside the 600s cap but the tightest yet (151 evals + 23.8s startup, cold compile cache for the new shape as planned). VRAM 3134.6MB (~2x, as predicted); params 4,286,026 unchanged.

Key Metrics:
- best_test_acc: 96.66% @ ep 148 (baseline 96.71, bar 96.81 — miss by 0.15pp; −0.05 vs baseline) (source: run.log summary block)
- num_epochs: 151 | num_steps: 7228 | training_seconds: 300.0 | total_seconds: 560.8 | startup: 23.8s | peak_vram_mb: 3134.6 | num_params: 4,286,026

## Verification Results

### Conditions Checked

0. **Pre-condition: contention sanity (Protocol Findings EXP-011)** — num_epochs 151 vs step-100 projection ~150 (within 1%); SLOW detector silent. CLEAN — conditions may be evaluated. (source: run.log summary; monitor bhxorltu2)
1. **Run completed within budget (≤ 600s total)** — `grep "^total_seconds:" run.log` → 560.8s, summary block present, exit 0. **PASS**. (source: run.log summary block; task bm70wed4e)
2. **best_test_acc ≥ 96.81 (baseline 96.71 + 0.1pp)** — `grep "^best_test_acc:" run.log` → 96.66. **FAIL** (−0.05pp vs baseline; −0.15pp vs bar). (source: run.log summary block)
3. **Validation at most once per epoch** — skipped per first-failure stop. (Informally compliant: 151 eval lines = num_epochs 151.)

### Informational Metrics

- Not collected (necessary condition 2 failed). Observed in summary: peak_vram_mb 3134.6 (~2x baseline, soft constraint fine), num_epochs 151 (throughput mechanism VALIDATED: +12 epochs), num_params 4,286,026 (unchanged).

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
