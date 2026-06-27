# EXP-060: CutMix substituted for RandomErasing at matched dose (p=0.5, α=1.0)

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-060.md
- **Plan**: plans/plan-060.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-060
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Four hunks to train.py exactly as planned: (1) constants `CUTMIX_P = 0.5`, `CUTMIX_ALPHA = 1.0`; (2) `transforms.RandomErasing(...)` removed from train_tf (comment notes the occlusion moved to per-batch CutMix at the same dose); (3) sync-free eager CutMix block in the charged loop after the H2D copies — CPU-RNG branch decision, λ ~ Uniform (Beta(1,1)), box math entirely in Python ints/floats, GPU work limited to one `randperm`, one in-place strided slice copy from the permuted batch, one `targets[perm]` gather; (4) loss inside the existing autocast becomes the area-weighted dual CE (`lam_adj·CE(y) + (1−lam_adj)·CE(y[perm])`, both on the SAME logits) when applied, plain CE otherwise. The compiled `model(inputs)` call is byte-identical in both branches — mixing and loss are eager, outside the graph; the existing 3-iter compile warmup needs no variants. CPU sanity ALL PASS: params exact; CutMix unit test (box from permuted source, complement untouched, lam_adj exact area complement, mixed loss finite + backward); RandomErasing absent / TA retained; smoke decreasing.

### Surprises & Discoveries
- (planning-time, recorded for posterity) Only `model` is wrapped by torch.compile in train.py, so the brainstorm's "compile-static mask" precaution was unnecessary — standard slice-assignment CutMix is compile-inert here. The binding correctness rule is instead sync-freedom of the charged path (no `.item()` on CUDA tensors before the loss).

### Decisions
- Box-center RNG via `torch.randint` on CPU (seeded by the existing `torch.manual_seed(42)`) rather than Python `random` (unseeded) — keeps the run's randomness under the established seed discipline without touching seeds.
- The train-loss print will read HIGHER than family (mixed targets inflate CE); pre-noted so trajectory judgment uses eval lines only (plan M4).

## Experimental Adjustments

## Run Log

### Run 0 (GPU probe — uncharged, pre-launch)

Metadata:
- **Job ID**: foreground Bash (probe ran clean first attempt)
- **Log file(s)**: probe stdout (recorded below)
- **Status**: completed

Description:
- Charged-step pricing per plan M2: 40 timed steps with CutMix forced ON every step (worst-case duty cycle). Branches: P ≤ 23.5 launch; (23.5, 24.5] launch with shifted bands; > 24.5 = presumed sync bug, fix and re-probe (≤2).

Observations:
- Gate clean (apps=0, load 8.3 < 40). **P_plain = 21.66ms ∈ [21.5, 23.5] (anchor PASS); P_cutmix = 21.77ms → worst-case toll 0.11ms/step, ≈ 0.05ms at p=0.5 duty (≈ −0.01pp by the deferral law — negligible).** Branch P ≤ 23.5 → LAUNCH with UNSHIFTED family bands: D0 [21.5, 23.5], steps [13,100, 13,600], epochs [133, 142]. Sync-free implementation confirmed in vivo: no .item()-class stall visible at 40-step resolution.

Key Metrics:
- P_plain = 21.66ms | P_cutmix = 21.77ms (forced every step; load 8.3) — source: probe stdout above

### Run 1

Metadata:
- **Job ID**: background task b5sc3wp6k (composite), bj2af4cz9 (watcher)
- **Log file(s)**: run.log (training); /tmp/exp060_composite_run1.log (gate/watchdog)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-11 09:04
- **Ended**: 2026-06-11 (RC=0 at watchdog tick 31, ~462s)

Description:
- Single gated run of the CutMix-substitution recipe via /tmp/exp060_composite.sh (standard single-phase launcher: dual gates apps==0 & load<60; GATE_KILL D0>26; contention 4×>max(26, D0×1.25); NaN/divergence guards; WALL_CAP). Expected family signatures: dt 22.0–23.0ms (≤ +0.3ms CutMix toll at p=0.5), 13,100–13,600 steps, 133–142 epochs, params 4,286,026, ~480–510s total. ep1 tripwire RELAXED to ≥25 (mixed-label supervision may slow the first-epoch read; divergence guard still armed). Decision per plan-060 pre-registered branches: (i) ≥96.81 → byte-identical replicate, pair MEAN decides; (ii) [96.41, 96.73] → absorption extends to augmentation TYPE, reg axis closed in type AND dose; (iii) (96.73, 96.81) → no-improvement; (iv) <96.41 → mixed-label over-pressure at constant dose; (v) infra → relaunch ≤2.

Observations:
- PRISTINE single-phase run. Gates poll 1 (apps=0, load 8); GATE_DECISION D0=22.7 ∈ [21.5, 23.5]; all windows ticks 5–30 read 22.0–22.8ms (slow_streak 0 throughout); PROC_EXITED tick 31, RC=0. The probe-predicted ≤+0.3ms CutMix toll is confirmed: window profile is byte-indistinguishable from family. (source: /tmp/exp060_composite_run1.log)
- Step/epoch ledger ON family bands: 13,285 steps ∈ [13,100, 13,600]; 137 epochs ∈ [133, 142]; startup 9.4s; training_seconds 300.0; total 453.0s. peak_vram 1,613MB — plan's "<3,300" expectation was mis-anchored on EXP-059's 1024-batch tail; 1,613 is the correct 512-batch family value (EXP-059's 3,157 was the double-shape run). (source: run.log summary)
- Trajectory family-shaped: ep1 36.26 (≥ the UNRELAXED 30 tripwire — mixed-label supervision did NOT slow the early read); plateau evals 96.59–96.69 tight; final 8 evals within 0.10. test_loss runs HIGHER than family (final 0.2241 vs ~0.185) exactly as pre-noted in Decisions — mixed-target CE arithmetic (EXP-036-class), accuracy unaffected. (source: run.log eval lines)

Key Metrics:
- best_test_acc: 96.69 @ ep133 (source: run.log summary; branch (ii) — family band [96.41, 96.73], mean+0.75σ)
- final_test_acc: 96.62 / final_test_loss: 0.2241 @ ep137 (source: run.log summary)
- training_seconds: 300.0 | total_seconds: 453.0 | startup: 9.4 | num_epochs: 137 | num_steps: 13,285 | num_params: 4,286,026 | peak_vram_mb: 1,613.0 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Integrity pre-condition** (plan-060 M4 / step 0): PASS — RC=0; no kill markers; D0 22.7 ∈ [21.5, 23.5]; all windows ≤ 22.8 < 28.4 thresh; steps 13,285 ∈ [13,100, 13,600]; epochs 137 ∈ [133, 142]; `num_params: 4,286,026`; `training_seconds: 300.0`; `total_seconds: 453.0` ≤ 600; 137 evals = 137 epochs; ep1 36.26 ≥ 25 (relaxed tripwire; also clears the standard 30); zero NaN lines. (source: run.log + /tmp/exp060_composite_run1.log)
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: FAIL — best_test_acc 96.69. Pre-registered branch (ii): ∈ [96.41, 96.73] family band → the absorption law extends to augmentation TYPE; the regularization axis is closed in TYPE as well as dose. First-failure-stop: remaining conditions not evaluated for the verdict (recorded informationally below).
- **Condition 2 — completes within budget**: PASS informationally (453.0 ≤ 600).
- **Condition 3 — validation ≤ once/epoch**: PASS informationally (137/137, structural).

### Informational Metrics

- peak_vram_mb: 1,613.0 | num_epochs: 137 | num_params: 4,286,026 (informational; Condition 1 failed)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
