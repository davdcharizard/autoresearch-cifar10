# EXP-062: Schedule-Free SGD — full anneal→averaging substitution, eval at x

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-062.md
- **Plan**: plans/plan-062.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-062
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Five hunks to train.py exactly as planned (M1): (1) constants — `WARMUP_FRAC`/`MOMENTUM` replaced by `WARMUP_STEPS=2000`, `SF_MOMENTUM=0.9`, `WEIGHT_LR_POWER=2.0`, `SF_R=0.0`, `REFRESH_BATCHES=24`; `PEAK_LR=0.4` retained as the constant post-warmup lr. (2) `lr_at` (cosine) deleted; `ScheduleFreeSGD` class added (~75 lines): params hold y, `state['z']` per param, foreach step implementing exactly the reference math (WD-at-y → `y.lerp_(z,ckp1)` → `y += grad·lr(β(1−ckp1)−1)` → `z −= lr·grad`; `ckp1 = lr_max²/Σ`; step-keyed warmup inside `step()`), plus guarded `eval_mode()`/`train_mode()` in-place lerp switches (y↔x). (3) optimizer instantiation keeps the SAME two param groups (selective WD); second persistent `refresh_loader` (num_workers=4) added. (4) timed loop: per-step lr assignment removed (warmup lives in the optimizer); lr print reads `scheduled_lr`; the charged window is otherwise byte-identical. (5) eval block (uncharged, between epochs): `optimizer.eval_mode()` → 24 forward-only bf16-autocast batches from refresh_loader to re-converge BN running stats at x → `evaluator.evaluate(base_model, device)` unchanged → `optimizer.train_mode()`. CPU sanity ALL PASS (14 checks): algebra vs fp64 pure-python reference exact to 4e-16 on y/z/x, roundtrip restores y to 6e-8 fp32, warmup sched factors exact, params 4,286,026, smoke loss decreasing, WD strictly confined to the decay group.

### Surprises & Discoveries
- None at implementation time. The reference's in-place x-recovery trick (`p.lerp_(z, 1−1/β)`) verified exactly against the explicit x-tracking reference in the sanity test — no separate x buffer needed (VRAM cost is only the z state, ~17MB).

### Decisions
- Step-keyed warmup (`WARMUP_STEPS=2000`) instead of time-keyed: faithful to the reference implementation; equivalent at constant dt (~15% of projected ~13.2k steps = the certified warmup duration).
- BN refresh wrapped in the same bf16 autocast as training forwards so refreshed stats come from the same numerics regime as training-time batch stats.
- `import math` left in place though now unused — minimal-diff principle.

## Experimental Adjustments

## Run Log

### Run 0 (GPU probe — uncharged, pre-launch)

Metadata:
- **Job ID**: foreground Bash (probe ran clean first attempt)
- **Log file(s)**: probe stdout (recorded below)
- **Status**: completed

Description:
- dt pricing of the schedule-free step (3 extra foreach ops on 4.3M params vs SGD) + the 24-batch eager BN refresh, per plan M2. Branches: P ≤ 23.5 launch; (23.5, 26] launch with priced toll; > 26 no launch. R > 0.8s → cut REFRESH_BATCHES to 12.

Observations:
- Gate clean (apps=0, load 33.0 < 40); params asserted 4,286,026 in vivo. **P = 23.28ms ≤ 23.5 → LAUNCH branch** (top of family band; the optimizer's foreach ops price at ~+0.6–0.8ms vs the 22.5–22.7 family probe reads — inside band, no toll adjustment). **R = 0.306s** per 24-batch refresh (GPU side) → REFRESH_BATCHES stays 24; projected wall +~50s incl. fetch.
- **Probe-revised bands (recorded pre-launch)**: steps ∈ [12,105, 12,830]; epochs ∈ [122, 133]; D0 ∈ [23.0, 24.6]; expected run dt ≈ 23.8–24.3 (probe +0.5–1.0 offset).

Key Metrics:
- P = 23.28ms (load 33.0); R = 0.306s — source: probe stdout above

### Run 1

Metadata:
- **Job ID**: background task bs7tzojpw (composite), bxflcshar (watcher)
- **Log file(s)**: run.log (training); /tmp/exp062_composite_run1.log (gate/watchdog)
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-11 (launched immediately post-probe)

Description:
- Single gated run of the Schedule-Free SGD substitution via /tmp/exp062_composite.sh (single-phase launcher: dual gates; GATE_KILL D0>26; contention 4×>max(26, D0×1.25); TAIL_THRESH=THRESH; STARTUP_KILL tick 12; NaN guard; divergence guard acc<20 after ep5; watchdog 44×15s backstop). Expected: params 4,286,026, steps/epochs per probe-revised bands, total ≈ 535–565s (family 480s + ~0.4s/epoch refresh), VRAM 1,650–1,900MB. ep1 tripwire relaxed to ≥25 (optimizer change); contamination judged by the step ledger. Decision per plan-062 branches: (i) ≥96.81 → byte-identical replicate, pair MEAN decides; (ii) [96.41, 96.73] → schedule axis closes at the FAMILY level; (iii) <96.41 → constant-lr basin-quality loss; (iv) (96.73, 96.81) → replicate-mean protocol; (v) BN-mismatch signature → REFRESH_BATCHES=50 engineering retry; (vi) infra → relaunch ≤2.

Observations:
- Infrastructurally PRISTINE run. Gates poll 1 (apps=0, load 13); GATE_DECISION D0=22.5ms; windows ticks 5–40 read 21.4–23.0ms, slow_streak 0 throughout; PROC_EXITED tick 41, RC=0. (source: /tmp/exp062_composite_run1.log)
- **Probe-band note**: D0 22.5 and steps 13,471 sit on the FAST side of the probe-revised bands (D0 [23.0, 24.6], steps [12,105, 12,830]) — the probe (P=23.28 at load 33) was load-inflated; the run signature matches the FAMILY ledger exactly (13,100–13,600 steps, dt 22.0–22.8). The schedule-free foreach step is dt-FREE (~22.27ms effective). Fast-side deviations cannot fake a higher metric (contamination only ever yields FEWER steps); integrity judged on the family ledger. (source: composite log + run.log summary)
- Ledger: 13,471 steps, 139 epochs, params 4,286,026 exact, startup 9.4s, 300.0s charged, **total 601.5s (1.5s OVER the 600s cap — wall-side: ~139 × (0.85s eval + ~0.5s BN refresh incl. fetch); the charged budget is untouched)**, VRAM 1,613.0MB (exactly family — z state absorbed by the allocator). (source: run.log summary)
- ep1 22.05 < 25 tripwire — NOT contamination (ledger clean): the genuine schedule-free signature, x at ep1 averages the first ~98 near-init warmup iterates. (source: run.log eval lines)
- **Trajectory: smooth, strictly monotone x-curve (test_loss falls nearly every eval 2.13→0.229) that is STILL CLIMBING at budget end** — new best at ep137/139, loss still falling over the final 10 epochs; only 3 evals within 0.15 of best (family 8–12; the plateau-length mechanism predicted ≥30). The paper's curve SHAPE delivered; the LEVEL is catastrophically behind: 94.87 at 139 epochs where the cosine reads ~96.6 at the same step count. test_loss 0.229 vs family 0.185 — worse basin AND unfinished convergence. No BN-mismatch signature (loss < 0.30, no depressed-gap vs own peak) — branch (v) excluded; this is the genuine constant-lr schedule-free trajectory at lr 0.4. (source: run.log eval lines; trajectory extract in analyze notes)

Key Metrics:
- best_test_acc: 94.87 @ ep137 (source: run.log summary; branch (iii): < 96.41, deficit −1.84 vs baseline, ~−10.6σ vs family mean)
- final_test_acc: 94.81 / final_test_loss: 0.2287 @ ep139 (source: run.log summary)
- training_seconds: 300.0 | total_seconds: 601.5 | startup: 9.4 | num_epochs: 139 | num_steps: 13,471 | num_params: 4,286,026 | peak_vram_mb: 1,613.0 (source: run.log summary)
- Status: completed | Ended: 2026-06-11 (RC=0 at watchdog tick 41, ~610s)

## Verification Results

### Conditions Checked

- **Integrity pre-condition** (plan-062 step 0): PASS — RC=0; no kill markers; windows 21.4–23.0 all < 28.1 thresh; step LEDGER 13,471 ∈ FAMILY band [13,100, 13,600] (probe-revised band rejected as mis-anchored — probe load-inflated; fast-side deviation is integrity-benign); `num_params: 4,286,026` exact; `training_seconds: 300.0`; 139 evals = 139 epochs; ep1 22.05 < 25 tripwire explained by the x-at-init mechanism with clean ledger (per plan: ledger is the binding gate); zero NaN; BN-mismatch screen NEGATIVE (test_loss 0.2287 < 0.30, monotone trajectory, no depressed-gap) → branch (v) does not fire, the read is the genuine schedule-free result. (source: run.log + /tmp/exp062_composite_run1.log)
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: FAIL — best_test_acc 94.87. Pre-registered branch (iii): < 96.41 → constant-lr hot training loses basin quality that explicit averaging cannot repair at this horizon; deficit −1.84 is ~12σ below the bar, far beyond any retune recovery in the record. First-failure-stop: remaining conditions informational only.
- **Condition 2 — total_seconds ≤ 600**: FAIL informationally — 601.5s (+1.5s, wall-side from the per-epoch BN refresh + eval; charged 300.0s exact). EXP-013 precedent (626.9s): recorded as no-improvement with metric, overage noted. A REFRESH_BATCHES=12 retry would shave ~30s but cannot change the research conclusion (−1.84).
- **Condition 3 — validation ≤ once/epoch**: PASS informationally (139/139, structural).

### Informational Metrics

- peak_vram_mb: 1,613.0 | num_epochs: 139 | num_steps: 13,471 | num_params: 4,286,026 | evals within 0.15 of best: 3 (mechanism check — plateau-length mechanism did NOT engage; the level never converged)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
