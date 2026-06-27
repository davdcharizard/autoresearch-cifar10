# Plan EXP-043: Full-alternation two-member ensemble (2 × 4x ResNet-20, per-step alternation, logit-mean inference)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-043.md

## Milestones

### Milestone 1: Code change implemented and sanity-checked (CPU)
- [x] On branch `autoresearch/exp-043` (cut from `autoresearch/dev`), edit `train.py`:
  (a) `ResNet` class UNCHANGED (the certified baseline net is the member architecture);
  (b) add `class MeanEnsemble(nn.Module)` holding `m1, m2`; `forward(x) = (m1(x) + m2(x)) / 2`;
  (c) construct `model1`, `model2` = two `ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULT)` instances sequentially under the existing seed (independent Kaiming draws), both `.to(device, channels_last)`; `base_model = MeanEnsemble(model1, model2)` (eager eval reference); `c1, c2 = torch.compile(model1), torch.compile(model2)`;
  (d) two optimizers `opt1, opt2`, each with the baseline selective-WD groups over its member's params (PEAK_LR 0.4, momentum 0.9 nesterov, WD 5e-4 on ndim>1);
  (e) compile warmup: 3 iters on `c1` then 3 on `c2` (same warm tensors), `zero_grad(set_to_none=True)` on BOTH optimizers after, sync, del;
  (f) training loop: `model.train()` → `base_model.train()` at epoch top; inside the step, select `active, opt = (c1, opt1) if step % 2 == 0 else (c2, opt2)` (parity BEFORE increment); set `lr_now` on `opt`'s groups only; forward/backward/step on `active`/`opt`; loss print = active member's CE (single-CE scale, family-comparable);
  (g) EVAL THINNING: evaluate only when `progress >= 0.6 or epoch % 3 == 1` (ep1 always evaluated; every epoch once past 60% of budget — the plateau the max-statistic harvests is fully sampled); skip both `evaluator.evaluate` and the eval print on skipped epochs; best_acc logic unchanged;
  (h) `num_params` = sum over `base_model.parameters()` (expect 2 × 4,286,026 = 8,572,052 — VERIFY constructed value)
- [x] CPU sanity A — INIT DIVERSITY: `model1.conv1.weight` ≠ `model2.conv1.weight` (and one deep layer differs)
- [x] CPU sanity B — ALTERNATION ISOLATION: 2-step CPU mini-loop with the real selection logic: step 0 changes model1 params only (model2 bit-identical snapshot), step 1 changes model2 only
- [x] CPU sanity C — EVAL CONTRACT: `base_model.eval()`; `base_model(x)` returns single (B,10) tensor == `(model1(x) + model2(x)) / 2` computed directly
- [x] CPU sanity D — params: constructed total == 8,572,052; `git diff --stat` shows train.py only
- [x] CPU sanity E — THINNING PREDICATE: pure-python sweep over (epoch, progress) confirms: ep1 evaluated; below progress 0.6 exactly epochs ≡1 (mod 3) evaluated; at/above 0.6 every epoch evaluated

### Milestone 2: Gated launch, clean run to completion
- [x] Copy `/tmp/exp041_composite.sh` → `/tmp/exp043_composite.sh` with ONE change: STARTUP_KILL tick 10 → 12 (two compile warmups; inductor cache for this exact baseline graph is warm, but allow margin). Baseline thresholds otherwise apply unchanged — dense 4x kernels, dt expected ≈ 22.4ms (+ negligible alternation overhead): contention > 27ms (off-rung), NaN guard, DIVERGENCE_KILL eval < 15% after ep5 (thinned evals at ep 1,4,7,… still feed it), WALL_CAP tick 44. Dual launch gates unchanged (GPU-0 zero compute apps AND load < 60)
- [x] Run completes: rc=0, num_epochs 139, 85 evals, dt 22.34ms, total 470.3s, expected eval-line count ≈ 28 (thinned phase) + ~56 (plateau phase) ≈ 80–90

### Milestone 3: Verification and exp-log complete
- [x] First-failure-stop verification executed (protocol below), recorded in `logs/exp-log-043.md § Verification Results`
- [x] Diagnostics recorded (third-shape outcome: gain real ~+0.4 but starvation ~−0.9 dominates), INCLUDING the pre-registered interpretive branch (brainstorm-043 § Idea Evaluation): (i) ≥96.81 improvement; (ii) best 96.6–96.8 AND final_test_loss ≤ ~0.165 → mechanism real, starvation-limited → mid-fork next; (iii) in-band best with family-equal test_loss (~0.185) → averaging dichotomy closed both halves
- [ ] run.log deleted after extraction (analyze housekeeping)

## Code Changes
- **train.py** (only editable file, ~45-line diff): duplicate the certified baseline trainee into two independently-initialized members that alternate full-batch steps, evaluated jointly as mean logits. Why this tests the hypothesis: each member runs the EXACT baseline recipe (architecture, constants, kernels, noise per step) at half the step count, so any plateau LEVEL above the starvation-priced single-member level is attributable to function-space averaging. Edge cases: (i) `model.train()` after eval must reach BOTH members → use `base_model.train()` (propagates to children); (ii) per-STEP alternation (not per-epoch) keeps members equally fresh (±1 step) at every eval — epoch-boundary law; (iii) 97 steps/epoch is odd, so batch-parity assignment swaps each epoch — over any two epochs both members see both step parities; (iv) eval print and best_acc update only on evaluated epochs; final summary's `final_test_acc/test_loss` reference the last eval, which is always the final epoch (progress ≥ 0.6 ⇒ every epoch evaluated); (v) the eval-thinning predicate uses `total_training_time / TIME_BUDGET_S` (charged progress), not wall time; (vi) loss print is the ACTIVE member's single CE — family-comparable scale, NaN guard unaffected.

## Configuration Changes
- Members: 2 × WIDTH_MULT 4 (unchanged widths 64/128/256, dense kernels — EXP-042's grouped path explicitly avoided). Total params 8,572,052; per-member step count ≈ 6,700 (~70 epoch-equivalents each); all training constants byte-identical to baseline per member.
- Eval cadence: every 3rd epoch below progress 0.6, every epoch at/above (once/epoch is a CEILING per goal; thinning validated EXP-031). Rationale: ensemble eval costs ~2× forward; un-thinned wall ≈ 360s of evals busts the 600s cap; the harvested statistic lives in the plateau, which stays fully sampled.

## Execution Environment
- Method: local, via `/tmp/exp043_composite.sh` with `run_in_background: true`
- Resources: GPU 0 ONLY (never GPU 1; wait if busy), ~3.0–3.5GB VRAM (two models + two optimizer states), host load < 60 at launch
- Estimated runtime: ~510–570s total (300s charged + startup ~15–35s two warm-cache compiles + ~84 evals × ~2.0s + stalls); cap 600s — wall is the watched risk, mitigated by thinning
- Log output: `uv run train.py > run.log 2>&1` (no tee); run.log deleted after extraction
- Tool skill: none (local)

## Abort Criteria
- STARTUP_KILL: no step line by tick 12 (~180s)
- CONTENTION_KILL: 4 consecutive 15s windows > 27ms (baseline off-rung threshold; dt unchanged at ~22.4ms). On kill: confirm contamination (nvidia-smi apps, /proc/loadavg), relaunch byte-identically when gates clear
- NaN guard: any `loss: nan` → kill
- DIVERGENCE_KILL: any eval < 15% after epoch 5
- WALL_CAP_KILL: still running at tick 44 (~660s)
- Experiment-specific monitor (not a kill): early ensemble evals are EXPECTED below family (members at half steps — ep4 ensemble may read ~45–58 vs family ~60); do not misread as a bug. Suspect a selection-logic bug only if evals sit below ~30 past ep10 (divergence guard still protects)

## Verification Protocol

### Verification Procedure
First-failure-stop; baseline via `exp-index.sh baseline` at verification time (currently 96.71; bar = 96.81).

**Pre-condition (run integrity):**
- Profile: 200-step quantization-safe windows (every 4th step-line pair) — require mean 22.0–23.5ms and 0 windows > 27ms; num_epochs in 130–145. If contaminated → rerun byte-identically, do not judge.
- Integrity: `num_params: 8,572,052`; `training_seconds: 300.0`; eval-line count == (count of epochs ≡1 mod 3 in the thinned phase) + (count of epochs at progress ≥ 0.6) — cross-check ≈ 80–90 and ≤ num_epochs.
- Timeout: greps on finished run.log; missing summary ⇒ crash (`tail -n 50 run.log`).

**Condition 1 — best_test_acc ≥ 96.81**: `grep "^best_test_acc:" run.log`. Fail → STOP, verdict `no-improvement` (rest incidental, but ALWAYS record the pre-registered interpretive branch via final_test_loss).

**Condition 2 — within budget**: composite rc == 0 AND `total_seconds` ≤ 600.0.

**Condition 3 — validation ≤ once/epoch**: `grep -c "eval ep" run.log` ≤ num_epochs.

**Diagnostics (always):** final_test_loss vs family ~0.185 — THE mechanism discriminator: ensemble averaging of diverse members must reduce loss markedly (≤ ~0.165) even if accuracy misses; family-equal loss ⇒ members too correlated/starved (branch iii). Plateau last-15 evaluated epochs: mean/spread vs ~96.5/±0.15 (hypothesis predicts mean shift UP with equal-or-less scatter). Early thinned evals (ep4/7/10) vs the starvation expectation (below family is expected, not a defect).

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~3,000–3,500
- num_epochs: `grep "^num_epochs:" run.log` — expect 135–143
- num_params: `grep "^num_params:" run.log` — expect 8,572,052
