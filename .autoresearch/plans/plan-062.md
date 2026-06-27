# Plan EXP-062: Schedule-Free SGD — full anneal→averaging substitution, eval at x
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-062.md

Baseline 96.71 @ 1990397 (bar ≥ 96.81; family mean 96.57, σ 0.16, band [96.41, 96.73]). Hypothesis and pre-registered branches per brainstorm-062 § Chosen Idea. Exact algorithm per knowledge/papers/schedule-free-road-less-scheduled.md (verified from the reference source 2026-06-11).

## Milestones

### Milestone 1: Code changes + CPU sanity
- [ ] train.py: `ScheduleFreeSGD` optimizer class implemented (foreach, per-group WD-at-y, step-keyed warmup, β=0.9, r=0, weight_lr_power=2) + `train_mode()`/`eval_mode()` switches; cosine `lr_at` removed; eval block gains x-swap + BN refresh at x via second persistent loader; timed-step semantics byte-identical (t0→synchronize window contains exactly: H2D, forward, loss, backward, optimizer.step)
- [ ] CPU sanity (`CUDA_VISIBLE_DEVICES="" uv run python /tmp/exp062_sanity.py`): (a) scalar-model algebra check — 20 steps of ScheduleFreeSGD vs a pure-python reference of the published equations, max |Δ| < 1e-6 on y, z, x; (b) train→eval→train roundtrip returns params to y within 1e-6 relative; (c) warmup sched factor correct at k=0, k=warmup_steps−1, k=warmup_steps; (d) params count 4,286,026 exact; (e) 3-step smoke at lr 0.01, loss decreasing; (f) WD applied only to ndim>1 group (zero grad-shift on a BN weight with wd=0 group)

### Milestone 2: GPU probe (uncharged, ~2 min)
- [ ] Gate: GPU-0 compute apps == 0 AND host load < 40 (probes inflate at load ≥ ~40 — infra-errors EXP-059)
- [ ] /tmp/exp062_gpu_probe.py: build model+optimizer exactly as train.py, compile, warm 3 iters, time 40 steps → P; ALSO time one 24-batch eager no_grad BN-refresh pass → R seconds
- [ ] Branches: P ≤ 23.5 → LAUNCH with probe-revised bands (steps ∈ [300000/(P+1.5), 300000/(P+0.1)], epochs = steps/97.65 ±2, D0 ∈ [P−0.3, P+1.3]); P ∈ (23.5, 26] → launch with priced deferral toll (−7 ep/ms ≈ −0.08pp/ms) noted for analysis; P > 26 → NO LAUNCH, cost-closure verdict (EXP-040/042 precedent). If R > 0.8s → reduce REFRESH_BATCHES 24→12 in train.py before launch (wall protection; 12 batches still replaces ~70% of y-stats)

### Milestone 3: Gated composite run
- [ ] /tmp/exp062_composite.sh (copy exp061_composite.sh, retarget): dual gates (zero GPU-0 compute apps AND load < 60, poll 30s×240) → rm -f run.log → `uv run train.py > run.log 2>&1` background → watchdog 44×15s, win = Δpct×3000/Δstep; GATE_KILL D0 > 26; CONTENTION_KILL 4 consecutive > max(26, D0×1.25); TAIL_THRESH = THRESH (single-phase); STARTUP_KILL tick 12; NaN/divergence guard; WALL_CAP 600s
- [ ] Launch via Bash run_in_background (stdout → /tmp/exp062_composite_run1.log) + until-grep watcher + TaskOutput(block=true)
- [ ] Clean completion: RC=0, no kill markers, D0 and all windows within probe-revised bands

### Milestone 4: Verification (first-failure-stop; integrity gates Condition 1)
- [ ] Integrity pre-condition: RC=0; D0 ∈ probe band; steps/epochs ∈ probe-revised bands (step LEDGER is the contamination gate); `num_params: 4,286,026` exact; `training_seconds: 300.0`; evals == epochs; ep1 ≥ 25 (relaxed: optimizer change); zero NaN; NO BN-mismatch signature (see branch (v))
- [ ] Condition 1: best_test_acc ≥ 96.81
- [ ] Condition 2: total_seconds ≤ 600
- [ ] Condition 3: validation ≤ once/epoch (structural)

## Code Changes

- **train.py** (only editable file; single experiment-scoped diff in 5 hunks):
  1. **Constants**: drop `WARMUP_FRAC` (no anneal exists; the only schedule is warmup); add `WARMUP_STEPS = 2000` (≈15% of the projected ~13.2k steps — preserves the certified warmup duration; step-keyed exactly as the reference; equivalent to time-keying at constant dt), `SF_MOMENTUM = 0.9` (interpolation β — replaces SGD momentum entirely; nesterov gone), `WEIGHT_LR_POWER = 2.0`, `SF_R = 0.0`, `REFRESH_BATCHES = 24` (92% replacement of y-stats at BN momentum 0.1; EXP-029: refresh data MUST be the augmented train distribution). `PEAK_LR = 0.4` retained as the constant post-warmup lr (certified peak; paper notes schedule-free optimal lr is often ≥ the scheduled peak — one-draw anchor, branches absorb both error directions).
  2. **`ScheduleFreeSGD` class** (~50 lines, replaces `lr_at`): param groups hold y; `state['z']` per param; `step()` implements exactly: `sched = min((k+1)/WARMUP_STEPS, 1)`, `lr = lr_base·sched`, `lr_max = max(...)`, `weight = lr_max**2`, `ckp1 = weight/weight_sum`; foreach: `grad += wd·y` (per-group wd) → `y.lerp_(z, ckp1)` → `y += grad·lr·(β(1−ckp1)−1)` → `z −= lr·grad`. `eval_mode()`: `p.lerp_(z, 1−1/β)` (y→x); `train_mode()`: `p.lerp_(z, 1−β)` (x→y); guarded by a mode flag so double-calls are no-ops. fp32 state/arithmetic exactly like current SGD (numerics-equivalence law: same compiled graph/kernels; optimizer math eager fp32).
  3. **Setup**: optimizer = ScheduleFreeSGD with the SAME two param groups (decay 5e-4 ndim>1 / 0.0 otherwise); second persistent DataLoader `refresh_loader` (same `train_set`, batch 512, shuffle, num_workers=4, pin_memory, persistent_workers — EXP-025/033 validated pattern) for BN refresh.
  4. **Timed loop**: per-step `lr_at`/group-lr assignment removed (warmup lives inside `step()`); print lr as `optimizer.scheduled_lr`. The t0→synchronize charged window is otherwise byte-identical — no new work inside it except the optimizer's own foreach ops (honest training compute).
  5. **Eval block** (between epochs, entirely OUTSIDE the timed window — timer/charged semantics untouched): `optimizer.eval_mode()` (params→x) → BN refresh: `base_model.train()`, `torch.no_grad()`, forward `REFRESH_BATCHES` batches from `refresh_loader` (eager, uncharged) → `evaluator.evaluate(base_model, device)` exactly as today (Eval untouched; x is the model under training — the optimizer's official evaluation point, same legitimacy class as EXP-011/032) → `optimizer.train_mode()` (params→y). Compile warmup loop unchanged (no optimizer.step → y=z=x at init, switches are exact no-ops there).
- **Why this tests the hypothesis**: the generating process changes (z never anneals; gradients at y), the readout is the theory's x point, and everything else — graph, kernels, batch, augmentation, WD structure, eval contract — is held at the certified recipe.
- **Risks/edge cases**: in-place y→x→y roundtrip rounding ~1e-7 fp32 (negligible, verified in sanity (b)); running stats carry x-statistics back into training (irrelevant — training uses batch stats); `refresh_loader` workers add host load (4 workers within the ~3% loader margin — refresh forwards are GPU-side, no new per-image CPU transform).

## Configuration Changes
- LR schedule: time-keyed cosine (peak 0.4, warmup 0.15) → warmup-to-0.4-then-CONSTANT (WARMUP_STEPS 2000) (the experiment variable)
- Optimizer: SGD(momentum 0.9, nesterov) → ScheduleFreeSGD(β 0.9, r 0, weight_lr_power 2) (the experiment variable; reference defaults)
- WEIGHT_DECAY 5e-4 selective: unchanged (applied at y per reference)
- BATCH_SIZE 512, LS 0.1, transforms, compile, warmup-iters 3: unchanged
- New: REFRESH_BATCHES = 24 (BN-at-x refresh; cut to 12 if probe R > 0.8s)

## Execution Environment
- Method: local, GPU 0 only; single gated run via /tmp/exp062_composite.sh launched with Bash run_in_background; `uv run train.py > run.log 2>&1` (no tee)
- Resources: GPU 0 (wait for free per goal constraints), host load gates (probe < 40, launch < 60); VRAM expectation ~1,650–1,900MB (family 1,613MB at 512 single-shape + z state 17MB + refresh buffers — do NOT anchor on EXP-059's 3,157MB double-shape figure)
- Estimated runtime: probe ~2 min; run ≈ 535–565s total (family 480s + ~135 × ~0.4s refresh) — under the 600s cap with ~40s margin; charged exactly 300.0s
- Log output: run.log (training, source of truth); /tmp/exp062_composite_run1.log (gate/watchdog telemetry); delete run.log after the experiment concludes

## Abort Criteria
- GATE_KILL: D0 > 26ms at GATE_DECISION (cost-closure branch if probe also read high)
- CONTENTION_KILL: 4 consecutive windows > max(26, D0×1.25) (foreign-process/host-load law; relaunch byte-identically, never analyze)
- STARTUP_KILL: no training prints by watchdog tick 12
- Divergence: any NaN in loss prints, or eval acc < 20% at/after ep5 (constant-lr instability — kill, record as research failure branch (iii) evidence only if numerics are clean; NaN → crash verdict)
- WALL_CAP: kill at 600s total
- ep1 tripwire ≥ 25 RELAXED for the optimizer change — contamination is judged by the step ledger (EXP-058)

## Verification Protocol

### Verification Procedure
First-failure-stop; integrity pre-condition gates Condition 1 (per goal-learnings protocol entries).

0. **Integrity pre-condition** (timeout: 2 min, from run.log + composite log): RC=0, no kill markers; D0 ∈ [P−0.3, P+1.3]; all windows ≤ max(26, D0×1.25); `num_steps` ∈ [300000/(P+1.5), 300000/(P+0.1)] (contamination gate); `num_epochs` = steps/97.65 ±2; `num_params: 4,286,026` exact; `training_seconds: 300.0`; eval lines == num_epochs; ep1 ≥ 25; zero NaN. **BN-mismatch screen (branch v)**: final test_loss < 0.30 AND no converged plateau sitting > 1.0pp below the trajectory's own earlier peak — if the signature fires (EXP-029 class: depressed acc + inflated loss with clean training), this is an ENGINEERING failure: set REFRESH_BATCHES = 50 and relaunch (counts toward the 2-retry limit), not a research verdict.
1. **Condition 1** — `grep "^best_test_acc:" run.log`, parse %; PASS iff ≥ 96.81 (= baseline 96.71 + 0.1; baseline confirmed via `bash .../exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-test-accuracy.tsv`). **Near-bar protocol (EXP-052)**: if Run 1 ≥ 96.81, run one byte-identical replicate through the same composite; the PAIR MEAN decides (mean ≥ 96.81 → improvement; else no-improvement). The max of the pair is never a decision input.
2. **Condition 2** — `grep "^total_seconds:" run.log` ≤ 600 (timeout: 1 min).
3. **Condition 3** — eval-line count == num_epochs (structural; validation once/epoch is a ceiling and the code evals exactly once/epoch).

Pre-registered decision branches (from brainstorm-062): (i) pair-mean ≥ 96.81 → improvement, commit; (ii) ∈ [96.41, 96.73] → no-improvement; anneal-equivalence extends to full substitution — schedule axis closed at the FAMILY level; (iii) < 96.41 → no-improvement; constant-lr hot training loses basin quality averaging cannot repair (EXP-016 mechanism generalized); (iv) ∈ (96.73, 96.81) → no-improvement per replicate-mean protocol unless pair mean clears; (v) BN-mismatch signature → engineering retry (≤2); (vi) infra kill → byte-identical relaunch (≤2).

Mechanism check (informational, for analysis regardless of branch): count evals within 0.15 of best — family ~8–12; the plateau-length mechanism predicts ≥ 30. This diagnoses WHICH of the paper's two claims (level vs plateau shape) transferred.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect 1,650–1,900MB
- num_epochs: `grep "^num_epochs:" run.log` — expect probe band (~126–138 at family P)
- num_params: `grep "^num_params:" run.log` — must be 4,286,026 (also an integrity gate)
