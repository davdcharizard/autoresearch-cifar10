# Plan EXP-028: Muon optimizer for conv weights (airbench-anchored hybrid, 2-point LR design)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-028.md

## Milestones

### Milestone 1: Hybrid optimizer implemented and sanity-checked
- [ ] On branch `autoresearch/exp-028` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. New constants after LABEL_SMOOTHING: `MUON_PEAK_LR = 0.24` (airbench anchor), `MUON_MOMENTUM = 0.6  # nesterov`.
  2. New module-level function `zeropower_via_newtonschulz5(G, steps=5)` — exactly the Jordan implementation recorded in knowledge/references/muon-optimizer.md (quintic coefficients 3.4445/−4.7750/2.0315, bf16, transpose-if-tall, norm-normalize, 5 iterations).
  3. Parameter split: `muon_params = [p for p in model.parameters() if p.ndim == 4]` (the 19 conv weights); the existing SGD keeps everything else with baseline behavior — groups: `fc weight (ndim==2) with WEIGHT_DECAY` and `ndim<=1 with 0.0` (same selective-WD rule as baseline, minus the convs which move to Muon).
  4. Muon state: `muon_bufs = [torch.zeros_like(p) for p in muon_params]`.
  5. In the timed step body, replace the single `optimizer.step()` with: `optimizer.step()` (SGD group, lr = `lr_now` as today) followed by an inline Muon step under `torch.no_grad()`: for each (p, buf): `g = p.grad; buf.mul_(MUON_MOMENTUM).add_(g); u = g.add(buf, alpha=MUON_MOMENTUM)` (nesterov); `O = zeropower_via_newtonschulz5(u.reshape(p.size(0), -1)).reshape_as(p)`; `scale = max(1, p.size(0) / u.reshape(p.size(0), -1).size(1)) ** 0.5`; decoupled WD `p.mul_(1 - muon_lr_now * WEIGHT_DECAY)`; `p.add_(O.to(p.dtype), alpha=-muon_lr_now * scale)`. `muon_lr_now = MUON_PEAK_LR * lr_now / PEAK_LR` (reuses the time-keyed one-cycle shape exactly).
  6. `optimizer.zero_grad(set_to_none=True)` already clears conv grads too (they remain leaves of the same module) — verify no grad-retention leak.
  7. Compile-warmup loop: the 3 warmup iterations call only forward/backward (no optimizer.step) — unchanged; Muon state stays zero until the timed loop.
- [ ] Sanity: `uv run python -c "import ast; ast.parse(open('train.py').read())"`; grep confirms exactly one `optimizer.step()` plus the Muon block inside the timed region (before `torch.cuda.synchronize()`); param-split assertion added temporarily or checked via startup print: 19 muon tensors; `num_params` print must still equal 4,286,026.

### Milestone 2: Run 1 (peak 0.24) launched with gates
- [ ] GPU-0 zero-compute-apps pre-check passes inside the composite launcher.
- [ ] Composite: pre-check → `rm -f run.log` → launch → 15s watchdog with: contention kill (4 consecutive >30ms... NOTE raised to >32ms here since legitimate dt may sit ~25–27ms — contention threshold = gate threshold +5ms), **early-dt gate: kill at 3 consecutive windows >27.0ms within first 7 ticks**, STARTUP_KILL tick 10, NaN/inf guard → `wait` → summary.
- [ ] Early readout (analysis while running): dt in 23–27ms ⇒ epochs ~115–135; required gross effect = epoch deficit (≈0.015pp/epoch) + 0.24 (bar minus mean 96.57) — recorded in exp-log.

### Milestone 3: Trajectory readout and (if needed) Run 2
- [ ] Mid-run readout at ~ep10: if Run 1 was killed by divergence OR ep10 test_acc < 70% (baseline family ~78), launch Run 2 with `MUON_PEAK_LR = 0.12` (pre-authorized one-knob adjustment); otherwise let Run 1 complete and stand.
- [ ] Completion: rc=0, total ≤600s, eval_lines = num_epochs, params 4,286,026, post-hoc profile clean (windows > [gate+5]ms ≤2, epochs within ±3 of 139 × 22.4 / mean_win_ms).

## Code Changes
- **train.py** (only file): constants + NS5 function + param split + hybrid step as specified in Milestone 1. The eval path, schedule function, loaders, compile path, and SGD group semantics are untouched.
- Why this tests the hypothesis: the only change is the update GEOMETRY of the conv weights — same data, same schedule shape, same budget accounting (the Muon math executes inside the timed region, honestly charged to the 300s).
- Risks/edge cases: (a) NS in bf16 on H20 — exactly its native design (airbench runs it in half precision); (b) compiled-model params are the same objects as base_model's (compile wraps, doesn't copy) — Muon mutates them in place like SGD does; (c) channels_last conv weights: `.reshape(p.size(0), -1)` may need contiguity — use `.reshape` (handles non-contiguous via copy) NOT `.view`; the per-step copy cost is part of the measured dt; (d) decoupled WD uses muon_lr_now (not the SGD lr) — pressure differs slightly from baseline's coupled WD; accepted, this is a big-swing experiment; (e) momentum 0.6 for convs is outside the SGD-bracketed noise law — that law was measured within SGD geometry; flagged in brainstorm.

## Configuration Changes
- MUON_PEAK_LR: 0.24 (Run 1; airbench's measured value — the only conv-net anchor) | 0.12 (Run 2, only if the §Milestone-3 trigger fires)
- MUON_MOMENTUM: 0.6, nesterov (airbench pair with 0.24)
- All baseline constants unchanged (PEAK_LR 0.4 still drives the SGD group and the schedule shape).

## Execution Environment
- Method: local composite background Bash (pre-check + launch + inline watchdog + wait + summary), branch `autoresearch/exp-028`, GPU 0 (`CUDA_VISIBLE_DEVICES=0`).
- Resources: VRAM ~1630MB (+~17MB Muon buffers); 8 loader workers.
- Estimated runtime: ~490–560s if dt lands 23–26ms; under the 600s cap (eval count drops with epochs, so wall stays bounded).
- Log output: `run.log`; watchdog WIN lines; post-hoc awk profile authoritative.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill (note: NS adds compile-free eager work; startup should still be ~13s).
- **Early-dt gate**: 3 consecutive windows >27.0ms within first 7 ticks → kill; arithmetic makes the bar implausible below ~112 epochs.
- **Contention kill**: 4 consecutive windows >32ms → kill, contaminated, rerun once.
- **Divergence**: NaN/inf loss, or any eval test_acc <15% after epoch 5 → kill → triggers Run 2 at peak 0.12 (Milestone 3), not a crash retry.
- **Wall cap**: >600s → kill, failure.
- **Crash** (rc≠0, import/shape errors): code-error fix + resubmit per execute-skill rules (max 2).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order, on the FINAL completed run (Run 2 supersedes Run 1 only if Run 1 triggered the Milestone-3 rule). Baseline from `exp-index.sh baseline`: **96.71** @ 1990397. Context from EXP-027: baseline mean ≈96.57, σ ≈0.16 — interpretation of any miss uses this calibration.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition: post-hoc profile — `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>32) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>32ms: %d of %d | mean win %.1f ms | expected epochs %.1f\n", c, n, s/n, 139*22.4/(s/n)}'` — require ≤2 slow windows AND epochs within ±3 of expected. Contaminated ⇒ rerun once.
   - Integrity sub-check on a bar-pass: final-7 median ≥ 96.6 (`tr '\r' '\n' < run.log | grep "eval ep" | tail -7`); `grep "^num_params:" run.log` = 4,286,026 (Muon adds no params); gain visible as a trajectory-wide/plateau-level shift, not a single-eval spike.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: `tr '\r' '\n' < run.log | grep -c "eval ep"` ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- Measured Muon dt overhead: mean win ms − 22.4 (the NS cost on H20, recorded to knowledge/references/muon-optimizer.md)
- num_epochs, VRAM (expect ~1630), startup
- Trajectory family comparison at ep1/5/10 (deferral check) and final-7 median vs best (plateau shape)
