# Plan EXP-055: FreezeOut-style tail freezing of stem+stage1 (FREEZE_FRAC=0.70, dual-graph warmup)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-055.md

## Context

Baseline 96.71 @ 1990397 (bar ≥ 96.81; recipe mean ≈ 96.57, σ ≈ 0.16 per EXP-027). Chosen idea: split params into group A (conv1 + bn1 + layer1 — ≈ ⅓ of conv FLOPs by the stage-balance arithmetic) and group B (rest). Group A follows a compressed one-cycle `lr_A(p) = lr_at(min(p / FREEZE_FRAC, 1.0))` that completes its full warmup+cosine by p = 0.70; at p ≥ 0.70 group A is frozen (`requires_grad_(False)`; its BN modules stay in train mode so running stats keep tracking) and backward drops A's entire subgraph (~22% of step time) → ~+900–1,300 extra tail steps for group B plus ~+10–13 extra plateau evals. Compile warmup is extended to pre-compile BOTH graph variants (unfrozen and frozen) on random data with no optimizer.step, so the p=0.70 flip hits a cached graph instead of a charged mid-run recompile (the dynamo `requires_grad` guard otherwise forces one — infra-errors EXP-021 adjacency).

**Honest heat accounting (planning note)**: the compressed schedule is the UNSCALED FreezeOut variant — group A's integrated heat over time is 0.70× its baseline allocation (compressing the shape into 70% of the budget shrinks the integral by exactly 0.70). The brainstorm's "heat preserved" claim holds per-FreezeOut-convention (A's anneal completes rather than being truncated), not in absolute integral. The amplitude-scaled variant (peak 0.4/0.70 ≈ 0.57 for A) would preserve the integral but violates the certified-peak heat law (EXP-010: hotter peak −0.57). The unscaled variant is the conservative pre-registered form; the scaled variant is a documented unexplored alternative if branch (iii) fires.

**Failed-approaches screen**: distinguished from EXP-025/033 (data-side tail lightening — here the data distribution stays at full pressure to the last step; what stops moving is a parameter subset whose anneal has completed) and from EXP-018 (init-time freezing = deferral at peak heat — this freezes at the schedule's END, the mirror point). No High-Importance or count ≥ 2 entry matches compute reallocation across layers/time; no standing law (deferral, numerics, noise, heat-bracket, throughput-floor, absorption) prices the mechanism — throughput-floor (EXP-048) bounds non-kernel OVERHEAD, while this removes kernel WORK.

## Milestones

### Milestone 1: Code changes implemented and passing CPU sanity
- [x] Branch `autoresearch/exp-055` created from `autoresearch/dev`
- [x] train.py: `FREEZE_FRAC = 0.70` constant; 4 optimizer param groups (B-decay, B-nodecay, A-decay, A-nodecay) with `tag` keys; per-group LR in the timed loop; one-shot freeze flip at p ≥ 0.70 with a single FREEZE marker print; dual-variant compile warmup (3 unfrozen + 2 frozen iters, random data, no step, requires_grad restored, grads zeroed)
- [x] `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes
- [x] CPU sanity `/tmp/exp055_sanity.py` (`CUDA_VISIBLE_DEVICES="" uv run python /tmp/exp055_sanity.py`, `sys.path.insert(0, <project root>)`) ALL PASS:
  - params exactly 4,286,026
  - partition check: A ∪ B covers every model parameter exactly once (disjoint, complete); |A| matches `sum(p.numel() for conv1+bn1+layer1)` and the 4 groups' element counts sum to the total
  - schedule probe: lr_A(0)=0; lr_A(0.105)=PEAK_LR (compressed warmup end = 0.15×0.70); lr_A(0.70)≈0 (anneal complete); lr_A(0.85)=lr_A(1.0)=lr_at(1.0); lr_B identical to baseline lr_at at all probe points
  - freeze semantics: after `requires_grad_(False)` on A + one forward/backward, every A param has `.grad is None` and every B param has a grad tensor; bn1 `running_mean` still CHANGES across a train-mode forward (stats keep tracking); after restore, A grads flow again
  - frozen-step optimizer no-op: with A frozen and grads zeroed via set_to_none, `optimizer.step()` leaves A weights byte-identical (SGD skips `grad is None` params — also stops WD/momentum on A, intended freeze semantics)
  - 2-step eager smoke on random data: loss decreases

### Milestone 2: Experiment launched and gates clear
- [x] `git status` clean except train.py; working tree on `autoresearch/exp-055`
- [x] Launch via composite `/tmp/exp046_composite.sh` (verified present, 4023 bytes): Bash `run_in_background` + second background until-grep watcher on the task output file for `GATE_DECISION|GATE_KILL|STARTUP_KILL`, then TaskOutput(block) to wait
- [x] GATES_CLEAR (GPU-0 apps = 0 AND load < 60) and GATE_DECISION D0 ∈ 22.3–22.8ms expected (pre-freeze dt; small print-cost ε from the per-group LR loop ≤ 0.1ms). D0 > 26ms → GATE_KILL = infra, relaunch (max 2) — D0 = 22.5ms, poll 1, thresh 28.1ms
- [x] Exp-log-055.md created with Implementation Notes before/while the run proceeds

### Milestone 3: Run completes with mechanism signals readable
- [x] FREEZE marker appears in run.log at progress ≈ 0.70 (~step ~9,300, ~210s charged) — Run 2: step 9355, progress 0.700 (Run 1's requires_grad mechanism was a silent no-op under compile; fixed via graph-visible detach flag, GPU-probe-validated 31.3% saving)
- [x] Watchdog windows AFTER the freeze tick drop (probe-revised band 14.0–17.5ms): 15.3–16.5ms; windows BEFORE it 22.0–22.7ms
- [x] No mid-run recompile signature: transition window 19.3ms only, ledger 15,026 (surplus +~1,550 delivered)
- [x] rc=0, summary block parses

### Milestone 4: Verification protocol executed and verdict rendered
- [x] Integrity pre-condition evaluated (see Verification Procedure) — PASS on Run 2 (Run 1 rejected: contention + mechanism no-op)
- [x] Necessary conditions checked in order, first-failure-stop — Condition 1 FAIL: 96.32 < 96.81
- [x] Pre-registered branch identified and recorded in exp-log-055.md — branch (iii): read < 96.41, class closed with sign

## Code Changes

All in `train.py` (the only editable file), on branch `autoresearch/exp-055`:

1. **Constant** (hyperparameter block, after WARMUP_FRAC): `FREEZE_FRAC = 0.70` — fraction of budget at which group A's compressed anneal completes and A freezes.

2. **Param regrouping** (replace the current 2-group optimizer construction, L168–179). Collect group A from the eager module (`base_model.conv1, base_model.bn1, base_model.layer1` — `base_model` is bound before `torch.compile`, weights shared) by `id()`; group B = complement. Build 4 groups, each carrying a `tag` extra key (preserved by `torch.optim.Optimizer`):
   - `{"params": B_decay, "weight_decay": WEIGHT_DECAY, "tag": "B"}` (FIRST — keeps the loop's `param_groups[0]["lr"]` print showing group B's LR, the still-training group, family-comparable)
   - `{"params": B_no_decay, "weight_decay": 0.0, "tag": "B"}`
   - `{"params": A_decay, "weight_decay": WEIGHT_DECAY, "tag": "A"}`
   - `{"params": A_no_decay, "weight_decay": 0.0, "tag": "A"}`
   Decay split unchanged: `p.ndim > 1` decays. Keep `group_a_params` (the flat A list) in scope for the warmup flip and the freeze flip.

3. **Dual-variant compile warmup** (extend L183–198): keep the existing 3 unfrozen iterations exactly as-is; then flip `p.requires_grad_(False)` for all A params, run 2 more autocast forward+backward iterations on the SAME random tensors (compiles+caches the frozen graph variant: AOTAutograd specializes on param `requires_grad`, so this is a distinct graph), then restore `p.requires_grad_(True)`, then the existing `optimizer.zero_grad(set_to_none=True)` + synchronize + del. No optimizer.step anywhere in warmup, random data only — the established legitimate uncharged-warmup pattern (EXP-006, extended exactly as the brainstorm pre-registered). Weights provably unchanged (backward only populates .grad).

4. **Timed-loop LR + freeze flip** (replace L222–225). Before the loop: `frozen = False`. In the step, after computing `progress`:
   ```python
   if not frozen and progress >= FREEZE_FRAC:
       for p in group_a_params:
           p.requires_grad_(False)
       frozen = True
       print(f"\nFREEZE: group A frozen at step {step + 1}, progress {progress:.3f}")
   lr_b = lr_at(progress)
   lr_a = lr_at(min(progress / FREEZE_FRAC, 1.0))
   for g in optimizer.param_groups:
       g["lr"] = lr_a if g["tag"] == "A" else lr_b
   ```
   The flip + print cost ~µs once, inside the charged region (honest accounting). After the flip the next forward dispatches to the warmup-cached frozen graph — no recompile. A params then have `grad is None` every step (zero_grad set_to_none + no backward accumulation), so SGD skips them entirely: no weight update, no WD, no momentum — true freeze. BN modules of A remain in train mode (model.train() is global), so running stats keep tracking batch statistics, per FreezeOut and per the BN-constants law (EXP-029/038/39: stats must stay train-distribution-sourced and fresh).

   Risks: (a) recompile leak despite dual warmup — detectable as a one-off ≥ 15s watchdog stall at the flip + collapsed step ledger → branch (iv), ONE fix attempt; (b) loader pressure rises ~25% post-freeze (17.5ms steps vs 22.5) — stalls land OUTSIDE charged dt, so the step surplus is unaffected, but total_seconds may grow ~+30–60s; family total is 470–510s so the 600s cap retains ≥ 40s margin (EXP-013 watch item; WALL_CAP kill in composite is the backstop, classified infra).

Nothing else changes: eval path (`evaluator.evaluate(base_model, device)`) untouched, timer semantics untouched, augmentation/batch/schedule constants untouched.

## Configuration Changes

- `FREEZE_FRAC`: (new) → 0.70 (brainstorm-055 choice: late enough that A's 0.70-compressed anneal still satisfies the heat envelope, early enough that ~30% of the budget converts at the ~22% backward saving; FreezeOut's own t_0 sweet spot is cited around 0.5–0.8 of training with accuracy flat at the high end)
- Group A LR schedule: `lr_at(progress)` → `lr_at(min(progress / 0.70, 1.0))` (compressed one-cycle, completes at p=0.70; peak unchanged at 0.4 — certified)
- Everything else byte-identical to the EXP-006 certified recipe.

## Execution Environment

- Method: local, via the validated composite launcher `/tmp/exp046_composite.sh` (dual gates: zero GPU-0 compute apps AND 1-min load < 60, 30s poll up to 2h; `rm -f run.log`; `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1 &`; inline watchdog 44×15s with D0 gate >26ms, contention streak ≥4 windows >max(26, D0×1.25), NaN, divergence <15% after ep5, wall cap). Launch with Bash `run_in_background`, plus a second background until-grep watcher for `GATE_DECISION|GATE_KILL|STARTUP_KILL` on the task output file; TaskOutput(block=true, timeout=600000) to wait for completion. No `sleep` polling in the main loop.
- Resources: GPU 0 ONLY (gate enforces); ~10 of 180 cores; VRAM ≈ family (~baseline peak; frozen phase saves backward activations, peak set pre-freeze).
- Estimated runtime: startup ~45–55s (baseline ~23s + frozen-variant inductor compile ~15–25s), 300.0s charged, ~148 evals ≈ 195s, stalls — total ≈ 500–560s (≤ 600s cap with margin; STARTUP_KILL threshold 180s is 3× the expected startup).
- Log output: `run.log` in project root (the source of truth); composite stdout (gate/watchdog/summary) in the background task output file. run.log deleted after the experiment concludes per goal procedure.
- Tool skill: none (local execution).

## Abort Criteria

- **GATE_TIMEOUT / GATE_KILL (D0 > 26ms) / CONTENTION_KILL / STARTUP_KILL** (composite-automated): infrastructure, not research — relaunch byte-identically once gates clear, max 2 relaunches, then Outcome failed (branch (v)).
- **NAN_KILL / DIVERGENCE_KILL** (composite-automated): research failure → Outcome failed, no blind retry (one fix-retry only for a demonstrable implementation bug, e.g. wrong group partition).
- **Mid-run recompile signature**: one-off ≥ 15s stall in watchdog windows at/after the freeze tick (window sequence shows a frozen step counter for ≥ 1 tick then resumes), OR final step ledger far below family (< ~13,000) with clean per-window dt — branch (iv): engineering failure; ONE fix attempt (strengthen the dual-warmup cache, e.g. 3 frozen iters / verify flip ordering), else Outcome failed.
- **WALL_CAP** (composite kills after watchdog window; goal hard-fails > 600s total): if caused by loader-stall growth, classify infra-adjacent (EXP-013 mechanism), one relaunch permitted only if a demonstrable transient (foreign load) contributed; otherwise Outcome failed.
- **Missing FREEZE marker** by ~80% progress with run still alive: implementation bug (flip never fired) — kill, fix, one retry.

## Verification Protocol

### Verification Procedure

All commands from the project root. Source of truth: `run.log` + composite task output. Timeout for the whole run: TaskOutput block 600s, re-block as needed; treat a composite that produces neither RC line nor kill marker after 20 min as infrastructure failure.

**Integrity pre-condition (gates all conditions; evaluated first)**:
1. Pristine-run check from composite output: GATES_CLEAR; D0 ∈ [21.5, 23.5]ms; pre-freeze windows ≤ 23.5ms mean, none > 27ms; no kill markers; rc=0.
2. Mechanism + ledger check from run.log (**bands revised pre-launch after Run 1's failed requires_grad mechanism was replaced by the detach-flag mechanism and the GPU probe measured the actual saving**: unfrozen 22.04ms, frozen 15.15ms = 31.3% saving, flip 0.016s no-recompile — /tmp/exp055_gpu_probe.py):
   - `tr '\r' '\n' < run.log | grep "FREEZE:"` — marker present at progress ≈ 0.700 ± 0.005
   - post-freeze watchdog windows ∈ ~[14.0, 17.5]ms (the dt drop IS the mechanism); if post-freeze windows stay ~22ms the freeze did not take (implementation failure, not a research read)
   - `num_steps` ≥ ~14,500, expect ~15,000–15,500 (family 13,400–13,500; ≈9,330 pre-freeze + ≈5,900 post-freeze at 15.2ms — an ~12% surplus, far above the <1% scatter band of EXP-048/053)
   - `num_epochs` ∈ [148, 162]; `num_params: 4,286,026` exact; `training_seconds: 300.0`; `total_seconds` ≤ 600 (watch item: ~157 evals ≈ 205s + loader demand 1.5× during the frozen tail — stalls are uncharged but inflate wall; WALL_CAP is the backstop); evals ≤ num_epochs (once per epoch ceiling)
   - trajectory criterion (EXP-048): ep1 ≥ 30%, trajectory rejoins family, plateau converged-flat, final_test_loss informative (~0.18–0.20 family band; group-A freeze may shift it slightly — informational, not gating)
   - no NaN/EMA spikes in the loss trail
3. If integrity fails on a contention/infra signature → relaunch (max 2). If it fails on the recompile signature → branch (iv). If it fails on absent dt-drop → fix-or-fail (one attempt).

**Necessary conditions (first-failure-stop, from the goal file, baseline via exp-index)**:
1. **best_test_acc ≥ 96.81** (= baseline 96.71 + 0.1; re-query at verification time: `bash .../exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-test-accuracy.tsv`). Extract: `tr '\r' '\n' < run.log | grep "^best_test_acc:"`. Empty grep = crash → `tail -n 50 run.log`.
   - **Escalation (pre-registered, EXP-052 protocol)**: a single read ≥ 96.81 does NOT decide. Launch a second byte-identical run via the same composite; improvement iff the MEAN of the two ≥ 96.81. Max is never a decision input. A sub-bar read → no escalation, branch (ii)/(iii) by the bands below.
2. **Run completes within budget**: rc=0 AND `total_seconds` ≤ 600 (grep `^total_seconds:`).
3. **Validation at most once per epoch**: eval-line count ≤ num_epochs (`tr '\r' '\n' < run.log | grep -cE "^  eval ep"` vs `^num_epochs:`).

**Pre-registered outcome branches (from brainstorm-055 Hypothesis, verbatim mapping)**:
- (i) read ≥ 96.81 → replicate-pair escalation; improvement iff MEAN ≥ 96.81.
- (ii) read ∈ [96.41, 96.73] (mean band) at family-shaped signatures WITH the step surplus visible in the ledger → freeze free but tail steps sub-σ; class closed, tail-conversion law sharpened. Verdict no-improvement.
- (iii) read < 96.41 → layer-1 tail refinement is load-bearing (parameter-side tail-pressure law, alongside the data-side EXP-025/033 law); class closed with sign. Verdict no-improvement.
- (iv) recompile signature → engineering failure, ONE fix attempt (dual-warmup cache), else Outcome failed (verdict crash if no usable metric, else no-improvement with the integrity caveat recorded).
- (v) gate/contention/startup kills → infra relaunch, max 2; if exhausted, Outcome failed, verdict crash.
- Reads in (96.73, 96.81): sub-bar, no escalation — record as no-improvement with the +σ note; do NOT single-draw-promote (EXP-052 protocol finding).

### Informational Metrics (Optional)

- peak_vram_mb: `tr '\r' '\n' < run.log | grep "^peak_vram_mb:"` — expect ≈ family (peak set pre-freeze)
- num_epochs: `grep "^num_epochs:"` — expect ≥ 145 (the converted tail)
- num_steps: `grep "^num_steps:"` — the step ledger; expect ~14,200–14,500 (mechanism quantification)
- num_params: `grep "^num_params:"` — must be 4,286,026 (also in integrity)
- startup_seconds: `grep "^startup_seconds:"` — expect ~45–55s (dual-compile cost; uncharged)
- FREEZE step index + post-freeze window dts from composite output — the direct mechanism record for the report
