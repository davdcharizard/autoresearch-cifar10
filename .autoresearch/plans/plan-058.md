# Plan EXP-058: Classifier weight decay ×4 (fc.weight WD 5e-4 → 2e-3) — dose-response along the measured slope
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-058.md

Baseline (exp-index): 96.71 @ 1990397 → bar = 96.81. σ context (EXP-027): recipe mean ≈ 96.57, σ ≈ 0.16; reads in (96.73, 96.81) are no-improvement by protocol.

**Idea**: EXP-057 measured the first directional slope on the frontier — fc decay pressure 0 → 5e-4 gains ~0.2pp (the classifier's WD margin cap is load-bearing under CE+LS + heavy aug). Test the unmeasured side: fc.weight gets its own param group at WD 2e-3 (×4), conv weights stay at 5e-4, BN/bias at 0. Optimizer-only diff; graph/schedule/loader/loop byte-identical; **no GPU probe** (EXP-057 validated this exact diff class in vivo: D0 22.7ms, family signatures throughout).

## Milestones

### Milestone 1: Code change + CPU sanity PASS
- [ ] Branch `autoresearch/exp-058` cut from `autoresearch/dev`; edit train.py: add `FC_WEIGHT_DECAY = 2e-3` constant + three-group optimizer split (~8 lines)
- [ ] CPU sanity (`CUDA_VISIBLE_DEVICES="" uv run python /tmp/exp058_sanity.py`, adapted from /tmp/exp057_sanity.py): (a) total params == 4,286,026; (b) fc.weight alone in the WD=2e-3 group; (c) conv group numel 4,277,952, fc group numel 2,560, no-decay numel 5,514; groups disjoint and exhaustive; (d) 3-step smoke at lr 0.01, losses decreasing

### Milestone 2: Gated launch, confirmed running
- [ ] Reuse `/tmp/exp046_composite.sh` verbatim; launch via Bash `run_in_background` + background until-grep watcher + TaskOutput(block=true, timeout=600000)
- [ ] GATE_DECISION shows D0 ∈ family band [21.5, 23.5]ms (graph unchanged — no probe offset)

### Milestone 3: Run completes with clean signatures
- [ ] rc=0, no NaN, no kill markers; charged 300.0s; total ≤ 600s
- [ ] Family signatures: dt 22.0–22.8ms, 138–140 epochs, steps 13,400–13,515, params 4,286,026, evals ≤ epochs, ep1 ≥ 30 tripwire

### Milestone 4: Verification (first-failure-stop) + exp-log complete
- [ ] Integrity pre-condition, then Condition 1 (metric vs bar), Conditions 2–3 informational; record in exp-log-058.md

## Code Changes
- **train.py** (only file; two hunks):
  1. Constants block (~L26): add `FC_WEIGHT_DECAY = 2e-3  # classifier-only cap tightening; conv stays at WEIGHT_DECAY` below WEIGHT_DECAY.
  2. Optimizer setup (~L168): three-group split —
  ```python
  # No weight decay on BN params/biases (ndim <= 1); conv weights decay at
  # WEIGHT_DECAY; the classifier weight gets its own, tighter decay: fc is the
  # only layer with no BN after it, so WD there directly caps the logit scale
  # (EXP-057 measured the cap as load-bearing).
  fc_weight_id = id(base_model.fc.weight)
  decay_params = [
      p for p in model.parameters() if p.ndim > 1 and id(p) != fc_weight_id
  ]
  no_decay_params = [p for p in model.parameters() if p.ndim <= 1]
  optimizer = optim.SGD(
      [
          {"params": decay_params, "weight_decay": WEIGHT_DECAY},
          {"params": [base_model.fc.weight], "weight_decay": FC_WEIGHT_DECAY},
          {"params": no_decay_params, "weight_decay": 0.0},
      ],
      lr=0.0,  # set per-step by lr_at()
      momentum=MOMENTUM,
      nesterov=True,
  )
  ```
  The in-loop `for g in optimizer.param_groups: g["lr"] = lr_now` and the `optimizer.param_groups[0]["lr"]` print are group-count-agnostic (group 0 remains the conv-decay group, so the printed lr is the live schedule). This is the entire diff — the only changed quantity vs baseline is the decay coefficient on the 2,560-element fc.weight.
  Risks: none mechanical (optimizer is eager; the third group adds one foreach partition of one tensor — dt effect unmeasurable, and the watchdog band would catch any surprise).

## Configuration Changes
- fc.weight weight_decay: 5e-4 → 2e-3 (×4; single variable). Rationale: dose-response continuation of EXP-057's measured positive slope (0 → 5e-4 = ~+0.2pp); ×4 chosen so a real effect clears one-draw resolution (≥ +0.3) while staying a moderate cap tightening (equilibrium fc norm scales sub-linearly in λ). No other constant changes.

## Execution Environment
- Method: local, GPU 0 ONLY (wait if busy — never GPU 1), gated composite launcher `/tmp/exp046_composite.sh` via Bash `run_in_background`; second background until-grep watcher on the composite log (`until grep -qE "GATE_DECISION|GATE_KILL|STARTUP_KILL|GATE_TIMEOUT" /tmp/exp058_composite_run1.log; do sleep 10; done`); then TaskOutput block
- Resources: 1× H20 (≈2GB used), host load < 60 at launch (both gates enforced by launcher)
- Estimated runtime: ~470–510s total (300.0s charged); watchdog wall cap 660s
- Log output: `run.log` in project root (source of truth; deleted after analysis); composite telemetry to /tmp/exp058_composite_run1.log

## Abort Criteria
(Enforced by the in-launcher watchdog; manual checks mirror them)
- GATE_KILL: D0 > 26ms | CONTENTION_KILL: 4 consecutive windows > max(26ms, D0×1.25) | STARTUP_KILL: no steps by tick 12 | NaN loss | divergence (best < 15% after epoch 5) | WALL_CAP > 660s
- Experiment-specific: any dt drift > ±1ms from D0 is contamination, not mechanism (graph unchanged); a strongly depressed EARLY trajectory (ep1 < 30) judged by the standing trajectory criterion (rejoins-family + plateau + family test_loss)

## Verification Protocol

### Verification Procedure
First-failure-stop. Integrity pre-condition gates Condition 1; integrity failure → contaminated, relaunch byte-identically (max 2), never analyzed.

0. **Integrity pre-condition** (run.log + watchdog log; timeout 2 min): rc=0; D0 ∈ [21.5, 23.5]; no kill markers; no window > 27ms; `num_params: 4,286,026`; `training_seconds: 300.0`; `total_seconds ≤ 600`; eval lines ≤ num_epochs; epochs ∈ [136, 141], steps ∈ [13,300, 13,600]; ep1 ≥ 30 (else trajectory criterion); no NaN.
1. **Condition 1 — metric beats baseline by ≥ 0.1pp**: `grep "^best_test_acc:" run.log` ≥ 96.81 PASSES. Pre-registered branches (all terminal):
   - (i) ≥ 96.81 → replicate-pair escalation (second byte-identical gated run; decision = MEAN ≥ 96.81; max never a decision input)
   - (ii) ∈ [96.41, 96.73] family band → no-improvement; slope saturates at/before 5e-4 — fc-WD axis closed FLAT above default (three measured points: 0 ↓, 5e-4 ✓, 2e-3 →)
   - (iii) ∈ (96.73, 96.81) → no-improvement by standing protocol
   - (iv) < 96.41 → no-improvement, over-constrained: optimum bracketed in (0, 2e-3) with 5e-4 the measured best — axis closed from above
   - (v) infra contamination → relaunch byte-identically (max 2)
2. **Condition 2 — completes within budget**: `grep "^total_seconds:" run.log` ≤ 600 (informational if C1 failed)
3. **Condition 3 — validation ≤ once/epoch**: eval-line count ≤ num_epochs (structural; informational)

Delete run.log after metrics are extracted into the exp-log/report.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1,600 (family)
- num_epochs: `grep "^num_epochs:" run.log` — expect 138–140
- num_params: `grep "^num_params:" run.log` — must be 4,286,026
