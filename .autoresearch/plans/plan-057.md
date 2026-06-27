# Plan EXP-057: Decouple the classifier from weight decay (fc.weight WD 5e-4 → 0)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-057.md

Baseline (exp-index): 96.71 @ 1990397 → bar = 96.81. σ context (EXP-027): recipe mean ≈ 96.57, σ ≈ 0.16; reads in (96.73, 96.81) are no-improvement by protocol and never single-draw promoted.

**Idea**: fc is the single BN-free, scale-sensitive layer — the one parameter class the WD-with-BN equilibrium argument does not cover. WD 5e-4 on fc.weight monotonically shrinks the logit scale; label smoothing ε=0.1 fixes a finite optimal logit gap (≈ 4.51). Move fc.weight from the decay group to the no-decay group and let CE+LS find its own self-limiting equilibrium. Zero graph / loader / schedule / loop change — training signatures byte-identical to the family by construction, so **no GPU probe is required** (the optimizer is eager; param-group membership is invisible to inductor; same two-group foreach structure).

## Milestones

### Milestone 1: Code change + CPU sanity PASS
- [ ] Branch `autoresearch/exp-057` cut from `autoresearch/dev`; edit train.py optimizer param-group construction only (~6 lines)
- [ ] CPU sanity (`CUDA_VISIBLE_DEVICES="" uv run python /tmp/exp057_sanity.py`): (a) total params == 4,286,026; (b) fc.weight is in the WD=0.0 group and NOT in the WD=5e-4 group (membership by `id()`); (c) every conv weight (ndim>1, non-fc) in the WD=5e-4 group; group numel ledger: baseline no-decay (19 BNs: 2×(64+6×64+6×128+6×256)=5,504, + fc.bias 10) = 5,514 → new no-decay = 5,514 + 2,560 = 8,074, new decay = 4,286,026 − 8,074 = 4,277,952; (d) 3-step smoke at lr 0.01 on one random batch, `losses[-1] < losses[0]` (lr 0.05 overshoots with nesterov — known test artifact)

### Milestone 2: Gated launch, confirmed running
- [ ] Reuse `/tmp/exp046_composite.sh` verbatim (dual gates: zero GPU-0 compute apps AND load < 60, poll 30s×240; `rm -f run.log`; background `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1 &`; watchdog 44×15s)
- [ ] Launch via Bash `run_in_background` (NEVER detached `& disown`) + background until-grep watcher on the composite log + TaskOutput(block=true, timeout=600000)
- [ ] GATE_DECISION shows D0 in family band [21.5, 23.5]ms (no probe offset applies — graph unchanged)

### Milestone 3: Run completes with clean signatures
- [ ] rc=0, no NaN, no CONTENTION_KILL/GATE_KILL/STARTUP_KILL; charged 300.0s; total ≤ 600s
- [ ] Family signatures: dt 22.0–22.8ms, 138–140 epochs, steps 13,400–13,515, params 4,286,026, evals ≤ epochs, ep1 ≥ 30 tripwire

### Milestone 4: Verification (first-failure-stop) + exp-log complete
- [ ] Integrity pre-condition, then Condition 1 (metric vs bar), Conditions 2–3 informational; record in exp-log-057.md

## Code Changes
- **train.py** (only file; optimizer setup block L168–175): replace the two-list comprehension with an fc-aware split —
  ```python
  # No weight decay on BN params/biases (ndim <= 1) and on the classifier weight:
  # fc is the only layer with no BN after it, so WD there directly shrinks the
  # logit scale that CE+LS wants at a finite equilibrium. Decay conv weights only.
  fc_weight_id = id(base_model.fc.weight)
  decay_params = [
      p for p in model.parameters() if p.ndim > 1 and id(p) != fc_weight_id
  ]
  no_decay_params = [
      p for p in model.parameters() if p.ndim <= 1 or id(p) == fc_weight_id
  ]
  ```
  Optimizer call, lr_at, warmup, loop: byte-identical. `id()`-based membership against `base_model` is the EXP-055-validated pattern (compiled wrapper shares the underlying Parameters). This is the entire diff — it tests the hypothesis because the ONLY changed quantity is the decay pressure on the 2,560-element fc.weight (fc.bias is ndim 1, already no-decay).
  Risks: none mechanical — same two-group SGD structure, same total params; the optimizer prints/lr assignment loop iterate over `param_groups` generically.

## Configuration Changes
- fc.weight weight_decay: 5e-4 → 0.0 (single variable; rationale in brainstorm-057 — LS-equilibrium vs WD shrinkage on the one BN-free layer). No other constant changes.

## Execution Environment
- Method: local, GPU 0 ONLY (wait for it if busy — never GPU 1), via the gated composite launcher `/tmp/exp046_composite.sh`, launched with Bash `run_in_background`; second background until-grep watcher (`until grep -qE "GATE_DECISION|GATE_KILL|STARTUP_KILL|GATE_TIMEOUT" <composite log>; do sleep 10; done`); then TaskOutput block
- Resources: 1× H20 (98GB; run needs ~4GB), host load < 60 at launch (both gates enforced by the launcher)
- Estimated runtime: ~470–510s total (300.0s charged); watchdog wall cap kills at 660s
- Log output: `run.log` in project root (source of truth; deleted after analysis); composite telemetry to the launcher's own log

## Abort Criteria
(Enforced by the in-launcher watchdog; manual checks mirror them)
- GATE_KILL: D0 (median dt at ~step 100) > 26ms
- CONTENTION_KILL: 4 consecutive 15s windows > max(26ms, D0×1.25)
- STARTUP_KILL: no training steps by tick 12 (180s after launch)
- NaN in loss, or divergence (best < 15% after epoch 5)
- WALL_CAP: total > 660s
- Experiment-specific: none beyond family bands — any dt drift > ±1ms from D0 is unexplained (graph is unchanged) and flags contamination, not mechanism

## Verification Protocol

### Verification Procedure
First-failure-stop. Integrity pre-condition must pass before Condition 1 is evaluated; if integrity fails, the run is contaminated/invalid — relaunch byte-identically (max 2) rather than rendering a verdict from it.

0. **Integrity pre-condition** (read run.log + watchdog log; timeout 2 min): rc=0; D0 ∈ [21.5, 23.5]; no kill markers; no window > 27ms; `num_params: 4,286,026`; `training_seconds: 300.0`; `total_seconds ≤ 600`; eval lines ≤ num_epochs (once-per-epoch ceiling); epochs ∈ [136, 141] and steps ∈ [13,300, 13,600]; ep1 acc ≥ 30 (trajectory tripwire — judge by rejoins-family if tripped); no NaN. PASS → proceed.
1. **Condition 1 — metric beats baseline by ≥ 0.1pp**: `grep "^best_test_acc:" run.log` → value ≥ 96.81 PASSES. Branches (pre-registered, all terminal):
   - (i) ≥ 96.81 → escalate to replicate-pair (second byte-identical gated run; decision = MEAN of the two ≥ 96.81; max never a decision input)
   - (ii) ∈ [96.41, 96.73] family band → no-improvement; per-layer WD coverage complete, fc WD measured redundant (corner closed)
   - (iii) ∈ (96.73, 96.81) → no-improvement by standing protocol (never single-draw promoted)
   - (iv) < 96.41 → no-improvement, sign-down: fc WD's margin cap is load-bearing regularization; corner closed from below
   - (v) infra contamination → relaunch byte-identically (max 2)
2. **Condition 2 — completes within budget**: `grep "^total_seconds:" run.log` ≤ 600 (informational if C1 already failed)
3. **Condition 3 — validation ≤ once/epoch**: count `eval ep` lines ≤ `num_epochs` (structurally guaranteed; informational)

Then delete run.log after metrics are extracted into the exp-log/report.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~family (≈ 5,400)
- num_epochs: `grep "^num_epochs:" run.log` — expect 138–140
- num_params: `grep "^num_params:" run.log` — must be 4,286,026
