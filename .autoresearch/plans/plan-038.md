# Plan EXP-038: BN running-stat momentum 0.1 → 0.02 — the last unmeasured (implicit) constant
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-038.md

## Milestones

### Milestone 1: One-constant change implemented and sanity-checked
- [x] On branch `autoresearch/exp-038` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. Constants block: add `BN_MOMENTUM = 0.02  # running-stat EMA horizon ~50 batches (default 0.1 ~10); never dosed — implicit framework constant (EXP-038)` after `LABEL_SMOOTHING`.
  2. All three `nn.BatchNorm2d(...)` construction sites (BasicBlock bn1, bn2; ResNet stem bn1) → `nn.BatchNorm2d(..., momentum=BN_MOMENTUM)`.
- [x] Sanity: AST parse OK; `git diff` shows 4 changed/added lines in train.py only; CPU check (`CUDA_VISIBLE_DEVICES="" uv run`): every `nn.BatchNorm2d` module reports `momentum == 0.02`, params 4,286,026 (unchanged), forward (4,3,32,32)→(4,10) OK. (Measured: 19 BN layers, momenta {0.02}, params 4,286,026, forward OK)
- NOTE: momentum is set at CONSTRUCTION, never mutated at runtime → no torch.compile guard/recompile risk (the EXP-035 SAM caveat applied to runtime toggling only).

### Milestone 2: Run 1 launched with gates and baseline watchdog
- [x] Launch gates: zero GPU-0 compute apps AND host 1-min load < 60, poll 30s up to 2h. (GATES_CLEAR poll 1)
- [x] Composite watchdog (15s ticks, 44): contention 4 consecutive windows >27ms (baseline thresholds — signatures must be byte-identical); STARTUP_KILL tick 10; NaN/inf; divergence eval <15% after ep5; wall cap 600s. (Never triggered; windows 21.7–22.8ms)
- [ ] Expected signatures: dt ≈ 22.4ms windows, ~139 epochs, VRAM 1613.0MB, total ~475–495s. **Monitoring note (diagnostic, not abort)**: hot-phase evals (ep ~5–60) may sit slightly BELOW the baseline family — a 50-batch stat horizon lags fast-moving weights; this is the predicted mechanism signature and is harmless to a plateau-taken max. The divergence guard (<15%) is far below any plausible lag effect.

### Milestone 3: Completion and readout
- [x] Full run: rc=0, total 499.3s ≤600s, epochs 139, eval_lines 139 = num_epochs, params 4,286,026. (best 96.27 — below baseline band; hypothesis inverted)
- [ ] No fallback runs: single clean run decides. Momentum 0.05 (mid-dose) or 0.005 (extreme) are NEW experiments contingent on this read's sign; a null at 0.02 closes the dial (estimator noise <0.1 contribution) without further doses.

## Code Changes
- **train.py** (only file): one new constant + three constructor arguments. Why this tests the hypothesis: the ONLY change is the BN running-stat EMA horizon — the constants every eval normalizes with — at byte-identical training-path execution (the EMA update is the same fused op regardless of momentum value; weights, gradients, schedule, noise all untouched). Any plateau shift is attributable to normalization-constant estimation quality, the exact in-regime sensitivity EXP-029 measured.
- Risks/edge cases: (a) hot-phase eval lag (cosmetic, pre-flagged above); (b) plateau-period lag if end-of-run weight drift is larger than assumed — bounded by the moderate dose (0.02, not 0.005); (c) zero risk in any closed currency — dt/heat/noise/VRAM/params unchanged.

## Configuration Changes
- BN momentum: 0.1 (PyTorch default, never set) → 0.02 on all 19 BN layers. Rationale: ~5× variance reduction of the stat estimator at the converged plateau where the max-statistic is taken; weights are quasi-static there so lag cost ≈ 0 (brainstorm-038 Candidate 1; EXP-029 constants-sensitivity law; EXP-027 plateau scatter). Dose chosen mid-range: meaningful smoothing without the extreme-lag regime.
- All other constants unchanged.

## Execution Environment
- Method: local composite background Bash (dual launch gates → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` → 44×15s watchdog → wait → rc + summary greps + eval tails), branch `autoresearch/exp-038`, GPU 0 only.
- Resources: VRAM ~1613MB (baseline-identical); 8 loader workers.
- Estimated runtime: ~475–495s total (baseline signature). ~105s margin under the cap.
- Log output: `run.log` (no tee); post-hoc awk profile authoritative; delete run.log after analysis.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill.
- **Contention**: 4 consecutive windows >27ms → kill, contaminated, rerun once after both gates re-clear.
- **Divergence**: NaN/inf loss, or any eval <15% after epoch 5 → kill, code error.
- **Wall cap**: >600s → kill, failure.
- **Crash** (rc≠0): code-error fix + resubmit per execute-skill rules (max 2).
- NOTE: moderately depressed HOT-PHASE evals are the predicted lag signature, NOT an abort signal; plateau evals are the health metric.

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** @ 1990397; bar = **96.81**. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition (profile): standard awk over step lines — require ≤2 windows >27ms AND mean ≤24ms AND num_epochs within 139±4. **Quantization note (EXP-037 protocol finding)**: 50-step windows quantize to 6ms rungs {18.0, 24.0, 30.0}; at true dt 22.4 the legal readings are 18.0/24.0, so the 27 threshold sits safely between rungs — any 30.0 reading IS real contention at this dt. Contaminated ⇒ rerun once (gates re-cleared first).
   - Integrity sub-check: `grep "^num_params:" run.log` = 4,286,026; training_seconds = 300.0; eval_lines = num_epochs; on a bar-pass confirm the plateau (last ~15 evals) sits above the baseline band, not a single-eval spike.
   - Timeout: greps instant; missing run.log = infrastructure failure.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: eval_lines ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- **Plateau statistics vs baseline family**: last-15-eval mean, scatter (max−min), and best−median gap — the mechanism predicts REDUCED scatter and a RAISED mean if BN-estimator noise was material; record all three either way (first direct measurement of the BN-noise share of run-level σ).
- **Hot-phase lag signature**: evals at ep 5/10/20/40 vs family (≈64/72/…) — confirms the estimator-lag mechanism operated.
- **dt / epochs / VRAM / params**: must be baseline-identical (22.4ms / 139±4 / 1613MB / 4,286,026) — any deviation flags an implementation error.
- **final_test_loss** vs family ~0.185 (same-loss-family comparison is honest here — training path identical, only eval constants differ).
