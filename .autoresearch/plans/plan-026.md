# Plan EXP-026: Activation modernization — ReLU → GELU at all three activation sites
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md

## Milestones

### Milestone 1: Code change implemented and sanity-checked
- [ ] On branch `autoresearch/exp-026` (cut from `autoresearch/dev` @ 1990397), edit `train.py` — exactly three one-token substitutions, `F.relu` → `F.gelu`:
  1. `BasicBlock.forward` post-bn1: `out = F.gelu(self.bn1(self.conv1(x)))` (currently train.py:64)
  2. `BasicBlock.forward` post-residual-add: `return F.gelu(out)` (currently train.py:71)
  3. `ResNet.forward` stem: `out = F.gelu(self.bn1(self.conv1(x)))` (currently train.py:103)
  NOTHING else changes — init (Kaiming √2 gain, airbench-consistent), schedule, optimizer, loaders, compile path, eval all untouched.
- [ ] Sanity check: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff` shows exactly 3 changed lines, all `relu→gelu`; `grep -c "F.relu" train.py` returns 0 and `grep -c "F.gelu" train.py` returns 3.

### Milestone 2: Experiment launched clean and confirmed running
- [ ] GPU-0 zero-compute-apps pre-check passes inside the composite launcher.
- [ ] Single background composite command: pre-check → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1 &` → inline 15s watchdog (standard batch-512 thresholds: kill at 4 consecutive >30ms windows; STARTUP_KILL at tick 10; NaN/inf divergence guard) → `wait` → `TRAIN_EXIT rc=$?` → summary grep.
- [ ] **Early dt gate (~step 100, EXP-008 protocol)**: windowed dt must be ≤ 23.0ms (GELU is pointwise-fused; tolerance +0.6ms over baseline 22.4). Projected epochs = 139 × 22.4 / win_ms must be ≥ 135. If dt > 23.5ms sustained over 3 windows → kill (the fused-pointwise premise failed; the experiment is not worth the budget at <133 epochs).

### Milestone 3: Run completed with clean signatures
- [ ] `TRAIN_EXIT rc=0`; num_epochs 139±3 (or matching the measured win_ms projection); VRAM ≈ 1613MB (GELU adds no state; ±30MB tolerance); params = 4,286,026 (activation has no parameters — MUST be exact).
- [ ] Post-hoc contention profile (authoritative): windows >30ms ≤ 2 AND epochs within ±3 of (139 × 22.4 / mean_win_ms).
- [ ] Trajectory check: ep1–5 evals at or above the baseline family (ep1 ~38–39%) — confirms no-deferral premise; a depressed early trail with later catch-up would contradict the mechanism and inform analysis.

## Code Changes
- **train.py** (only file): 3 one-token edits, `F.relu` → `F.gelu` (default erf-exact form; do NOT pass `approximate='tanh'` — keep the canonical op that inductor fuses).
- Why this tests the hypothesis: the nonlinearity is the only recipe component never probed; both regime-matched budget-race lineages (airbench GELU, cifar10-fast CELU) use smooth activations. GELU eliminates dead units and is smooth everywhere → better per-step optimization from step 0 → converged plateau forms higher. The change is active the entire run (no deferral), pointwise-fused (no epoch cost), parameter-free (params unchanged), and noise-neutral.
- Risks/edge cases: (a) dt regression if fusion is worse than expected — caught at the early gate, kill at >23.5ms; (b) interaction with the LR/noise optimum certified under ReLU — graceful converged miss; (c) torch.compile builds a slightly different graph — same input signature, startup comparable; warmup loop unchanged covers compilation; (d) no BN-ordering trap (activation follows BN, same as both references).

## Configuration Changes
- None. All hyperparameter constants identical to baseline. The activation is an architecture (code) change, not a constant change.

## Execution Environment
- Method: local, single composite background Bash command (pre-check + launch + inline watchdog + wait + summary). Project root, branch `autoresearch/exp-026`.
- Resources: GPU 0 only (`CUDA_VISIBLE_DEVICES=0`); VRAM ~1613MB; 8 loader workers.
- Estimated runtime: ~480–540s total (baseline-identical signatures expected), under the 600s cap.
- Log output: `run.log` via `> run.log 2>&1`; watchdog WIN lines; post-hoc awk windowed profile authoritative.
- Tool skill: none (local run).

## Abort Criteria
- **Startup gate**: no `step` lines by watchdog tick 10 (150s) → kill.
- **Early dt gate**: windowed dt > 23.5ms sustained across 3 consecutive windows at steps ≤ 600 → kill (projected epochs < ~133; fused-pointwise premise failed). 22.4–23.0ms = proceed.
- **Contention kill**: 4 consecutive 15s windows >30ms → kill, contaminated, rerun once.
- **Divergence**: NaN/inf loss, or any eval test_acc < 15% after epoch 5 → kill, research failure.
- **Wall cap**: total runtime exceeding 600s → kill, failure per goal constraint.
- **Crash**: TRAIN_EXIT rc≠0 → handle per execute-skill failure rules (max 2 code-error retries; contaminated reruns don't count).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** (commit 1990397). All commands from project root after the run completes. Timeout: all greps are instant; treat a missing run.log as a crash.

1. **best_test_acc ≥ 96.81** (baseline + 0.1pp absolute):
   - Command: `grep "^best_test_acc:" run.log`
   - Pass: value ≥ 96.81. Empty grep ⇒ crash — `tail -n 50 run.log`, classify as crash.
   - Pre-condition: clean post-hoc profile — `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>30) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>30ms: %d of %d | mean win %.1f ms | expected epochs %.1f\n", c, n, s/n, 139*22.4/(s/n)}'` — require windows>30ms ≤ 2 AND epochs within ±3 of expected. Contaminated ⇒ discard, rerun once.
   - Integrity sub-check: on a bar-pass, compute final-7-evals median (`tr '\r' '\n' < run.log | grep "eval ep" | tail -7`) — require median ≥ 96.6 (win via plateau level, not an outlier spike); confirm `grep "^num_params:" run.log` = 4,286,026 exactly (parameter-free change); confirm the gain mechanism is trajectory-wide (early evals at/above family, plateau shifted up) not a tail anomaly.
2. **Run completes without crashing within budget**:
   - Command: `grep "^total_seconds:" run.log` + TRAIN_EXIT. Pass: rc=0 AND total_seconds ≤ 600.
3. **Validation at most once per epoch**:
   - Command: `tr '\r' '\n' < run.log | grep -c "eval ep"` vs `grep "^num_epochs:" run.log`. Pass: eval count ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1613 (±30)
- num_epochs: `grep "^num_epochs:" run.log` — expect 139±3 (or win_ms projection)
- num_params: `grep "^num_params:" run.log` — must equal 4,286,026
- Early trail (analysis-only): eval lines ep1–5 vs baseline family (ep1 ~38–39%) — no-deferral confirmation
