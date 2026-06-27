# Plan EXP-024: Noise-increasing momentum trade — MOMENTUM 0.8 + PEAK_LR 0.8 (lr/(1−β) = 4 held)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md

## Milestones

### Milestone 1: Code change implemented and sanity-checked
- [ ] On branch `autoresearch/exp-024` (cut from `autoresearch/dev` @ 1990397), edit `train.py`: `MOMENTUM = 0.9` → `0.8`, `PEAK_LR = 0.4` → `0.8`. NOTHING else changes — batch 512, default-mode compile, foreach/nesterov SGD, schedule shape, WD, LS, augmentation byte-identical.
- [ ] Sanity check: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --stat` shows exactly 1 file / 2 changed lines.

### Milestone 2: Experiment launched clean and confirmed running
- [ ] GPU-0 zero-compute-apps pre-check passes inside the composite launcher (abort if any PID present).
- [ ] Single background composite command: pre-check → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1 &` → inline 15s watchdog (windowed ms from pct_done deltas; CONTENTION_KILL at 4 consecutive >30ms — standard batch-512 thresholds, clean dt ~22.4ms; STARTUP_KILL if no steps by tick 10 = 150s) → `wait` → `TRAIN_EXIT rc=$?` → summary grep.
- [ ] Early signature gate (~step 100): windowed dt ≈ 22–23ms. Execution path is byte-identical to baseline; any dt deviation >10% with GPU-0 free is contamination, not the experiment.

### Milestone 3: Run completed with baseline-identical signatures and metrics extracted
- [ ] `TRAIN_EXIT rc=0`; num_epochs 139±3; dt mean ≈ 22.4ms; VRAM ≈ 1613MB; params = 4,286,026.
- [ ] Post-hoc contention profile (authoritative, standard thresholds): windows >30ms ≤ 2 AND epochs within ±3 of (139 × 22.4 / mean_win_ms). If contaminated: discard, rerun once.
- [ ] Trajectory + plateau-integrity check: a bouncy hot phase is acceptable (EXP-012 precedent); the tail must converge. Record the final-7-evals spread and median — a win must come from plateau LEVEL (median ≥ baseline family's ~96.6), not from variance inflation of the max.

## Code Changes
- **train.py** (only file): two constants in the hyperparameter block (L23, L25):
  - `PEAK_LR = 0.8  # noise-increasing trade: doubled with momentum 0.9->0.8, lr/(1-beta) = 4 held`
  - `MOMENTUM = 0.8  # paired with PEAK_LR 0.8 — first-order effective step unchanged, averaging horizon halved`
  - `lr_at()` is multiplicative in PEAK_LR, so lr(p)/(1−β) equals baseline's at every progress point; the optimizer consumes β as a scalar — same foreach kernels, numerics-equivalent.
- Why this tests the hypothesis: the EXACT mirror of EXP-023 — the only moved variable is the averaging horizon (10→5 steps), raising effective gradient noise at held first-order step. EXP-023 validated that this design produces baseline-identical signatures, so the metric delta is attributable to the noise variable alone. This measures the unmeasured INCREASE side of the gradient-noise law (goal-learnings § Patterns).
- Risks: per-step magnitude is 2× baseline at the peak — bouncier hot phase expected (EXP-012 precedent: bouncy but convergent at lr 0.8); divergence unlikely (effective step unchanged, bf16 wide-range) and caught by abort criteria. Failure graceful: a converged miss brackets the noise curve and certifies baseline as the noise optimum.
- Failed-approaches justification (heat axis count 2, Medium): EXP-010/014 closed UNCOMPENSATED heat increases — peak 0.6 at β 0.9 raised lr/(1−β) to 6. Here peak 0.8 at β 0.8 HOLDS lr/(1−β) = 4; integrated first-order heat is unchanged, the admissible compensated-trade class. EXP-023 (same class, opposite direction) confirmed the class executes as designed.

## Configuration Changes
- MOMENTUM: 0.9 → 0.8 (averaging horizon 10→5 steps — raises effective gradient noise; the one direction the noise law leaves open)
- PEAK_LR: 0.4 → 0.8 (compensator: 0.4×(1−0.8)/(1−0.9) = 0.8 holds lr/(1−β) = 4)

## Execution Environment
- Method: local, single composite background Bash command (pre-check + launch + inline watchdog + wait + summary; cd inside the command). Project root, branch `autoresearch/exp-024`.
- Resources: GPU 0 only (`CUDA_VISIBLE_DEVICES=0`); VRAM ~1613MB expected; NUM_WORKERS=8 unchanged.
- Estimated runtime: ~480–540s total, under the 600s cap.
- Log output: `run.log` via `> run.log 2>&1` (no tee/stream); watchdog WIN lines for live windowed dt; post-hoc awk profile authoritative.
- Tool skill: none (local run).

## Abort Criteria
- **Startup gate**: no `step` lines by watchdog tick 10 (150s) → kill.
- **Contention kill**: 4 consecutive 15s windows >30ms → kill, contaminated, rerun once after GPU-0 frees.
- **Signature break**: windowed dt persistently >25ms or <20ms at ~step 100 with GPU-0 free → kill and inspect the diff (the change cannot move dt).
- **Divergence**: NaN/inf loss in step lines, or smoothed train loss climbing monotonically for 3+ consecutive watchdog checks during the post-warmup hot phase, or any eval test_acc < 15% after epoch 5 → kill, research failure (per-step instability at lr 0.8 — the brainstorm's named tail-risk).
- **Wall cap**: total runtime exceeding 600s → kill, failure per goal constraint.
- **Crash**: TRAIN_EXIT rc≠0 → handle per execute-skill failure rules (max 2 code-error retries; contaminated reruns don't count).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** (commit 1990397). All commands from project root after the run completes.

1. **best_test_acc ≥ 96.81** (baseline + 0.1pp absolute):
   - Command: `grep "^best_test_acc:" run.log`
   - Pass: value ≥ 96.81. Empty grep ⇒ crash — read `tail -n 50 run.log`, classify as crash.
   - Pre-condition before judging ANY metric: clean post-hoc profile — `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>30) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>30ms: %d of %d | mean win %.1f ms | expected epochs %.1f\n", c, n, s/n, 139*22.4/(s/n)}'` — require windows>30ms ≤ 2 AND epochs within ±3 of expected. Contaminated ⇒ discard, rerun once; never analyze.
   - Integrity sub-check (reward-hacking guard from brainstorm-024): if the value passes the bar, additionally compute the final-7-evals median and spread (`tr '\r' '\n' < run.log | grep "eval ep" | tail -7`). A pass driven by an outlier spike over a flat-or-lower plateau (median < ~96.6, spread ≫ baseline's ~±0.1pp) is variance harvesting, not a genuine improvement — flag for the analyze phase to weigh toward no-improvement in substance.
2. **Run completes without crashing within budget**:
   - Command: `grep "^total_seconds:" run.log` + TRAIN_EXIT from the launcher.
   - Pass: rc=0 AND total_seconds ≤ 600.
3. **Validation at most once per epoch**:
   - Command: `tr '\r' '\n' < run.log | grep -c "eval ep"` vs `grep "^num_epochs:" run.log`.
   - Pass: eval count ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1613 (baseline-identical)
- num_epochs: `grep "^num_epochs:" run.log` — expect 139±3
- num_params: `grep "^num_params:" run.log` — must equal 4,286,026
