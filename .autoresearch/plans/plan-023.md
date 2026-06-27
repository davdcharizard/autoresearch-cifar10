# Plan EXP-023: Heat-constant momentum trade — MOMENTUM 0.95 + PEAK_LR 0.2 (lr/(1−β) = 4 held)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md

## Milestones

### Milestone 1: Code change implemented and sanity-checked
- [ ] On branch `autoresearch/exp-023` (cut from `autoresearch/dev` @ 1990397), edit `train.py`: `MOMENTUM = 0.9` → `0.95`, `PEAK_LR = 0.4` → `0.2`. NOTHING else changes — batch 512, default-mode compile, foreach/nesterov SGD, schedule shape, WD, LS, augmentation all byte-identical.
- [ ] Sanity check: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --stat` shows exactly 1 file / 2 changed lines (plus comment edits on those same lines only).

### Milestone 2: Experiment launched clean and confirmed running
- [ ] GPU-0 zero-compute-apps pre-check passes inside the composite launcher (abort if any PID present).
- [ ] Single background composite command: pre-check → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1 &` → inline 15s watchdog (windowed ms from pct_done deltas; CONTENTION_KILL at 4 consecutive >30ms windows — STANDARD batch-512 thresholds, clean dt is ~22.4ms; STARTUP_KILL if no step lines by tick 10 = 150s) → `wait` → `TRAIN_EXIT rc=$?` → summary grep.
- [ ] Early signature gate (~step 100): windowed dt ≈ 22–23ms. The execution path is BYTE-IDENTICAL to baseline (optimizer constants only), so ANY dt deviation >10% from 22.4ms with GPU-0 free indicates contamination, not the experiment.

### Milestone 3: Run completed with baseline-identical signatures and metrics extracted
- [ ] `TRAIN_EXIT rc=0`; num_epochs within 139±3; dt profile mean ≈ 22.4ms; VRAM ≈ 1613MB; params = 4,286,026. Signature equality to baseline is itself a hypothesis check — the change must be free in epochs/dt by construction.
- [ ] Post-hoc contention profile (authoritative, standard thresholds): windows >30ms ≤ 2 AND epochs within ±3 of (139 × 22.4 / mean_win_ms). If contaminated: discard, rerun once (does not consume code-error retries).
- [ ] Trajectory check: mid-run evals at-or-above the baseline family (hypothesis predicts a smoother hot phase); converged-plateau length/level vs baseline's ~10 converged evals.

## Code Changes
- **train.py** (only file): two constants in the hyperparameter block (L23, L25):
  - `PEAK_LR = 0.4` → `PEAK_LR = 0.2  # heat-constant trade: halved with momentum 0.9->0.95, lr/(1-beta) = 4 held` (replaces the linear-scaling comment).
  - `MOMENTUM = 0.9` → `MOMENTUM = 0.95  # paired with PEAK_LR 0.2 — first-order effective step unchanged`.
  - `lr_at()` scales multiplicatively in PEAK_LR (warmup `PEAK_LR·p/WARMUP_FRAC`, cosine `PEAK_LR·0.5(1+cos)`), so lr(p)/(1−β) = 4·(baseline shape) at EVERY progress point — the heat profile is first-order invariant by construction. `optimizer = optim.SGD(..., momentum=MOMENTUM, nesterov=True)` consumes the new β with zero execution-path change (same foreach kernels — numerics-equivalence preserved).
- Why this tests the hypothesis: the ONLY degree of freedom moved is the gradient-averaging horizon (1/(1−β): 10→20 steps) at fixed effective step — isolating second-order smoothing of the search direction. Signatures (dt/epochs/VRAM/params) must equal baseline, making contamination and premise-failure separately detectable.
- Risks: momentum transients (warmup ramp memory, Nesterov lookahead, BN effective-LR coupling) make the match first-order only — the run may sit slightly off the heat optimum in either direction (EXP-010/014/015: heat misses cost 0.2–0.6pp). Failure is graceful (converged no-improvement) and completes the constant-bracketing certification.
- Failed-approaches justification (required — heat axis count 2, Medium): EXP-010/014 closed UNCOMPENSATED heat increases; that entry's own Insight names "heat-compensated reshapes" as the remaining admissible class. This plan holds integrated first-order heat exactly constant — it is the compensated trade, not a retry of the closed approach. EXP-011 (EMA, smoothing-class) differs mechanistically: EMA smoothed the EVALUATED weights (variance clip on the max-statistic); higher β smooths the SEARCH DIRECTION and can move the mean.

## Configuration Changes
- MOMENTUM: 0.9 → 0.95 (averaging horizon 10→20 steps; Sutskever et al. 2013 effective-step framing)
- PEAK_LR: 0.4 → 0.2 (compensator: 0.4×(1−0.95)/(1−0.9) = 0.2 holds lr/(1−β) = 4)

## Execution Environment
- Method: local, single composite background Bash command (pre-check + launch + inline watchdog + wait + summary in ONE call; cd inside the command). Run from project root on branch `autoresearch/exp-023`.
- Resources: GPU 0 only (`CUDA_VISIBLE_DEVICES=0`); VRAM ~1613MB expected (baseline-identical); NUM_WORKERS=8 unchanged; loader-stall margin untouched (no CPU-side change).
- Estimated runtime: ~480–540s total (300s timed + ~13s warm startup + ~139 evals × ~0.85s + stalls), under the 600s cap.
- Log output: `run.log` in project root via `> run.log 2>&1` (no tee/stream); watchdog prints `WIN step N: Xms` lines; post-hoc awk windowed profile is the authoritative contention gate.
- Tool skill: none (local run).

## Abort Criteria
- **Startup gate**: no `step` lines in run.log by watchdog tick 10 (150s) → kill (warm compile is ~13s, cold ~23s).
- **Contention kill**: 4 consecutive 15s windows >30ms → kill, contaminated, rerun once after GPU-0 frees (standard batch-512 thresholds).
- **Signature break**: windowed dt persistently >25ms or <20ms at ~step 100 with GPU-0 confirmed free → something other than the planned change is in play (the diff cannot move dt); kill and investigate the diff before any retry.
- **Divergence**: any eval test_acc < 15% after epoch 5, or NaN/inf loss in step lines → kill, research failure (would indicate the trade is dynamically unstable despite first-order matching).
- **Wall cap**: total runtime exceeding 600s → kill, failure per goal constraint.
- **Crash**: TRAIN_EXIT rc≠0 → handle per execute-skill failure rules (max 2 code-error retries; contaminated reruns don't count).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** (commit 1990397). All commands from project root after the run completes.

1. **best_test_acc ≥ 96.81** (baseline + 0.1pp absolute):
   - Command: `grep "^best_test_acc:" run.log`
   - Pass: value ≥ 96.81. Empty grep ⇒ crash — read `tail -n 50 run.log`, classify as crash.
   - Pre-condition before judging ANY metric: clean post-hoc profile — `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>30) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>30ms: %d of %d | mean win %.1f ms | expected epochs %.1f\n", c, n, s/n, 139*22.4/(s/n)}'` — require windows>30ms ≤ 2 AND actual epochs within ±3 of expected. Contaminated ⇒ discard and rerun once; never analyze.
2. **Run completes without crashing within budget**:
   - Command: `grep "^total_seconds:" run.log` + TRAIN_EXIT from the launcher.
   - Pass: rc=0 AND total_seconds ≤ 600.
3. **Validation at most once per epoch**:
   - Command: `tr '\r' '\n' < run.log | grep -c "eval ep"` vs `grep "^num_epochs:" run.log`.
   - Pass: eval count ≤ num_epochs.

On first failure: stop, classify (no-improvement if converged metric below bar; crash if rc≠0/no metrics; invalid if integrity broken), proceed to analyze.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1613 (baseline-identical; deviation flags a premise break)
- num_epochs: `grep "^num_epochs:" run.log` — expect 139±3 (epoch-freedom is part of the hypothesis)
- num_params: `grep "^num_params:" run.log` — must equal 4,286,026 (architecture untouched)
