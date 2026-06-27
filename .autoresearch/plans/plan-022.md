# Plan EXP-022: Batch 1024 with √-scaled peak LR (0.566) — fix EXP-012's diagnosed LR-rule defect
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md

## Milestones

### Milestone 1: Code change implemented and sanity-checked
- [ ] On branch `autoresearch/exp-022` (cut from `autoresearch/dev` @ 1990397), edit `train.py`: `BATCH_SIZE = 512` → `1024`, `PEAK_LR = 0.4` → `0.566`. NOTHING else changes — compile stays `torch.compile(model)` (DEFAULT mode), SGD stays foreach/nesterov, schedule shape/warmup/augmentation/WD/LS byte-identical.
- [ ] Sanity check: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --stat` shows exactly 1 file / 2 changed lines.

### Milestone 2: Experiment launched clean and confirmed running
- [ ] GPU-0 zero-compute-apps pre-check passes inside the composite launcher (abort if any PID present).
- [ ] Single background composite command: pre-check → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1 &` → inline 15s watchdog (windowed ms from pct_done deltas; CONTENTION_KILL at 4 consecutive >60ms windows — threshold doubled because clean windowed dt at batch 1024 is ~41ms, see Abort Criteria; STARTUP_KILL if no step lines by tick 10 = 150s) → `wait` → `TRAIN_EXIT rc=$?` → summary grep.
- [ ] Early signature gate (~step 100, from run.log): windowed dt ≈ 41–43 ms (EXP-012 measured family). If dt projects < 70 epochs (>61 ms), kill per goal-learnings dt gate.

### Milestone 3: Run completed with clean signatures and metrics extracted
- [ ] `TRAIN_EXIT rc=0`; `grep -c "eval ep" run.log` ≈ 148–153 (EXP-012 measured 151 at 1024); `best_test_acc` line present.
- [ ] Post-hoc contention profile (authoritative, threshold scaled to batch-1024 dt): windows >60ms ≤ 2 AND epochs within ±3 of (151 × 41.0 / mean_win_ms). If contaminated: discard, rerun once (does not consume code-error retries).
- [ ] Trajectory check against hypothesis: mid-run evals within ~1pp of the baseline family (EXP-012's linear-0.8 run showed an ~18pp mid deficit — its absence/presence is the mechanism readout).

## Code Changes
- **train.py** (only file): two constants in the hyperparameter block (L22–23):
  - `BATCH_SIZE = 512` → `BATCH_SIZE = 1024` — re-applies EXP-012's measured throughput gain (+8% img/s, 151 vs 139 epochs) within the SAME default-compile execution regime (numerics-equivalence requirement, EXP-021).
  - `PEAK_LR = 0.4` → `PEAK_LR = 0.566  # sqrt scaling: 0.4 x sqrt(1024/512)` — replaces the linear rule (0.8) that EXP-012 diagnosed as the trajectory-damaging defect with the √-scaling rule (Hoffer et al. 1705.08741) that preserves gradient-noise scale.
  - The comment on L23 (`# linear scaling: 0.1 x (512/128)`) is replaced by the sqrt-scaling comment so the constant's provenance stays accurate.
  - Note: the warmup tensor `warm_x`/`warm_y` (L185–188) is sized by `BATCH_SIZE`, so compile warmup automatically matches the new batch shape — no further edits. `lr_at()` reads `PEAK_LR` — no schedule code changes.
- Why this tests the hypothesis: both halves of EXP-012's result are reproduced except the one diagnosed defect; if √-scaling holds the trajectory, the +12 epochs convert (~+0.24pp by EXP-006 arithmetic) and the metric clears baseline +0.1.
- Risks: per-example heat 0.566/1024 = 0.71× baseline's 0.4/512 — possible mid-cold result (EXP-015 cold-side reading); failure mode is graceful (converged no-improvement), and a miss closes the batch axis permanently (cold/middle/hot points all measured at 1024).

## Configuration Changes
- BATCH_SIZE: 512 → 1024 (throughput measured clean at this value in EXP-012: dt ~41ms windowed, 151 epochs, VRAM ~2.6GB — well within H20)
- PEAK_LR: 0.4 → 0.566 (√-scaling 0.4×√2; canonical alternative to the linear rule whose hot bouncy phase EXP-012 isolated as the −0.05 deficit's cause; sits between untried-cold and measured-hot 0.8 on the 1024-LR axis)

## Execution Environment
- Method: local, single composite background Bash command (pre-check + launch + inline watchdog + wait + summary in ONE call; Bash cwd persists — cd inside the command). Run from project root on branch `autoresearch/exp-022`.
- Resources: GPU 0 only (`CUDA_VISIBLE_DEVICES=0`); expected VRAM ~2.6GB (EXP-012); CPU loaders NUM_WORKERS=8 unchanged — per-image CPU work unchanged, so loader-stall margin (infra-errors EXP-013) is untouched.
- Estimated runtime: ~480–500s total (300s timed budget + ~13s warm startup + ~150 evals × ~0.85s + loader stalls), well under the 600s cap. Watchdog adds 15s granularity, max wait ~10min before STARTUP_KILL/CONTENTION_KILL resolves.
- Log output: `run.log` in project root via `> run.log 2>&1` (no tee/stream per goal procedure); watchdog prints `WIN step N: Xms` lines to the Bash result for live windowed dt; post-hoc awk profile over step lines is the authoritative contention gate.
- Tool skill: none (local run).

## Abort Criteria
- **Startup gate**: no `step` lines in run.log by watchdog tick 10 (150s) → kill (covers hung compile; cold compile is ~23s, warm ~13s).
- **Contention kill**: 4 consecutive 15s windows with windowed ms > 60 → kill, treat as contaminated, rerun once after GPU-0 frees. (Threshold is 60ms, NOT the baseline 30ms: clean windowed dt at batch 1024 is ~41ms — EXP-012; 30ms would self-kill instantly. 60ms ≈ 1.5× clean, same ratio as 30/22.4.)
- **Early dt gate**: at ~step 100, projected epochs = 151 × 41.0 / measured_win_ms < 70 (i.e., win_ms > ~88) with GPU-0 confirmed free → kill, research failure (not contention).
- **Divergence**: any eval line with test_acc < 15% after epoch 5, or loss NaN/inf in step lines → kill, research failure (hot-phase instability would indicate √-scaling still too hot).
- **Wall cap**: total runtime exceeding 600s (10min) → kill, failure per goal constraint.
- **Crash**: TRAIN_EXIT rc≠0 → handle per execute-skill failure rules (max 2 code-error retries; contaminated reruns don't count).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** (commit 1990397). All commands from project root after the run completes.

1. **best_test_acc ≥ 96.81** (baseline + 0.1pp absolute):
   - Command: `grep "^best_test_acc:" run.log`
   - Pass: value ≥ 96.81. Empty grep ⇒ crash — read `tail -n 50 run.log`, classify as crash.
   - Timeout: n/a (file read). Pre-condition before judging ANY metric: the post-hoc contention profile must be clean — `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>60) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>60ms: %d of %d | mean win %.1f ms | expected epochs %.1f\n", c, n, s/n, 151*41.0/(s/n)}'` — require windows>60ms ≤ 2 AND actual epochs within ±3 of expected. Contaminated ⇒ discard and rerun once; never analyze.
2. **Run completes without crashing within budget**:
   - Command: `grep "^total_seconds:" run.log` + TRAIN_EXIT from the launcher.
   - Pass: rc=0 AND total_seconds ≤ 600.
3. **Validation at most once per epoch**:
   - Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log`.
   - Pass: eval count ≤ num_epochs (loop structure unchanged guarantees this; confirm anyway).

On first failure: stop, classify (no-improvement if converged metric below bar; crash if rc≠0/no metrics; invalid if integrity broken), proceed to analyze.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~2600 (EXP-012 family; soft constraint, H20 has 98GB)
- num_epochs: `grep "^num_epochs:" run.log` — expect 148–153; this IS the throughput-delivery readout
- num_params: `grep "^num_params:" run.log` — must equal 4,286,026 (architecture untouched)
