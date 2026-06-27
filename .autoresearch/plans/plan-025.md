# Plan EXP-025: Final-phase clean-data alignment — aug-off tail at progress ≥ 0.85 (FixRes-style)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md

## Milestones

### Milestone 1: Code change implemented and sanity-checked
- [ ] On branch `autoresearch/exp-025` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. New constant in the hyperparameter block: `ALIGN_FRAC = 0.85  # final fraction of budget trained on clean (test-distribution) data`.
  2. After `train_loader` construction: build `clean_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])` (EXACTLY the Eval transform from prepare.py L15–20), `clean_set = datasets.CIFAR10(DATASET_DIR, train=True, download=True, transform=clean_tf)`, and `clean_loader = DataLoader(clean_set, <identical kwargs to train_loader>)`.
  3. In the `while` loop, replace `for inputs, targets in train_loader:` with an epoch-level source switch: `loader = clean_loader if total_training_time >= ALIGN_FRAC * TIME_BUDGET_S else train_loader` then `for inputs, targets in loader:`.
  NOTHING else changes — schedule, optimizer, compile, batch, eval untouched. The timed step body is byte-identical.
- [ ] Sanity check: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --stat` shows 1 file, ~+12/−1 lines; grep confirms `for inputs, targets in loader:` and both loaders share batch_size/num_workers/pin_memory/drop_last/persistent_workers.

### Milestone 2: Experiment launched clean and confirmed running
- [ ] GPU-0 zero-compute-apps pre-check passes inside the composite launcher.
- [ ] Single background composite command: pre-check → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1 &` → inline 15s watchdog (standard batch-512 thresholds: kill at 4 consecutive >30ms windows; STARTUP_KILL at tick 10) → `wait` → `TRAIN_EXIT rc=$?` → summary grep.
- [ ] Early signature gate (~step 100): windowed dt ≈ 22–23ms (GPU step work unchanged pre-switch by construction).

### Milestone 3: Run completed with clean signatures and the switch behavior verified
- [ ] `TRAIN_EXIT rc=0`; num_epochs 139±3 (clean-tail steps are GPU-identical; the one-time clean-loader spin-up lands OUTSIDE dt per the t0-after-yield accounting — but total_seconds must stay ≤600); VRAM ≈ 1613MB; params = 4,286,026.
- [ ] Post-hoc contention profile (authoritative, standard thresholds): windows >30ms ≤ 2 AND epochs within ±3 of (139 × 22.4 / mean_win_ms). Caveat: the clean-loader's first-fetch stall could surface as ONE slow window in the profile around the switch (~85% progress, ~step 11400) — a single window >30ms exactly there is the expected switch artifact, not contention; >2 windows or windows elsewhere = contamination, discard and rerun once.
- [ ] Switch observable in the trajectory: eval lines after ~ep 120 should show the alignment effect (hypothesis: dip ≤1pp for 1–2 epochs, then plateau above baseline family ~96.6+). Train loss will DROP sharply at the switch (clean data is easier) — expected, not divergence.

## Code Changes
- **train.py** (only file):
  - `ALIGN_FRAC = 0.85` constant — switch point chosen so the clean phase (~20 epochs) is long enough for BN running stats (momentum 0.1, 98 steps/epoch — re-converge within ~1 epoch) plus weight fine-tuning at cosine-tail lr (lr ≤ ~0.022 at p ≥ 0.85), while leaving 85% of the budget on the full augmentation recipe whose pressure dose is certified optimal.
  - Second dataset/loader on the SAME train split with the EXACT Eval transform (ToTensor + Normalize((0.4914,0.4822,0.4465),(1,1,1)) — verified against prepare.py L13–20). Identical DataLoader kwargs (batch 512, shuffle, 8 workers, pin_memory, drop_last, persistent_workers).
  - Epoch-level source switch keyed to `total_training_time` (the timed budget), not epoch count — consistent with the time-keyed schedule philosophy (anneal and alignment phase both complete under any throughput).
- Why this tests the hypothesis: the model currently NEVER trains on the distribution the frozen evaluator measures; the final ~15% of budget aligns weights + BN running stats to it. GPU step work, kernel set, compiled graph (input signature unchanged), schedule, optimizer are all byte-identical — the only change is WHICH images the tail sees.
- Risks/edge cases: (a) clean_set construction re-reads the CIFAR archive from `data/` (cached — no download, startup +~1s); (b) +8 worker processes spawn at the switch (host RAM fine; spin-up stall outside dt but inside the 600s wall — headroom ~70–110s); (c) overfitting in the tail is bounded by lr ≤ 0.022 + WD + LS still active; (d) if the dip does not recover (plateau below family), the mechanism is refuted cleanly — graceful no-improvement; (e) torch.compile does NOT recompile (same input shapes/dtypes/layout).
- Failed-approaches screen: pressure-axis closure (constant-dose changes, EXP-003/004/009/013/015) does not cover time-varying doses — explicitly a different approach to a probed idea (allowed); EXP-013's loader-cost trap is inverted here (transforms REMOVED on the active path). Four-laws screen documented in brainstorm-025 § Experimental History Review.

## Configuration Changes
- ALIGN_FRAC: (new) 0.85 — FixRes-style short alignment phase; single-knob follow-up at 0.92 exists if the dip recovery is too slow or overfit appears.

## Execution Environment
- Method: local, single composite background Bash command (pre-check + launch + inline watchdog + wait + summary). Project root, branch `autoresearch/exp-025`.
- Resources: GPU 0 only (`CUDA_VISIBLE_DEVICES=0`); VRAM ~1613MB; up to 16 persistent loader workers after the switch (host margin fine — EXP-013's bound was per-image transform COST, which drops here).
- Estimated runtime: ~485–545s total (baseline ~484–534 + ~1s extra dataset init + a few seconds of one-time clean-loader spin-up outside dt), under the 600s cap.
- Log output: `run.log` via `> run.log 2>&1`; watchdog WIN lines; post-hoc awk profile authoritative (with the documented single-switch-window caveat).
- Tool skill: none (local run).

## Abort Criteria
- **Startup gate**: no `step` lines by watchdog tick 10 (150s) → kill.
- **Contention kill**: 4 consecutive 15s windows >30ms → kill, contaminated, rerun once. (The switch artifact is at most ONE slow window — it cannot trip a 4-consecutive kill.)
- **Divergence**: NaN/inf loss, or any eval test_acc < 15% after epoch 5 → kill, research failure. NOTE: a sharp train-loss DROP at ~85% progress is the expected switch signature; a test-acc dip ≤1pp for ≤2 epochs post-switch is the predicted transient — neither is an abort signal.
- **Post-switch collapse**: if test_acc falls >3pp below the pre-switch level and is still falling after 3 post-switch evals → kill, research failure (alignment destabilized the tail).
- **Wall cap**: total runtime exceeding 600s → kill, failure per goal constraint.
- **Crash**: TRAIN_EXIT rc≠0 → handle per execute-skill failure rules (max 2 code-error retries; contaminated reruns don't count).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** (commit 1990397). All commands from project root after the run completes.

1. **best_test_acc ≥ 96.81** (baseline + 0.1pp absolute):
   - Command: `grep "^best_test_acc:" run.log`
   - Pass: value ≥ 96.81. Empty grep ⇒ crash — `tail -n 50 run.log`, classify as crash.
   - Pre-condition: clean post-hoc profile — `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>30) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>30ms: %d of %d | mean win %.1f ms | expected epochs %.1f\n", c, n, s/n, 139*22.4/(s/n)}'` — require windows>30ms ≤ 2 (one may be the documented switch artifact at ~85% progress) AND epochs within ±3 of expected. Contaminated ⇒ discard, rerun once.
   - Integrity sub-check: on a bar-pass, compute final-7-evals median and spread (`tr '\r' '\n' < run.log | grep "eval ep" | tail -7`) — require median ≥ 96.6 (win via plateau level, not an outlier spike). Also confirm `grep "^num_params:" run.log` = 4,286,026 and that the gain mechanism is the planned one (post-switch evals visibly above pre-switch family level).
2. **Run completes without crashing within budget**:
   - Command: `grep "^total_seconds:" run.log` + TRAIN_EXIT. Pass: rc=0 AND total_seconds ≤ 600.
3. **Validation at most once per epoch**:
   - Command: `tr '\r' '\n' < run.log | grep -c "eval ep"` vs `grep "^num_epochs:" run.log`. Pass: eval count ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1613
- num_epochs: `grep "^num_epochs:" run.log` — expect 139±3
- num_params: `grep "^num_params:" run.log` — must equal 4,286,026
- Post-switch trajectory (analysis-only): eval lines from the switch epoch onward — dip depth/duration and plateau level vs pre-switch family
