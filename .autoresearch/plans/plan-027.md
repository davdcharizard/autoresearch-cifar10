# Plan EXP-027: Baseline variance replicate ×2 — zero-diff measurement of run-level σ
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md

## PRE-REGISTRATION (binding)
This experiment makes **no code change** (zero diff; `git diff` on the experiment branch must be empty at all times). The verdict is **pre-registered as `no-improvement` regardless of measured values** — including if a replicate lands ≥ 96.81. Rationale: identical code cannot improve on itself; adopting a lucky draw would be variance harvesting, barred by the max-statistic law (goal-learnings § Patterns) and the no-seed-hacking constraint. The TSV metric recorded will be the MEAN of the two replicate best_test_acc values. The baseline (96.71 @ 1990397) does not move.

## Milestones

### Milestone 1: Branch confirmed at baseline, zero diff
- [ ] On branch `autoresearch/exp-027` (cut from `autoresearch/dev` @ 1990397): `git diff --stat` is EMPTY; `git rev-parse --short=7 HEAD` = 1990397; `grep -c "F.relu" train.py` = 3 (baseline activations intact).

### Milestone 2: Replicate 1 launched clean and completed
- [ ] GPU-0 zero-compute-apps pre-check passes inside the composite launcher.
- [ ] Standard composite (pre-check → `rm -f run.log` → launch → 15s watchdog: contention kill 4×>30ms, startup gate tick 10, NaN/inf guard — NO early-dt gate needed; baseline dt is certified) → `TRAIN_EXIT rc=0`.
- [ ] Post-hoc profile clean (windows >30ms ≤ 2, epochs within ±3 of 139 × 22.4 / mean_win_ms); record best_test_acc as R1. Save the eval tail (final 7) and summary block to the exp-log BEFORE the next run overwrites run.log.

### Milestone 3: Replicate 2 launched clean and completed
- [ ] Same procedure; record best_test_acc as R2. Both replicates' summaries and final-7 evals recorded in the exp-log.
- [ ] σ analysis: with draws {96.71 (standing), R1, R2}, compute spread and sample σ; evaluate the brainstorm's pre-registered sub-predictions: (a) R1, R2 < 96.81; (b) |R1−R2| ≤ 0.2; (c) min(R1,R2) < 96.71.

## Code Changes
- **NONE.** train.py is byte-identical to 1990397. Any accidental modification voids the experiment (re-checkout and restart).
- Why this tests the hypothesis: the only varying factor across the three draws is nondeterminism (cudnn.benchmark autotuning, bf16 atomics reduction orders) — exactly the quantity being measured.

## Configuration Changes
- None.

## Execution Environment
- Method: local, TWO sequential composite background Bash commands (one per replicate; the second launches only after the first completes and its results are recorded). Project root, branch `autoresearch/exp-027`.
- Resources: GPU 0 only (`CUDA_VISIBLE_DEVICES=0`); VRAM ~1613MB; 8 loader workers.
- Estimated runtime: ~480–540s per replicate, ~17–19 min total including recording between runs. Each run individually respects the 600s cap.
- Log output: `run.log` per run (overwritten between replicates — summaries extracted and persisted to the exp-log in between); watchdog WIN lines; post-hoc awk profile authoritative per run.
- Tool skill: none (local runs).

## Abort Criteria
- **Startup gate**: no `step` lines by watchdog tick 10 (150s) → kill.
- **Contention kill**: 4 consecutive 15s windows >30ms → kill; that replicate is contaminated → rerun once (does not count against the two-replicate design).
- **Divergence**: NaN/inf loss, or any eval test_acc < 15% after epoch 5 → kill (would indicate an environment fault, not research signal — investigate).
- **Wall cap**: total runtime exceeding 600s on either replicate → that replicate fails the goal constraint; record and note in analysis.
- **Crash**: TRAIN_EXIT rc≠0 → infra failure handling per execute skill (max 2 retries per replicate).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** (commit 1990397). PRE-REGISTRATION OVERRIDE: even if condition 1's numeric check passes on a replicate, the integrity sub-check FAILS by construction (no intervention — a zero-diff run cannot satisfy "gain through the intended intervention class"), so the experiment verdict is `no-improvement` in every branch. Verification is still run faithfully to document the outcome.

1. **best_test_acc ≥ 96.81** (evaluated on EACH replicate):
   - Command (per run, before overwriting): `grep "^best_test_acc:" run.log`
   - Numeric pass/fail recorded per replicate. Integrity sub-check: requires a code-change mechanism — zero-diff ⇒ FAILS unconditionally ⇒ verdict `no-improvement` regardless of numerics (pre-registered).
   - Pre-condition per replicate: clean post-hoc profile — `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>30) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>30ms: %d of %d | mean win %.1f ms | expected epochs %.1f\n", c, n, s/n, 139*22.4/(s/n)}'` — require ≤2 slow windows AND epochs within ±3 of expected; contaminated ⇒ rerun that replicate once.
2. **Run completes without crashing within budget** (per replicate): rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation at most once per epoch** (per replicate): `tr '\r' '\n' < run.log | grep -c "eval ep"` ≤ `grep "^num_epochs:" run.log` value.

### Informational Metrics (Optional)
- R1, R2 best_test_acc; spread |R1−R2|; sample σ of {96.71, R1, R2}; final-7-evals median per replicate (plateau level vs the max — how much the max-statistic harvests above the plateau)
- Standard signatures per replicate: num_epochs (expect 139±3), VRAM (~1613), params (4,286,026), startup (~13s warm)
