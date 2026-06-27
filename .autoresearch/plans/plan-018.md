# Plan EXP-018: Zero-init residual — γ=0 in each BasicBlock's final BN (bn2)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md

## Milestones

### Milestone 1: Code change implemented and syntax-checked
- [x] Edit `train.py` `ResNet.__init__`: after `self.apply(self._weights_init)`, add a loop zeroing every BasicBlock's `bn2.weight` (`init.zeros_(m.bn2.weight)` for `m` in `self.modules()` if `isinstance(m, BasicBlock)`) with a one-line comment citing the zero-γ trick
- [x] Syntax check passes: `uv run python -c "import ast; ast.parse(open('train.py').read())"` (do NOT import train.py — module level instantiates Eval())
- [x] Diff review: ~3 added lines, NOTHING else changed — all training constants, architecture ([3,3,3], 4x), schedule, aug, compile path byte-identical to baseline @ 1990397

### Milestone 2: Run launched cleanly
- [x] Pre-launch: GPU 0 has zero compute apps (composite script aborts otherwise)
- [x] Launch via the standard composite background script (pre-check + `uv run train.py > run.log 2>&1` + inline contention watchdog: windowed pct_done-delta dt every 15s, auto-kill on 4 consecutive >30ms windows). Do NOT arm a separate-turn Monitor for early signals (infra-errors: it first-polls after the run ends; the watchdog and post-hoc checks cover everything)
- [x] `run.log` params line reads exactly `params: 4,286,026` (init change adds zero params); any deviation → kill and fix

### Milestone 3: Early signals (post-hoc acceptable given Monitor limitation)
- [x] Throughput signature unchanged: windowed dt ≈ 22.4ms, projecting ~139 epochs (the diff cannot change the compiled graph, so any dt shift indicates contention, not the experiment)
- [x] Record the early eval trail (eps 1–20) in the exp-log for the hypothesis's faster-onset claim — informational, not a gate (nearest same-throughput comparator: EXP-017's trail 63.76@5 / 75.06@8)

### Milestone 4: Completion and verification
- [x] TRAIN_EXIT rc=0, summary block present in run.log
- [x] Contention sanity (pre-condition): num_epochs ≈ 139 ± 10% AND post-hoc windowed profile ~0 windows >30ms
- [x] Verification protocol executed in order (below), first-failure stop

## Code Changes
- **train.py** (only file; one ~3-line addition in `ResNet.__init__`):
  ```python
  self.apply(self._weights_init)
  # Zero-init residual: gamma=0 in each block's last BN makes every block an
  # identity map at init (Bag of Tricks, arXiv 1812.01187; Goyal et al. 1706.02677)
  for m in self.modules():
      if isinstance(m, BasicBlock):
          init.zeros_(m.bn2.weight)
  ```
  Tests whether identity-at-init eases the early high-LR phase (peak 0.4 one-cycle, batch 512 — the literature's exact regime), pulling convergence onset earlier and lengthening the converged plateau the max-over-evals metric harvests. Risks/edge cases: `bn2.weight` is in the no-decay group (ndim ≤ 1) so WD does not pin it at 0 — it learns away from 0 via gradients, as intended; shortcuts are parameter-free (pad/stride) so blocks are exactly identity at init within stages and a strided slice at transitions (literature setting covers both); compile sees an identical graph (init values don't affect tracing), so all throughput signatures must match baseline exactly.

## Configuration Changes
- None. Every hyperparameter stays at its certified-optimal value (goal-learnings § Patterns High). The intervention is init-time only — params 4,286,026, FLOPs, dt, VRAM all unchanged by construction, giving perfect attribution.

## Execution Environment
- Method: local, single run via the standard composite background script (one Bash `run_in_background` chain: GPU-0 zero-compute-apps pre-check → `rm -f run.log` → `uv run train.py > run.log 2>&1 &` → inline watchdog sampling windowed dt every 15s with auto-kill on 4 consecutive >30ms windows → `wait` + TRAIN_EXIT echo + summary grep)
- Resources: GPU 0 only (wait if busy — never GPU 1); ~1613MB VRAM (identical to baseline)
- Estimated runtime: ~480–515s total (300s timed + ~139 evals + startup; compile cache for this exact graph is warm from baseline runs → startup ~13s likely)
- Log output: `run.log` in project root (full redirect, no tee/stream per goal Procedure); deleted after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- **Contention** (infra): watchdog auto-kill on 4 consecutive >30ms windows; post-kill `nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader` to confirm foreign presence. Quarantine contaminated runs (never analyze), relaunch into a clean window, max 2 infra retries. An architecture-priced steady slowdown is IMPOSSIBLE here (identical graph) — any sustained >30ms is contention by elimination.
- **Wrong init / wrong model**: params line ≠ 4,286,026 → kill, fix, relaunch (code-error retry). Additionally, if the FIRST eval (ep 1) is degenerate (≤ 15%, i.e., near-random — would suggest blocks failed to turn on), let the run continue to ~ep 10; if still < 40% (far below any prior clean run's trail) kill as a research failure, do not retry.
- **No output**: no step lines within 90s of launch → inspect run.log tail for a crash.
- **Wall cap**: > 600s total = failure per hard constraints (data/ is cached; no download inflation expected).

## Verification Protocol

### Verification Procedure

Pre-condition (contention sanity, before any condition): num_epochs within ~10% of ~139 AND post-hoc windowed profile ≈ 0 windows >30ms:
`tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>30) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>30ms: %d of %d | mean win %.1f ms\n", c, n, s/n}'`
If contaminated: quarantine and relaunch (infra path) — never verify a contaminated run.

Conditions in goal-file order, FIRST-FAILURE STOP:
1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1, via `exp-index.sh baseline`): `grep "^best_test_acc:" run.log` — pass iff ≥ 96.81. Empty grep = crash (read `tail -n 50 run.log`). Timeout: 1 min (metrics already on disk).
2. **Run completes within budget**: `grep "^total_seconds:" run.log` ≤ 600 and TRAIN_EXIT rc=0. Timeout: 1 min.
3. **Validation at most once per epoch**: `tr '\r' '\n' < run.log | grep -c "eval ep"` equals the `num_epochs` value (code structure guarantees exactly 1/epoch). Timeout: 1 min.

Cleanup after verdict: delete `run.log`; on no-improvement discard via `git checkout -- . && git clean -fd -e .autoresearch/ -e data/` (NEVER bare `-fd` — it deletes the dataset cache, infra-errors Warning).

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — must be ≈ 1613 (attribution check)
- num_epochs: `grep "^num_epochs:" run.log` — must be ≈ 139 (attribution check)
- num_params: `grep "^num_params:" run.log` — must be 4,286,026 (attribution check)
- Early trail + plateau shape: `tr '\r' '\n' < run.log | grep "eval ep"` — eps 1–20 vs EXP-017's 63.76@5 / 75.06@8 (onset claim); final ~10 evals flatness and level (plateau claim)
