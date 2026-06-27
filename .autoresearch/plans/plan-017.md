# Plan EXP-017: Per-stage depth redistribution [3,3,3] → [2,3,4] at constant FLOPs
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md

## Milestones

### Milestone 1: Code change implemented and syntax-checked
- [x] Edit `train.py`: `NUM_BLOCKS = 3` → `NUM_BLOCKS = (2, 3, 4)` (per-stage block counts; comment notes RegNet third-stage-heavy allocation at depth 20)
- [x] Edit `ResNet.__init__`: the three `_make_layer` calls take `num_blocks[0]`, `num_blocks[1]`, `num_blocks[2]` respectively
- [x] Edit the startup print: depth formula `6 * NUM_BLOCKS + 2` → `2 + 2 * sum(NUM_BLOCKS)` (= 20, unchanged)
- [x] Syntax check passes: `uv run python -c "import ast; ast.parse(open('train.py').read())"` (do NOT import train.py — module level instantiates Eval())
- [x] Diff review: NOTHING else changed — all training constants (batch 512, peak 0.4, warmup 0.15, WD 5e-4, LS 0.1, aug stack, compile path) byte-identical

### Milestone 2: Run launched cleanly
- [x] Pre-launch: GPU 0 has zero compute apps (composite script aborts otherwise)
- [x] Launch via the standard composite background script (pre-check + `uv run train.py > run.log 2>&1` + inline contention watchdog: windowed pct_done-delta dt every 15s, auto-kill on 4 consecutive >30ms windows)
- [x] `run.log` shows the params line: expect `params: 5,392,714` — confirmed exactly

### Milestone 3: Early signals (informational dt-gate, step ~100–150)
- [x] Windowed dt from step prints ≈ 22.4ms ± 5% — measured 21.5ms mean (post-hoc, 278 windows; live Monitor armed late per the known turn-scheduling pattern, inline watchdog covered the run)
- [x] Projected epochs ~135–139 expected — actual 144 (dt came in BELOW baseline; FLOPs-neutrality confirmed with margin)

### Milestone 4: Completion and verification
- [x] TRAIN_EXIT rc=0, summary block present in run.log
- [x] Contention sanity (pre-condition): 144 epochs on-model at 21.5ms; 0/278 windows >30ms — CLEAN
- [x] Verification protocol executed in order, first-failure stop: condition 1 FAILED (96.43 < 96.81); conditions 2–3 skipped

## Code Changes
- **train.py** (only file; 3 edits):
  1. `NUM_BLOCKS = 3` → `NUM_BLOCKS = (2, 3, 4)` — moves one block from stage 1 (74k params, 64ch@32×32) to stage 3 (1.18M params, 256ch@8×8). Per-block MACs are equal across stages (36,864×1024 = 589,824×64 ≈ 75.5M), so total FLOPs are unchanged by construction; params 4.29M → 5.39M (+26%). Tests whether capacity ALLOCATION (vs the closed uniform-scaling axis) moves the metric when epochs are preserved.
  2. `self.layer1/2/3 = self._make_layer(..., num_blocks, ...)` → `num_blocks[0]/[1]/[2]` — wires the per-stage counts; `_make_layer` itself already handles arbitrary counts.
  3. Startup print depth formula → `2 + 2 * sum(NUM_BLOCKS)` — keeps the "ResNet-20" label correct for a tuple.
  - Risks: torch.compile may price 256ch/8×8 convs off the FLOPs model (project-insights Medium — never project across regimes) — covered by Milestone 3's measured-dt check; alignment is safe (256 = 8×32, all widths unchanged); channels_last + bf16 path untouched.

## Configuration Changes
- NUM_BLOCKS: 3 → (2, 3, 4) (RegNet population result: optimized allocations are third-stage-heavy/first-stage-light at matched compute — arXiv 2003.13678; brainstorm-017 § Chosen Idea)
- Everything else byte-identical to baseline @ 1990397 — the certified single-knob optimum (goal-learnings § Patterns High) must not be disturbed, so the result is attributable to allocation alone.

## Execution Environment
- Method: local, single run via the standard composite background script (one Bash `run_in_background` command chain: pre-launch GPU-0 zero-compute-apps check → `rm -f run.log` → `uv run train.py > run.log 2>&1 &` → inline watchdog loop sampling windowed dt every 15s, auto-kill on 4 consecutive >30ms windows → `wait` + exit-code echo + summary grep). NEVER arm a Monitor in a separate turn (infra-errors Important: it can first poll an already-finished run).
- Resources: GPU 0 only (wait for it to free if busy — never GPU 1); ~1.7GB VRAM expected (baseline 1613MB; stage-3 weights up, stage-1 activations down — record actual)
- Estimated runtime: ~510s total (300s timed budget + ~140 evals + ~13s warm-cache startup); hard cap 600s
- Log output: `run.log` in project root (full stdout+stderr redirect, no tee/stream per goal Procedure); deleted after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- **Contention** (infra, not research): watchdog auto-kills on 4 consecutive >30ms windows. Post-kill, check `nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader`: if a foreign PID is present or windowed dt was ALTERNATING (24/48ms pattern), it is contention — quarantine the run (never analyze), relaunch into a clean window, max 2 infra retries. If NO foreign app and dt was STEADY >30ms from the start, it is architecture-priced throughput (compile repriced the conv mix) — treat as a research result, not infra; rerun once only to confirm if ambiguous.
- **Wrong model wired**: params line deviates >1% from 5,392,714 → kill immediately, fix, relaunch (code-error retry).
- **No output**: no step lines within 90s of launch (startup is ~13s warm-cache; allow compile-cache-miss slack) → inspect run.log tail for a crash.
- **Wall cap**: any run exceeding 600s total is a failure per the goal's hard constraints (watch for download-inflated startup — data/ is preserved by the `-e data/` clean rule, so this should not occur).

## Verification Protocol

### Verification Procedure

Pre-condition (contention sanity, before any condition): num_epochs within ~10% of the step-100 projection (~135–139 expected at flat dt) AND post-hoc windowed profile ≈ 0 windows >30ms:
`tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>30) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>30ms: %d of %d | mean win %.1f ms\n", c, n, s/n}'`
If contaminated: quarantine and relaunch (infra path) — do NOT verify a contaminated run.

Conditions in goal-file order, FIRST-FAILURE STOP:
1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1, from `exp-index.sh baseline`): `grep "^best_test_acc:" run.log` — pass iff value ≥ 96.81. Empty grep = crash (read `tail -n 50 run.log`). Timeout: metrics are already on disk, 1 min.
2. **Run completes within budget**: `grep "^total_seconds:" run.log` ≤ 600 and TRAIN_EXIT rc=0. Timeout: 1 min.
3. **Validation at most once per epoch**: count `grep -c "eval ep" run.log` ≤ `grep "^num_epochs:" run.log` value (code structure guarantees exactly 1/epoch; confirm from the log). Timeout: 1 min.

Cleanup after verdict: delete `run.log`; on no-improvement discard via `git checkout -- . && git clean -fd -e .autoresearch/ -e data/` (NEVER bare `-fd` — infra-errors Warning: it deletes the dataset cache).

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — VRAM cost of the reallocation (expect ~1.6–1.8GB)
- num_epochs: `grep "^num_epochs:" run.log` — the FLOPs-neutrality claim's direct test (expect ~135–139)
- num_params: `grep "^num_params:" run.log` — expect 5,392,714
- Eval trajectory shape: `tr '\r' '\n' < run.log | grep "eval ep"` tail — converged-plateau check (final ≈ best, flat tail; project-insights Medium: plateau length is the metric's currency)
