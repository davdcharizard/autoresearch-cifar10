# Plan EXP-058: Shallower-but-wider ResNet-14 (6 blocks, k=5) — the dt-reducing capacity quadrant

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-058.md

Baseline = **96.45%** (EXP-054, commit 86161d9); bar = **96.55%**. Every prior capacity experiment INCREASED per-step dt (EXP-004/009 widen → compute-bound; EXP-038 FLOP-neutral realloc → +31% dt; EXP-044 deeper-narrower → +50% dt) and hit the epoch wall. This loop tests the untested INVERSE: REDUCE block count (9→6) to LOWER the launch-bound dt, reinvesting into width (k=4→k=5). The config is iso-param (4,290,874 ≈ baseline 4,299,866, −0.2%), making it the clean mirror of EXP-044's deeper-narrower iso-param test.

## Milestones

### Milestone 1: Code implemented and smoke-tested
- [ ] In train.py, change `NUM_BLOCKS` 3→2 and `WIDTH_MULT` 4→5 (the only two edits; the `ResNet`/`BasicBlock` classes already parameterize both). No other changes — augmentation, optimizer, schedule, batch, seed, compile, Cutout all unchanged.
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --name-only` = train.py only; no new imports.
- [ ] Smoke: build the model and confirm `num_params == 4,290,874`, `6*NUM_BLOCKS+2 == 14` (prints "ResNet-14"), a (8,3,32,32) forward gives (8,10) finite. (Already verified offline: 4,290,874 params, out (8,10).)

### Milestone 2: Running and dt/epoch + contention feasibility confirmed
- [ ] **Pre-launch idle-GPU check**: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader` + `--query-compute-apps=pid,used_memory`; launch on a GPU with util ~0% and mem <700MiB (EXP-038/056 contention lesson — a contended run is an unfair dt-budget test).
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background). Confirm run.log writes.
- [ ] **Early gate** (~60-90s wall, post-compile): two-point window (Δsteps via log, Δwall via ps etimes). Compute (a) steady mean dt, (b) **wall/Σdt ratio** (contention tell — expect ~1.3-2×, NOT ~10×), (c) projected epochs = (300 / mean_dt_s) / 390. **ABORT if: dt > ~10.5ms (epochs < ~73) → wide-conv memory wall dominates even at 6 blocks (hypothesis negative branch confirmed cheaply); OR wall/Σdt ≫ 2.5 → contention, relaunch on a clean GPU.**
- [ ] Early signal: ep1 test_acc normal (~45-50%), no NaN, dt steady.

### Milestone 3: Completes and verified
- [ ] Summary prints; `total_seconds < 600`.
- [ ] Extract best_test_acc, num_epochs, num_steps, dt dist, total_seconds, peak_vram_mb, final_test_loss; compare best_test_acc to bar **96.55**.
- [ ] Remove `run.log`.

## Code Changes
- **train.py L19-20** — two constant edits:
  ```python
  NUM_BLOCKS = 2  # ResNet-14 = 6*2+2 (was 3 / ResNet-20); fewer sequential blocks → lower launch-bound dt
  WIDTH_MULT = 5  # k=5 stages {80,160,320} (was 4 / {64,128,256}); widen to keep capacity iso-param (~4.29M)
  ```
  - **Why this tests the hypothesis**: shallower (6 vs 9 blocks) cuts kernel-launch/sequential-layer cost → lower dt (the binding constraint); wider (k=5) holds capacity iso-param. Tests whether a fewer-wider-blocks point reaches a better accuracy-per-budget than k=4/9-blocks — the one untested (dt-reducing) quadrant of the capacity surface.
  - **Risks / edge cases**: (a) **wide-conv memory-bandwidth wall** — k=5 320-ch convs may raise dt past the headroom that fewer blocks buys → epoch wall (gated, abort dt>10.5ms). (b) **capacity-per-stage loss** — 2 blocks/stage give fewer residual-refinement steps; at iso-param the net may plateau ≤ baseline even at adequate epochs (epoch-saturation means extra epochs don't rescue a converged smaller-depth net). (c) channels_last + compile: the model is rebuilt identically (same classes), so torch.compile(reduce-overhead) and channels_last apply unchanged. (d) num_params changes by design (architecture change) — this is allowed (num_params is informational, not a hard constraint; VRAM is soft).

## Configuration Changes
- `NUM_BLOCKS`: 3 → 2 (ResNet-20 → ResNet-14; 9 → 6 BasicBlocks).
- `WIDTH_MULT`: 4 → 5 (stages {64,128,256} → {80,160,320}).
- Iso-param: 4,290,874 vs baseline 4,299,866 (−0.2%). No optimizer/schedule/aug/batch/seed/compile changes. The depth↔width reallocation is the single conceptual variable (mirror of EXP-044, opposite direction).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background. MUST use `uv run`.
- Resources: single NVIDIA H20. Shared node GPUs 0/1 — **verify idle before launch** (EXP-038/056 contention lesson); pick util ~0%, mem <700MiB.
- Estimated runtime: ~300s Σdt; wall ~400-450s if dt 8-10ms. Target < 600s. dt is the open question (6 wider blocks vs 9 narrower) — gated.
- Log output: stdout/stderr → `run.log`.
- Tool skill: none (local run).

## Abort Criteria
- **dt > ~10.5ms** at Milestone-2 (epochs < ~73) → wide-conv memory wall → abort (hypothesis negative branch; record as no-improvement-by-epoch-wall, no retry — this IS the result).
- **wall/Σdt ≫ 2.5** early → GPU contention → TaskStop, relaunch on a clean idle GPU (NOT a code failure).
- Loss NaN/inf or diverging — check ep1.
- Total wall approaching ~595s without a summary → kill. [Unlikely — dt-bound.]
- No output / log not advancing > 3 min after launch.

## Verification Protocol

### Verification Procedure
Run after completion; stop at the first failed necessary condition.

1. **Baseline**: `bash /SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/improve-cifar10-test-accuracy.tsv` → 96.45, bar **96.55**.
2. **Necessary condition 1 — `best_test_acc >= 96.55`**: `grep -aE "^best_test_acc:" run.log` → parse float. PASS iff `>= 96.55`; else no-improvement. (Absent ⇒ crash → `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:|^num_params:" run.log` → summary printed, **`total_seconds < 600`**, `num_params == 4,290,874` (the intended new architecture). No NaN/traceback (`grep -ic "nan\|traceback" run.log` → 0).
4. **Necessary condition 3 — no hard-constraint violations**: `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps; seed 42 unchanged; ran on an uncontended GPU (fair dt-budget).
5. Remove `run.log`.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.45: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — the KEY diagnostic (does shallower buy epochs? expect >91 if dt<8ms, <91 if dt>8ms).
- total_seconds: `grep -aE "^total_seconds:" run.log`.
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — the binding-constraint readout (does shallower-wider lower dt below the 8ms floor, or does the wide-conv wall raise it?).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — vs baseline 0.195/EXP-054 0.1968 (underfit tell if ≫ 0.20).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`.
