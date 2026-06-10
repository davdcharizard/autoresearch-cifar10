# Plan EXP-055: Richer per-image AugMix chains on a subset — RandomApply([AugMix(mixture_width=4)], p)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-055.md

Baseline = **96.45%** (EXP-054, commit 86161d9); bar = baseline + 0.1 = **96.55%**. Push the only validated lever (augmentation chain-COUNT diversity, confirmed 3× — EXP-012/052/054) into its one untested region: raise AugMix `mixture_width` 3→4 (richer per-image mixture) delivered to a subset via `RandomApply`. Coverage `p` is the wall-control knob, set conservatively per the EXP-054 tight-wall lesson.

## Wall-feasibility reasoning (drives the p choice)
- Goal-learnings § Protocol Findings (EXP-054): RandomApply(heavy CPU aug) has HIGH run-to-run wall variance; the early-window projection UNDER-estimates the final wall (EXP-054 projected ~535s, finished **593.0s**, +58s spread). **Target projected ≤ ~540s, not ~585s; do NOT trust a tight (<600s) wall as reproducible.**
- This session's isolated dataloader probe (8 workers, 150 batches, post-warmup): `RandomApply(AugMix w4) p=0.4` = **12.2 ms/batch**; `p=0.35` = **11.4 ms/batch**; EXP-054 reference `RandomApply(AugMix w3) p=0.5` = **12.1 ms/batch** (and that shipped at the tight 593.0s wall).
- **Therefore p=0.4 (12.2ms ≈ EXP-054's 12.1ms) would reproduce EXP-054's ~593s tight wall → too risky.** Lead with **p=0.35** (11.4ms, real buffer), gated by an early REAL-load wall check with a conservative +60s safety buffer, and a **p=0.3 fallback** if the gate still projects tight.

## Milestones

### Milestone 1: Code change implemented and smoke-tested
- [ ] In `train.py` `train_tf` (line ~171), replace `transforms.RandomApply([transforms.AugMix()], p=0.5)` with `transforms.RandomApply([transforms.AugMix(mixture_width=4, chain_depth=-1)], p=0.35)` (richer 4-chain AugMix on ~35% of images; the rest get only RandomCrop+Flip, with GPU Cutout still applied in the train loop). Update the adjacent comment to reflect w4/p=0.35.
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --name-only` = `train.py` only; `AugMix(mixture_width=4, chain_depth=-1)` instantiates (no new dep — torchvision-native).
- [ ] Smoke: pull ~10 augmented samples through `train_tf` (RandomApply+AugMix(w4) runs, output shape (3,32,32) after ToTensor); confirm `num_params == 4,299,866` is unaffected (aug-only change).

### Milestone 2: Experiment running and FEASIBILITY (wall-clock) confirmed
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background) on an idle GPU (0 or 1; both idle per nvidia-smi). Confirm run.log is being written.
- [ ] **Early REAL-load wall gate** (after ~60–90s wall, post torch.compile): from run.log, measure eval-inclusive ms/step over a window (Δsteps / Δwall; strip leading zeros from step numbers with `sed 's/^0*//'`). Project `total_wall = effective_ms/step × 35300 + measured_startup_s`. Apply a **+60s conservatism buffer** (EXP-054 spread). **If projected+buffer > 595s → ABORT (TaskStop) and go to the p=0.3 contingency.** Else proceed.
- [ ] Early signal: dt steady ~8ms (GPU step unchanged — AugMix is CPU-side), ep1 test_acc normal (~45–48%), no NaN.

### Milestone 3: Run completes and is verified
- [ ] Run prints the summary block; `total_seconds < 600` (binding constraint) and total wall < 10 min.
- [ ] Extract best_test_acc, num_epochs, num_steps, dt dist, total_seconds, peak_vram_mb; compare best_test_acc to bar **96.55**.
- [ ] Remove `run.log` before the next experiment.

## Code Changes
- **train.py** (one line in `train_tf`, line ~171): `transforms.RandomApply([transforms.AugMix()], p=0.5)` → `transforms.RandomApply([transforms.AugMix(mixture_width=4, chain_depth=-1)], p=0.35)` (+ comment update).
  - **Why this tests the hypothesis**: The diversity lever is chain COUNT (EXP-053 closed magnitude; EXP-052/054 raised count via width 2→3 and richer-on-subset). Width 3→4 is the single untested extension — more parallel chains convex-combined per augmented image → a richer per-image mixture. Tests whether richer 4-chain diversity (on a 35% subset) beats the 3-chain/50% winner.
  - **Risks / edge cases**: (a) **Likely null/saturation** — the AugMix paper suggests clean-accuracy gains saturate past width 3 (regime-specific prior; our subset+Cutout+300s-budget regime differs). (b) **Confounded** — width↑ (3→4) and coverage↓ (p 0.5→0.35) co-move; a null cannot cleanly separate width-saturation from coverage-loss (documented for analysis; acceptable for a within-lever probe). (c) **Wall** — gated by the Milestone-2 real-load check + p=0.3 contingency; Σdt/epochs unaffected (dt 8ms, CPU-side aug).

### Contingency (Run 2, only if Run 1 trips the Milestone-2 wall gate)
- Re-run at `RandomApply([transforms.AugMix(mixture_width=4, chain_depth=-1)], p=0.3)` (lower coverage → lower average CPU cost, est ~11.0ms/batch → safer wall). Same mechanism, 30% coverage. If that also trips the gate, record wall-infeasibility for w4 and proceed to analysis with the partial finding.

## Configuration Changes
- Augmentation: `RandomApply([AugMix() w3,d-1], p=0.5)` (EXP-054 winner) → `RandomApply([AugMix(mixture_width=4, chain_depth=-1)], p=0.35)` (richer 4-chain on ~35%). No model/optimizer/schedule/seed/batch/compile changes. num_params unchanged (4,299,866).
  - Rationale for p=0.35 (not 0.4): isolated probe puts p=0.4 at 12.2ms/batch ≈ EXP-054's 12.1ms, which shipped at the tight 593.0s wall; the EXP-054 protocol finding mandates a conservative wall target. p=0.35 (11.4ms) buys real margin while still delivering w4 richness to a meaningful subset.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background (`run_in_background: true`). MUST use `uv run` (bare python lacks torchvision).
- Resources: single NVIDIA H20. Shared node GPUs 0/1 (both idle per nvidia-smi); launch on either.
- Estimated runtime: ~300s Σdt budget; wall ~560s projected at p=0.35 (target < 600s, monitored at Milestone 2; p=0.3 fallback if tight).
- Log output: stdout/stderr → `run.log` in project root.
- Tool skill: none (local run).

## Abort Criteria
- **Wall-clock projection + 60s buffer > 595s** at the Milestone-2 real-load check → abort, go to p=0.3 contingency.
- Loss NaN/inf or diverging.
- dt rises well above 8ms and stays (GPU-side issue — not expected; augmentation is CPU-side).
- No output / log not advancing > 3 min after launch.
- Total wall-clock actually reaching ~595s without a summary → kill (constraint breach = failure).

## Verification Protocol

### Verification Procedure
Run after completion; stop at the first failed necessary condition.

1. **Baseline**: `bash /SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/improve-cifar10-test-accuracy.tsv` → 96.45, bar **96.55**.
2. **Necessary condition 1 — `best_test_acc >= 96.55`**: `grep -aE "^best_test_acc:" run.log` → parse float. PASS iff `>= 96.55`; else no-improvement. (Absent ⇒ crash → `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:|^num_params:" run.log` → summary printed, **`total_seconds < 600`** (binding here), `num_params == 4,299,866`. No NaN/traceback (`grep -ic "nan\|traceback" run.log` → 0).
4. **Necessary condition 3 — no hard-constraint violations**: `git diff --name-only` = `train.py` only; prepare.py/eval untouched; `evaluate()` once/epoch (loop unchanged); no new deps (AugMix/RandomApply are torchvision); seed 42 unchanged; no seed hacking.
5. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.45: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 / ~35.3k (Σdt budget unaffected by CPU-side aug).
- total_seconds (wall): `grep -aE "^total_seconds:" run.log` — feasibility-critical (~560s expected at p=0.35).
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect steady 8ms.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare to EXP-054's 0.1968 (loss+top1 both moving = real gain signature).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` — expect ~454 MB.
