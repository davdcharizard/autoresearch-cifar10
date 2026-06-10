# Plan EXP-025: Large-batch throughput exploitation (batch 256 + linear LR scaling)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md

## Milestones

### Milestone 1: Code changes implemented and passing local checks
- [ ] Edit `train.py`: `BATCH_SIZE` 128→256, `PEAK_LR` 0.2→0.4, `WARMUP_FRAC` 0.05→0.08.
- [ ] Edit the `DataLoader(...)` call: add `persistent_workers=True` and `prefetch_factor=4` (wall-clock safeguard for the faster-per-image GPU; core-torch kwargs, no new dep).
- [ ] `git diff --name-only` shows ONLY `train.py` changed.
- [ ] Smoke check: `uv run python -c "import ast; ast.parse(open('train.py').read())"` parses clean; confirm `num_params` will be unchanged (no architecture touched).

### Milestone 2: Experiment launched and confirmed running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Within ~60s, `run.log` shows `params: 4,299,866`, clean compile, device cuda, and step lines appearing with no NaN.

### Milestone 3: Run completes; throughput/epoch diagnostic captured
- [ ] Run exits 0 and prints the summary block (`best_test_acc:` … `num_params:`).
- [ ] Capture the KEY diagnostic: mean `dt` (ms/step) and `num_epochs` vs baseline (91 @ batch 128, ~8ms/step). Records whether the net stayed launch-bound (dt ~flat ⇒ more effective epochs) or went compute-bound (dt ~2× ⇒ epochs flat).
- [ ] `total_seconds < 600`.

## Code Changes
- **train.py (L22)**: `BATCH_SIZE = 128` → `256`. Doubles images/step. Under the compute-`dt`-gated budget (L242, `total_training_time += dt`, timer starts after the dataloader yields at L218), if `dt` stays ~flat in the launch-bound regime this ~doubles effective epochs within the 300s budget — the core mechanism being tested.
- **train.py (L23)**: `PEAK_LR = 0.2` → `0.4`. Linear LR scaling rule (Goyal 2017): batch ×2 ⇒ LR ×2, to keep the SGD noise scale / per-epoch progress comparable at the larger batch. The time-fraction cosine schedule (`lr_at_fraction`, L35-41) consumes `PEAK_LR` directly, so it still warms to 0.4 then anneals to ~0 over the 300s — no schedule-shape retuning needed.
- **train.py (L24)**: `WARMUP_FRAC = 0.05` → `0.08`. Slightly longer warmup to absorb the higher peak LR (0.4) without early instability (Goyal 2017 gradual warmup). 0.08 of the budget ≈ 24s of compute warmup.
- **train.py (L172-179, DataLoader)**: add `persistent_workers=True` (avoid respawning the 8 workers each epoch — matters more now that epochs may increase) and `prefetch_factor=4` (keep larger 256-image batches staged ahead of the faster-per-image GPU). Pure wall-clock safeguard: the accuracy mechanism is compute-`dt`-gated and independent of dataloader speed; this only protects the <600s wall-clock constraint. Both are core `torch.utils.data.DataLoader` kwargs — no new dependency.

## Configuration Changes
- BATCH_SIZE: 128 -> 256 (double images/step; exploit launch-bound headroom for more effective epochs under the compute-gated budget)
- PEAK_LR: 0.2 -> 0.4 (linear scaling rule for 2× batch; Goyal 2017)
- WARMUP_FRAC: 0.05 -> 0.08 (longer gradual warmup for the higher peak LR)
- DataLoader: add persistent_workers=True, prefetch_factor=4 (wall-clock safeguard only; no accuracy effect)

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, run in background.
- Resources: single NVIDIA H20 (GPU 0). VRAM at batch 256 ≈ ~1 GB of 98 GB (baseline ~0.5 GB at 128) — non-binding.
- Estimated runtime: ~300s training compute + ~30s startup/compile + per-epoch eval. If epochs rise to ~140-160, eval calls rise proportionally; expect total wall-clock ~430-540s. Must stay < 600s.
- Log output: `run.log` in project root (tee'd via redirection); source of truth. Final summary block carries best_test_acc / num_epochs / num_steps / total_seconds / peak_vram_mb / num_params.
- Tool skill: none (local run).

## Abort Criteria
<!-- Conditions that signal the experiment should be stopped early. -->
- **NaN/inf loss** at any point (the higher 0.4 peak LR could destabilize despite warmup) → kill, treat as failed.
- **Loss not decreasing / diverging after warmup** (smoothed `loss` higher at ~step 1000 than at end of warmup) → LR too high, kill.
- **Wall-clock runaway**: if the process is still running past ~580s wall-clock (e.g. dataloader starvation inflating wall-clock with many eval calls), kill to respect the 10-min / <600s hard constraint and treat as failed.
- **No output / hang**: no new step lines in `run.log` for >120s → kill.
- CUDA OOM (not expected at ~1 GB) → kill, treat as failed.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22**, so the pass threshold is **best_test_acc ≥ 96.32**.

Run the conditions in order; stop at the first failure (→ no-improvement, or invalid for a constraint breach).

1. **Cond 1 — primary metric clears bar.** After the run completes, extract:
   `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`
   PASS iff `best_test_acc ≥ 96.32`. Empty `best_test_acc:` ⇒ crash (`tail -n 50 run.log`) → crash verdict.
   Timeout: the run itself ≤ 600s; verification grep is instant.
2. **Cond 2 — clean completion within budget.** PASS iff the summary block printed, no `Traceback` in `run.log` (`grep -c Traceback run.log` == 0), and `total_seconds < 600`.
3. **Cond 3 — no constraint violations.** PASS iff: `git diff --name-only` lists only `train.py`; `num_params == 4,299,866` (unchanged — no architecture change); eval-count == num_epochs (one `eval ep` line per epoch — `grep -c "eval ep" run.log` == `num_epochs`); no new deps (only core-torch DataLoader kwargs added); seed still 42 (unchanged in code).

**Mandatory attribution note (per project's epoch-wall pattern, project-insights High):** record mean `dt` (ms/step) and `num_epochs`. Interpretation for the report regardless of verdict:
- dt ~flat (~8-9ms) AND epochs ↑ (>91): launch-bound premise CONFIRMED; accuracy delta is a clean test of "does the TA recipe want more epochs + tolerate batch 256".
- dt ~2× (~16ms) AND epochs ~flat (~91): compute-bound — launch-bound premise FALSIFIED; any regression is the large-batch generalization gap at equal epochs (clean either way).

### Informational Metrics (Optional)
- peak_vram_mb: `grep -a "^peak_vram_mb:" run.log` — expect ~1 GB.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — the throughput diagnostic vs baseline 91 ep.
- training throughput (img/s): from the per-step `img/s` field in step log lines (mean of late-run samples) — compare to baseline ~15,600 img/s at batch 128.
