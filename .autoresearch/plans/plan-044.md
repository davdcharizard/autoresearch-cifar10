# Plan EXP-044: Depth↔width reallocation — deeper-narrower iso-param ResNet (ResNet-32, k=3)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-044.md

## Milestones

### Milestone 1: Code change implemented and architecture verified
- [ ] Edit `train.py`: `NUM_BLOCKS` 3→5 and `WIDTH_MULT` 4→3 (only these two constants).
- [ ] Confirm the printed banner reads `ResNet-32 | params: 4,166,970` (= 96.9% of baseline 4,299,866), widths {48,96,192} all multiples of 16.
- [ ] Sanity-check no other code path depends on the old width/depth (grep for hardcoded 256/128/64 channel assumptions — there are none; `_make_layer` derives widths from `width_mult`).

### Milestone 2: Run launched on an idle GPU and confirmed healthy
- [ ] Confirm a GPU is idle (`nvidia-smi`) — shared H20 node; contention silently inflates dt (infra-errors). Pick the idle index.
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` in the background.
- [ ] Within ~90s confirm: device line printed, params line = 4,166,970, no traceback, loss decreasing, first `dt:` readings visible.

### Milestone 3: Throughput (dt) and epoch count verified — the FAIRNESS GATE
- [ ] After completion, extract the dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`.
- [ ] Record steady-state dt and `num_epochs`. **This is the load-bearing check**: a fair depth test requires dt ≈ 8ms and `num_epochs ≥ 77` (the EXP-007 saturation point). If `num_epochs < 77`, the result is dt-confounded (underfit), not a clean depth test — note it explicitly in analysis.

### Milestone 4: Accuracy verified against baseline
- [ ] Extract `best_test_acc` and compare to bar 96.32 (baseline 96.22 + 0.1).

## Code Changes
- **train.py (L19-20)**: `NUM_BLOCKS = 3` → `NUM_BLOCKS = 5`; `WIDTH_MULT = 4` → `WIDTH_MULT = 3`. This reallocates the ~4.3M-param budget from width into depth: 3 blocks/stage @ {64,128,256} (ResNet-20, k=4) → 5 blocks/stage @ {48,96,192} (ResNet-32, k=3). Tests the hypothesis that depth is more parameter-efficient for generalization than width on CIFAR (He 2016), via the one capacity dimension untested in 45 experiments. No other lines change — the recipe (optimizer, LR schedule, augmentation, label smoothing, Cutout, torch.compile, bf16, channels_last, seed) is byte-identical, making this a clean single-variable depth-vs-width test.
- **Risks/edge cases**: (a) `ResNet.__init__` and `_make_layer` already derive all per-stage widths from `width_mult` and accept `num_blocks` uniformly — no hardcoded channel counts, so the two-constant change is self-consistent. (b) The compiled-graph banner `ResNet-{6*NUM_BLOCKS+2}` will read ResNet-32. (c) 15 sequential blocks (vs 9) is the primary dt risk — more sequential conv+BN layers; mitigated by the ≈iso-FLOP sizing (97.8%) and CUDA-graph launch-overhead amortization, but must be verified empirically in Milestone 3.

## Configuration Changes
- `NUM_BLOCKS`: 3 → 5 (depth: ResNet-20 → ResNet-32; +6 blocks total). Rationale: 5 blocks/stage at k=3 lands params at 96.9% and FLOPs at 97.8% of baseline — the closest iso-param + iso-FLOP deeper config among the candidates evaluated (verified by replicating the exact baseline param count 4,299,866).
- `WIDTH_MULT`: 4 → 3 (width: {64,128,256} → {48,96,192}). Rationale: narrowing compensates the added depth to hold the param/FLOP budget; all three widths stay multiples of 16 for the tensor-core path (EXP-038: non-multiple-of-8 widths ran ~5× slower).
- No other hyperparameters change.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, launched in the background (Bash `run_in_background: true`); harness re-invokes on completion (~6-8 min).
- Resources: single NVIDIA H20 (shared node, 2 GPUs idx 0/1). VRAM trivial (baseline ~491MB of 98GB; narrower net uses less). Fixed 300s training budget.
- Estimated runtime: ~6-8 min wall (300s timed training + ~30-60s startup/compile + per-epoch eval). Must be < 10 min total or it is a failure.
- Log output: `run.log` in the project root (tee'd via redirection). dt lines use `\r` — extract with `tr '\r' '\n'`. Primary source of truth for success/dt/epochs.
- Tool skill: none (local run).

## Abort Criteria
- Loss diverges (NaN/inf) or fails to fall below ~1.0 within the first few epochs.
- A traceback / OOM / shape error appears in `run.log` (a deeper net with stride-2 downsample blocks at 5/stage is structurally standard, so none expected).
- No `dt:` output or no epoch-eval lines after ~120s (silent hang).
- Total wall-clock approaches 10 min without the summary block printing → kill and treat as failure.
- GPU contention detected mid-run (dt steady-state ≫ 8ms while another user's job is co-resident) → discard as contention-confounded and re-run on an idle GPU (do NOT attribute the dt to the architecture).

## Verification Protocol

### Verification Procedure
Baseline (from experiment index) = **96.22%**; bar = **96.32%** (baseline + 0.1).

1. **Run completes cleanly within budget** — after the background run finishes, check the summary block exists:
   `grep -aE "^best_test_acc:|^training_seconds:|^total_seconds:|^num_epochs:|^num_steps:|^peak_vram_mb:" run.log`.
   Pass: `best_test_acc:` is present and non-empty, `total_seconds` < 600, `training_seconds` ≈ 300. Empty `best_test_acc` ⇒ crash (inspect `tail -n 50 run.log`). Timeout for the run: 600s wall.
2. **Fairness gate (dt / epochs)** — `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` and read `num_epochs`. Record steady dt and epoch count. If `num_epochs ≥ 77` and dt ≈ 8ms → fair test. If `num_epochs < 77` → dt-confounded; the verdict will still be rendered on best_test_acc but the analysis must flag the regression as epoch-wall-driven, not a clean depth refutation.
3. **Primary necessary condition** — extract `best_test_acc` (`grep -aE "^best_test_acc:" run.log`). Pass iff `best_test_acc ≥ 96.32`. (Improvement is `≥ baseline + 0.1`.)
4. **No hard-constraint violations** — confirm only `train.py` was modified (`git diff --name-only` shows just `train.py`), `prepare.py`/eval untouched, `evaluate()` still called once per epoch (unchanged loop), no new deps, no seed hacking (seed 42 unchanged).
5. Remove `run.log` before the next experiment (keep the tree clean).

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` — expect < baseline 491MB (narrower net).
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — the throughput/depth trade-off signal (the fairness gate above).
- training throughput (img/s) & dt distribution: from the `tr '\r' '\n'` extraction above — confirms launch-bound vs compute-bound behavior of the deeper net.
