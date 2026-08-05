# Plan EXP-020: Cosine one-cycle decay (vs linear triangular LR shape)
- **Created**: 2026-06-30

## Summary
Change ONLY the post-warmup LR decay shape from the current **linear** triangular one-cycle to **cosine**, behind a `SCHEDULE` env, all in `train.py`. The linear-decay shape was set in EXP-001 and never tuned; the fastest CIFAR recipes / fastai one-cycle use cosine, and the MosaicML benchmark found cyclic/linear schedules underperform cosine by up to ~0.5% val-acc on CIFAR-class CNNs. It is **exactly throughput-free** (a per-step scalar formula change → num_epochs identical), so it sidesteps the under-anneal trap behind 5+ prior nulls, and targets the low-LR anneal tail that EXP-001 showed dominates final accuracy. Tested same-session against a `tri` control, verdict keyed on `cA=cos` with unchanged `PCT_START=0.15`, with the EXP-019-hardened protocol.

Baseline (current): **96.38** (commit 07c3760). Bar: **best_test_acc ≥ 96.48** (baseline+0.1pp) AND cA > same-session c0 by >0.1pp, replicated on a confirmation re-run.

## Milestones

### Milestone 1: Cosine schedule implemented + LR-trace smoke passes
- [ ] Add `import os` and `import math` near the top of train.py.
- [ ] Add env reads: `SCHEDULE = os.environ.get("SCHEDULE", "tri")` and make `PCT_START` env-overridable: `PCT_START = float(os.environ.get("PCT_START", "0.15"))` (keeps default 0.15; lets the diagnostic cB vary warmup without code branching).
- [ ] In the LR block (train.py:286-290), add a `cos` branch while keeping the `tri` (else) branch's expression **byte-identical** to the current formula (so `SCHEDULE=tri` is bit-for-bit baseline).
- [ ] Smoke A (LR-trace, pure-python, no GPU): print sampled `progress, lr_tri, lr_cos` at progress ∈ {0, 0.075, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0} and assert:
  - `tri` matches the closed-form baseline `PEAK*(1-progress)/(1-PCT_START)` for progress≥PCT_START (and the ramp below); both schedules share the warmup branch exactly.
  - `cos` ramps identically 0→PEAK over PCT_START, is monotone non-increasing after, equals PEAK at q=0 and **exactly 0** at progress=1.0.
  - `cos` HOLDS HIGHER than `tri` early-post-warmup, crosses near mid-decay, and is LOWER than `tri` late (the corrected mechanism); report the fraction of post-warmup samples with lr < 0.05·PEAK for each (cos should have more low-LR mass).
- [ ] Smoke B (crash-guard, GPU): in `/tmp/exp020_smoke.py`, build `ResNet9()` and run exactly 2 manual fwd+bwd+opt steps on RANDOM tensors with the cos `lr` formula — **NO `evaluator.evaluate`, NO CIFAR dataloader, NO metric recording, NO repo writes** (the schedule change cannot affect the model; this only guards a typo). Assert loss finite.
- [ ] Smoke scripts live at `/tmp/exp020_smoke.py` (OUTSIDE the repo) and touch no benchmark machinery; verify `git status --porcelain` shows only `train.py` (the `run_*.log` outputs are gitignored — confirmed EXP-019).

### Milestone 2: Official same-session cells run on GPU 1
- [ ] Start a background `nvidia-smi -l 5 > /tmp/exp020_smi.log` sampler for the session; log a point `nvidia-smi` before each cell (contention guard, infra-errors EXP-010/014).
- [ ] c0 (control): `SCHEDULE=tri` — baseline anchor + regression check.
- [ ] cA (**PRIMARY — chosen hypothesis; determines the verdict**): `SCHEDULE=cos` (PCT_START=0.15 unchanged).
- [ ] cB (**DIAGNOSTIC ONLY — informational, cannot trigger an `improvement` verdict**): `SCHEDULE=cos PCT_START=0.10` — shorter warmup pairing for cosine; if cB beats c0 but cA does not, that is EXP-021 material, NOT a credited EXP-020 win (no schedule-search on the test metric, EXP-019 cB rule).
- [ ] Each: `CUDA_VISIBLE_DEVICES=1 SCHEDULE=... [PCT_START=...] timeout 600 uv run train.py > run_{cell}.log 2>&1` (hard `timeout 600` wall cap — exit 124 = wall-kill = failure, the goal's 10-min rule, auditable from the runner).
- [ ] Record best_test_acc, num_epochs, ep25 (early eval line), training_seconds, total_seconds, peak_vram_mb per cell; cross-check the per-step `img/s` trace for any mid-run drop.

### Milestone 3: Verification + confirmation
- [ ] All cells fit num_epochs ≥ 135 AND within the clean ~142–154 band, ALL equally uncontended. **Pre-registered contention rule (set BEFORE seeing accuracy, no selective latitude)**: a cell is contention-flagged iff `/tmp/exp020_smi.log` shows ANY non-our PID on GPU 1 during its run OR its in-log sustained `img/s` drops below **22,000** (clean band ~25–26k; 22k ≈ 0.85× → ~127 ep < the 135 gate). If ANY cell is flagged → preserve ALL contaminated logs, mark them invalid in the exp-log, and re-run the ENTIRE same-session set (never a single cell) once GPU 1 is idle. NEVER collapse a contention-flagged under-anneal into no-improvement.
- [ ] Verdict keyed on **cA vs c0**: if cA ≥ 96.48 AND cA − c0 > 0.1pp → run a **confirmation re-run** of {cA, fresh c0} as a second same-session pair; require the delta to replicate (>0.1pp).
- [ ] If cA − c0 < 0.15pp on the first pass, treat as no-signal unless the confirmation pair replicates (low-c0-draw lesson, EXP-016/017/019).

## Code Changes
- **train.py** (the only editable file):
  - **Imports**: add `import os`, `import math` (baseline imports neither).
  - **Env reads** (near hyperparameters): `SCHEDULE = os.environ.get("SCHEDULE", "tri")`; change `PCT_START = 0.15` to `PCT_START = float(os.environ.get("PCT_START", "0.15"))`.
  - **LR schedule** (train.py:286-290) — add the cosine branch, keep `tri` expression unchanged:
    ```python
    progress = min(1.0, total_training_time / TIME_BUDGET_S)
    if progress < PCT_START:
        lr = PEAK_LR * progress / PCT_START
    elif SCHEDULE == "cos":
        q = (progress - PCT_START) / (1.0 - PCT_START)   # 0..1 post-warmup
        lr = PEAK_LR * 0.5 * (1.0 + math.cos(math.pi * q))
    else:  # tri (baseline, unchanged)
        lr = PEAK_LR * (1.0 - progress) / (1.0 - PCT_START)
    ```
- **Why this tests the hypothesis**: the only change is WHERE in training the optimizer spends its steps (the anneal-tail shape). Cosine holds LR higher early then drops steeply, spending more time at very low LR — directly probing whether the decay shape selects a better-generalizing minimum, with zero per-step cost (so any effect is attributable to the schedule, not epochs — which EXP-014 proved are worth ≈0 here).
- **Risks / edge cases**:
  - *tri regression*: the `else` branch keeps the exact original expression → `SCHEDULE=tri` is bit-identical to baseline. Smoke A guards this.
  - *EMA/TTA gates*: key on `progress`, NOT on LR (EMA warmup `progress>=0.15`, TTA `progress>=0.8`) → untouched by the schedule change. No new params, no VRAM/dtype change.
  - *Full anneal*: cosine finishes at exactly 0 (q=1 → cos(π)=−1 → lr=0), preserving the complete-the-anneal property; EMA denoises the steep tail.
  - *PCT_START env*: only the LR warmup fraction reads PCT_START (line 287-288); EMA_WARMUP_FRAC (0.15) and TTA_START_FRAC (0.8) are separate constants, so the cB PCT_START=0.10 override affects ONLY the LR warmup (intended) — verified by reading train.py:262,308,342.

## Configuration Changes
- `SCHEDULE`: new env, default `tri` (baseline). Cells: `tri` (c0), `cos` (cA), `cos` (cB).
- `PCT_START`: now env-overridable, default `0.15` (unchanged for c0/cA); cB uses `0.10`.
- No change to PEAK_LR/WD/LS/EMA/aug/batch — only the decay shape changes (single-variable test).

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=1 SCHEDULE=... [PCT_START=...] uv run train.py > run_{cell}.log 2>&1` from project root.
- Resources: single NVIDIA H20, **GPU 1** (mandatory — GPU 0 in use). VRAM ~1.6 GB (unchanged).
- Estimated runtime: ~300s training + ~150s eval/startup per cell ≈ 7–9 min wall each; 3 cells + smokes + (conditional) confirmation pair ≈ 35–55 min.
- Log output: per-cell `run_{cell}.log` in project root, parsed via `grep`; deleted after recording.
- Tool skill: none (local run).

## Abort Criteria
- Foreign job on GPU 1 (`nvidia-smi` non-our PID at high util / multi-GB) → throughput drop → kill, re-run FULL set once idle (infra-errors EXP-010/014).
- Any cell's loss NaN/inf or eval collapses (<50% past ep20) → kill, inspect (schedule bug).
- Run exceeds 600s wall → killed, treated as failure (goal hard cap).
- No eval line after ~120s → silent hang; kill and inspect.
- num_epochs < 135 on ANY cell while another sits ~150 → since the change is throughput-free, this signals contention (not the schedule) → infra-confound, re-run.

## Verification Protocol

### Verification Procedure
Baseline = **96.38** (`exp-index.sh baseline`). Bar = **96.48**. Run cells c0 → cA → cB in one session on GPU 1.

1. **Completion + budget (necessary)**: for each cell, `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:\|^num_epochs:\|^peak_vram_mb:" run_{cell}.log`. Empty `best_test_acc` ⇒ crash (read `tail -n 50`). Confirm `total_seconds` < 600. Timeout: kill any run > 600s wall.
2. **Under-anneal / equal-contention gate (necessary for trustworthy comparison)**: each cell `num_epochs ≥ 135` and in the clean ~142–154 band (the change is throughput-free, so all three should land near c0's epoch count); all equally uncontended via the background `nvidia-smi` sampler AND the in-log img/s trace. If any cell under-anneals (< 135) or either contention signal fires ⇒ infra-confound, re-run the full set; do NOT label no-improvement.
3. **Primary metric (necessary) — keyed on cA only**: `cA.best_test_acc` ≥ 96.48 AND cA − c0 > 0.1pp. If `cA < 96.48` OR `delta ≤ 0.1` ⇒ no-improvement. **cB is diagnostic/informational** — a cB-only win is a next-loop lead, NOT an EXP-020 improvement.
4. **Confirmation re-run (necessary on an apparent cA win)**: re-run {cA, fresh c0} as a second same-session pair; require the >0.1pp delta to replicate. Non-replicating ⇒ no-improvement (EXP-019 lesson).
5. **Full-anneal / ep25 sanity (necessary for a clean read)**: cA fully annealed (best≈final, LR→0); cos changes early-LR so ep25 WILL differ from c0 — judge on the annealed best, not ep25 parity. A still-climbing best==final at budget end ⇒ under-anneal confound (shouldn't happen — throughput-free).
6. **Integrity / anti-gaming (necessary, ALWAYS run — performed DURING verification while `run_*.log` still exist; logs are deleted only later in analyze housekeeping)**: `git diff --quiet -- prepare.py`; `git status --porcelain` lists only train.py (the `run_*.log` are gitignored, so this is consistent); exactly one `evaluator.evaluate` per epoch (loop unchanged); seed 42; summary `best_test_acc` == max over per-epoch `eval ep` lines for every run (computed from each log before deletion); `SCHEDULE=tri` LR-trace ≡ baseline (Smoke A).

**On a confirmed cA win — BAKE step (necessary for reproducibility)**: because the goal procedure runs the bare `CUDA_VISIBLE_DEVICES=1 uv run train.py` with NO env, change the committed default to cosine — set `SCHEDULE` default to `"cos"` (`os.environ.get("SCHEDULE", "cos")`) so the bare command reproduces the win — then run one bare-command confirmation (no SCHEDULE env) and verify best_test_acc matches the cA result within noise. The experiment cells still set env explicitly for the controlled comparison; the bake makes the default reproduce the winner.

Render: **improvement** only if conditions 1–6 pass for cA AND the cA confirmation pair replicates AND the bare-command bake reproduces it; else **no-improvement** (cB-only signal → next-loop lead); **invalid** on any integrity failure (prepare.py touched, >1 eval/epoch, seed hack, summary≠per-epoch-max); **crash** on no metrics or unresolved GPU-1 contention.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run_{cell}.log` — expect ~1635 MB (unchanged).
- num_epochs / training_seconds / total_seconds: `grep` — confirms throughput-free premise (all cells ≈ same epoch count) and the under-anneal gate.
- ep25 trajectory + full LR-anneal: first few `eval ep` lines + final LR — schedule-shape health.
