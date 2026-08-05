# EXP-008: Stronger augmentation — Cutout 8→12 + light RandomErasing

## Execution

Overall Status & Info:
- **Created**: 2026-06-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-008
- **Commit**: 07c3760 (on experiment branch; merged to integration branch)
- **PR**: N/A — local-only repo (no git remote); no PR created
- **Outcome**: completed (valid run; best_test_acc 96.38% ≥ 96.10 bar → **improvement**, +0.38pp, throughput preserved)

## Implementation Notes

### Summary
Implemented the EXP-008 augmentation change on the EXP-004 base (96.00%). Two edits inside the `train_tf` Compose in `main()` (`train.py:205-213`): `Cutout(8)`→`Cutout(12)` and an appended `transforms.RandomErasing(p=0.25, scale=(0.02,0.15), ratio=(0.3,3.3), value=0.0)` as the final transform. Everything else (architecture, PEAK_LR=0.4, schedule, optimizer, EMA, whitening, batch size, TTA, `forward`, seeds) is byte-identical to EXP-004. Milestone 1 passed: `py_compile` clean; `git diff ae31206 -- train.py` confined to exactly those two lines; `git diff --name-only ae31206` lists only `train.py`; `ls *.py` = `prepare.py train.py` (no stray importable module); `PEAK_LR`=0.4 unchanged. In-process smoke confirmed: the exact pipeline produces a finite `[3,32,32]` tensor in normalized range; `RandomErasing(p=1.0,value=0.0)` does zero a rectangle (mechanism live); `Cutout(12)` zeroes ≤144 px (clipped); and **`num_params == 7,784,627` UNCHANGED** (augmentation-only change ⇒ zero param/architecture delta — the C2 cross-check).

### Surprises & Discoveries
- None. `transforms.RandomErasing` is a first-class torchvision transform (torchvision 0.24.1) usable directly in `Compose` on the post-Normalize CHW tensor; value=0.0 = mean-fill in the std=1 normalized space, matching the existing `Cutout` zero-fill convention. No new dependency.
- Confirmed (per plan review concern #2) the timing model: `t0` is set AFTER the DataLoader yields each batch, and `total_training_time` accumulates only GPU-step `dt`, so DataLoader-wait is OFF the 300s budget — a worker bottleneck would inflate WALL (`total_seconds`), not cut `num_epochs`. `total_seconds` is therefore the primary worker-saturation diagnostic.

### Decisions
- **Held all HPs (PEAK_LR=0.4) and architecture fixed** for clean single-variable attribution vs EXP-004; the only change is the augmentation pipeline.
- **RandomErasing settings kept moderate** (p=0.25, area ≤15%) per the plan — enough combined regularization (with cutout12) to plausibly clear the ~0.1pp noise floor, while light enough to avoid tipping ~142 epochs into under-fit. The under-fit risk is monitored via the ep25 trajectory, not pre-emptively softened.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (local, background) — exit code captured to run_exit.txt
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-28
- **Ended**: 2026-06-28 (exit 0)

Description:
- Official EXP-008 run on GPU 1. Cutout 8→12 + light RandomErasing, all else byte-identical to the EXP-004 recipe. Tests whether spending the diagnosed epoch surplus on stronger (throughput-free, CPU-side) augmentation lifts `best_test_acc` from 96.00% to ≥96.10%. Expectation: num_epochs ~142–150 and total_seconds ~440s (throughput preserved); early/mid trajectory at or modestly below EXP-004 (ep25 near ~92.6%) with the annealed tail closing the gap and finishing higher.

Observations:
- **IMPROVEMENT: best_test_acc 96.38%** (baseline 96.00%, **+0.38pp**) — ~4× the ~0.1pp noise floor, a decisive clear of the 96.10 bar. Clean run, exit 0. (source: run.log:372 `best_test_acc: 96.38%`)
- **Throughput FULLY PRESERVED — the hypothesis's key prediction confirmed.** num_epochs **150** / 14480 steps (in the EXP-004/006 142–150 band), total_seconds **447.6** (≈ EXP-004's ~440s — NO worker-saturation wall inflation), steady img/s ~25.3–25.6k (≈ EXP-004's ~26k). So the stronger CPU-side augmentation did NOT slow the GPU step or eat off-budget wall — it is genuinely throughput-free, unlike the EXP-005/007 capacity adds. (source: run.log step lines + summary; num_epochs=150, total_seconds=447.6)
- **Trajectory confirms the mechanism (slower convergence → higher annealed ceiling), NOT under-fit.** ep25 92.31% (vs EXP-004's ~92.6% — modestly below, the expected "harder-but-not-broken aug" signature), ep50 93.75%, ep100 95.13%, then the annealed tail closes and OVERTAKES: ep147 96.32 → ep150 96.38. The early dip is mild (~0.3pp at ep25), not the collapse that would signal too-strong augmentation. (source: run.log eval ep 25/50/100 + tail)
- Mild still-rising tail at the very end (ep147 96.32 → ep150 96.38, best==final) but at a level +0.38pp above baseline — indicates possible additional headroom with a touch more annealing/epochs, not a failure. peak VRAM 1635.4 MB (unchanged), startup 1.2s, num_params 7,784,627 (unchanged — augmentation-only). (source: run.log:373-381)

Key Metrics:
- best_test_acc: 96.38% @ ep150 (source: run.log:372; max per-epoch test_acc=96.38 == summary)
- final_test_acc: 96.38% @ ep150 | final_test_loss: 0.3120 (source: run.log:373,374)
- training_seconds: 300.0 | total_seconds: 447.6 | startup_seconds: 1.2 (source: run.log:375-377)
- num_epochs: 150 | num_steps: 14480 (throughput preserved; EXP-004 ref 142) (source: run.log:379,380)
- peak_vram_mb: 1635.4 | num_params: 7,784,627 (both unchanged vs EXP-004) (source: run.log:378,381)
- trajectory: ep25 92.31% / ep50 93.75% / ep100 95.13% / ep150 96.38% (source: run.log eval lines)

## Verification Results

### Conditions Checked

1. **C1 — Clean run within wall guard** — PASS. `RUN_EXIT=0` (not 124); exactly one `^best_test_acc:` line; `total_seconds 447.6` < 600. (source: run_exit.txt, run.log:372,376)
2. **C2 — Full training budget + scope/integrity** — PASS. `training_seconds 300.0` ≥ 295; `prepare.py` byte-unchanged vs ae31206; tracked diff = only `train.py`; `ls *.py` = `prepare.py train.py` (no stray importable module); `git status --porcelain` tracked = ` M train.py` (plus untracked `run_exit.txt`, a non-`.py` artifact; `run.log` is gitignored); diff confined to the two augmentation lines (Cutout 8→12 + RandomErasing); `PEAK_LR`=0.4 unchanged; `num_params 7,784,627` == EXP-004 (augmentation-only, nothing structural moved). (source: git, run.log:381)
3. **C3 — Improvement ≥ +0.1pp + genuineness** — **PASS → improvement**. `best_test_acc 96.38%` ≥ bar 96.10% (**+0.38pp** vs the 96.00 baseline, well clear of the ~0.1pp noise floor — NOT a sub-noise null). Genuine: max per-epoch `test_acc` 96.38 == summary; exactly one `evaluator.evaluate(` site (≤1 eval/epoch); seeds `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)` unchanged; gain from the augmentation change only. (source: run.log)

**All necessary conditions PASS → verdict: improvement.** Attribution is clean (num_epochs 150 and total_seconds 447.6 both in their normal bands → no throughput confound; the +0.38pp is attributable to the augmentation).

### Informational Metrics
- num_epochs / num_steps: **150 / 14480** (source: run.log:379,380) — in the 142–150 band; GPU-step throughput unaffected.
- total_seconds: **447.6** (source: run.log:376) — ≈ EXP-004's ~440s; no worker-saturation wall inflation (the primary throughput-free check).
- img/s: ~25.3–25.6k steady (source: run.log step lines) — ≈ EXP-004's ~26k.
- mid-trajectory: ep25 92.31% / ep50 93.75% / ep100 95.13% (source: run.log) — modestly below EXP-004 early (slower convergence from harder aug), not a collapse.
- peak_vram_mb: 1635.4 (source: run.log:378) — unchanged.
- num_params: 7,784,627 (source: run.log:381) — unchanged (augmentation-only).

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
