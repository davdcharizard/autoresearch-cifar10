# Sweep EXP-010: Muon conv-weight optimizer — PEAK_LR_MUON sweep
**Created**: 2026-06-29

<!-- Rationale + per-param justification live in 01-brainstorm.md (## Sweep Plan). -->

## Operating Context

- **Machine / resources**: single NVIDIA H20, GPU **1** (`CUDA_VISIBLE_DEVICES=1`); ~1.6 GB VRAM. 300s training-time budget (frozen in prepare.py).
- **Baseline operating point**: EXP-008 recipe (DavidNet/ResNet-9 + whitening + ReZero + EMA + flip-TTA + cutout12 + RandomErasing), baseline **96.38%** at ~150 epochs / ~25–26k img/s under full throughput. The sole change under test: conv weights use Muon (orthogonalized momentum, ns=3, airbench weight-renorm, no L2) instead of SGD; fc/BN/α stay on the unchanged EXP-008 SGD (lr 0.4, mom 0.9, wd 5e-4).
- **Other coupled config / IMPORTANT throughput note**: the 4 grid TRIALS ran during a window when **GPU 1 was contended by an external ~18.5 GB job** (PID 971585, 100% util), which **halved our throughput** to ~9.8k img/s → only **59–64 epochs/trial** (vs the normal ~140–150). So the trials' *absolute* accuracies are throughput-confounded (severely under-annealed) and NOT comparable to the 96.38 baseline; only their **relative ranking** (all trials equally slowed) is valid. The **confirmation run of the winner was executed later at restored full throughput** (~24.5k img/s, 132 epochs) and IS the comparable, fully-annealed result.

## Search Space & Optimizer & Budget (as used)

- **Optimizer**: grid (`itertools.product`)
- **Direction**: maximize (best_test_acc)
- **Trials**: 4 (MAX_PARALLEL: 1 — single GPU, sequential)
- **Parameters**:
  - `PEAK_LR_MUON`: {0.03, 0.06, 0.10, 0.14} (anchored below EXP-009's divergent 0.24; ~log spacing)
  - all else fixed at the EXP-009 verified-correct Muon recipe (ns_steps=3, momentum 0.9, weight-renorm, no Muon WD; PEAK_LR_SGD=0.4; EMA/whitening/TTA/aug/seed byte-identical to EXP-008)

## Trial Table

Full data: `experiments/010/trials.tsv` (+ per-trial logs in `experiments/010/trial_logs/`).
**⚠ Trial objectives below are at HALF throughput (~59–64 epochs, externally contended GPU) — valid for RANKING only, not vs baseline.**

| rank | trial_idx | PEAK_LR_MUON | best_test_acc (≈60ep, confounded) | num_epochs |
|------|-----------|--------------|-----------------------------------|------------|
| 1 | 0 | **0.03** | 95.90 | 59 |
| 2 | 1 | 0.06 | 95.54 | 59 |
| 3 | 2 | 0.10 | 94.42 | 64 |
| 4 | 3 | 0.14 | 93.33 | 64 |

Ranking is clean and monotonic: **lower PEAK_LR_MUON → higher accuracy and more stability** (cf. EXP-009's 0.24 which fully diverged to ~random). Winner = **0.03** (lowest grid point; the trajectory was stable, no collapse). The monotonic trend with the winner at the grid boundary hints the optimum could be ≤0.03, but the full-throughput confirmation (below) shows 0.03 is already at SGD's level.

## Confirmed Winner

Confirmation run at **restored full throughput** (GPU 1 free), `PEAK_LR_MUON=0.03`, full 300s budget, seed 42, all else byte-identical to EXP-008.

- **Winning parameters**: `PEAK_LR_MUON=0.03` (conv Muon group); `PEAK_LR_SGD=0.4`, `MUON_NS_STEPS=3`, momentum 0.9, weight-renorm, no Muon WD.
- **Primary metric**: **best_test_acc = 96.33%** (final 96.19%, peak at ep~125) — **baseline 96.38, delta −0.05pp**, below the +0.10 bar (96.48). (source: run.log summary; trajectory ep25 89.66 → ep50 92.15 → ep75 94.65 → ep100 95.81 → ep125 96.20/best 96.33 → ep132 96.19)
- **Necessary Conditions**:
  - NC1 (completes in budget, valid metric, ≤10min): **PASS** — RUN_EXIT=0, training_seconds=300.0, total 445.7s, best_test_acc=96.33% printed.
  - NC2 (≥ baseline +0.10pp = ≥96.48): **FAIL** — 96.33 < 96.48 (−0.05 vs baseline, within the ~0.1pp noise floor → a statistical tie, not a win). Anti-bookkeeping: max per-epoch trace 96.33 == summary best 96.33 (no tampering).
  - NC3 (genuine/in-scope): PASS — `git status --porcelain` only `M train.py`; prepare.py byte-unchanged; num_params 7,784,627 unchanged; seeds intact; ≤1 eval/epoch.
  - → fails NC2 ⇒ **no-improvement**. No fallback trial confirmed: trials 1–3 (0.06/0.10/0.14) were monotonically worse even at matched throughput and carry MORE high-LR stress at full throughput, so none can beat 0.03 — recording best-effort per sweep protocol.
- **Informational Metrics**: num_epochs 132 (vs EXP-008's 150 — img/s ~22.7–24.5k, mild residual host load; but schedule is time-based so the run fully annealed — best peaked at ep125 then dipped, i.e. converged, not truncated); peak_vram_mb 1635.3; num_params 7,784,627. (source: run.log summary)
- **Applied to**: `train.py` on branch `autoresearch/maximize-cifar10-test-accuracy-010` — `PEAK_LR_MUON` default set to 0.03 (env-overridable). Will be **discarded** in analyze (no-improvement).

## Errors & Dead Ends

### 2026-06-29 — sweep trials throughput-confounded by external GPU-1 contention
- Error: all 4 grid trials ran at ~9.8k img/s / 59–64 epochs (vs normal ~25k / ~150) because an external ~18.5 GB job (PID 971585) shared GPU 1 at 100% util during the sweep window.
- Root cause: shared-host GPU-1 contention (GPU 0 always busy on this box; GPU 1 intermittently borrowed by others).
- Resolution: ranking is throughput-invariant (all trials equally slowed) so the winner is still valid; the WINNER's comparable result comes from a separate full-throughput confirmation run after GPU 1 freed. Recorded for future sweeps: check `nvidia-smi` GPU-1 occupancy before trusting absolute epoch counts; prefer confirmation runs when GPU 1 is uncontended.

## Human Notes

> User confirmed GPU 1 was free before the winner-confirmation run was launched.
