# EXP-019: Whitening init for conv1 (patch-eigenvector filters ± negations, learnable)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md
- **Plan**: plans/plan-019.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-019
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (run completed cleanly; necessary condition 1 not met — 96.45 < 96.81)

## Implementation Notes

### Summary

One 20-line block inserted in `main()` between model creation and `base_model = model`, exactly per plan: 5000 raw `train_set.data` images → mean-normalized → 500k disjoint 3×3 patches (stride-3 unfold, deterministic, zero RNG consumed) → 27×27 covariance → `eigh` → filters vᵢ/√(λᵢ+1e-4) written into `conv1.weight[:27]` and their negations into `[27:54]`; filters 54–63 keep Kaiming init. Weights remain learnable and in the decay group. Syntax checked via `ast.parse`; numerical pre-validation run on CPU before launch (below). Diff is +20 lines, nothing else.

### Surprises & Discoveries

- **The plan's pre-validation criterion (output covariance ≈ I) was wrong as written**: natural-image patch covariance has a steep eigenspectrum (λ from 1.34 down to 1.1e-5), so the ε=1e-4 floor intentionally suppresses near-null directions — 12 of 27 directions get >10% variance suppression, and the smallest-λ direction sits at ~0.10 output variance, "failing" the |diag−1|<0.05 check while off-diagonals passed at 0.0012. This is the designed noise-floor behavior, not a bug, and bn1 absorbs per-channel scale regardless.
- Re-validated against the ANALYTIC expectation diag = λᵢ/(λᵢ+ε): max error 8e-5, off-diag 0.0012 — implementation exactly matches the intended math. Filter L2 norms span 0.86–94.8 (the 1/√λ scaling); bn1 normalizes each output channel so the scale spread is absorbed.

### Decisions

- **Kept ε=1e-4 and corrected the validation criterion instead of shrinking ε** (straightforward fix, no plan-uncovered design decision): shrinking ε to make diag ≈ I would amplify near-null patch directions by up to ~300x, turning those stem channels into BN-amplified noise — the opposite of the information-adding mechanism. The suppression is what airbench's ε does too. Validation now asserts (a) exact decorrelation (off-diag < 0.05) and (b) diag matches λᵢ/(λᵢ+ε) to <0.01 — both passed.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: b22ult0yv (composite background script: pre-check + train + inline watchdog)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 10:54:28 (GPU 0 confirmed free at launch by the script's pre-check)
- **Ended**: 2026-06-10 11:02:43 (TRAIN_EXIT rc=0)

Description:
- Single 300s-budget run of the baseline recipe with whitening-initialized conv1 (27 eigenvector filters + 27 negations + 10 Kaiming), on GPU 0 via the standard composite launcher. Tests whether a data-aligned decorrelating stem at init raises learning per unit of early schedule heat — the only intervention class exempted from the EXP-018 deferral closure, and the only one with wall-clock-regime evidence (airbench/hlb speedruns). Expected: early trail FASTER than EXP-017's 63.76@5 / 75.06@8; signatures unchanged (params 4,286,026, dt ≈ 22.4ms, ~139 epochs, VRAM ≈ 1613MB; startup +2–5s for the one-time eigh); best_test_acc ≥ 96.81.

Observations:
- Pristine execution: zero watchdog SLOW events; 0 of 267 windows >30ms, mean 22.4ms; 139 epochs / 13,418 steps; startup 12.3s — the covariance+eigh cost is unmeasurable (baseline 13.2s); VRAM 1613.0MB, params 4,286,026 — all signatures byte-identical to baseline, attribution perfect (source: task b22ult0yv output; run.log; profile command).
- **The faster-onset prediction did NOT materialize**: ep1 38.95 vs EXP-017's 38.20, ep5 53.80 (bouncy dip) vs 63.76, ep10 75.84 vs ~75 — the early trail is within eval noise of the baseline-family comparator, neither faster (hypothesis) nor slower (EXP-018's inversion). The whitening stem produced no visible early advantage (source: run.log eval trail eps 1–10).
- Mid-run on-family (87.08@60, 93.23@100) and a proper converged plateau: final six evals 96.37–96.45, final ≈ best (Δ0.02) — converged flat at −0.26pp. Notably final_test_loss 0.1888 is among the best on record, but the max-statistic pays in accuracy, not loss (source: run.log eval trail).
- Mechanism reading: the wash-out risk from the plan materialized, with a sharper architectural explanation — in our net `conv1` is immediately followed by `bn1`, which renormalizes every stem channel; airbench/hlb feed their whitening conv into an activation DIRECTLY (no BN between), so the variance-equalization half of whitening is precisely the part bn1 already does/undoes. What remains is only the data-aligned basis, and at 139 epochs (vs airbench's ~10) the stem learns an equivalent basis early enough that nothing is gained.

Key Metrics:
- best_test_acc: 96.45% @ ep 137/139 (source: run.log summary + eval trail) — baseline 96.71, bar 96.81: −0.26pp
- total_seconds: 487.6 | training_seconds: 300.0 | startup_seconds: 12.3 (source: run.log summary)
- num_epochs: 139 | num_steps: 13,418 | windowed dt mean 22.4ms, 0/267 >30ms (source: run.log + profile)
- peak_vram_mb: 1613.0 | num_params: 4,286,026 — byte-identical to baseline (source: run.log summary)
- final_test_acc: 96.43% | final_test_loss: 0.1888 (source: run.log summary)

## Verification Results

<!-- Filled after the experiment completes successfully.
     If ANY necessary condition fails, remaining conditions are not evaluated. -->

### Conditions Checked

- **Pre-condition (contention sanity)**: num_epochs 139 = clean projection exactly; 0 of 267 windows >30ms — CLEAN, analyzable (source: profile command output).
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: `grep "^best_test_acc:" run.log` → 96.45%. **FAILED** (−0.26pp vs baseline).
- **Condition 2 — total ≤ 600s, rc=0**: skipped — aborted after prior failure (informally: 487.6s, rc=0 — would have passed).
- **Condition 3 — eval at most once per epoch**: skipped — aborted after prior failure (informally: 139 eval lines = 139 epochs — would have passed).

### Informational Metrics

Not collected per protocol (necessary condition failed). Informal values from run.log for the report: peak_vram_mb 1613.0; num_epochs 139; num_params 4,286,026 — all identical to baseline as the init-only claim requires.

## Errors & Dead Ends

<!-- Append only. Never delete. Agent reads this before proposing any next action.
     Include source pointers for traceability. -->

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
