# Report EXP-020: Cosine one-cycle LR decay (vs linear triangular)
- **Created**: 2026-06-30

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within the fixed 300s training budget, editing only `train.py`. Baseline 96.38 (EXP-008, commit 07c3760). Improvement bar: best_test_acc ≥ 96.48 (baseline + 0.1pp) AND clearly above the same-session control beyond the ~0.1–0.2pp noise floor, replicated on a confirmation re-run.

## Idea & Hypothesis
Chosen idea (Codex idea-review scored it 8/7, the top finalist): replace the time-based **linear triangular** one-cycle LR decay with **cosine** decay over the same warmup→0 envelope. Selected because it was the only remaining cheap, throughput-free lever with BOTH an internal flag (EXP-012 noted schedule shape as untested) AND external evidence (MosaicML: cyclic/linear underperform cosine by up to ~0.5% val-acc on CIFAR-class CNNs; fastai one-cycle uses cosine). Hypothesis: cosine holds LR higher mid-training then spends more time in the very-low-LR tail, sharpening the final minimum and lifting clean test accuracy by ≥0.1pp at matched epochs — without changing per-step cost (so num_epochs is unchanged).

## Approach
Single-file change in `train.py`: added `import math`/`import os`; made `PCT_START` env-overridable (default 0.15); added a `SCHEDULE` env (default `tri`). In the LR block, inserted a `cos` branch (`q=(progress−PCT_START)/(1−PCT_START); lr=PEAK·0.5·(1+cos(πq))`) while keeping the `tri` else-branch byte-identical to the original linear formula — so `SCHEDULE=tri` is bit-for-bit baseline. The change is throughput-free (a few scalar ops per step), so epoch count should be unchanged. An on-win BAKE step (flip the default to `cos`) was pre-planned but, given the no-improvement verdict, was not executed.

## Execution
Two same-session pairs on GPU 1, all cells full 300s-budget runs under `timeout 600`, throughput sampled via background `nvidia-smi`.
- **Run 1 (session)**: c0 (tri) 96.04, cA (cos) 96.36, cB (cos, PCT_START=0.10 diagnostic) 96.13 — all 150 epochs. cA showed +0.32pp same-session, but c0 drew anomalously low (96.04 vs stored 96.38) and cA's absolute 96.36 < 96.48, so a confirmation pair was required.
- **Run 2 (confirmation)**: fresh c0b (tri) 96.35, cAb (cos) 96.39 — both 150 epochs. With a normal-draw control, the cosine lead collapsed to +0.04pp.
- One operational hiccup: the confirm runner was launched with a trailing `&` inside a `run_in_background` Bash wrapper, so the wrapper's completion notification fired before the work finished and an early grep misread a half-written log. Resolved by polling for the script's own `CONFIRM DONE` sentinel; no data lost (logged in 03-execute § Errors & Dead Ends).

## Results
- **Primary metric**: 96.39 (verdict-bearing cos cell, confirmation draw) vs baseline 96.38 (delta +0.01, +0.01%). Below the 96.48 floor.
- **Observations**: The cosine vs linear same-session delta was entirely draw-dependent: +0.32pp when the control drew low (c0 96.04), +0.04pp when the control drew normally (c0b 96.35). The cos cells themselves were remarkably stable across sessions (96.36 / 96.39); the *apparent* effect came from control variance, not from cosine. cB confirmed PCT_START=0.15 is the better cosine warmup (shorter 0.10 underperformed). All 5 cells hit exactly 150 epochs → the change is throughput-free as designed; peak_vram unchanged at 1635 MB.
- **Analysis**: The intervention achieved its intended LR-shape effect (smoke-verified: cosine spends ~2.8× more of the post-warmup window below 0.05·PEAK), but that local effect did not translate into a test-accuracy gain. Cosine is statistically **tied** with the incumbent linear schedule on this recipe — the schedule shape is not the binding constraint at 300s. This is the exact pattern as EXP-019 SE (+0.28 same-session that collapsed to +0.02 on confirmation): a low-control-draw inflates a one-session delta that evaporates under replication.
- **Key Learning**: LR-schedule shape (cosine vs linear) is a non-lever on this saturated recipe — the +0.32 same-session lead was a low-control-draw artifact that collapsed to +0.04 (tie) under a normal-draw confirmation.

## Verification
- **Conditions**: NC1 FAILED — cAb 96.39 < 96.48 floor, and the same-session delta did not replicate (+0.32 → +0.04, below the 0.1pp bar). Integrity/hard-constraints PASS (only train.py; SCHEDULE=tri byte-identical baseline; best==per-epoch-max all 5 cells; 150 epochs all cells; foreign GPU0 noise immaterial at img/s ≥25.3k).
- **Review Notes**: Results confirmed trustworthy. The two-pair design directly exposed the control-variance mechanism; no parsing/cache/seed anomalies (best matches per-epoch max everywhere).
- **Verdict**: no-improvement
- **Verdict Basis**: valid result, verification condition NC1 failed (metric below floor + delta did not replicate). No hard-constraint violation, so not invalid.

## Unexplored Avenues
- **Cosine combined with a higher peak LR**: cosine holds LR higher mid-training, so it may tolerate / benefit from a larger PEAK_LR than linear's 0.4. Untried; the current test held PEAK_LR fixed, which may have masked any cosine-specific headroom. Low-confidence — likely still ceiling-bound.
- **Other tail shapes** (e.g. linear-then-flat-floor, or a short constant low-LR tail): not meaningfully different in expectation from cosine given this result; treat the schedule-shape axis as closed.

## Next Steps
- **Treat the schedule-shape axis as exhausted** (high confidence) — cosine, the strongest-evidence variant, ties; do not revisit linear/cyclic/cosine permutations.
- **Pivot to a wholesale different backbone** (medium confidence) — 15 straight nulls across capacity, optimizer, all 3 input-aug mechanisms, reg-scalars, SAM, throughput, BN-noise, downsampling, channel-attention, and now schedule-shape strongly indicate the DavidNet/ResNet-9 family is at its generalization ceiling at 300s. The remaining high-EV move is a structurally different architecture (e.g. a wider-but-shallower or attention-augmented stem) that the saturated levers cannot reach.
- **Revisit the two near-misses for composition** (low confidence) — none individually cleared the bar, but stacking the largest deterministic-mechanism candidates is the only within-family option left.

## Exit Action Results
- No exit actions defined for this goal — skipped.
