# Report EXP-002: Widen to 8x (stage widths 128/256/512)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md
- **Plan**: plans/plan-002.md
- **Log**: logs/exp-log-002.md

## Goal
Maximize best_test_acc (%) on CIFAR-10 within the fixed 300s training budget, modifying only train.py. Direction: higher. Baseline at experiment start: 95.23% (EXP-001, 4× width, commit bd0976e). Question: does the width gradient measured at 4× (+2.07pp) continue to 8×?

## Idea & Hypothesis
Chosen over RandomErasing and a ResNet-9 swap: continue the capacity direction with WIDTH_MULT 4 → 8 (~17M params), recipe unchanged. Evidence: WRN width study (gains to 8–12× at 16-layer depth with full training) and the steep project-measured gradient. Hypothesis: best_test_acc ≥ 95.6%. Known risk, explicitly flagged in the brainstorm: undertraining at ~30–45 epochs.

## Approach
One-constant diff: `WIDTH_MULT = 8`. Everything else byte-identical to EXP-001. No deviations.

## Execution
Single run, clean (exit 0, 366.2s total, no errors). Early signal: 17,124,490 params confirmed; epoch-1 acc 22.63% (above abort threshold, below 4×'s 39.20% — first sign of slower convergence per wall-clock).

## Results

- **Primary metric**: best_test_acc 94.41% (baseline: 95.23%, delta: −0.82 pp, −0.86%)
- **Observations**:
  - The 8× net got only 40 epochs / 3,807 steps (vs 114 / 10,965 at 4×) — per-image cost grew ~2.9× while WRN's full-training gains assume convergence, which never arrived.
  - final = best (94.41%) — the model was still improving when the budget expired; this is undertraining, not overfitting or instability.
  - No divergence at peak 0.4 — the LR-transfers-across-width claim held; the failure is purely a compute-allocation one.
  - 94.41% at 40 epochs is still far above EXP-000's 93.16% — width works; 8× just overshoots the budget-matched optimum.
- **Analysis**: The hypothesis failed on its flagged risk. Under a FIXED TIME budget, capacity and epochs trade off through throughput: accuracy(width) is unimodal, rising while capacity gains dominate and falling when epoch starvation dominates. 4× (114 epochs) is left of or near the optimum; 8× (40 epochs) is right of it. The optimum plausibly sits at 5–6× width — OR at 8× with cheaper epochs (e.g., GPU-resident data pipeline raising img/s, recouping the epoch deficit). The width *idea* is not exhausted; this specific width/throughput combination is.
- **Key Learning**: Fixed-time-budget scaling is not fixed-epoch scaling — capacity increases must be paired with throughput increases or they starve the schedule; the width-epoch optimum at current throughput lies between 4× and 8×.

## Verification
- **Conditions**: condition 2 failed (best_test_acc 94.41 < 95.33); condition 1 passed (clean, 366.2s); condition 3 skipped per protocol
- **Review Notes**: result trustworthy — valid single-variable run, clean exit, metrics parsed from the standard summary block; failure mechanism (epoch starvation) clearly identified in the data
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure — metric below baseline+0.1 bar; no hard-constraint violation, no crash

## Unexplored Avenues
- **Intermediate width (5–6×)**: directly targets the unimodal optimum; ~6.7–9.6M params, ~60–80 epochs expected. Simple, likely positive, but possibly a small gain if the curve is flat near 4×.
- **8× width + GPU-resident data pipeline**: raise img/s so the big net gets enough epochs — attacks the root cause (throughput) rather than retreating on capacity. Host DataLoader is a known bound (goal-learnings § Patterns).
- **4× width + RandomErasing**: the orthogonal regularization lever, unaffected by this failure; EXP-001's train/test gap supports it.
- **Larger batch (1024) at 6–8× width**: better GPU utilization per step for wide nets; pairs with LR rescale.

## Next Steps
1. **GPU-resident data pipeline (+ keep 4× width this loop)** — high confidence for throughput, medium-high for accuracy: removes the host DataLoader bound (~2–3× more epochs at 4×, or enables 6–8× later); attribution stays clean.
2. **Intermediate width 6×** — medium confidence: targets the optimum directly but may land within noise of 4×.
3. **4× + RandomErasing** — medium-high confidence: orthogonal, cheap, supported by the observed train/test gap.

## Exit Action Results
