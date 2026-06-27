# Report EXP-051: Confidence-weighted CE — detached w = p_true^0.7, mean-normalized (GCE-style aug-noise filtering)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-051.md
- **Plan**: plans/plan-051.md
- **Log**: logs/exp-log-051.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged budget by modifying train.py only. Baseline: **96.71** @ 1990397; bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16 (EXP-027).

## Idea & Hypothesis

Suppression-side test of EXP-050's destroyed-view hypothesis (heavy aug creates effectively-wrong-label views whose gradients corrupt training). Chosen idea: weight per-sample CE+LS by detached, batch-mean-normalized p_true^0.7 — the canonical GCE gradient geometry (Zhang & Sabuncu 2018) that sends p→0 samples' gradients toward zero (~25× suppression) while genuine samples keep ≈ full gradient. The sign-flip burden from EXP-050 was discharged by construction: this suppresses the exact population 050's uniform margin amplified. Hypothesis: best ≥ 96.81 if destroyed-view gradient corruption binds; branch (ii) mean band = hypothesis unsupported; branch (iii) < 96.42 = suppression negative (hard-genuine samples load-bearing).

## Approach

`GCE_Q = 0.7` constant; timed-loop loss → per-sample CE+LS (`reduction="none"`) × detached normalized weight (no_grad: softmax → gather → float pow → mean-normalize); warmup mirrored. Diff 1 file, +16/−4. CPU sanity passed first run: q=0 identity to 1e-6; weights (0.0398, 0.6156, 0.9647) for p = (0.01, 0.5, 0.95) with mean exactly 1; per-row logit gradient equals w_i × unweighted row gradient (allclose) — GCE geometry confirmed pre-launch; params exact; smoke decreasing. No plan deviations.

## Execution

Single pristine run: gates clear poll 1; D0=22.7ms; 29 post-gate windows 21.7–22.8ms, slow_streak 0; RC=0; 139 epochs / 13,417 steps / 300.0s charged / 469.7s total. No retries, no kills, no errors.

## Results

- **Primary metric**: best_test_acc 95.32 (baseline: 96.71, delta: −1.39, −1.44%)
- **Observations**: 95.32 = mean − 7.8σ — the largest valid active negative in the project's history. Distinct failure signature from EXP-050: test_loss ELEVATED (0.239 vs family 0.185; 050 was 0.150) and the tail still climbing at cutoff (best at the final epoch) — the undertrained shape. The intervention slowed effective learning rather than misdirecting a converged solution.
- **Analysis**: Pre-registered branch (iii), decisively. Root cause: on clean-label CIFAR-10, low-p_true views under heavy aug are overwhelmingly hard-but-GENUINE — the most informative gradient population — not destroyed-label noise. Weighting by p^0.7 built a permanent anti-curriculum: the model never receives full gradient on examples until it already finds them easy, so boundary refinement stalls and the budget ends mid-climb. Two conclusions: (1) the destroyed-view hypothesis from exp-report-050 is REFUTED as a binding limitation — if destroyed views dominated low-p mass, filtering them should have helped or been neutral; instead the genuine-hard population dominates and is load-bearing. EXP-050's own negative needs re-reading: the margin's damage was plausibly generic over-pressure on logit gaps (capacity diverted from boundary placement to gap inflation everywhere), not specifically destroyed-label amplification. (2) The per-sample loss-weighting class is now closed from BOTH sides with opposite CE signatures: uniform up-pressure → CE improves, accuracy −2.4σ (050); confidence-keyed down-pressure → CE degrades, accuracy −7.8σ (051). The recipe's plain mean CE+LS sits between two measured cliffs — the loss is at a local optimum in the per-sample-treatment dimension, mirroring every other audited axis. GCE's published gains presuppose dataset-level label noise; aug-induced "noise" is not that — augmented hard views still carry correct-label signal that the network can and does exploit.
- **Key Learning**: Low-confidence views under heavy aug are the load-bearing training signal, not noise — suppressing them (−7.8σ, undertrained signature) is far worse than over-pushing them (−2.4σ). The loss is at a measured local optimum in the per-sample dimension; destroyed-view hypothesis refuted.

## Verification

- **Conditions**: Integrity pre-condition PASSED (windows 21.7–22.8ms; 139 epochs; 13,417 steps within ~1% of family ledger; params exact; 300.0s; 139 evals ≤ 139; early trajectory family-shaped as predicted — weights ≈ uniform at low confidence). Condition 1 (best ≥ 96.81) FAILED: 95.32. Conditions 2–3 skipped per first-failure-stop (informationally pass: 469.7s ≤ 600; 139 ≤ 139).
- **Review Notes**: Results trustworthy — pristine profile, family throughput signatures, and the elevated-test-loss/still-climbing shape is exactly what a slowed-learning mechanism produces (no wiring-failure ambiguity; a no-op would read family). Eval untouched.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure — valid active-negative result (pre-registered branch iii).

## Unexplored Avenues

- **Band-pass weight w = 4·p(1−p) (brainstorm-051 Idea 2)**: now bracketed by BOTH failures it interpolates — it suppresses p→0 (051's measured −7.8σ damage direction is suppression of low-p mass) AND emphasizes p≈0.5 (050's boundary-emphasis direction, −2.4σ). No remaining mechanism argues it escapes both cliffs. Treat as closed by interpolation.
- **Anti-GCE (w = (1−p)^q, hard-example emphasis / focal)**: the mirror image — predicted negative by 050's amplification result plus the focal-loss caveat already documented in brainstorm-050. Closed by symmetry.
- **Stratified/class-balanced batches**: still law-priced (gradient-noise law); documented rejection in brainstorm-051 Idea 3.

## Next Steps

1. **Declare the loss axis fully closed and stop sampling it** (high confidence): plain mean CE+LS is bracketed by measured cliffs in target-distribution (036, 009), logit-geometry (050), and per-sample-weighting (051) directions — the brainstorm frontier must move off the loss entirely.
2. **Re-read EXP-050's negative as generic over-pressure, not destroyed-label amplification** (medium confidence): 051 refuted the destroyed-view mechanism; update the goal-learnings entry so future candidates aren't built on the refuted reading.
3. **Remaining unfalsified space after 45 consecutive non-improvements** (high confidence): compound interventions of certified components (EXP-009 precedent against) and nothing else constructible under the measured laws — the next brainstorm should weigh whether any compound has a mechanism story that survives all laws simultaneously, per the standing directive to keep generating candidates.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
