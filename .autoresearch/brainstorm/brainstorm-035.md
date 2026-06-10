# Brainstorm EXP-035
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Ideation only. Metric/direction/constraints/verification live in the goal file;
     baseline (96.22, commit 6c417a4) lives in experiment-indices/improve-cifar10-test-accuracy.tsv. -->

## Web Search & Literature Review

No new external sources consulted this loop — the relevant literature is already distilled in the knowledge base and goal-learnings. The grounding for this experiment comes from project-internal evidence:
- **DeVries & Taylor 2017 (Cutout)** and **Müller & Hutter 2021 (TrivialAugment)** — the augmentations being cooled down; already in the recipe (EXP-002/012).
- **SWA (Izmailov et al. 2018)** — the constant-LR-tail mechanism, tested and closed for top-1 here (EXP-019/020); informs why a *small re-annealed* (not constant) clean-phase LR is the right shape, not a flat floor.
- **"fine-tune on clean data at the end of augmented training"** — the FixRes / train-test resolution-discrepancy intuition (Touvron et al. 2019): augmentation creates a train/test distribution gap; a short clean-data tail re-aligns the model to the test distribution. This is the cooldown's mechanism (EXP-033/034).

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, commit 6c417a4); bar = 96.32 (+0.1). 34 experiments run; ~26 axes mapped.

**The single OPEN axis with a generalization (not polish) mechanism — augmentation cooldown** (EXP-033/034, goal-learnings Low Importance):
- EXP-033 @ COOLDOWN_FRAC 0.15 (fired ep77/frac0.85, base 95.43) → 96.10.
- EXP-034 @ COOLDOWN_FRAC 0.10 (fired ep83/frac0.91, base 96.05) → **96.26** (best non-baseline; +0.04 over baseline, within noise; loss 0.1951≈baseline). Later/shorter is strictly better (fine-tune from a higher base).
- **Decisive limitation found**: the cooldown's marginal lift over a full-aug cosine tail is only ~+0.04. The clean tail peaks ~4–9 clean epochs in (ep87) then DECLINES to ep91.

**What's CLOSED (do not revisit)**: width/depth/capacity past k=4 (EXP-004/009, compute wall); the ENTIRE LR-schedule axis as a *global* lever — peak 0.2 (EXP-016/017), floor cosine-to-0 (EXP-019/020), shape single-cosine (EXP-029 SGDR regressed −0.67pp); weight averaging EMA/SWA (EXP-006/019/020, count≥3, loss-not-top1); GC (EXP-030/031, loss-not-top1 polish); Bag-of-Tricks/LS-down (EXP-026/023, loss-not-top1); the whole augmentation FAMILY strength/policy (EXP-011/013/014/018/021); architecture (SE/SiLU/preact/ResNet-D/BlurPool/multi-scale-head); batch size (EXP-025).

**The untried GAP**: EXP-034's clean fine-tune fired at frac 0.91, where cosine LR ≈ 0.001–0.005 (near-frozen). The clean-distribution adaptation is **LR-starved** — the model cannot take meaningful gradient steps toward the clean-data optimum in the final 9% because the cosine schedule has already annealed to ~0. No experiment has given the clean-data tail its OWN LR budget. This is the specific, mechanistically-motivated untried lever.

## Candidate Ideas

### 1. Clean-tail LR reheat (augmentation cooldown @0.10 + small re-annealed LR 0.02→0 on the clean phase)
**Summary**: Take the proven EXP-034 config (augmentation cooldown at COOLDOWN_FRAC = 0.10: disable TrivialAugment+Cutout for the final 10% of the budget, keep RandomCrop+Flip). Add ONE new element: when the cooldown fires, override the LR for the remaining clean-data phase with a small re-annealed cosine from `CLEAN_LR0 = 0.02` down to 0 across the remaining budget, instead of letting the global cosine sit at its near-zero tail value (~0.001–0.005). Concretely, during the clean phase, `lr = CLEAN_LR0 * 0.5 * (1 + cos(pi * clean_progress))` where `clean_progress` runs 0→1 over the final 10%. Everything else identical to EXP-034 (params 4,299,866, throughput-neutral, ~91 ep, dt 8ms).

**Reasoning**: EXP-034 proved the cooldown reaches a clean-data state but gains only +0.04 over baseline. The diagnosed cause (goal-learnings Low Importance + EXP-034 report § Results): the clean fine-tune happens at near-frozen LR (cosine ≈ 0.001–0.005 at frac 0.91+), so the model can barely move toward the clean-distribution optimum. Giving the clean phase ~4–20× more LR (0.02 vs 0.001–0.005) lets it actually adapt weights+BN to the clean/test distribution, then re-anneal to 0 to settle (preserving the proven cosine-to-0 endpoint benefit). This targets a generalization mechanism (train/test distribution re-alignment, FixRes-style), NOT the closed loss-polish cluster. The 0.02 level is grounded: EXP-020's best SWA result used a 0.02 floor; it is small enough (10% of peak) to be gentle on the converged solution.

**Distinct from closed axes**: This is NOT the closed global LR-schedule axis (peak/floor/shape were swept on the *augmented* full-budget schedule) — it is a clean-phase-specific gentle re-anneal coupled to the cooldown. It is NOT SGDR (EXP-029 restarted to FULL peak 0.2 and split the budget 50/50, destroying cycle-1's minimum; this is a 0.02 micro-reheat in only the final 9% on *clean* data). It is NOT SWA (no weight averaging; single trajectory; re-annealed not constant).

**Sources**: EXP-034 report (reports/exp-report-034.md § Results — the near-zero-LR diagnosis); goal-learnings Low Importance (cooldown entry, "LEAD"); EXP-020 (SWA floor 0.02 = best weight-avg point); FixRes (Touvron et al. 2019, train/test distribution gap).

**Estimated Effort**: low — re-apply the 4 EXP-034 cooldown edits (documented, verified to fire correctly) + add ~4 lines for the clean-phase LR override. One run.

**Risk Assessment**: (a) A 0.02 reheat could mildly DESTABILIZE the converged solution (the SGDR lesson — re-raising LR can destroy a converged minimum). Mitigation: 0.02 is 10× smaller than SGDR's 0.2 restart and re-anneals to 0; worst case ~−0.2 to −0.5pp, not catastrophic. (b) It may just reproduce the loss-polish pattern (top-1 flat). (c) Throughput-neutral and safe (no NaN risk at 0.02). Worst case: clean no-improvement that definitively closes the cooldown axis.

### 2. GC + augmentation cooldown @0.10 (combine the two best compute-neutral near-misses)
**Summary**: Stack EXP-031's throughput-neutral compiled Gradient Centralization onto the EXP-034 cooldown@0.10 config. GC improves the optimization basin (loss 0.1894); cooldown re-aligns to the clean distribution (96.26). Hope they stack to clear 96.32.

**Reasoning**: The NEVER-STOP directive explicitly suggests "combine previous near-misses." These are the two best compute-neutral results (96.26, 96.14) and both improve loss via orthogonal mechanisms.

**Sources**: EXP-031 report (GC compiled), EXP-034 report (cooldown), directive guidance.

**Estimated Effort**: medium — re-apply BOTH the GC compiled code AND the cooldown code; higher edit surface and recompile risk.

**Risk Assessment**: WEAK top-1 mechanism — goal-learnings explicitly closes GC for top-1 ("GC joins the convergence-POLISH cluster: loss↓, top-1 flat"). Combining a top-1-flat polish lever with cooldown most likely yields loss-polish + cooldown's +0.04, i.e. ~96.2–96.3, not a breakthrough. Higher effort, weaker mechanism than #1.

### 3. Cooldown @0.07 bracketing probe (pre-staged axis-closer)
**Summary**: The pre-staged EXP-034 lead — re-run cooldown with COOLDOWN_FRAC = 0.07 (one constant change) to bracket the cooldown-window optimum.

**Reasoning**: The window trend is monotone (0.15→96.10, 0.10→96.26); 0.07 brackets the optimum.

**Sources**: EXP-034 report § Next Steps; goal-learnings Low Importance LEAD.

**Estimated Effort**: low — one constant change.

**Risk Assessment**: LOW confidence to clear the bar (my own EXP-034 report: "expect ~96.25–96.30, likely still < 96.32"). It is purely axis-closing — it tunes the WINDOW but does not address the diagnosed LR-starvation that caps the cooldown's gain. Lowest information of the three.

## Idea Evaluation

All three operate on the single open axis (augmentation schedule). The decisive question is which best addresses *why* the cooldown caps at +0.04.

- **Mechanism clarity & expected impact**: Idea #1 directly attacks the diagnosed root cause (LR-starved clean fine-tune) with a clear causal story — give the clean phase real LR so it can adapt to the test distribution. #3 only re-tunes the window (a knob already known monotone and converging to baseline) and does nothing about the LR starvation, so it cannot exceed ~96.30 by my own prior analysis. #2's mechanism is explicitly weakened by the goal-learnings (GC closed for top-1).
- **Evidence**: #1's LR level (0.02) is grounded in EXP-020's best SWA point and the FixRes distribution-gap rationale; its diagnosis is recorded in the EXP-034 report. #3 has the monotone-trend evidence but that same evidence predicts a sub-bar result. #2 has two near-miss data points but a closed-axis warning against the top-1 mechanism.
- **Risk**: #3 and #1 are both low-effort and safe; #1 carries a small destabilization risk (mitigated by the gentle 0.02 level + re-anneal-to-0). #2 is medium-effort with the highest edit/recompile surface and the weakest mechanism.
- **Information value**: #1 is the highest — it tests a real hypothesis; success breaks the plateau, failure definitively closes the augmentation-schedule axis (LR-starvation ruled out). #3 only closes the window sub-knob. #2 likely confirms the polish pattern.

Idea #1 wins on mechanism clarity, expected impact, and information value at comparable (low) cost.

## Chosen Idea
**Selected**: Clean-tail LR reheat (augmentation cooldown @0.10 + small re-annealed LR 0.02→0 on the clean phase)

**Why this idea**:
It is the only candidate that targets the *diagnosed* reason the cooldown caps at +0.04 — the clean-data fine-tune is LR-starved (cosine ≈ 0.001–0.005 at frac 0.91+), so the model cannot move toward the clean/test distribution optimum. Giving the clean phase its own gentle re-annealed LR (0.02→0) supplies the missing adaptation budget through a generalization mechanism (train/test distribution re-alignment), not the closed loss-polish cluster. It builds directly on the proven EXP-034 base (96.26), is a clean single-variable addition, is throughput-neutral and low-effort, and is mechanistically distinct from the closed global-LR-schedule, SGDR, and SWA axes. Either outcome is decisive: a gain breaks the plateau; a null closes the augmentation-schedule axis for good (LR-starvation ruled out as the cap).

**Hypothesis**:
Adding a small re-annealed LR (CLEAN_LR0 = 0.02 → 0) over the clean-data phase of the EXP-034 cooldown will let the model adapt to the clean/test distribution, lifting best_test_acc above EXP-034's 96.26 — targeting ≥96.32 (the bar) by converting the cooldown's currently-LR-starved +0.04 into a larger gain. Throughput-neutral (~91 ep, dt ~8ms, params 4,299,866). Falsified if best_test_acc ≤ ~96.30 (cooldown axis then closed) or if the 0.02 reheat destabilizes the converged solution (regression), which would cap CLEAN_LR0 lower.
