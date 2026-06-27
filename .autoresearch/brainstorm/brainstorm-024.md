# Brainstorm EXP-024
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only. -->

## Web Search & Literature Review

No new searches. Grounding from prior loop documents:

- **Sutskever et al. 2013** (effective-step framing, already cited in brainstorm-023): lr/(1−β) is the first-order step size; EXP-023 just validated empirically that this matching holds in practice here (signatures byte-identical to baseline at β 0.95 + peak 0.2).
- **Keskar et al. 2016 / Smith & Le 2018 (noise-scale literature, known context)**: SGD noise scale ~ lr·(dataset/batch); flat-minima accounts predict an optimal noise band for generalization. The campaign now has the reduction side of that band measured in-project three ways.
- **In-project laws**: deferral (8 confirmations), numerics equivalence (EXP-021), max-statistic plateau (EXP-011/016), gradient-noise law NEW (EXP-011/022/023 — reductions lose even when free in heat/epochs/numerics).

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006). **Eighteen consecutive misses (EXP-007…023).**
- **EXP-023 completed the recipe certification**: every train.py constant is now bracketed alone or in a compensated trade. The EXP-006 recipe is a certified local optimum over its entire visible parameter set.
- **The gradient-noise law is one-sided**: EMA (−0.25), 2× batch (−0.14), 2× momentum horizon (−0.30) — all REDUCTIONS. No experiment has ever measured the INCREASE side at held first-order step. The noise curve's shape above baseline noise is the last unmeasured direction in recipe-space.
- **Methodology validated**: EXP-023 proved the compensated-trade design works as intended — signatures byte-identical to baseline, isolating the noise variable cleanly. The mirror trade inherits this design.
- **Standing exclusions checked**: variance-increasing TAIL tricks (LR floor) are barred as reward hacking — they harvest the max-statistic via eval noise. A full-trajectory noise trade is different in kind: it must win by raising the CONVERGED PLATEAU LEVEL, judged exactly like every prior experiment. Also rejected at filter stage: TTA-wrapper around base_model (games the frozen eval interface AND ~doubles per-eval cost, busting the 600s cap); stochastic depth (a 5th regularizer — pressure axis closed with four points, sign predicted negative).

## Candidate Ideas

### 1. Noise-increasing momentum trade: MOMENTUM 0.9→0.8 + PEAK_LR 0.4→0.8 (lr/(1−β) = 4 held)
**Summary**: The exact mirror of EXP-023 — halve the averaging horizon (1/(1−β): 10→5 steps) and double peak LR so the first-order effective step is unchanged at every schedule point. Two constants; execution byte-identical; batch 512, default compile, same kernels.

**Reasoning**: The noise law has three measured points below baseline noise and zero above it. If baseline noise is below the generalization optimum, this run gains; if baseline IS the optimum, this run loses symmetrically (~−0.3) and the noise curve is bracketed once and for all — either outcome is the most informative measurement still available in recipe-space. The mechanism sign is the only one currently unexcluded by all four laws: free in first-order heat (by construction), epochs (no execution change), numerics (same kernels), and on the noise axis it moves in the one untested direction. Methodology is de-risked: EXP-023 demonstrated the trade design produces baseline-identical signatures, so any metric difference is attributable to the isolated variable.

**Sources**: exp-report-023.md § Next Steps #1 and § Unexplored Avenues (cold direction flagged genuinely open); goal-learnings § Patterns (GRADIENT-NOISE LAW entry); Sutskever et al. 2013.

**Estimated Effort**: minimal — two constants; standard batch-512 launcher; expected signatures identical to baseline (dt ~22.4ms, ~139 epochs, VRAM ~1613MB).

**Risk Assessment**: Main risk is per-step instability: peak lr 0.8 per-step magnitude is 2× baseline (EXP-012 saw a bouncy-but-convergent hot phase at lr 0.8 with β 0.9 at batch 1024; here the momentum buffer is shorter, which damps oscillation buildup but raises per-step noise). Divergence is unlikely (effective step unchanged; bf16 has wide range) and the abort criteria catch it; the likely failure mode is a converged plateau below baseline — graceful, and it completes the noise bracket. Integrity: must win via plateau LEVEL, not plateau variance — analysis will check the final-evals spread against baseline's (±0.1pp) and treat a variance-inflated max with a flat-or-lower plateau median as no-improvement in substance.

### 2. Smaller-batch noise increase: BATCH_SIZE 512→256 + PEAK_LR 0.4→0.2 (linear-down scaling)
**Summary**: Raise SGD noise via smaller minibatch at proportionally scaled LR — the other lever on the noise scale lr·N/batch.

**Reasoning**: Same noise-law sign as Idea 1, via the batch lever instead of momentum. But it is NOT free in epochs: dt does not halve with batch on H20 (throughput sublinearity, project-insights EXP-007 law) — at ~14–16ms/step for 98 steps/epoch the epoch time likely RISES vs baseline, paying the deferral tax that has killed 8 changes; also doubles loader pressure (more batches/s) against the ~3% CPU margin (infra-errors EXP-013).

**Sources**: goal-learnings (batch axis entry — closed at 2×, untested at 0.5× but the throughput penalty is measured physics); project-insights (deferral High; throughput-exponent Medium).

**Estimated Effort**: low (two constants) plus a mandatory dt spot-measurement gate (kill if projected epochs <135).

**Risk Assessment**: Pays epochs for the same mechanism Idea 1 gets free — strictly dominated by Idea 1 unless momentum-noise and batch-noise act differently (no evidence they do at this granularity). Run only if Idea 1's outcome motivates a second point on the increase side.

### 3. Loss-asymmetry late phase: anneal LABEL_SMOOTHING 0.1→0 over the final 15% of the budget
**Summary**: Keep LS 0.1 during the noisy/hot phases (where its regularization pays) and linearly remove it in the cold tail so the final converged weights optimize the true 0/1-calibrated objective the evaluator measures.

**Reasoning**: Out-of-recipe (a schedule on a LOSS constant, never tried — only constant LS was probed); literature (Müller et al. 2019) shows LS hurts calibration/logit sharpness, which matters most exactly at convergence; the change costs zero dt, zero epochs, identical kernels (label_smoothing is a scalar in F.cross_entropy).

**Sources**: Müller et al. 2019 "When does label smoothing help?" (known context); goal-learnings (regularization axis peaked at constant dose — this varies the DOSE over time, a different approach).

**Estimated Effort**: low — one-line change inside the loop (`ls_now = 0.1·min(1,(1−progress)/0.15)`-style), plus passing it to F.cross_entropy.

**Risk Assessment**: The cold tail is where the converged plateau forms; changing the objective there directly perturbs the plateau the metric harvests — could sharpen logits (helping accuracy at the margin) or destabilize the BN-coupled equilibrium (EXP-016 showed late-phase dynamics dominate the metric). Within the pressure-axis-peaked reading, removing pressure late is a mild pressure DECREASE — EXP-015 measured pressure-down as −0.30. Sign genuinely uncertain; graceful failure.

## Idea Evaluation

**Evidence strength**: None of the three has comparable-regime external evidence — recipe-space is past that frontier. Idea 1 stands on the strongest in-project scaffolding: a freshly validated experimental design (EXP-023's compensated trade, signatures proven baseline-identical) pointed at the one unmeasured direction of a law built from three in-project measurements. Idea 2 uses the same law but pays a measured tax (throughput sublinearity + deferral). Idea 3 has plausible external theory (Müller et al.) but its in-project priors cut against it (pressure-down −0.30; late-phase perturbations dominate the metric, EXP-016).

**Mechanism clarity**: Idea 1: raise gradient-noise scale at held first-order step → if baseline noise < generalization optimum, converged plateau rises; binary, falsifiable, and informative in BOTH outcomes (completes the noise bracket). Idea 2: same sign, but confounded by epoch loss. Idea 3: two opposing mechanisms (sharper logits vs late pressure-drop) — sign unclear.

**Expected impact**: Idea 1 if the increase side pays: +0.1–0.3pp plausible (mirror-symmetric to the measured reduction losses). Ideas 2–3: smaller or negative.

**Risk profile**: Ideas 1 and 3 graceful (converged no-improvement); Idea 1 carries a small instability tail-risk caught by abort criteria. Idea 2 risks cap-bust via loader pressure plus guaranteed epoch loss.

**Feasibility**: Ideas 1–2 are two-constant diffs; Idea 3 is a small loop change. Idea 1's signature check (must equal baseline) doubles as the strongest possible contamination/premise gate.

## Chosen Idea
**Selected**: Noise-increasing momentum trade — MOMENTUM 0.8 + PEAK_LR 0.8 (lr/(1−β) = 4 held)

**Why this idea**:
It probes the single unmeasured direction left in recipe-space — the increase side of the gradient-noise curve — using an experimental design validated one loop ago to isolate exactly this variable. All four campaign laws are satisfied or untested in its favor: free in first-order heat, epochs, and numerics; noise sign is the only one not yet excluded. Either outcome is maximally informative: a win breaks the 18-miss streak through a mechanism the campaign derived itself; a loss brackets the noise curve and certifies baseline as the noise optimum, cleanly ending recipe-space exploration.

**Hypothesis**:
With the effective step held at baseline, halving the averaging horizon raises effective gradient noise; if baseline sits below the noise optimum for generalization, the converged plateau LEVEL rises — **best_test_acc ≥ 96.81 with a final-7-evals plateau whose median is at-or-above baseline's** (win must come from plateau level, not variance). Signatures must be byte-identical to baseline (dt ~22.4ms, ~139 epochs, VRAM ~1613MB, params 4,286,026); a hot bouncy mid-phase is acceptable per EXP-012 precedent so long as the tail converges. A converged miss certifies the baseline noise scale as the optimum and closes recipe-space entirely.
