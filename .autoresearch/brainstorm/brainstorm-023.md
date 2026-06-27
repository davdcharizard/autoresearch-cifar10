# Brainstorm EXP-023
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only. -->

## Web Search & Literature Review

No new searches. Grounding already in the knowledge base / prior loop documents:

- **Sutskever et al. 2013 ("On the importance of initialization and momentum")**: with momentum SGD the relevant first-order quantity is the effective step lr/(1−β); configurations matched on it explore similar loss-surface scales while differing in gradient-averaging horizon (1/(1−β) steps). This is the canonical justification for trading β against peak LR at constant effective step.
- **knowledge/README (cifar10-fast lineage)**: short-budget CIFAR recipes near-universally use β=0.9 with one-cycle; β=0.95+low-LR variants appear in some speedrun configs (airbench uses high-β small-LR for its few-epoch regime). No comparable-regime evidence at ~139 epochs — this is acknowledged.
- **In-project laws governing this ideation**: deferral (project-insights High, 8 confirmations), numerics equivalence (EXP-021), max-statistic plateau preference (EXP-011/016), batch axis closed at both LR points (EXP-012/022).

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006). **Seventeen consecutive misses (EXP-007…022).**
- **EXP-022 closed the last open measured axis**: batch 1024 bracketed at both canonical LR rules (√ 0.566 → 96.57, linear 0.8 → 96.66) — the deficit is the batch's own gradient-noise reduction, unfixable by LR tuning. Batch 512 is THE optimum.
- **Axis inventory is now fully closed with both-sided measurements**: constants (heat ±, pressure ±, smoothing), capacity (±, allocation), schedule (heat, warmup, family), init (±), topology (shortcuts, depth/width), batch×LR-rule, throughput (numerics tier). 
- **The one constant never touched in 23 experiments: MOMENTUM (0.9 since EXP-000)**. It was never probed alone (a bare β bump is a heat increase — closed) but the HEAT-CONSTANT trade (β 0.95 + peak 0.2, holding lr/(1−β)=4) has been queued in exp-report-021/022 § Next Steps as the last untried in-recipe candidate.
- **Other listed untried gaps**: width asymmetry — widen stage 3 only (goal-learnings EXP-017 entry names it "the one untried capacity-where-cheap move"); data-order interventions (never probed, no evidence either way).

**Synthesis check**: seventeen misses; the honest posterior is that 96.71 is the family optimum and everything remaining is low-EV. Per the standing directive the loop continues, so the ranking criterion is mechanism quality under the established laws. The momentum trade is the ONLY remaining candidate that is free in BOTH early heat (first-order heat held constant at every progress point by construction) AND epochs (zero execution change — foreach SGD takes β as a scalar; dt unchanged) AND numerics (same kernels, same batch, same compile mode). Nothing else surviving has that property.

## Candidate Ideas

### 1. Heat-constant momentum trade: MOMENTUM 0.9→0.95 + PEAK_LR 0.4→0.2 (lr/(1−β) = 4 held)
**Summary**: Double the gradient-averaging horizon (1/(1−β): 10→20 steps) while halving peak LR so the first-order effective step lr/(1−β) is unchanged at every point of the time-keyed schedule (warmup and cosine scale multiplicatively in PEAK_LR). Two constants in `train.py`; execution byte-identical otherwise.

**Reasoning**: The effective-step match neutralizes the deferral objection by construction: lr(p)/(1−β) is identical to baseline for all p, so no phase of the schedule gets hotter or colder to first order; dt, epochs, kernels, batch and compile mode are untouched, so the epochs/numerics objections are void too. What changes is second-order: averaging over ~20 steps instead of ~10 smooths gradient noise, which plausibly (a) lowers trajectory variance during the hot phase (less wasted bouncing) and (b) yields an equal-or-longer converged plateau for the max-statistic to harvest. It is also the completion of the constant-bracketing program — after this, every constant in the file has been probed alone or in a compensated trade.

**Sources**: exp-report-022.md § Next Steps #1; exp-report-021.md § Next Steps #1; Sutskever et al. 2013 (effective-step framing); goal-learnings § Patterns (heat closure EXP-010/014/015 — why a bare β bump is inadmissible).

**Estimated Effort**: minimal — two constants (`MOMENTUM`, `PEAK_LR`); standard launcher; signatures must equal baseline exactly (dt ~22.4ms, ~139 epochs, VRAM ~1613MB, params 4,286,026).

**Risk Assessment**: The first-order match is not exact dynamics-equivalence: momentum's transient response differs (longer memory across the warmup ramp; interaction with BN effective-LR and Nesterov lookahead), so the run may land slightly off the heat optimum in either direction — EXP-010/014/015 say ±heat costs 0.2–0.6pp. Failure is graceful (converged no-improvement) and closes the momentum axis, completing the recipe's certification. EXP-011's law cuts against pure smoothing (EMA collapsed the max by trading variance for mean) — the counter-argument is that EMA smoothed the EVALUATED weights only, while higher β smooths the SEARCH DIRECTION, which can improve the mean rather than just clip variance. Prior: low (~10%).

### 2. Width asymmetry: widen stage 3 only, (64,128,256) → (64,128,320)
**Summary**: +25% stage-3 width at 8×8 resolution (cheap FLOPs), all stages keep depth; 320 is 64-aligned (H20 law). The "capacity where it is cheap" move goal-learnings explicitly left open after EXP-017.

**Reasoning**: Stage-3 convs run at 8×8 so the FLOPs cost of widening is ~¼ that of earlier stages; alignment is satisfied; EXP-017 partially rehabilitated the direction by isolating its failure to stage-1 REMOVAL, not stage-3 addition.

**Sources**: goal-learnings § Failed Approaches EXP-017 entry; project-insights High (H20 alignment law).

**Estimated Effort**: low-medium — `_make_layer` widths are derived from `width_mult`; needs a one-off stage-3 width override; dt must be spot-measured (project-insights: never project across widths).

**Risk Assessment**: NOT free in epochs — any dt increase pays the deferral tax that has now killed 8 changes across 5 mechanism classes; fc layer and layer3 in_ch change ripples params +~1.4M. Estimated +1–2ms/step → −6–12 epochs → needs the capacity gain to beat ~0.15–0.25pp of starvation before clearing +0.1. Against a closed capacity axis (both directions), prior very low (~5%).

### 3. Class-balanced batch composition (data order, execution unchanged)
**Summary**: Replace the uniform shuffle with a sampler that makes every batch exactly class-balanced (51–52 per class at batch 512); loader-side only, zero GPU cost, dt unchanged.

**Reasoning**: Balanced batches reduce the variance of the per-batch gradient's class composition — a noise reduction orthogonal to LR/batch-size, never probed on any axis; touches no measured optimum.

**Sources**: none comparable-regime (acknowledged); CIFAR-10 train is exactly balanced (5000/class) so a balanced sampler is implementable with a permutation-of-class-indices scheme in `train.py`.

**Estimated Effort**: medium — custom Sampler; must preserve `drop_last`, `persistent_workers`, per-epoch reshuffle; risk of loader-side CPU overhead (infra-errors EXP-013: ~3% margin) is near-zero since sampling is index math.

**Risk Assessment**: Mechanism is double-edged under the max-statistic law: reducing gradient-noise variance is exactly what EXP-022 just measured as HARMFUL at 2× batch (less noise → worse generalization); balanced batches are a milder dose of the same medicine. Prior very low; also the EXP-011/022 noise lessons predict the sign is wrong.

## Idea Evaluation

**Evidence strength**: All three are weakly evidenced (the campaign's frontier is past all strong evidence). Idea 1 has the strongest structural support: it is the designated next step in two consecutive reports, completes a systematic program, and its admissibility (heat-constant) is derived from in-project measurements rather than external transfer. Idea 2 fights the single most-confirmed law in the project (deferral, 8 confirmations). Idea 3's expected SIGN is wrong by the project's own freshest result (EXP-022: less gradient noise hurt).

**Mechanism clarity**: Idea 1: first-order heat invariance by construction + second-order smoothing of the search direction — precise, falsifiable (signatures must equal baseline; any epoch/dt drift means the premise broke). Idea 2: capacity-where-cheap, but cannot articulate why +params beats −epochs when EXP-017's faster variant already lost. Idea 3: noise-shaping with a predicted-harmful sign.

**Expected impact**: All low. Idea 1 if right: smoother hot phase + equal-or-longer plateau → +0.1–0.2pp. Ideas 2–3: more likely negative than positive.

**Risk profile**: Ideas 1 and 3 are graceful (converged no-improvement). Idea 2 risks compounding starvation + alignment surprises. Idea 1 additionally has the cleanest interpretation either way — it finishes the constant-certification program.

**Feasibility**: Idea 1 is a two-constant diff with baseline-identical signatures (strongest possible contamination check). Idea 3 needs a custom sampler. Idea 2 needs architecture surgery plus a dt spot-measurement gate.

## Chosen Idea
**Selected**: Heat-constant momentum trade — MOMENTUM 0.95 + PEAK_LR 0.2 (lr/(1−β) = 4)

**Why this idea**:
It is the only surviving candidate that is simultaneously free in early heat (held constant at every schedule point by construction), free in epochs (zero execution change), and numerics-clean (same kernels/batch/compile) — i.e., the only one that escapes all three established failure laws rather than arguing against one of them. It is also the designated last untried in-recipe candidate from two consecutive reports; win or lose, it completes the bracketing certification of every constant in `train.py`.

**Hypothesis**:
With the effective step held at baseline, doubling the averaging horizon smooths the search direction through the hot phase: mid-run trajectory at-or-above the baseline family, signatures byte-identical to baseline (dt ~22.4ms, ~139 epochs, VRAM ~1613MB, params 4,286,026), and a converged plateau at-or-above baseline's — **best_test_acc ≥ 96.81** if the smoothing converts to mean improvement rather than variance clipping. A converged miss closes the momentum axis and certifies the EXP-006 recipe as a completed local optimum over its entire constant set, routing future loops to explicitly radical, out-of-recipe mechanisms only.
