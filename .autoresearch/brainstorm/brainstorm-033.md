# Brainstorm EXP-033
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **knowledge/README.md scan**: relevant prior entries — swa-stochastic-weight-averaging.md (EXP-032: accuracy is decision-boundary-limited, not calibration-limited), progressive-resizing.md (EXP-031: gains require unchanged training distribution), airbench row (heavy-aug speedrun recipes), WRN row.
- **Augmentation annealing / curriculum-out augmentation** (background knowledge): the practice of REDUCING augmentation strength near the end of training appears in fastai community recipes and in "data echoing/curriculum" literature; the cleanest published relative is the FixRes/"fine-tune on test distribution" family — which EXP-025 already measured here in its EXTREME form (fully clean tail: one-time +0.35 alignment jump, then clean-data overfit, net −0.87). The original ResNet paper (He et al. 2015, cited in train.py) trained CIFAR with EXACTLY crop(32,pad=4)+flip — i.e., "light aug" has a canonical anchor: it is the original recipe's augmentation.
- **No new web search needed**: the decisive evidence is in-project — EXP-025's measured transient and failure mechanism, EXP-032's boundary-limited diagnosis. External sources would only add fixed-epoch results (transfer record 0-for-12).

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (top of distribution; mean ≈96.57, σ ≈0.16); bar 96.81 ⇒ true effect ≥ +0.3. **27 consecutive misses (007–032).**
- **Closed axes**: capacity (both directions), all recipe constants (bracketed), gradient noise (bracketed), batch+LR, schedule shape, STATIC augmentation pressure (dose-response peaked at current recipe), init, activations, head/shortcut topology, optimizer geometry, throughput tiers (kernel/pipeline/resolution), eval-time BN, weight averaging (EMA + SWA, both ends), FixRes clean tail.
- **EXP-032's diagnosis (the live thread)**: accuracy here is DECISION-BOUNDARY-limited — solution-smoothing moved loss, not argmaxes. The only intervention class that moves boundaries with the weights co-adapting is the TRAINING DISTRIBUTION while training continues.
- **EXP-025's two-sided measurement (the unexploited near-miss)**: switching the final 15% to FULLY CLEAN data (normalize-only — no crop, no flip, no TA, no RE) produced an immediate +0.35 alignment jump — larger than the bar requires — then overfit-flatlined (test_loss rising) because ALL training pressure vanished. The interval between "full pressure" (baseline plateau) and "zero pressure" (EXP-025) has never been probed.
- **Augmentation SCHEDULE is an open axis**: the dose-response (RE +0.83, TA +0.17, mixup −0.46, reflect −0.14, WD-half −0.30) is entirely STATIC — constant pressure for the whole run. No experiment has varied pressure over TIME except EXP-025's extreme endpoint.
- **Protocol assets**: load-gate + GPU-free launch gates (EXP-032 infra entry), composite watchdog, σ calibration, per-segment profiles.

## Candidate Ideas

### 1. Augmentation taper: revert the tail to the ORIGINAL ResNet augmentation (crop+flip only) at 88% of budget
**Summary**: Add `AUG_TAPER_FRAC = 0.88`. Build a second DataLoader at startup (`tail_loader`) over a dataset whose transform is the He-2015 original: RandomCrop(32, pad=4) + RandomHorizontalFlip + ToTensor + Normalize — i.e., baseline transform MINUS TrivialAugmentWide and RandomErasing. The epoch loop picks `train_loader` while progress < 0.88, `tail_loader` after. Nothing else changes: LR cosine anneals to zero as in baseline (the full anneal is preserved — unlike EXP-032), eval is `base_model` every epoch, timed step untouched, dt unchanged.

**Reasoning**: This is the measured interpolation the history begs for. EXP-025 proved the upside exists: dropping the heavy train/test distribution gap yields an immediate +0.35 — bar-sized — and proved the failure mode: with ZERO pressure the model overfits the clean set within ~5 epochs. Keeping crop+flip (worth ~classical 3–4pp on CIFAR ResNets — the dominant regularizer) maintains training pressure while still removing the two heaviest distribution distorters (TA's global photometric/geometric warps, RE's occlusions). BN stats and weights co-adapt over the remaining ~17 epochs of live annealing (the EXP-029-safe pattern — training continues). The mechanism is exactly the class EXP-032 diagnosed as the remaining lever: move decision boundaries via the training distribution, late (deferral-safe), at zero charged cost (CPU-side transforms get CHEAPER; dt and epochs unchanged ~139).

**Sources**: exp-report-025.md (the +0.35 transient AND the zero-pressure failure mechanism); exp-report-032.md § Results (boundary-limited diagnosis) + § Next Steps; exp-report-029.md (training-continues stat safety); train.py:134–158 (transform/loader), :222 (epoch loop); He et al. 2015 § 4.2 (crop+flip is the original CIFAR recipe — the anchor for "light").

**Estimated Effort**: low-medium — second transform/dataset/loader at startup (~10 lines), one-line loader selection per epoch, watchdog reuse.

**Risk Assessment**: (a) The interpolation could inherit EXP-025's overfit (if crop+flip pressure is insufficient at this capacity) — visible in-run as rising test_loss in the final epochs; graceful no-improvement that BRACKETS the pressure-schedule axis (zero / light / full all measured). (b) The +0.35 jump may have been specific to fully-clean BN alignment; the light-aug jump could be ~half — still bar-relevant stacked on the completing anneal. (c) Two semi-tuned anchors (0.88; crop+flip as "light") — flagged: a miss leaves interiors, but the endpoints would then bound the axis tightly. (d) Second persistent loader: +8 workers (~ok at 180 cores, load-gated launch), worker spin-up ~2s uncharged on first tail epoch.

### 2. ResNet-26 depth probe at 4× width with a strict early-dt gate
**Summary**: NUM_BLOCKS 3→4 (ResNet-26, +33% blocks/params, widths 64/128/256 unchanged); early-dt gate kills within ~90s unless measured dt ≤ 24.5ms (≥ ~127 epochs projected).

**Reasoning / why weak**: The last unbracketed capacity direction (deeper at constant width). But it contradicts the two strongest laws (capacity-starvation, deferral), prior depth evidence is negative in both directions (EXP-008 shallow-wide, EXP-017 within-budget reallocation), WRN says width>depth on CIFAR, and projected dt ~29ms → ~104 epochs sits in the historically always-losing range. Value is INFORMATION (closing the axis for ~90s of GPU time if the gate fires; one full run if it doesn't), not expected gain.

**Sources**: exp-report-017.md, exp-report-008.md, knowledge/README.md WRN row; goal-learnings § capacity entries.
**Estimated Effort**: trivial (one constant + gate). **Risk Assessment**: graceful; near-certain no-improvement; cheap.

### 3. LR floor for a variance-raising plateau (lr_min ≈ 0.005 instead of →0)
**Summary**: Clamp the cosine at a small floor so plateau iterates keep bouncing — more independent draws for the max-statistic (the direction goal-learnings' EXP-011 insight flags as what the metric rewards).

**Reasoning / why weak**: Arithmetic caps it sub-bar: E[max of ~20 draws] gains ~+1.9σ_plateau; even doubling plateau σ from 0.05 adds only ~+0.09 expected, and the floor's extra heat lowers the mean (EXP-010/014 dose-response). Cannot reach +0.24 alone; a refinement to bank only if a level-raising change lands first.

**Sources**: goal-learnings § EMA-merged entry insight; exp-report-027.md (plateau σ, max harvest ≤0.03).
**Estimated Effort**: trivial. **Risk Assessment**: graceful but expected-value ≈ +0.05; weak.

## Idea Evaluation

**Evidence strength**: Idea 1 rests on a measured IN-PROJECT positive transient of bar-exceeding size (+0.35, EXP-025) with a diagnosed, specifically-repairable failure mode — the strongest evidence configuration available after 27 misses (the same "repair the diagnosed flaw" template as EXP-032, but here the upside was directly measured rather than imported from fixed-epoch literature). Idea 2's evidence is uniformly negative; Idea 3 is arithmetically sub-bar. **Mechanism clarity**: Idea 1 = remove train/test distribution distortion while keeping classical pressure and a live anneal, so boundaries and BN stats co-adapt to a nearer-test distribution — precisely the boundary-moving class EXP-032 isolated. **Expected impact**: Idea 1 is the only candidate whose central estimate (+0.2–0.35 sustained) brackets the needed +0.24-over-mean; the others cap below. **Risk profile**: Idea 1 fails gracefully and informatively (the pressure-schedule axis gets bracketed at three points); overfit is observable in-run (rising test_loss). **Feasibility**: ~12 lines, no new packages, dt-neutral, wall-neutral (~493s).

## Chosen Idea
**Selected**: Idea 1 — Augmentation taper to the original-ResNet light transform (crop+flip) at 88% of budget

**Why this idea**:
It exploits the campaign's largest unexploited measured positive (+0.35 on switch to nearer-test data, EXP-025) while repairing that experiment's diagnosed failure (zero pressure → clean-set overfit) by keeping the canonical crop+flip regularizer — and it operates in the one intervention class EXP-032's diagnosis left open: moving decision boundaries through the training distribution while training continues. Late-only, dt-free, anneal-preserving, fully within train.py.

**Hypothesis**:
Switching the final ~12% of the charged budget (~17 epochs) from the full TA+RE pipeline to crop+flip-only data yields an alignment gain that, unlike EXP-025's, is SUSTAINED (crop+flip pressure prevents the overfit flatline) and rides the completing cosine anneal: evals jump ≥ +0.15 within 2 epochs of the taper and the final plateau sits ≥ +0.25 above the baseline mean ⇒ best_test_acc ≥ 96.81 at unchanged dt/epochs (~22.4ms / ~139) and wall ~495s. Falsifiable: (a) no jump within 3 taper epochs → the EXP-025 transient required full cleanliness; (b) test_loss reversing upward before run end → light pressure still insufficient (axis closes bracketed either way).
