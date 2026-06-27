# Report EXP-020: Projection shortcuts at stage transitions (ResNet option B, WRN-faithful)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md
- **Plan**: plans/plan-020.md
- **Log**: logs/exp-log-020.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: after fourteen consecutive misses and an explicit EV synthesis check, does modernizing the last structural 2016-era component — the option-A strided-slice + zero-pad transition shortcut — to the learned 1×1+BN projection used by the WRN reference at exactly our width regime hold a gain?

## Idea & Hypothesis

Chosen idea: replace the transition shortcut in `BasicBlock` (used in layer2[0] 64→128 s2 and layer3[0] 128→256 s2 only) with `Conv2d(in, out, 1, stride=s, bias=False)` + `BatchNorm2d(out)`; identity shortcuts elsewhere unchanged. Evidence: WRN (arXiv 1605.07146) uses projection shortcuts at depths 16–40 on CIFAR-10 at our widths — the closest evidence transfer available in the remaining candidate space; He 2015 found option B marginally better than A. Hypothesis: half the transition output channels currently receive a zero shortcut and the strided slice discards 75% of spatial positions; a learned full-rank shortcut raises what is learned per unit of schedule heat from step 1 (not deferral) → early trail at-or-above family, signatures preserved, best_test_acc ≥ 96.81.

## Approach

+12/−8-line diff confined to `BasicBlock.__init__`/`forward`: `self.shortcut` = conv+BN Sequential at transitions, `nn.Identity()` elsewhere; `out += self.shortcut(x)`. Projection conv picked up by the existing Kaiming pass; its BN landed in the no-decay group via the existing `ndim` split; BN γ left at 1 (EXP-018 guard). CPU pre-validation pinned params at exactly 4,327,754 (+41,728, +0.97%), 2 projections / 7 identities, forward OK. Zero deviations from plan.

## Execution

One run, no retries (task bnfojwnrw, launched 11:16:29 via composite launcher + inline watchdog into a verified-free GPU 0). Pristine: zero watchdog events, 0/259 windows >30ms (mean 23.0ms), 135 epochs / 13,044 steps, total 495.2s, VRAM 1661.2MB, params exactly the pre-validation pin. startup 22.6s — fresh inductor compile for the new graph topology (matches EXP-006's 22.8s cold compile). The projection kernels cost a real +0.6ms/step → 135 epochs vs baseline's 139, inside the plan's 133–141 window.

## Results

- **Primary metric**: best_test_acc = 96.58% (baseline: 96.71, delta: −0.13pp, −0.13%); bar was 96.81
- **Observations**: The hypothesis's faster-or-equal early trail did NOT materialize — the run started SLOWER: ep1 34.55 vs the family's 38.2–38.95, ep5 59.84 vs 63.76, ep10 best 74.20 vs ~75. It closed onto the family mid-run (87.67@60, 93.00@100) and produced a proper converged plateau (final eight evals 96.44–96.58, final ≈ best, Δ0.08). final_test_loss 0.1882 is on par with the best on record — but the metric pays in max accuracy.
- **Analysis**: Two stacked costs, both predicted by the campaign's own laws and neither by the WRN evidence: (1) a mild EXP-018-style deferral — the two projection convs (Kaiming-initialized, not identity-like) must be LEARNED during the hottest schedule phase, visibly taxing epochs 1–10, whereas the WRN result was measured at fixed epochs where early-phase cost is free; (2) a throughput tax — +0.6ms/step bought 4 fewer epochs, and by EXP-006's conversion arithmetic ~4 epochs ≈ −0.08pp by itself. The net −0.13pp converged deficit means the full-rank shortcut's representational benefit roughly paid for one of the two costs but not both. The deeper lesson: the option-A pad shortcut was never a binding defect — zero-padded channels are produced by a residual branch that learns to fill them, and 135 epochs are ample to do so (the same "the budget learns it anyway" wash-out as EXP-019's whitening basis). This is the fifteenth consecutive miss and the FOURTH structural perturbation to land below baseline; the certified optimum now also covers shortcut topology, and the WRN-reference evidence class (fixed-epoch architecture comparisons at matched regimes) joins RegNet and Bag of Tricks in failing to transfer to fixed wall clock.
- **Key Learning**: Even reference-faithful architecture modernizations priced in early heat + epochs lose under fixed wall clock: the pad shortcut costs nothing the 135-epoch budget can't repay, while projection costs both early heat and 4 epochs.

## Verification

- **Conditions**: pre-condition contention sanity CLEAN (0/259 windows >30ms; 135 epochs within the predicted 133–141 given +0.6ms/step); condition 1 FAILED (best_test_acc 96.58 < 96.81); conditions 2–3 skipped per first-failure stop (informally: 495.2s ≤ 600 rc=0; 135 evals = 135 epochs — both would have passed)
- **Review Notes**: trustworthy — metric matches the eval trail (best 96.58 @ ep 134); num_params printed exactly the pre-validated pin (4,327,754) so the diff that ran is the diff that was validated; epoch count fully explained by the measured per-step cost, no contention
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid clean run; primary-metric necessary condition not met)

## Unexplored Avenues

- **Zero-init or identity-biased projection** (initialize the 1×1 as channel-copying for the first `in_ch` channels): would remove the early-heat learning cost — but it converges toward exactly the pad shortcut it replaces, and EXP-018 showed identity-at-init has its own deferral failure. Low interest.
- **ResNet-D variant (avgpool + 1×1 stride 1)**: keeps all spatial positions; but it ADDS cost per step on top of projection and its evidence (ImageNet ResNet-50, fixed epochs) is a weaker transfer than WRN's, which already failed. Closed in spirit.
- **Projection WITHOUT BN**: saves a little step time and removes a BN from the shortcut path; does not address the dominant early-heat cost. Low interest.

## Next Steps

1. **GPU-side augmentation to reclaim loader-stall time** — the designated follow-up from brainstorm-020's synthesis check: throughput conversion at unchanged hyperparameters is the only validated +pp mechanism since EXP-006 (~50s of baseline loader stalls ≈ +20 epochs potential); the hard part is faithful per-image TrivialAugmentWide on GPU — any augmentation-semantics drift confounds on the peaked regularization axis. Confidence: medium-low (strong mechanism, high implementation risk). **[SUPERSEDED — brainstorm-021: the premise is wrong. `t0` is set after the loader yields (train.py L215–216), so stalls live outside the timed budget and reclaiming them adds ZERO epochs (EXP-013 confirms: stalls 50→197s, epochs unchanged at 139). Discarded; the honest throughput lever is dt itself (compile tier, fused optimizer).]**
2. **Heat-constant momentum trade (0.95 + peak 0.2, lr/(1−β) held at 4)** — the only never-touched recipe constant, admissible only as a compensated trade. Confidence: low.
3. **Synthesis check, sharpened**: fifteen misses; four structural perturbations below baseline. Remaining candidate classes are now (a) throughput-at-fixed-hparams (the one validated mechanism), and (b) heat-constant multi-knob trades. Fixed-epoch architecture evidence — from ANY source, however well-matched the regime — should be treated as non-transferable to this budget unless the change is free in both early heat AND epochs. Confidence: n/a (process note).

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
