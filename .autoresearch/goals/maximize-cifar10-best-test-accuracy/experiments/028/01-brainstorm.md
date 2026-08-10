# Brainstorm EXP-028
**Created**: 2026-08-06

## Web Search & Literature Review

- **Positive-Negative Momentum** (`knowledge/papers/positive-negative-momentum.md`; ICML 2021)
  Alternating momentum streams improved CIFAR ResNet generalization without a second gradient, but paper recurrence/decay do not match PyTorch SGD scale and require an explicit noise-only normalization.
- **Gradient Centralization** (`knowledge/papers/gradient-centralization.md`; ECCV 2020)
  Zero-centering each convolution filter's gradient is a one-reduction projected-gradient regularizer with broad vision evidence and no forward-graph change.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`; BMVC 2016)
  Rectifier/dropout changes can complement width, but published regimes are longer and wider; accepted short strong-view fit makes extra suppression risky.

## Experimental History Review

- EXP010 remains the 94.15% frontier. Width-2 capacity, all-parameter decay, 80% LR hold, N1/M7, 50% CutMix, and the simultaneous low-LR hard weak transition are now jointly protected by multiple successes/failures.
- EXP020 Nesterov and EXP022 Lookahead hit candidate-only concentration despite lower short loss. Any new optimizer path needs exact scale/state recurrence and immutable multi-step safety; lower loss cannot clear a class transient.
- EXP027 confirms that removing regularization before the LR drop is harmful even when RandAugment remains: hard N1/M7 collapsed at 70% and finished 0.46 below baseline. Preserve the entire data curriculum.
- Representation changes repeatedly trade fit for NLL or destabilize recruitment: preactivation/zero-gamma, pooling shortcuts, raw-max readout, abrupt width, and ECA failed. A fixed activation change must prove initial/update scale and full exposure.
- Backward is 75.46% of counted step time, but optimizer overhead is only 1.67%. PNM and gradient centralization have plausible accuracy mechanisms; their extra reductions/state must still preserve >=26,629 steps.

## Collected Ideas

- **Scale-matched PNM** — Maintain alternating momentum streams with paper normalization but analytically match accepted steady-gradient drift and coupled decay. Targets generalization through amplified gradient noise; direct CIFAR evidence is strong, local optimizer transients are the main risk.
- **Conv-only gradient centralization** — After backward, subtract each Conv2d filter gradient mean across input/spatial dimensions, then use unchanged SGD. Targets flat/generalizable filters without changing data or forward activations.
- **Fixed LeakyReLU** — Replace ReLU with a small fixed negative slope and use matching Kaiming gain. Targets dead/one-sided feature flow, but changes every block and may lose the accepted sparse representation bias.
- **Tail-only gradient centralization** — Activate GC only after the 80% switch to regularize low-LR refinement. Lower risk to strong fit but little horizon and a new phase-dependent optimizer rule.
- **Standard Option-B projections** — Learn same-lattice transition channel transport. It remains untested, but EXP017 makes projection+BN a likely NLL liability.
- **Short weak-tail EMA** — Track recent parameters rather than uniform SWA. EXP010 already ends at its best, so demonstrated ceiling is low.
- **Channels-last plus capacity** — Use any measured layout speedup to fund extra width. This could attack backward efficiency and representation together but destroys attribution and FP32 layout evidence is weak.
- **Moonshot SAM-lite** — Periodically perform a sharpness-aware two-pass update rather than every step. It imports a strong generalization mechanism but creates an arbitrary cadence and major fixed-time exposure loss.

## Combinations

- **Gradient centralization + PNM**: projection controls filter-gradient mean while alternating momentum amplifies residual noise, potentially separating drift and noise better than either. Their optimizer effects interact nonlinearly and must be isolated first.
- **Gradient centralization + LeakyReLU**: smoother feature flow plus centered filter updates could improve conditioning, but both are global and a failure would be uninterpretable.
- **Channels-last + GC**: layout speed might pay for centralization reductions, but the accuracy mechanism belongs to GC and should be measured alone before systems compensation.

## Candidate Ideas

### Fixed Small-Slope LeakyReLU

Replace all 19 dynamic ReLU activations with fixed slope-0.01 LeakyReLU and use the analytically matched Kaiming gain, changing no parameter count, branch, data rule, optimizer, or schedule. The intervention may preserve weak signed evidence and inactive-feature gradients, but it alters every block and the final pooled feature; BN also weakens the premise that dying ReLUs are the current limiter. Full specification: `proposals/idea-03.md`.

### Signal-Scale-Matched Positive-Negative Momentum

Replace accepted momentum SGD with two alternating `rho=0.81` momentum streams and the paper's `+2/-1` geometry, but multiply the raw PNM direction by a closed-form step-dependent scale so its response to a constant decay-augmented gradient exactly equals accepted PyTorch momentum at every step. This preserves first-step and coherent-signal scale while testing the paper's negative-history/noise geometry. Direct CIFAR evidence is the strongest of the candidates, but EXP020/022 make optimizer-path concentration and exposure the dominant risks. Full specification: `proposals/idea-01.md`.

### Conv2d-Weight-Only Gradient Centralization

After backward and before unchanged SGD, subtract each of the 19 Conv2d weight gradients' mean over input/spatial dimensions; leave FC, BN, biases, forward graph, initialization, data, schedule, and all-parameter coupled decay untouched. This is a small, RNG-neutral projected-gradient regularizer with ECCV vision evidence and no representation drift, though 38 tiny GPU operations could reduce exposure and the proposed pre-decay ordering is narrower than the paper's strict subspace theorem. Full specification: `proposals/idea-02.md`.

## Review

Claude's independent adversarial review (`01-idea-review.md`) selected **signal-scale-matched PNM**. It judged PNM and GC equally strong on evidence/reasoning (7/10), but gave PNM the higher impact ceiling (8/10 versus 5/10); fixed LeakyReLU scored 3/10 on both axes because BN weakens its unmeasured dying-ReLU premise and the intervention touches every block in a locally fragile family.

The review's central objection is valid: the closed-form normalization matches accepted SGD only for a constant decay-augmented gradient, whereas PNM's intended mechanism operates on changing stochastic gradients. The immutable-corpus preflight therefore must promote the changing-gradient update-ratio distribution to a veto: abort if its median candidate/control ratio exceeds 1.30, in addition to the existing concentration and spike gates. The strong-phase transfer mismatch is also explicit: a switch checkpoint in `[88.73, 89.73)` followed by a top-1 miss is the predicted harmful-noise signature, not an anomalous outcome.

PNM is not an unchanged retry of EXP020/022. It exactly matches accepted first-step direction, avoids Nesterov's 1.9x first direction, never pulls parameters away from their optimizer state, and implements the scale-matched PNM revisit requested by EXP020. Those distinctions justify one gated attempt, while the two prior optimizer-path failures justify treating any lower loss as irrelevant if concentration or update-ratio gates fail.

## Idea Evaluation

- **Scale-matched PNM** — Best evidence/upside combination and a clean test of negative-history geometry after analytically removing coherent signal-scale mismatch. High preflight mortality is acceptable because the exact failure mode is measurable on an immutable production corpus before the one production run. Advance with the review's median update-ratio veto.
- **Conv-only gradient centralization** — Best fallback. It preserves the initial function and accepted optimizer, but its pre-decay `P(g)+lambda*w` operator is weaker than the cited theorem, its likely gain is close to the 0.10-point gate, and 38 small kernels face a strict 1% exposure budget. Retain for a future experiment if PNM is vetoed or misses.
- **Fixed LeakyReLU** — Reject for EXP028. No accepted-run diagnostic establishes dead channels as a limiter, fixed slope-0.01 has weaker support than learned PReLU, and signed post-add/GAP features introduce a global representation change after repeated local strong-phase suppression failures.

## Chosen Idea
**Selected**: Signal-scale-matched Positive-Negative Momentum

**Why this idea**:
It is the only candidate with a directly comparable published CIFAR ResNet accuracy gain and has the largest plausible upside. The recurrence is materially distinct from the failed Nesterov and Lookahead paths: accepted first-step direction and constant-gradient signal scale are exact, and parameters never move independently of optimizer state. The remaining stochastic-noise risk is observable and now has conservative immutable-corpus concentration, spike, and median-ratio vetoes.

**Hypothesis**:
With beta0=1, `rho=0.81`, paper `+2/-1` alternating history, all-parameter coupled decay, and a closed-form per-step scale matching accepted momentum's constant-gradient response, PNM will amplify useful stochastic-gradient noise without suppressing the protected strong-phase fit, preserve at least 26,091 updates, and improve seed-42 `best_test_acc` from 94.15% to at least 94.25%. Point prediction is 94.30%; a valid lower result is no-improvement, and any immutable-corpus concentration or median update-ratio above 1.30 invalidates this exact candidate before production.
