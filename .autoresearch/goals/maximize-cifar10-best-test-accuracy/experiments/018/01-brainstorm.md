# Brainstorm EXP-018
**Created**: 2026-08-06

## Web Search & Literature Review

- **Generalization Analysis of Stochastic Weight Averaging with General Sampling** (`experiments/018/papers/swa-general-sampling.md`; https://proceedings.mlr.press/v235/wang24bl.html)
  ICML 2024 establishes sharper SWA stability bounds than SGD in nonconvex, without-replacement settings matching the shuffled CIFAR loader.
- **How to Scale Your EMA** (`experiments/018/papers/ema-scaling.md`; https://papers.nips.cc/paper_files/paper/2023/hash/e7681dd6fe16052433ab68cd1555bdc9-Abstract-Conference.html)
  NeurIPS 2023 supports model EMA as a supervised robustness/generalization tool but makes its update timescale an explicit algorithmic choice.
- **On the Generalization Benefit of Noise in Stochastic Gradient Descent** (`experiments/018/papers/sgd-noise-generalization.md`; https://proceedings.mlr.press/v119/smith20a.html)
  ICML 2020 shows that lower optimization loss at reduced gradient noise need not improve test accuracy, strengthening the case against revisiting large batches for modest throughput.
- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`)
  Late averaging can complement annealing at low memory cost, but BatchNorm state and the short averaging window require an explicit protocol.
- **ECA-Net** (`knowledge/papers/eca-net.md`)
  Tiny channel-axis attention can recalibrate residual features with very few parameters, but CutMix interaction and exact identity initialization require local validation.

## Experimental History Review

- EXP-010 remains the 94.15% frontier: width-2 postactivation ResNet-20, N1/M7 plus 50% alpha-1 CutMix through 80%, then a hard weak cosine tail. Its 89.73% switch fit, 26,898 steps, and 0.1934 final NLL are the accepted reference.
- EXP-011 showed stronger CutMix probability suppresses strong fit; EXP-008/009 showed changing all-parameter 1e-4 decay hurts the width-2 balance. Preserve both operating points.
- EXP-012 and EXP-015 show that identity-oriented residual initialization/order changes can underfit the short strong phase. EXP-014 shows zero output alone is not update-safe; any new branch needs production-batch first-update gates.
- EXP-017 is the sharpest new diagnostic: learned normalized transition shortcuts raised switch and first-weak accuracy by 0.47/0.29 points with 98.73% step retention, yet worsened final NLL and missed the frontier by 0.06. Short-horizon fit is not sufficient; late generalization and confidence are now the clearest accuracy limiter.
- The systems profile remains current at `7c1e7d8`: backward is 75.46% of counted work, visible loader/host overhead is negligible, and memory has enormous headroom. Candidates must either be nearly free or produce enough representation/late-solution benefit to justify lost exposure.
- Untried gaps include late weight averaging on the accepted trajectory, isolated Nesterov without EXP-001's short-LR confound, exact-identity channel recalibration, ghost BN at unchanged batch 128, and deterministic anti-aliased Option-A downsampling without projection or new BN.

## Collected Ideas

- **Late arithmetic SWA during cosine refinement** - Maintain an online average of parameters only after the 80% switch and evaluate the averaged solution on the unchanged schedule. This directly targets noisy late generalization while preserving the optimizer trajectory and batch-128 gradient noise; ICML 2024 provides without-replacement support, but BN buffers and counted update overhead are load-bearing design issues.
- **Fixed-timescale EMA shadow** - Track an exponential parameter average across the strong-to-weak boundary and evaluate that shadow. EMA can smooth trajectory noise more responsively than uniform SWA, but a decay introduces a tunable effective window and averaging stale strong-phase weights may bias the refined solution.
- **Exact-identity ECA residual recalibration** - Add a zero-initialized channel-axis Conv1d and multiply residual outputs by `2*sigmoid(gate)` so the initial function is exactly unchanged while the network learns cross-channel allocation. This targets representation efficiency rather than transition paths; its global descriptor is plausible from ECA-Net, though even tiny per-block attention adds backward launches.
- **Isolated Nesterov momentum** - Change only SGD's `nesterov` flag while preserving LR, momentum, decay, data, architecture, timer, and evaluator. EXP-001 never isolated this lever from its harmful 15% LR hold, so it remains untested; look-ahead gradients could improve progress during the fixed plateau, but may also change the beneficial noise/regularization balance.
- **Deterministic pool-first Option-A shortcut** - Replace strided slicing with 2x2 average pooling while retaining channel zero padding and adding no learned projection or BN. This isolates the anti-alias component that EXP-017 confounded with a random normalized channel basis, at essentially no parameters but some pooling backward cost.
- **Ghost BatchNorm at unchanged optimizer batch** - Compute BN statistics on virtual groups of 64 inside each batch of 128 while retaining the same gradient batch, LR, and update count. NeurIPS/ICML evidence links ghost statistics to large-batch generalization, but batch 128 may already be below the harmful regime and custom reshaping could add material overhead.
- **Late-only stochastic depth** - Activate a very small, schedule-fixed drop-path probability only during the hard weak tail, leaving strong fit untouched and ensembling nearby subnetworks during refinement. It is a risky bet because only 60 counted seconds remain, identity-oriented residual changes have repeatedly underfit, and per-sample masking perturbs BN/shortcut semantics.
- **Two-point tail interpolation** - Retain the online model at the 80% switch and form a fixed interpolation with the final online weights for evaluation, seeking a wider basin without per-step shadow updates. This is a moonshot simplification of averaging, but one predeclared coefficient has weak local evidence and the switch model is much less accurate than the tail model.

## Combinations

- **Nesterov + late SWA**: Nesterov may traverse the high-LR plateau faster, while SWA could damp its noisier terminal path. The combination plausibly improves both optimization and late generalization, but it changes two independent mechanisms and should not precede isolated tests because a result would be hard to attribute.
- **Identity-scale ECA + late SWA**: ECA reallocates representation capacity and averaging smooths the learned gates and backbone together. This could outperform either alone if recalibration increases useful diversity, but the combined exposure cost and two-source causal ambiguity make it a later composition candidate.
- **Pool-first Option-A + identity-scale ECA**: deterministic pooling preserves all four spatial samples while ECA reallocates channels without a learned transition projection. It separates spatial anti-aliasing from channel mixing more cleanly than EXP-017, yet combining them immediately would obscure which component repaired or repeated the NLL deficit.

## Candidate Ideas

### Identity-Scale Final-Stage ECA Recalibration
**Summary**: Add ECA only to the three `layer3` residual outputs. Each module globally pools the 128-channel residual, applies a zero-initialized bias-free length-5 Conv1d, and multiplies by `2*sigmoid(logit)` before shortcut addition. This adds exactly 15 parameters and starts at gate one with bitwise-equal initial outputs/shared gradients while immediately recruiting the new weights. See `proposals/idea-02.md`.

**What it targets**: High-level channel allocation under the accepted width-2 capacity, without touching the Option-A transitions that EXP-017 changed or suppressing early feature extraction. Paired timing must protect exposure because model backward is 75.46% of counted cost.

**Reasoning**: ECA-Net reports inexpensive residual channel recalibration without an SE bottleneck. Final-stage-only scope addresses semantic features, leaves early CutMix/RandAugment processing intact, and uses an exact-identity centered gate to avoid the residual suppression seen in EXP-012/015.

**Sources**: `knowledge/papers/eca-net.md`; `proposals/idea-02.md`; EXP-012, EXP-014, EXP-015, and EXP-017 analyses.

**Estimated Effort**: medium

**Risk Assessment**: Channel adjacency may have little meaning; a global descriptor can blur CutMix regions; `(0,2)` gates can amplify residuals after one update; and small sequential pool/Conv1d/sigmoid/multiply kernels may exceed the proposed 2% exposure budget.

### Late Arithmetic SWA with In-Budget BN Recalibration
**Summary**: Preserve the accepted online trajectory through 86% counted time, average parameter snapshots from weak-tail epoch endpoints in `[86%,98%)`, then reserve the final 2% of the same 300-second budget to install the mean and cumulatively recompute BatchNorm moments using at least one full weak-loader pass. Every shadow update, copy, reset, and refresh forward is charged to counted time; the terminal SWA evaluation replaces rather than supplements the terminal online evaluation. See `proposals/idea-01.md`.

**What it targets**: Late solution quality and calibration, now the clearest limiter after EXP-017 improved switch fit but worsened final NLL. It avoids the 75.46% backward bottleneck during BN refresh but deliberately sacrifices roughly 2% of optimizer exposure.

**Reasoning**: ICML 2024 provides SWA stability/generalization support for nonconvex without-replacement training, and ICML 2025 reports averaging can complement annealing. Arithmetic averaging is directly aimed at late generalization rather than another representation change, while explicit in-budget BN refresh avoids stale or averaged running-statistics invalidity.

**Sources**: `experiments/018/papers/swa-general-sampling.md`; `experiments/018/papers/ema-scaling.md`; `knowledge/papers/weight-averaging.md`; EXP-010 and EXP-017 analyses.

**Estimated Effort**: high

**Risk Assessment**: The short tail may yield too few correlated snapshots; equal averaging can lag an improving trajectory; six refresh seconds cost updates; and earlier online evaluations remain part of `best_test_acc`, so a positive result requires careful attribution between online and terminal SWA models.

### Isolated PyTorch Nesterov Momentum
**Summary**: Add only `nesterov=True` to the accepted single SGD group, preserving momentum 0.9, all-parameter coupled decay 1e-4, LR/data/model/timer/evaluator, and every other semantic. This resolves EXP-001's Nesterov confound with its harmful 15% LR hold. See `proposals/idea-03.md`.

**What it targets**: Basin selection and response speed along the fixed noisy optimization trajectory at essentially unchanged model cost. It keeps batch-128 stochastic-gradient noise, which ICML 2020 identifies as a generalization-relevant mechanism.

**Reasoning**: PyTorch Nesterov adds the current decay-augmented gradient to the momentum direction without a second forward. EXP-001 cannot assign it a sign because schedule, persistence, and evaluation all changed simultaneously; an exact one-keyword experiment is the only clean test of this surviving optimizer question.

**Sources**: `experiments/018/papers/sgd-noise-generalization.md`; `proposals/idea-03.md`; EXP-001, EXP-002, and EXP-010 analyses; installed PyTorch 2.9.1 SGD semantics.

**Estimated Effort**: low

**Risk Assessment**: The first update is intentionally 1.9 times ordinary momentum, the current-gradient correction can amplify CutMix/RandAugment noise, coupled decay is filtered differently, and faster training loss or switch fit may repeat EXP-017's failure to improve late generalization.

## Review

Mandatory external Claude review completed successfully with no fallback and selected late arithmetic SWA. It judged SWA the only candidate aligned with the diagnosed late-generalization/NLL limiter, while ECA repeated the fit-side representation family and Nesterov lacked positive accuracy evidence. The central concern was that `[90%,98%)` might average overly correlated low-LR iterates from a monotone improving tail. I accept the need for more spread but reject moving all the way to 80%: EXP-010's first weak checkpoints are rapidly adapting from the strong distribution and materially below its peak. The plan-review-hardened fixed window is `[86%,98%)`, spanning approximately LR 0.0080 to 0.00034 and providing snapshot-count margin, with epoch-endpoint parameter-distance diagnostics and at least seven snapshots. The final 2% remains an in-budget cumulative BN refresh.

I also adopt Claude's attribution correction. Final SWA NLL is pre-registered against EXP-010's 0.1934 as the primary mechanistic diagnostic, while `best_test_acc >=94.25%` remains the goal's sole formal metric. The run records the best online accuracy before SWA installation separately: if an online checkpoint supplies the formal maximum or final SWA does not improve NLL, the result can be a protocol improvement but does not support the averaging mechanism. No second terminal evaluation is added. Full critique and scores are in `01-idea-review.md`.

## Idea Evaluation

Claude scored SWA 8/10 for evidence/reasoning and 6/10 for impact, ahead of ECA at 5/10 and 5/10 and Nesterov at 4/10 and 3/10. I adopt the named pick. The selection is not based on safety: ECA had the strongest implementation gates and Nesterov the smallest diff, but SWA is the only finalist whose proposed effect directly addresses the late NLL/generalization gap exposed by EXP-017. The window/NLL/attribution refinements above satisfy the significant review concerns without changing the selected mechanism into EMA, cyclic LR, or a multi-lever schedule experiment.

## Chosen Idea
**Selected**: Late Arithmetic SWA with In-Budget BN Recalibration

**Why this idea**:
Late parameter averaging is the most evidence-backed untried intervention aimed at the current limiter: the accepted online model already fits strongly, while recent representation changes either suppress fit or improve it without improving late confidence. A parameter-only arithmetic mean leaves the accepted SGD path unchanged through most of the run, and a charged cumulative weak-data BN refresh makes the evaluated averaged model internally consistent. The fixed `[86%,98%)` window balances post-switch adaptation, iterate spread, and snapshot-count margin, and the final 2% reserve prevents uncounted recalibration compute or an extra evaluator opportunity.

**Hypothesis**:
Uniformly averaging at least seven weak-tail epoch-endpoint parameter states from `[86%,98%)`, then installing them and cumulatively recomputing BatchNorm statistics for at least one full weak-loader pass within the last 2% of the 300-second counter, will retain at least 97% of EXP-010's optimizer steps and make the final SWA model reach at least 94.25%. A mergeable improvement requires both formal `best_test_acc >=94.25%` and final SWA accuracy at least 94.25% and at least the recorded pre-install online best. Mechanism support additionally predicts final SWA test loss below 0.1934 and pre-registered nondegenerate endpoint spread; no diagnostic can create a rerun.
