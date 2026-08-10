# Brainstorm EXP-011
**Created**: 2026-08-06

## Web Search & Literature Review

- **Averaging Weights Leads to Wider Optima and Better Generalization** (`knowledge/papers/stochastic-weight-averaging.md`)
  Late trajectory averaging improves CIFAR residual networks with little training overhead, but canonical SWA relies on trajectory diversity and requires explicit BatchNorm-statistics handling.
- **How to Scale Your EMA** (`knowledge/papers/how-to-scale-your-ema.md`)
  EMA decay must be derived from update cadence and effective horizon; this goal's wall-clock schedule makes a copied scalar especially hard to interpret.
- **A Modern Look at the Relationship between Sharpness and Generalization** (`knowledge/papers/modern-sharpness-generalization.md`)
  Sharpness is setup-dependent, so combining averaging with SAM must be judged by direct accuracy rather than a flatness story alone.

## Experimental History Review

- The lineage progresses from BASE 91.51 to WRN EXP-001 94.62, front-loaded CutMix EXP-002 95.23, and clean-tail period-two SAM EXP-004 95.40. EXP-004 remains the global best with 25,560 steps and final-equals-best accuracy.
- EXP-004's four children tested self-distillation, CutMix/manifold substitution, ASAM, and paired RandAugment; none tested weight averaging. EXP-006 explicitly recommends EMA after its final four checkpoints spanned 0.15 points.
- The current limiter is detectable generalization gain from a strong recipe rather than memory. An averaging method can use abundant memory while avoiding another model forward, but it must not obtain free uncharged training through BN recalibration or multiple validation models.
- EXP-005 shows preserving optimizer steps while reducing independent-image exposure can still hurt. EXP-006 shows replacing validated mechanisms muddies attribution. EXP-011 should therefore be additive, retain the exact CutMix/SAM package, and evaluate one preregistered model per epoch.
- Transient logs require exact durable state/metric audit. Wall-clock throughput changes alter SAM and averaging dose, so update counts, horizon, state distance, and evaluation routing must be logged.

## Collected Ideas

- **Horizon-derived clean-tail EMA** - Start when CutMix ends and SAM begins, initialize a shadow from the live model, and update it after each optimizer step with decay derived from a preregistered time/update half-life. Swap one full floating state into the model for each evaluation and restore exactly afterward. This directly smooths late iterates without another forward, but averaged BatchNorm buffers may lag distribution changes.
- **Epoch-end uniform SWA** - Uniformly average model parameters at each clean-tail epoch boundary, producing roughly 30-35 trajectory samples with almost no charged overhead. Evaluate only the averaged model after a minimum sample count. It follows the SWA literature more directly, though cosine LR collapse may leave insufficient trajectory diversity and canonical BN recalibration is unavailable.
- **SAM-pulse-conditioned averaging** - Update a uniform or exponential shadow only after optimizer steps that used SAM's perturbed gradient. This preferentially averages flatness-aware trajectory points and halves update overhead. The conditioning may provide a stronger mechanism than averaging all steps, but it entangles averaging dose with SAM cadence and could overweight a nonrepresentative subset.
- **Last-window Polyak average** - Begin only in the final 10% of charged time when LR is near its floor and average every optimizer step. It minimizes stale-state risk and BN drift, but the window may be too short and iterates too correlated to yield a detectable 0.10-point improvement.
- **Slow-weight Lookahead** - Maintain slow weights updated toward live weights every fixed number of steps and periodically sync them back into training. This converts averaging from evaluation-only to an optimizer stabilizer, potentially reducing oscillation, but it changes the trajectory and could interfere with validated SAM momentum dynamics.
- **EMA/live fixed interpolation** - Maintain a horizon-derived EMA but evaluate a preregistered convex blend of live and EMA parameters, reducing over-smoothing while retaining some variance reduction. It adds a mixing coefficient with weak evidence and makes BatchNorm compatibility harder.
- **SWA plus flat LR tail** - Replace the final cosine collapse with a small constant or cyclic LR while uniformly averaging clean-tail iterates, matching canonical SWA's diversity mechanism. This has larger upside but bundles two changes and risks sacrificing EXP-004's successful low-LR convergence.
- **Parameter-only EMA with live BN buffers** - Average trainable parameters but always use current running statistics, avoiding stale buffer averaging and uncharged recalibration. It is simple and cheap, but parameter/statistic mismatch can invalidate the averaged function precisely when weights move most.
- **Moonshot: two-point low-rank trajectory ensemble** - Track the mean and dominant difference direction of sparse late checkpoints, then choose a fixed central point for single-pass evaluation. It aims to capture trajectory geometry with low memory, but selection without a validation sweep is underdetermined and implementation complexity exceeds its evidence.

## Combinations

- **Horizon EMA + SAM-pulse cadence**: derive the EMA half-life in number of SAM pulses and update only after those optimizer steps. This aligns samples with the validated late mechanism and reduces overhead, but may be stronger than either component only if SAM points better represent flat solutions than intervening SGD points.
- **Uniform SWA + late-window start**: average epoch-end checkpoints only after the LR falls below a fixed progress-derived level. Compared with full clean-tail SWA, this avoids early stale weights; compared with last-step Polyak averaging, epoch spacing may preserve more diversity.
- **EMA weights + averaged BN buffers**: update all floating tensors with the same horizon while copying integer counters from live state. This avoids an uncharged BN pass and gives a self-contained averaged model, but buffer EMA is an approximation rather than canonical recalibration.

## Candidate Ideas

### Epoch-end full-state SWA
**Summary**: Uniformly average restored online parameters and floating BatchNorm buffers at each completed clean-tail epoch boundary, charge that update, and switch the single evaluation source to SWA after five samples. Expect roughly 24-25 samples separated by 195 updates; copy integer counters from the latest sample and restore online state exactly after evaluation.

**What it targets**: The same late checkpoint variance through an extremely low-overhead central trajectory point, while leaving the validated online trajectory untouched.

**Reasoning**: This is the most direct averaging interpretation and costs less than 1 GiB total state traffic. Epoch spacing may preserve more diversity than per-step averaging, and uniform recurrence avoids a decay hyperparameter. However canonical SWA uses constant/cyclic LR, while EXP-004's cosine tail collapses from about 0.034 to 0.002; uniform early-tail weight can lag the final solution.

**Sources**: `experiments/011/proposals/idea-02.md`; `knowledge/papers/stochastic-weight-averaging.md`; EXP-004 and EXP-006 reports.

**Estimated Effort**: medium.

**Risk Assessment**: The parent exposes no best-to-final gap, expected gain is only 0.05-0.20 points, averaged BN state is approximate, and the trajectory may be too correlated. A five-sample switch also hides later online checkpoints by preregistered design.

### SAM-pulse-conditioned EMA
**Summary**: Initialize a full floating-state EMA at the first clean-tail SAM optimizer step and update only after subsequent SAM steps, with a fixed 512-pulse half-life (`beta=2**(-1/512)`). Evaluate EMA only after activation, average parameters and floating BN buffers, leave integer counters live, and restore the online model exactly after the sole evaluator call.

**What it targets**: Late variance while preferentially sampling the flatness-aware subsequence responsible for EXP-004's accepted gain.

**Reasoning**: A 512-pulse horizon corresponds to about 1,024 optimizer steps, 15.7 charged seconds, and 5.25 epochs. EMA favors the recent low-LR state over uniform full-tail averaging, and each sampled state includes all prior ordinary updates. Conditioning halves cadence relative to all-step EMA and composes directly with SAM.

**Sources**: `experiments/011/proposals/idea-03.md`; `knowledge/papers/how-to-scale-your-ema.md`; `knowledge/papers/stochastic-weight-averaging.md`; EXP-004 report.

**Estimated Effort**: medium-high.

**Risk Assessment**: Even-step post-SAM states are an unproven stroboscopic subsequence and may encode a two-cycle or data-order bias. Roughly 2,449 model-sized EMA updates add much more charged traffic than the other candidates, BN moment EMA remains approximate, and any gain is attributable only to this conditioned package.

### Horizon-derived clean-tail EMA
**Summary**: Start a full-state EMA at charged progress 0.75, sample the restored post-optimizer online state every 31 steps, and derive decay from actual elapsed charged time with an 18.75-second half-life (four half-lives across the 75-second tail). The odd cadence alternates ordinary/SAM step parity. Average parameters and floating BatchNorm buffers, copy integer counters, and evaluate exactly one EMA state per epoch after activation through an exception-safe swap/restore.

**What it targets**: Late-solution/model-selection variation in the global-best CutMix+SAM trajectory, without replacing any input, gradient, optimizer update, or evaluator call.

**Reasoning**: SWA provides CIFAR residual-network evidence for central late solutions, while EMA scaling says cadence and horizon must be coupled. Actual-time decay is stable under wall-clock throughput changes and sparse 31-step updates limit charged overhead. Four half-lives bound the initial tail state's final weight at 6.25%. Full floating-state averaging avoids live-BN mismatch without a free recalibration pass.

**Sources**: `experiments/011/proposals/idea-01.md`; `knowledge/papers/stochastic-weight-averaging.md`; `knowledge/papers/how-to-scale-your-ema.md`; EXP-004 and EXP-006 reports.

**Estimated Effort**: medium.

**Risk Assessment**: SAM and EMA may be redundant because EXP-004 final already equals best; cosine decay may collapse trajectory diversity; EMA of BN running moments is only an approximation; the fixed horizon may lag late improvement. Swap leakage would corrupt online training, so full-state restoration and parent-relative overhead gates are mandatory.

## Review

Claude selected horizon-derived EMA. It found the three candidates share an unverified transfer of late variance from EXP-006 because EXP-004 itself ended final-equals-best; trajectory-distance auditing is therefore promoted to a named secondary outcome. It also caught that cadence 32 always samples even SAM steps, so the selected policy now uses odd cadence 31 to alternate optimizer-step parity. The expected range is tempered from 95.70-95.95 to 95.45-95.70 while preserving the formal 95.50 and meaningful 95.70 thresholds. Per-BN EMA/live variance ratios are added to diagnose approximate running-stat averaging. Full feedback is in `experiments/011/01-idea-review.md`.

## Idea Evaluation

The Claude verdict is adopted. Horizon EMA scored 7/10 for evidence and 6.5/10 for impact, ahead of SAM-pulse EMA at 6/10 and 6/10 and epoch-SWA at 5/10 and 4/10. It provides the same plausible smoothing ceiling as dense pulse EMA at roughly one-sixteenth the shadow-update cadence, and its actual-time decay is robust to wall-clock throughput. Epoch-SWA's uniform stale early-tail weight and absent constant/cyclic LR make its expected effect sub-noise.

## Chosen Idea
**Selected**: Horizon-derived clean-tail EMA

**Why this idea**:
It is the lowest-risk implementation of the averaging hypothesis: the complete EXP-004 training path stays online and exact, sparse updates are charged, decay is derived from elapsed time, and a single predetermined EMA model is evaluated after activation. Odd cadence 31 removes implicit SAM-only sampling, while full-state swap/restore and online BN-buffer EMA avoid extra evaluation passes or uncharged recalibration. Even if accuracy is flat, consecutive-sample and EMA-to-live distances will resolve whether the parent tail had meaningful geometry to average.

**Hypothesis**:
On one fixed-seed GPU-0 run, cadence-31 time-horizon EMA will retain at least 25,200 optimizer steps, preserve exact CutMix/SAM semantics, produce finite nonzero trajectory and EMA-to-live distances with zero restoration mismatches, and reach at least 95.50% best test accuracy versus EXP-004 at 95.40%. A result at or above 95.70% is the stronger detectable-effect target; below 95.50% or a valid preflight failure is no improvement.
