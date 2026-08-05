# Brainstorm EXP-013
**Created**: 2026-07-24

## Web Search & Literature Review

- `knowledge/papers/weight-averaging.md`: carefully windowed averaging can mildly improve generalization at low overhead.
- `knowledge/papers/label-smoothing.md`: mild target smoothing can reduce overconfidence, but stacking soft targets needs calibration.
- `knowledge/papers/time-matters-regularization.md`: early regularization can retain benefits after removal, supporting preservation of the accepted hard-label tail.

No network search was performed; the session is offline and these distilled local sources are sufficient.

## Experimental History Review

- Accepted WRN plus early mixup remains 94.07% at 141.9 passes. It reaches near-zero late training loss, so generalization rather than fitting throughput is limiting.
- Three low-resolution architecture probes are now closed: width 94.11, full block 94.15 with worse loss, and rank-64 bottleneck 93.74. More architecture is no longer the leading direction.
- BF16 exposure, stronger/shorter mixup, CutMix, dropout, late decay removal, and cosine-to-zero all failed. A generalization mechanism that does not corrupt targets/features or alter the optimizer schedule is the main gap.

## Diagnosis

The model has ample exposure and capacity to fit but limited held-out generalization. Architecture changes produced diminishing or negative returns, while additive regularization repeatedly hurt. The narrow next test is late iterate averaging on the accepted model; initialization geometry and transition-only smoothing are distinct lower-evidence controls.

## Collected Ideas

Quick pass; candidates are recorded below.

## Combinations

Quick pass; combinations are excluded to preserve single-variable attribution.

## Candidate Ideas

### Late Whole-State EMA
**Summary**: Initialize a detached FP32 EMA of all parameters and floating BatchNorm buffers on the first hard-label step at 65%; copy integral BN counters, update every subsequent step with decay 0.999, and evaluate only the EMA state at existing cadences via exception-safe swap/restore.

**What it targets**: Late specialization/calibration noise in an accepted model that already fits nearly perfectly, without added capacity or target corruption.

**Reasoning**: The existing semantic boundary yields roughly 9,700 averaging updates and a 0.999 decay has an approximately 1,000-update horizon. The fully developed EXP-012 proposal defines state, cadence, restoration, and a 95% retention/134.8-pass gate.

**Sources**: `experiments/012/proposals/idea-01.md`; `knowledge/papers/weight-averaging.md`; EXP-002/011/012 analyses.

**Estimated Effort**: medium.

**Risk Assessment**: Terminal-best accepted trajectories imply EMA lag may outweigh variance reduction; averaged BN moments approximate rather than exactly recalibrate EMA weights. Swap/restore integrity is mandatory.

### Zero-Initialized Residual Endpoints
**Summary**: Preserve accepted construction RNG, then overwrite all six basic blocks' `conv2.weight` tensors with zero. This safely starts residual corrections closed; literal pre-ReLU `bn2` zeroing is forbidden because it permanently kills gradients.

**What it targets**: Optimization geometry and early basin selection without compute, parameter, or explicit regularization changes.

**Reasoning**: The treatment retains full exposure and branches open after one update, but it lacks direct local evidence. Detailed gradient semantics are already developed in EXP-012.

**Sources**: `experiments/012/proposals/idea-03.md`; accepted `PreActBlock`; EXP-011/012 analyses.

**Estimated Effort**: low.

**Risk Assessment**: The shallow WRN may not benefit, and removing early random residual features can worsen the fixed schedule.

### Transition-Only Label Smoothing
**Summary**: Use epsilon 0.05 label-smoothed cross-entropy only from 65% to 85% counted time, then exact hard labels for the final 15%; preserve accepted mixup before 65%.

**What it targets**: Overconfidence while retaining a clean terminal convergence window.

**Reasoning**: Label smoothing is literature-backed and temporally isolated from mixup, but local additive/stronger regularization has repeatedly regressed and the accepted 35% hard tail is valuable.

**Sources**: `knowledge/papers/label-smoothing.md`; `knowledge/papers/time-matters-regularization.md`; EXP-003/005/006.

**Estimated Effort**: low.

**Risk Assessment**: It may simply extend soft-target underfitting into the hard-label tail and repeat the established over-regularization failure.

## Review

The blind offline critic selected late whole-state EMA (7.0 evidence / 6.0 impact). I accept that checkpoint variance is inferred rather than observed and that averaged BN moments are approximate. The fixed policy therefore requires strict swap/state integrity and a normal-exposure miss closes only this exact window. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer pick. EMA targets generalization without shortening the accepted hard-label tail or changing the optimization basin; zero endpoints remain less evidenced, and transition smoothing conflicts with repeated local regularization failures.

## Chosen Idea
**Selected**: Late Whole-State EMA

**Why this idea**:
It is the strongest remaining orthogonal treatment after architecture and additive regularization failures. The exact implementation is already developed in `experiments/012/proposals/idea-01.md` and can be rejected cleanly.

**Hypothesis**:
Whole-state EMA initialized on the first hard-label update at 65%, updated every later step with decay 0.999, will retain at least 95% throughput / 134.8 projected passes and reach `best_test_acc >=94.17%` in one fixed-seed run.
