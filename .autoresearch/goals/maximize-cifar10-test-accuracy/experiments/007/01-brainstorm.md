# Brainstorm EXP-007
**Created**: 2026-07-24

## Web Search & Literature Review

- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early regularization can retain its generalization benefit after removal, motivating an isolated late weight-decay-off candidate.
- **Prior zero-floor proposal and blind critique** (`experiments/006/proposals/idea-04.md`; `experiments/006/01-idea-review.md`): a zero cosine floor is clean and quantitative, but must be decoupled from the 0.002 warmup start to test only late settling.

No network search was performed because this session is explicitly offline; the local knowledge base and completed EXP-006 review provide the relevant evidence.

## Experimental History Review

- EXP-001 established WRN-16-2 with a counted-time warmup/cosine schedule at 93.38%; EXP-002 added alpha-0.2 mixup through 65% and remains the 94.07% accepted baseline.
- Four subsequent regularization changes failed at normal exposure: shared CutMix 93.72%, a 50% mixup cutoff 93.91%, alpha-0.4 mixup 93.57%, and p=0.10 block dropout 93.52%. The last two also worsened test loss, so additional stacked regularization is not the current priority.
- WRN-16-3 was screened out before EXP-006 because it retained only 56.8% of width-2 throughput and projected about 80.6 passes. Capacity is available in VRAM but not cheaply in counted time.
- The accepted run finishes at its best with loss 0.2432 after continued hard-tail gains. The remaining narrow gap is most plausibly an unisolated optimization endpoint or input-conditioning choice, not insufficient exposure or an obviously unstable training path.

## Collected Ideas

Quick pass; candidate ideas are recorded directly below.

## Combinations

Quick pass; candidates remain isolated and are not combined.

## Candidate Ideas

### Disable Weight Decay for the Hard-Label Tail
**Summary**: Keep selective `5e-4` weight decay through the first 65% of counted training, then set the decay parameter group's `weight_decay` to zero exactly when mixup ends. Every architecture, LR, stochastic operation, and evaluation setting remains accepted.

**What it targets**: The final hard-label phase must refine already regularized features and margins. Continuing norm shrinkage throughout this low-LR tail may oppose that clean-label fit even though early decay supplied its main generalization effect.

**Reasoning**: The local Time Matters note supports removing regularization after the critical early period, and EXP-002 validates the 65% boundary for a different regularizer. This is a low-overhead optimization intervention that does not add stochastic noise after four regularization regressions.

**Sources**: `knowledge/papers/time-matters-regularization.md`; EXP-002 analysis; `03-experiment-learnings.md`.

**Estimated Effort**: low.

**Risk Assessment**: Weight decay and mixup need not share an optimal cutoff. Training loss is already near zero, so removing decay may only enlarge norms, worsen calibration, or overfit without improving class decisions. The mechanism has weaker direct local evidence than the schedule-floor probe.

### Decoupled Cosine-to-Zero Floor
**Summary**: Add `WARMUP_START_LR = 0.002`, set `MIN_LR = 0.0`, initialize SGD at the warmup-start value, and use the new constant only in the first 5% linear warmup. The post-warmup cosine then reaches exactly zero while the entire accepted warmup trajectory remains unchanged.

**What it targets**: The accepted schedule still applies LR 0.002 at the 300-second boundary. This candidate tests whether residual late Nesterov motion limits final hard-label settling, without the warmup coupling identified by the EXP-006 critic.

**Reasoning**: Relative to the accepted schedule, the change is negligible soon after warmup but reduces LR by 26.4% at 90%, 59.3% at 95%, and 100% at the endpoint; it cuts hard-tail LR area by about 7.9% with no operation, RNG, memory, or throughput change. The schedule floor was bundled into EXP-001 and has never been isolated.

**Sources**: `experiments/006/proposals/idea-04.md`; `experiments/006/01-idea-review.md`; EXP-001 and EXP-002 analyses.

**Estimated Effort**: low.

**Risk Assessment**: The accepted endpoint is stable and still improving, so no observed oscillation proves the floor is harmful. Cosine-to-zero may freeze useful margin refinement, and its plausible gain may be smaller than the required 0.10 points.

### Evaluator-Consistent In-Model Channel Standardization
**Summary**: Register fixed CIFAR-10 channel-standard-deviation buffers in `WideResNet` and divide the already mean-centered input at the start of `forward`. Because the frozen train and test transforms both subtract the same means with unit scale, applying division inside the model standardizes both paths identically without touching `prepare.py`.

**What it targets**: The first convolution is the only major layer not preceded by BatchNorm, yet it receives channels with unequal raw standard deviations. Canonical in-model scaling may improve first-layer conditioning and allocate gradient signal more evenly across channels.

**Reasoning**: The code explicitly uses `(1, 1, 1)` as transform standard deviations for historical compatibility. Moving fixed scale correction into the model preserves evaluator consistency and costs only a broadcast divide, while Kaiming initialization is better matched to roughly unit-variance inputs.

**Sources**: `train.py` and frozen `prepare.py` transform definitions; EXP-006 analysis recommendation.

**Estimated Effort**: low.

**Risk Assessment**: First-layer and later BatchNorm parameters may already compensate for channel scale, so the change can be neutral. It changes effective input magnitude and therefore early optimization more broadly than the schedule probe; an incorrect buffer shape/device or train/eval mismatch would invalidate the run and must be preflighted.

## Review

The blind critic selected late weight-decay removal because the local temporal-regularization evidence applies to the exact regularizer being changed and its plausible headroom exceeds the narrowly late cosine-floor probe. The main concern is that 65% is validated only for mixup, not weight decay; EXP-007 will therefore treat it as a deliberately isolated first cutoff, change only the decayed optimizer group's value once, verify the no-decay group remains zero, and record matrix-parameter norms around the switch. A negative completed run will not authorize a result-conditioned cutoff adjustment.

The decoupled zero floor remains the cleanest attribution experiment but targets no observed endpoint instability and likely has a sub-threshold ceiling. Channel standardization was downgraded because immediate stem-output BatchNorm cancels much of the scale effect and the original `fan_out` initialization rationale was technically incorrect. Full concerns and scores are in `01-idea-review.md`.

## Idea Evaluation

Adopt the critic's verdict in `01-idea-review.md`. Late weight-decay removal scored 6.5/10 for evidence and 5.5/10 for impact, offering the best balance after four added/altered regularizers regressed. The choice preserves all early regularization and tests only whether coupled L2 shrinkage constrains the accepted clean-label tail.

## Chosen Idea
**Selected**: Disable Weight Decay for the Hard-Label Tail

**Why this idea**:
It is a one-shot, low-overhead optimization change grounded in evidence that early weight decay can retain its generalization benefit after removal. Unlike the failed experiments, it adds no stochastic or target regularization and leaves the validated first 65% intact. It also has more plausible multi-tenth headroom than the zero-floor schedule probe.

**Hypothesis**:
Setting only the decayed SGD parameter group's `weight_decay` from `5e-4` to `0.0` at the existing 65% mixup transition will preserve early norm regularization while allowing more effective hard-label margin refinement. With unchanged exposure, the run should achieve `best_test_acc >= 94.17%` and final test loss no worse than 0.2432; worse loss or accuracy with normal exposure will indicate that continued late decay remains beneficial.
