# Brainstorm EXP-008
**Created**: 2026-07-24

## Web Search & Literature Review

- **Prior zero-floor proposal and critique** (`experiments/006/proposals/idea-04.md`; `experiments/006/01-idea-review.md`)
  A cosine-to-zero endpoint is an attribution-clean optimization probe only if its floor is decoupled from the accepted 0.002 warmup start; its likely ceiling is modest because the accepted run ended stably at its best.
- **Weight averaging** (`knowledge/papers/weight-averaging.md`)
  Late parameter averaging can improve generalization at low arithmetic cost, but the local review identified BatchNorm-state mismatch and lag as material risks under this short, still-improving trajectory.

No network search was performed because this session is explicitly offline. Existing local knowledge and the prior adversarial review cover the narrow remaining schedule question.

## Experimental History Review

- EXP-001 established the accepted WRN-16-2 architecture and counted-time cosine schedule at 93.38%; EXP-002 added alpha-0.2 mixup through 65% and remains best at 94.07% with 141.9 data passes and final test loss 0.2432.
- Five controlled follow-ups regressed at normal exposure: CutMix 93.72%, a 50% mixup cutoff 93.91%, alpha-0.4 mixup 93.57%, early block dropout 93.52%, and removing late matrix decay 93.74%. The last two sharply worsened test loss, so continuous `5e-4` matrix decay and mixup alone are fixed for this loop.
- The accepted run finished at its best and continued improving during its hard-label tail, which is evidence against a severe endpoint-settling problem. However, the bundled 0.002 cosine floor has never been isolated, and it still drives several thousand late Nesterov updates after most representation learning is complete.
- The inline limiter diagnosis is therefore narrow: the accepted recipe has no obvious exposure or stability failure, but its late update amplitude and first-layer input conditioning remain untested. A useful experiment must preserve the validated early trajectory and have essentially zero throughput cost.

## Collected Ideas

Quick pass; candidate ideas are recorded directly below.

## Combinations

Quick pass; candidates remain isolated and are not combined.

## Candidate Ideas

### Late Momentum Taper
**Summary**: Preserve LR and weight decay exactly, but reduce SGD momentum smoothly from 0.9 to zero over the final 10% of counted training by updating each live parameter group's `momentum`. The first 90% of the accepted path remains unchanged.

**What it targets**: If persistent velocity rather than scalar LR is the source of late overshoot, damping momentum can let the hard-label endpoint settle while retaining the accepted nonzero LR for direct gradient refinement.

**Reasoning**: This separates late velocity memory from learning-rate amplitude and costs no additional model operations. It is materially different from EXP-007 because matrix decay remains continuous. The final 10% contains roughly 2,800 steps, enough for a smooth taper rather than an abrupt optimizer-state reset.

**Sources**: `train.py` optimizer configuration; EXP-002 endpoint trajectory; EXP-007 evidence requiring continuous matrix decay.

**Estimated Effort**: low.

**Risk Assessment**: There is no direct local evidence that momentum overshoot is limiting accuracy, and mutating momentum changes optimizer dynamics while retaining a buffer accumulated under 0.9. The accepted stable endpoint argues against the diagnosis; reduced momentum may simply slow useful late movement.

### Evaluator-Consistent Channel Standardization
**Summary**: Register fixed CIFAR-10 channel-standard-deviation buffers in `WideResNet` and divide the already mean-centered input at the start of `forward`. The frozen train and test transforms subtract identical channel means with unit scale, so in-model scaling applies consistently to both paths without touching `prepare.py`.

**What it targets**: The first convolution is the only major convolution not preceded by BatchNorm and currently receives channels with unequal sub-unit variance. Fixed channel scaling tests whether stem conditioning limits early optimization.

**Reasoning**: Canonical CIFAR channel standard deviations are already known locally, the operation is a single broadcast divide, and model-level placement preserves evaluator consistency. This is orthogonal to failed target, spatial, dropout, and decay interventions.

**Sources**: `train.py` and frozen `prepare.py` transform definitions; EXP-007 brainstorm and analysis recommendations.

**Estimated Effort**: low.

**Risk Assessment**: The following residual-stack BatchNorm layers may compensate for stem scale, so the change may be neutral. It changes the effective input magnitude throughout training and is broader than the schedule probe; fixed buffers and evaluator agreement must be preflighted.

### Decoupled Cosine-to-Zero Floor
**Summary**: Add `WARMUP_START_LR = 0.002`, set `MIN_LR = 0.0`, initialize SGD at the warmup-start value, and use the new constant only for the first 5% linear warmup. Post-warmup cosine decay then reaches exactly zero while the accepted warmup path remains unchanged.

**What it targets**: The accepted schedule still applies LR 0.002 at the 300-second boundary. This tests whether residual late Nesterov motion limits final hard-label settling without the warmup coupling identified by the EXP-006 critic.

**Reasoning**: The proposed LR is 0.05983 at the 65% transition versus 0.06123 accepted, 0.00542 at 90% versus 0.00736, and zero at the endpoint. It reduces hard-tail LR area by about 7.9% and final-10% area by about 52.2% without changing operations, RNG draws, memory, exposure, decay, or evaluation cadence. The schedule floor was bundled into EXP-001 and remains unisolated.

**Sources**: `experiments/006/proposals/idea-04.md`; `experiments/006/01-idea-review.md`; EXP-001, EXP-002, and EXP-007 analyses.

**Estimated Effort**: low.

**Risk Assessment**: Final equaled best and the hard-label tail still made progress, so no observed oscillation proves the floor is harmful. Cosine-to-zero may freeze useful margin refinement, and any benefit may be smaller than the required 0.10 points. A lower loss without at least 94.17% remains a strict no-improvement.

## Review

The offline blind critic selected the decoupled cosine-to-zero floor as the cleanest controlled test. I accepted its significant cautions: the accepted final-equals-best trajectory is contrary evidence, the expected ceiling is modest, and the implementation must use `WARMUP_START_LR = 0.002` for both optimizer initialization and the entire 0-5% warmup while retaining continuous `5e-4` matrix decay. Schedule values will be asserted at 0%, 5%, 65%, 90%, 95%, and 100%; one sub-threshold run will not trigger a nearby floor sweep. Full concerns and scores are in `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. The zero-floor candidate scored 7/10 for evidence and 4.5/10 for impact: its upside is limited, but it uniquely isolates an untested schedule parameter with exact arithmetic and no throughput, RNG, exposure, or evaluator confound. Channel scaling has a slightly higher speculative ceiling but is largely canceled by the immediate stem-output BatchNorm; momentum taper entangles buffer memory and effective gradient gain without trajectory evidence of overshoot.

## Chosen Idea
**Selected**: Decoupled Cosine-to-Zero Floor

**Why this idea**:
It preserves every accepted stochastic and regularization choice while testing the sole bundled schedule endpoint that has not been isolated. Decoupling the warmup start resolves the prior critic's main attribution concern, and exact schedule checks make the intervention falsifiable even though the stable accepted endpoint limits confidence.

**Hypothesis**:
Preserving the accepted 0.002-to-0.2 warmup but annealing post-warmup LR to zero will reduce unproductive late Nesterov motion during the hard-label tail. With unchanged exposure and continuous matrix decay, the run should reach `best_test_acc >= 94.17%`, with final accuracy at or near best and final test loss no worse than 0.2432. Any valid result below 94.17% falsifies the intervention for this loop regardless of loss.
