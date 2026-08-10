# Claude Adversarial Idea Review: EXP-008

Claude Opus reviewed the randomized candidates in this order: EMA, ECA, RandAugment. It had read-only access to the goal, system diagnosis, parent code/report, learnings, paper summaries, and all three full proposals.

## Prioritized Feedback

### Horizon-Matched Full-State EMA

- **Fatal: the proposed 15-second time constant samples a nearly stationary tail.** About 63% of the EMA mass would come from the final 15 charged seconds, where LR has already decayed from roughly 0.0034 to 0.002. This conflicts with the cited warning that averaging requires trajectory diversity.
- **Fatal: its own expected effect is below the protocol resolution.** The proposal estimates only +0.05 to +0.20 and EXP-004 finished at its best checkpoint, so the claimed variance limiter was not measured on the actual parent.
- **Fatal-adjacent: the fixed evaluator switch has a poor information tradeoff.** Once evaluation switches to EMA, the once-per-epoch constraint prevents observing the parent online tail. Averaged BatchNorm buffers also need not match averaged parameters, while recalibration would add an unfair uncharged data pass.
- If revisited, Claude recommended an earlier start and at least a 60-second horizon, but still judged the likely effect too small for this loop.

### Fixed Torchvision RandAugment

- **Major: use a published CIFAR WRN operating point rather than generic library defaults.** Direct CIFAR evidence is the candidate's main advantage, so the preregistered package should cite the matched source values rather than treating `(N=2,M=9)` as evidence-backed for this model.
- **Major: keep the clean SAM tail clean.** Leaving RandAugment active after progress 0.75 changes the input regime for the mechanism responsible for EXP-004's +0.17 gain. Gate it off with CutMix so the validated late path remains intact.
- **Moderate: loader work is partly outside the charged timer.** The proposal's preflight must benchmark the complete RNG-isolation wrapper and report total as well as charged runtime. Claude otherwise approved the private per-worker generator-state swap as the correct way to preserve crop/flip RNG semantics.

### Identity-Preserving Residual ECA

- **Moderate: the proposed descriptor comes from raw post-convolution output.** Claude suggested deriving the descriptor from the normalized `bn2`/ReLU input while applying the gate to `conv2` output, closer to the paper's controlled descriptor regime.
- **Moderate: global weight decay pulls zero-initialized attention kernels toward the identity gate.** A separate zero-weight-decay parameter group would remove that restoring force.
- The ImageNet effect ceiling is weakly calibrated to a six-block, ten-class CIFAR model; realistic upside may be below the 0.30-point resolution floor.
- Claude approved the RNG-neutral raw parameter, exact identity initialization, nonzero first gradient, fixed kernel sizes, SAM integration, and launch-latency gate.

### Constraint Screen

All three candidates respect the file, dependency, seed, hardware, timing, and evaluation constraints. None is an exact retry or evaluator manipulation.

## Scored Verdict

### Fixed Torchvision RandAugment Before CutMix
- **Evidence and reasoning: 7/10.** Only candidate with literature matched to CIFAR-10 and the WRN family, discounted for the original proposal's scalar and clean-tail choices.
- **Potential impact: 8/10.** Only finalist with a credible path beyond the roughly 0.30-point resolution floor; input diversity previously produced the lineage's largest post-architecture gain.

### Identity-Preserving Residual ECA
- **Evidence and reasoning: 6/10.** Strongest engineering design, but effect evidence is transferred from a much deeper ImageNet model and descriptor placement needs refinement.
- **Potential impact: 5/10.** Shallow-model transfer and identity-restoring weight decay make a small or null effect plausible.

### Horizon-Matched Full-State EMA
- **Evidence and reasoning: 2/10.** The fixed horizon conflicts with the trajectory-diversity premise and targets variance not observed in EXP-004.
- **Potential impact: 2/10.** Self-declared effect is below the evidentiary floor and the evaluator switch increases downside.

## Pick

**Fixed RandAugment in the regularized phase.** Claude selected it because its task/architecture evidence and effect ceiling are strongest. The selection is conditional on preregistering a source-matched CIFAR WRN configuration and disabling the policy at the 0.75 transition so the clean periodic-SAM tail is preserved.
