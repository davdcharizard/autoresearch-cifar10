# Adversarial Idea Review — EXP-006

The accepted baseline is 94.07%; EXP-006 must reach at least 94.17% in one
fixed-seed, 300-second run. The width-3 option is correctly absent from the
finalists: its measured 56.8% relative throughput and projected 80.6 passes
fail the preregistered gates in `proposals/idea-01.md`. The remaining choice is
therefore among two low-ceiling endpoint interventions and one higher-variance
internal regularizer.

## Prioritized Feedback

**1. [Early block dropout] The proposal assumes feature co-adaptation is the
remaining limiter, but neither the local trajectory nor its cited notes measure
that condition.** `03-experiment-learnings.md` establishes that the accepted
model benefits from mild input mixup and regresses when that input regularizer
is strengthened; it does not show redundant or co-adapted internal features.
The local WRN note supports shallow width, but does not document dropout, and
the `Time Matters` note summarizes weight decay, augmentation, and mixup rather
than residual-branch dropout. The most likely failure is therefore additive
over-regularization: alpha-0.2 mixup plus six branch masks weakens an already
small 691,674-parameter WRN, reproducing EXP-005's higher-loss regression by a
different route. This is not fatal because feature masking is materially
different from stronger interpolation and has a larger plausible ceiling than
the endpoint ideas. **Improve it** by treating p=0.10 as a preregistered
exploratory treatment, retaining the exact `bn2/ReLU -> dropout -> conv2`
placement and 65% one-shot disable, and requiring the proposed warm matched-path
exposure gate; do not claim the cited notes directly validate this setting.

**2. [EMA] The evidence used to motivate variance reduction is mostly evidence
that little removable endpoint variance remains.** EXP-002 finished at its best
after accuracy continued improving through the hard-label tail
(`experiments/002/04-analysis.md`), while EXP-001 finished only 0.04 points
below best (`experiments/001/04-analysis.md`). A cosine schedule already drives
LR to 0.002. Evaluating only EMA after 65% can therefore hide a still-improving
live-model peak, and decay 0.999 can lag a trajectory that is not yet stationary.
The local paper note supports carefully windowed *checkpoint averaging* in
general, but does not establish this EMA decay, start point, or CIFAR/BN policy.
**Improve it** by explicitly classifying it as a low-headroom test and retaining
the proposed exposure preflight; there is no clean way to preserve live-model
peak detection under the one-evaluation-per-epoch comparison without changing
the experiment's metric selection policy.

**3. [EMA] Averaging BatchNorm buffers is coherent but not the state of the
averaged network, leaving a concrete evaluation mismatch.** As
`proposals/idea-02.md` acknowledges, EMA `running_mean` and `running_var` are
time averages of live-model statistics, not population statistics produced by
the EMA weights. This can erase a gain of only a few tenths. Exact post-training
recalibration would consume another augmented data pass outside the accepted
training path and is not an acceptable hidden free operation. **Improve it** by
keeping buffer averaging and reporting it as the treatment, as proposed, but
make a negative result reject only this EMA-plus-buffer approximation, not
weight averaging broadly. Do not use uncounted BN recalibration to rescue it.

**4. [Zero-floor cosine] The intervention is attribution-clean in throughput
and RNG terms, but its mechanism is weakly diagnosed.** The accepted final
equals best and improves through the hard-label tail; neither result shows the
late oscillation or peak erosion expected if residual Nesterov motion is the
limiter. Moreover, `MIN_LR` controls both the cosine floor and optimizer/warmup
start (`train.py:24`, `train.py:112`, `train.py:184`), so the advertised
hard-tail test also changes the first 5% of training. The proposal quantifies
that coupling honestly, but one-line simplicity does not make it more isolated.
**Improve it** by introducing a fixed `WARMUP_START_LR = 0.002` and changing
only the cosine floor to zero if the scientific question is specifically late
settling. Even then, the 7.9% reduction in hard-tail LR area implies a modest
ceiling close to the 0.10-point gate.

**5. [Cross-cutting] The hard acceptance margin is as large as the expected
gain of the endpoint candidates, with no variance estimate available.** The
goal forbids seed rerolling, and all history entries are single fixed-seed runs.
EMA and zero-floor cosine can be directionally correct yet fail the required
94.17% threshold. This is not permission to repeat a near miss; it is a reason
to favor a sound idea with enough upside to clear the bar and to interpret any
sub-threshold result strictly as no improvement.

**6. No finalist has a hard-constraint, reward-hacking, or unchanged-retry
failure.** All remain `train.py`-only, preserve the frozen evaluator and
evaluation cadence, charge added training work to the 300-second budget, and
retain seed 42. The proposed timing gates for EMA and dropout are legitimate
unscored feasibility checks, provided they never inspect test accuracy.

## Scored Verdict

| Candidate | Strength of evidence and reasoning | Potential impact |
|---|---|---|
| **Early p=0.10 WRN block dropout** | **5.5/10** — placement and temporal removal are coherent, but the cited local evidence does not establish co-adaptation or dropout benefit in this small, already-mixup-regularized WRN. | **7/10** — internal feature regularization is genuinely orthogonal and could move several tenths without sacrificing capacity, though additive over-regularization is the leading failure mode. |
| **EMA over the hard-label tail** | **6/10** — carefully specified and supported by general averaging literature, but stable final checkpoints and continued late improvement contradict the claimed variance-limited diagnosis; BN state remains approximate. | **4.5/10** — a mild gain is plausible, but lag, EMA-only evaluations, and the already low LR make a result above the 0.10-point bar unlikely. |
| **Zero-floor cosine** | **6.5/10** — the cleanest controlled schedule probe with exact quantitative analysis, but there is no observed endpoint instability and `MIN_LR` also changes warmup. | **4/10** — essentially free and possibly helpful, but it alters only a small fraction of update area and has the lowest plausible ceiling. |

**Strongest idea: Early p=0.10 WRN Block Dropout.** It wins because it is the
only finalist that tests a distinct representational-generalization mechanism
with a credible effect larger than the acceptance margin while preserving the
validated model, mixup schedule, hard-label tail, and nearly all exposure.
This is an exploratory pick, not a claim of stronger direct evidence: its
additive-regularization risk is real and must remain explicit. EMA loses because
the accepted endpoint is already stable and still improving, undermining both
its variance diagnosis and its ceiling; zero-floor cosine is cleaner but targets
an unobserved settling problem and likely moves less than the required margin.
