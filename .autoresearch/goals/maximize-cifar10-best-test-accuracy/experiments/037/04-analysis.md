# Report EXP-037: Mean-Centered Stem Convolution
- **Created**: 2026-08-06

## Goal

Test whether a bounded, stem-only representation change can raise seed-42 CIFAR-10 `best_test_acc` from the moving 94.15% baseline to at least 94.25%, while changing only `train.py` and preserving the fixed one-H20, 300-second training protocol.

## Idea & Hypothesis

The candidate subtracts each output filter's coefficient mean only in the image-facing convolution. It was selected because the projection removes a local DC/common-mode response, cannot expand effective weight norm, and is much narrower than prior all-convolution centralization. The hypothesis was that this would create a persistent post-BN/pooled representation change without unstable class geometry or material fixed-budget cost, then improve generalization.

## Approach

`MeanCenteredConv2d` derived from `nn.Conv2d` and used `weight - weight.mean((1,2,3), keepdim=True)` through `_conv_forward`; only `ResNet.conv1` used the subclass. Stored parameters, Kaiming initialization, residual convolutions, BN, optimizer, decay, data policy, schedule, evaluator, and parameter count remained unchanged. The reviewed plan required exact construction and FP64 projection oracles, followed by a control-relative mechanism-survival test at initialization and after 64 registered strong batches. Full safety, timing, and production were conditional on that test.

## Execution

One local preflight was run with no retries or production attempt. Static/scope checks and exact construction passed: 19 convolutions, 19 BN layers, one linear layer, 1,073,962 parameters, and only `conv1` projected. The FP64 oracle matched outputs and input gradients exactly, with effective filter means at `6.17e-18` and projected raw-weight gradient means at `3.95e-16`. The controller durably wrote its report, then exited nonzero because both step-64 mechanism checks failed. The warning about converting an oracle tensor to a scalar was diagnostic-only.

## Results

- **Primary metric**: NaN (baseline: 94.15%, delta: N/A; production was not authorized)
- **Observations**: At initialization, exact accepted controls matched while the candidate changed pooled features by relative L2 `0.2498`/`0.2129` and logits by `0.1988`/`0.1793` on hard/CutMix views. After 64 shared batches, accepted control/control divergence itself reached `0.6530` and `0.7927`. Candidate maximum divergence was `1.0518` and `1.1741`, only `1.61x` and `1.48x` the matching control floor rather than the required `5x`. Candidate loss ratios (`0.9465`, `0.9665`) and logit scales remained bounded, with no candidate-only concentration.
- **Analysis**: The implementation achieved its exact local projection and clearly changed the initial representation, so it was not a mathematical no-op. It did not, however, remain distinguishable enough from the accepted trajectory's 64-step numerical divergence under the preregistered generic feature/logit test. That evidence neither establishes an accuracy regression nor proves BatchNorm completely cancels the projection; it makes this exact stem-only point unqualified for the expensive single production run. The generic long-horizon divergence statistic is also weakly identifiable once accepted repeats have chaotically separated.
- **Key Learning**: Stem centering changes initial representations, but its 64-step effect was only 1.48-1.61x accepted control divergence and failed the required persistence margin.

## Verification

- **Conditions**: Static scope, construction, inventory, RNG/state equality, FP64 projection, finite ratios, and concentration checks passed; both step-64 control-relative mechanism-survival conditions failed. Full safety, timing, and production verification were intentionally not run.
- **Review Notes**: The report hashes bind the controller (`79bac4d8…`) and tracked source (`5e51ebad…`), exact controls matched at initialization, and both hard/CutMix failures agree. There is no stale production metric to misclassify.
- **Verdict**: invalid
- **Verdict Basis**: The prospective plan explicitly classified a mechanism veto before production as invalid/NaN. No hard constraint was breached and no infrastructure crash occurred, but no trustworthy primary metric was produced.

## Unexplored Avenues

- An effect-specific probe of stem DC-response removal could remain identifiable after training better than whole-feature/logit divergence, but it would require a newly reviewed hypothesis and protocol rather than relaxing EXP037 post hoc.
- All-layer mean centering or variance-standardized weights may produce a stronger effect, but prior all-Conv centralization cost and scale-reparameterization failures make them separate high-risk ideas.

## Next Steps

- **High confidence**: preserve the accepted stem and explore a mechanism with an intrinsic output bound or direct generalization rationale rather than another small projection.
- **Medium confidence**: reconsider the deferred fixed-scale cosine head only with scale sensitivity bounded analytically and CutMix-compatible magnitude checks.
- **Medium confidence**: prioritize architecture or throughput changes that preserve the accepted short-phase fit and can be judged directly under the fixed budget.
