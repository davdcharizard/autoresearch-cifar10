# Report EXP-043: Convolution-Only Gradient Centralization
- **Created**: 2026-07-27

## Goal

Increase fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.48% frontier under the unchanged 300-second single-H20 budget. This experiment asked whether removing per-output-filter common-mode convolution data-gradient motion could improve generalization while retaining at least 127 data passes.

## Idea & Hypothesis

Apply gradient centralization to all 18 convolution weights after every backward and before unchanged coupled-decay Nesterov SGD. The pooled residual head, classifier, forward graph, loss, data, RNG, schedule, and optimizer state semantics remain accepted. The hypothesis predicted at least 94.58% best accuracy and at least 127 realized passes if common-mode convolution motion was a harmful generalization direction.

## Approach

Added a pure helper that subtracts each convolution filter gradient's mean across input-channel and spatial axes, cached the 18 convolution parameters after CUDA model construction, and invoked the helper between backward and SGD. The tracked diff was 11 additions in `train.py`; no parameter or configuration changed. An ignored harness directly invoked the production helper and independently loaded `a7c42dc:train.py` to verify exact common state/forward/raw gradients, selected membership, FP32/FP64 projection algebra, excluded gradients, coupled-decay ordering, fresh/preseeded Nesterov updates, RNG, temporal controls, and paired wall-time exposure.

## Execution

Semantic qualification passed after one harness-only `sys.path` correction made before any model check. Timing passed with median retention 0.987468, all four paired retentions 0.98518-0.99306, projected 128.671 passes, ratio CVs below 0.38%, and 622.712 MiB candidate allocation. The sole scored run then exited zero in 341.2 wall seconds with correct one-way transitions, 27 unique evaluations, one H20, and one finite summary. No production correction, score retry, or treatment adjustment occurred.

## Results

- **Primary metric**: 93.88% (baseline: 94.48%, delta: -0.60 points, -0.64%)
- **Observations**: The run retained 25,353 steps or 129.80736 passes, so exposure was not the cause of the miss. The projector achieved its intended local effect, but it removed 97.85-98.80% of the stem gradient norm on fixed hard/early fixtures and roughly 19-48% in deeper convolutions. Training still nearly interpolated the hard tail, while final test accuracy/loss worsened from accepted 94.45%/0.2456 to 93.87%/0.2661.
- **Analysis**: The normal-exposure result falsifies the hypothesis for this exact global rule. Common-mode filter directions were not benign noise: the unnormalized stem and positive residual inputs appear to carry substantial useful learning in those directions, and 1x1 shortcuts also lack a redundant spatial dimension. Since test loss and top-1 both worsened while terminal train loss approached zero, centralization changed the learned boundary adversely rather than merely slowing convergence. This does not prove every gradient projection is harmful, but it removes the rationale for immediate layer exclusions, partial strengths, schedules, alternate axes, or linear additions without a new diagnosis.
- **Key Learning**: Full-run per-filter convolution gradient centralization preserves exposure but removes useful common-mode learning, especially the dominant stem direction.

## Verification

- **Conditions**: Completion/resource contract passed; primary improvement and hypothesis accuracy conditions failed.
- **Review Notes**: Results are trustworthy. The production diff was isolated to `train.py`, accepted source and evaluator remained frozen, semantic and timing gates passed, the score was unique, and 129.80736 realized passes rule out low exposure as an explanation.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid fixed-budget score of 93.88% failed both the 94.48% baseline and 94.58% required threshold at normal exposure.

## Unexplored Avenues

- Layer-selective centralization that preserves the stem and 1x1 shortcuts could avoid the largest destructive projections, but this is an immediate post-result rescue and lacks an independent diagnosis; it is deliberately not promoted.
- Spatial-only kernel centering would preserve input-channel common modes while removing only spatial DC components, but 1x1 tensors make the rule inapplicable and no local evidence identifies spatial DC motion as harmful.
- Fusing reductions could reduce launch overhead, but timing was already non-limiting and cannot address the 0.60-point accuracy regression.

## Next Steps

- **High confidence**: Preserve raw convolution, pooled-head, and classifier gradients; seek a new mechanism that adds generalization signal without deleting learned directions.
- **Medium confidence**: Revisit the offline literature/knowledge base for a low-overhead target or representation intervention not adjacent to closed mixup, masking, pooling, or classifier families.
- **Low confidence**: Test optimizer-transition state only if a direct boundary diagnostic is first established; the previously proposed momentum reset has too short a supported effect to lead by default.
