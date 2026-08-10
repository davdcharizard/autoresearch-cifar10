# External Claude Idea Review - EXP-017

The mandatory external Claude review completed successfully. No fallback reviewer was used.

## Prioritized Feedback

1. All candidates have indirect mechanism-to-limiter evidence: the diagnosed limiter is generalization under the short strong phase, while transition transport, conditional channel allocation, and basin selection are plausible but not locally proven causes. Preserve switch accuracy near EXP010's 89.73% as the main mechanism diagnostic.
2. Identity-scale ECA has the weakest transfer and a live timing-veto risk. EXP012's all-block SE probe cost 1.233x; nine ECA chains remain sequential, `2*sigmoid` departs from published ECA, global CutMix descriptors conflict semantically, and channel adjacency has no guaranteed meaning.
3. Nesterov has excellent attribution but an intrinsically low ceiling. Its 94.30 point estimate is only five test examples above the formal gate, with no local evidence that it helps the accepted recipe.
4. Learned transition shortcuts have the sharpest first-update risk. Kaiming projection plus gamma-one BN injects full-scale random signal into all channels immediately, while the RMS gate permits up to 4x residual scale. Keep the catastrophic gate but flag any value above 2x during analysis.
5. Prior EXP012 review judged plain projection-shortcut ceiling low, but pool-first ResNet-D transport is materially different. Direct CIFAR ResNet20 evidence that downsampling configuration matters is the load-bearing source; ImageNet ResNet-D evidence supports only the mechanism.
6. Exposure and evaluation-count fairness are sound for all finalists; none grants extra max-metric observations, and thresholds match their exposure arithmetic.
7. Describe the chosen method precisely as shortcut information transport, not full anti-aliasing, because the residual stride-2 path remains unchanged.

## Scored Verdict

| Candidate | Evidence and reasoning | Potential impact |
|---|---|---|
| Learned transition shortcuts | **7/10** - converging ResNet-D and direct CIFAR ResNet20 evidence, a coherent mechanism, and fully active accepted residual branches; transfer and ceiling remain indirect. | **7.5/10** - highest leverage at width-expansion boundaries and a plausible 94.25-94.50 range. |
| Identity-scale ECA | **5/10** - elegant identity initialization, but weak shallow-network transfer, modified gate semantics, CutMix conflict, and strong timing precedent. | **5.5/10** - modest ceiling with a material chance of preflight timing no-go. |
| Isolated Nesterov | **6/10** - cleanly isolated and genuinely untested, but lacks evidence for this recipe. | **4.5/10** - point estimate and ceiling sit close to single-seed resolution. |

## Pick

**Learned Anti-Loss Transition Shortcuts.** It has the strongest product of evidence and upside, modifies the stage transitions where width introduces its proven capacity gain, and preserves all nine accepted residual branches at full activity. Its main risk is observable and pre-registered: a full-scale random learned shortcut can dominate or interfere with the residual path, so hard/soft first-update, 200-batch trajectory, timing, and exposure gates must all pass before production.
