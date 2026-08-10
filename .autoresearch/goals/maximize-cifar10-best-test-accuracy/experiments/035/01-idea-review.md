# Adversarial Review — EXP-035 Finalists

## Prioritized Feedback

1. **EMA's terminal-noise premise conflicts with local evidence.** EXP002 measured only a 0.01-point best/final gap, and EXP018's online checkpoint beat its uniform average. A 0.999 EMA over a monotonic cosine tail may simply lag. If pursued, shadow-versus-online weak-corpus behavior should be load-bearing, not merely diagnostic.
2. **EMA is genuinely distinct and cannot damage the strong phase, but it attacks a repeatedly weak region.** EXP018/030/032 all worsened tail behavior. Per-step exponential weighting, uninterrupted SGD, and no BN refresh distinguish this proposal from EXP018, yet measured headroom is small.
3. **Prior block-wide failures do not directly settle SiLU.** EXP012/015 changed residual ordering or branch scale/recruitment; SiLU preserves all three. The real local risks are signed-feature cancellation before GAP and loss of useful ReLU sparsity. Keep pooled-feature sign balance and strong-fit trajectory evidence load-bearing.
4. **SiLU can fail timing without producing accuracy evidence.** Sigmoid-backed backward touches the 75.46% bottleneck, and EXP029 showed nominally cheap code can cost 1.97%. The paired 2% timing gate is necessary, but an invalid timing veto is a likely information-limited outcome.
5. **Reflection has thin evidence and low predicted margin.** Its 94.27 point prediction is only two images above the formal gate under a one-seed protocol. Published-practice priors could strengthen evidence, but the ceiling remains modest.
6. **Reflection can still affect strong difficulty.** Roughly 13.4% expected crop area changes in the protected strong phase; it is a boundary prior, but not functionally negligible. The paired trajectory/switch-fit screen is warranted.
7. **Shared assessment:** none of the finalists violates scope, reward-hacks the metric, or exactly repeats a failed experiment. All correctly limit causal claims from a ten-image single-seed threshold.

## Scored Verdict

### Fixed SiLU Throughout ResNet-20

- **Evidence/reasoning: 6.5/10** — real ICLR signal-propagation mechanism and rigorous gates, but weak transfer to a shallow BN network.
- **Potential impact: 8/10** — the highest-ceiling finalist and a genuinely untried activation-family intervention targeting representation quality.

### Weak-Tail Full-State EMA

- **Evidence/reasoning: 6/10** — well differentiated from EXP018 and literature-backed, but its noise premise is contradicted by the nearly monotonic local tail.
- **Potential impact: 6/10** — bounded upside but uniquely unable to suppress strong-phase fit.

### Reflection-Padded Strong and Weak Crops

- **Evidence/reasoning: 5.5/10** — clean mechanism with weak direct empirical support and a prediction essentially on the gate.
- **Potential impact: 5/10** — low ceiling; flatness or a small regression is more likely than a clear gain.

## Pick

**Fixed SiLU Throughout ResNet-20.** It targets the diagnosed representation limiter rather than a weak tail with little measured headroom, is not shadowed by a direct local failure, and preserves the residual mechanisms implicated in prior block-wide regressions. Its throughput and signed-feature risks are real but explicitly measurable before the one scored run. EMA is the fallback; reflection is sound but lower ceiling.
