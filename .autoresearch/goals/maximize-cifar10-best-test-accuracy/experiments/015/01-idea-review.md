# EXP-015 Adversarial Idea Review

**Reviewer**: External Claude idea critic
**Status**: completed successfully; no fallback reviewer used

## Prioritized Feedback

1. **EMA targets variance contradicted by local evidence.** EXP-002 found only a 0.01-point best/final gap and accepted annealed trajectories are monotonic with final at best. A one-epoch-half-life EMA on a still-improving low-LR path is more likely to introduce lag than remove the 0.10 points of variance needed for success. If retained, frame it as an unproven sub-epoch-noise test rather than smoothing known variance.
2. **EMA has an unresolved BatchNorm-state confound.** Current online buffers are not exact moments for EMA parameters, and the proposal admits this mismatch can erase the target-sized gain. EMA buffers double-lag, while recalibration adds another mechanism/data pass, so there is no clean local resolution.
3. **EMA forfeits the known-good endpoint.** After 90% it evaluates only EMA. Since accepted final equals best, this bets a lagged estimator against a still-improving online endpoint; evaluating both would bias the observation budget.
4. **BF16 attacks the systems bottleneck, not the diagnosed accuracy limiter.** The system profile says generalization under the short strong phase limits accuracy and explicitly notes that more exposure has not been causally tested. A clean speedup therefore does not establish the accuracy hypothesis.
5. **BF16 changes both throughput and the gradient field.** A pass could be a net BF16-method gain but cannot be attributed solely to exposure. The proposal correctly treats exposure-miss/accuracy-pass as unattributable.
6. **BF16 may end at preflight.** Small CIFAR maps, BatchNorm/add launches, and existing accelerated FP32 kernels could miss the 1.15x gate, producing no accuracy result.
7. **Zero-gamma's main risk is a small effect, not the one-step recruitment delay.** Unlike preactivation or stronger CutMix, it adds no regularization pressure and preserves topology/exposure. One delayed branch update among roughly 21k strong steps is negligible; neutrality at ResNet-20 depth is more likely than harm.
8. **Zero-gamma literature transfer is directional.** Goyal et al.'s 0.24-point ImageNet gain used a deeper projection-shortcut model and different regime. The reviewer noted that safe non-padded `layer1[0]` could optionally also be zeroed to enlarge effect, but this would change the predeclared six-block candidate.
9. **Zero-gamma has the strongest safety protocol.** Finite/nonzero gamma gradients, a bounded first update, replay-loss/class-concentration checks, second-backward recruitment, and a 64-step fit check directly internalize EXP-014's first-update-scale lesson. Leaving padded transitions active avoids the proven dead-channel mechanism.
10. **Any bare pass is weak evidence.** The 0.10-point single-seed threshold is ten test examples; do not overclaim causality.

## Scored Verdict

| Candidate | Evidence & Reasoning | Potential Impact |
|---|---:|---:|
| Same-Width Zero-Gamma Identity Init | **8/10** — canonical, compute-neutral, carefully scoped, and strongly de-risked; docked for thin shallow-network transfer. | **6/10** — modest but credible 0.10-0.30 point ceiling with likely null rather than catastrophic failure. |
| H20 BF16 Autocast Exposure | **5/10** — rigorous systems gates, but unsupported exposure-to-generalization link and precision confound. | **6/10** — high nominal ceiling if both speed and extra exposure help, with a plausible preflight-null outcome. |
| Late Weak-Tail Parameter EMA | **5/10** — rigorous protocol and external support, but its local variance premise is contradicted. | **3/10** — low local headroom, structural BN mismatch, and loss of the known-good online endpoint. |

## Pick

**Same-Width Residual Identity Initialization: zero-gamma on the six ordinary blocks.**

It is the only finalist that directly engages the diagnosed optimization/generalization limiter while preserving the accepted exposure and postactivation recipe. It has clean attribution, directional literature evidence, avoids Option-A transition dead channels, and includes first-update gates tailored to EXP-014's failure. Its most likely miss is a sub-threshold null rather than a regression.

The reviewer placed BF16 second because its measured systems premise does not yet establish an accuracy premise, and EMA third because it fights the observed monotonic endpoint trajectory. Keep the six-block scope and honest single-seed framing; do not weaken the safety gates.
