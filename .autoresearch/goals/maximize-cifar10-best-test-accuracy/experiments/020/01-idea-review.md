I have all the context I need. Here is my adversarial review.

---

# Adversarial Review - EXP-020 Candidate Ideas

## Prioritized Feedback (most important first)

**1. [FATAL as pinned - PNM] The approximately 22.36x deterministic-scale collapse makes severe underfit the expected outcome in this fixed-budget regime, and the proposal forbids the one fix.**
The proposal's own math (idea-01.md L67-74) is correct: each PNM stream converges to `d` (decay `mu^2=0.81`, injection `0.19`), so the steady direction is `d/sqrt(5)`, while PyTorch momentum converges to `d/(1-0.9)=10d`. At the same numeric LR 0.1, PNM takes steady steps approximately 22.36x smaller. With stream time-constant approximately five steps, this dominates the entire approximately 21.5k-step strong phase - the model trains the whole plateau at an effective LR near 0.0045. On a model whose recipe depends on a healthy 89.73% switch, and which has repeatedly failed when strong fit is suppressed, a 22x step shrink is the largest fit-suppressor of any candidate here. The proposal even pre-registers a switch below 87.08 as expected but forbids LR rescue. **Fix:** scale the direction by approximately 22x or set LR to match EXP-010's effective step magnitude so PNM becomes a clean noise-geometry-only test. As pinned, the paper's 0.53-point gain is unreachable.

**2. [PNM] The cited evidence is doubly non-transferable, not just LR-mismatched.** Xie et al.'s 4.48% versus 5.01% result was obtained in a long-horizon 200-epoch regime and with the official code's `decoupled=True` decay, whereas this proposal pins coupled `decoupled=False`. The headline rests on a different horizon and decay coupling. **Fix:** state the evidence as directional-only, match the paper's decoupled decay, and match effective LR; otherwise the closest-positive-result claim is not evidence for this configuration.

**3. [Stochastic depth] This lands squarely inside a count-2 recurring failure family on the exact model that triggers it.** The learnings record identity-oriented residual changes suppressing the short strong phase in EXP-012 and EXP-015. Returning `x` from a residual branch is an identity-oriented residual change on shallow ResNet-20. The proposal concedes even 5% dropping can recreate the recurring underfit signature. **Fix:** there is no clean in-scope fix; external evidence is for deeper ResNets, and per-sample drop-path cannot conditionally skip kernels. Retire for ResNet-20 unless a distinct sub-mechanism is justified.

**4. [Stochastic depth] The compute-savings mechanism is marginal and likely illusory.** The proposal accounts for only 2.81% schedule-weighted forward-MAC reduction and demands at least 1% realized speedup. Python branch decisions, mask draws, and scaling add overhead. Even a pass lacks a demonstrated accuracy link because extra accepted-recipe exposure has not been causally tested. The backward-bottleneck framing does not connect to the accuracy limiter while the regularization half fights shallow underfit. **Fix:** none within scope that preserves the idea.

**5. [Stochastic depth] Reward-hack / measurement-gaming vector via evaluation count.** Faster steps create more epochs and potentially more per-epoch evaluations, inflating a max metric. The proposal correctly caps at 19 evaluations, but any pass at a higher count would be exposure/evaluation-confounded. The BN train/eval distribution mismatch further muddies attribution.

**6. [Nesterov - evidence weakness, not fatal] Direct evidence for Nesterov beating ordinary momentum on CIFAR is absent; the case is attribution and cleanliness, not expected magnitude.** The papers are directional only, and extra current-gradient weight could amplify CutMix/RandAugment noise or weaken useful historical filtering. The predicted 94.30% and 94.25% gate sit near the ten-image noise floor. **Mitigating fact:** unlike PNM, Nesterov has no scale pathology - its steady-state direction is `d + mu*b = 10d`, identical to ordinary momentum's `10d`; the 1.9x first step is a transient. This is the only candidate that does not fight shallow underfit or carry a transfer gap. **Fix:** frame its value as resolving EXP-001's confound and pre-register switch fit near 89.73% so deviation is a genuine signal.

**7. [All three] The +0.10-point gate on one un-rerollable seed is approximately ten test images.** Fixed-time CUDA trajectory variance constrains the credible impact of all three and favors the cleanest causal read per unit risk.

---

## Scored Verdict

**Default-beta0 PNM (idea-01)**
- Evidence/Reasoning: **5/10** - closest positive external CIFAR/ResNet result, but the pinned configuration nullifies transfer through effective-LR collapse, decay mismatch, and long-horizon-to-short-budget gap.
- Potential Impact: **3/10** - high ceiling in principle but foreclosed by expected underfit at pinned LR 0.1.

**Isolated Nesterov (019/idea-01)**
- Evidence/Reasoning: **5/10** - thin direct evidence, but internally consistent and pathology-free: same steady-state scale, clean one-keyword isolation, and a real unresolved confound.
- Potential Impact: **4/10** - modest but reachable ceiling, negligible cost risk, and no direct conflict with shallow-model fit.

**Low-Rate Batchwise Stochastic Depth (idea-03)**
- Evidence/Reasoning: **3/10** - overlaps a count-2 identity-oriented failure family, external evidence favors deeper nets, and compute/attribution are weak.
- Potential Impact: **3/10** - likely underfit, marginal exposure upside, and evaluation-count confounding.

### Named Pick: **Isolated Nesterov**

It wins on merit, not caution. PNM has the strongest paper but its impact is neutralized by its own pinned constraints; had it permitted LR/scale matching it would be the front-runner. Stochastic depth retries a count-2 failure family for marginal compute benefit. Nesterov is the only candidate whose modest upside is attainable given its mechanics, provides clean single-variable attribution, resolves a standing confound, and carries negligible catastrophic-underfit risk. Given the +0.10 single-seed gate, the highest merit-adjusted expected value is the clean, reachable, well-attributed intervention: **Nesterov**.
