# Adversarial Review — EXP-002 Candidate Ideas

Three candidates, all built on the EXP-001 WRN-16-2 / 93.38% baseline, all needing ≥93.48% (+0.10 pp). The brainstorm's own diagnosis is explicit: **the limiter is generalization, not capacity or optimization** — training loss is ~0.005 while test plateaus at ~93.3%. I judge each primarily by how well its mechanism matches that diagnosis.

## Prioritized feedback (most important first)

**1. [WRN-16-3 — fatal mechanism/limiter mismatch] Adding capacity attacks the wrong bottleneck.**
The diagnosis (`01-brainstorm.md:21`, `idea-03.md:14-21`) says the model already reaches near-zero train loss — it is generalization-limited, not capacity-limited. Tripling width to 48/96/192 (~1.55M params) *increases* memorization capacity with **no added regularizer** (the proposal explicitly forbids dropout/mixup/Cutout, `idea-03.md:58-61`). The WRN paper it cites uses dropout precisely because wide nets overfit CIFAR. So the most likely outcome is a *wider* train/test gap, not a smaller one. This is the idea most likely to regress, and it contradicts its own limiter section.
*Path to improve:* only worth running paired with a regularizer, but that breaks the clean attribution the proposal is built around — i.e., it should not be the first EXP-002 move. Reserve until width-2 regularization is exhausted.

**2. [WRN-16-3 — compounding data-exposure loss] Fewer passes hurt exactly the limiter in play.**
2.25× conv work drops dataset passes from ~146 to ~80 (`idea-03.md:122-124`). Fewer image exposures on a generalization-bound problem is doubly bad, and LR 0.3 (a 50% jump) adds non-finite/instability risk with only 15s warmup. The proposal's own failure-mode table shows it can fail four different ways — a sign the bet is unfocused under a tight +0.10 target.

**3. [EMA — the headroom argument is self-defeating] The evidence cited for stability is evidence *against* the mechanism.**
The proposal's core justification is that EXP-001's final (93.34%) and best (93.38%) differ by only **0.04 pp** (`idea-01.md:24-27`), calling this a "clean case." But EMA removes late-iterate *variance* — and 0.04 pp *is* the observed variance. If the noise to be smoothed is ~0.04 pp, it is implausible that averaging reliably yields ≥0.10 pp. Compounding this: cosine already anneals LR to 0.002, so the late iterates barely move — a low terminal LR is *itself* implicit averaging, leaving little residual for EMA to capture.
*Path to improve:* this specific concern is hard to design around because it's inherent to a converged low-LR tail; a stronger version would raise the LR floor (so there's genuine iterate variance for EMA to average) — but that changes the proven schedule and loses attribution.

**4. [EMA — real downside risk, not just "small upside"] Switching evaluation entirely to the EMA view can lose the live peak.**
After 70%, the proposal evaluates *only* EMA (`idea-01.md:97-100`), required by the one-eval-per-epoch constraint. But EXP-001 was still improving late — it crossed baseline at epoch 120 and peaked at epoch 145 (`001/04-analysis.md:22`). A 0.995 / ~2000-step EMA *lags a still-improving trajectory* (the proposal admits this, `idea-01.md:166-168`). So `best_test_acc` could end up **below** what the live model would have scored — the experiment can underperform the baseline even when the underlying run reproduces EXP-001. That is a genuine loss mechanism, not merely "gain too small."
*Path to improve:* not cleanly fixable under the one-eval cap; a later BN-recalibrated variant or a higher LR floor would be needed — both out of scope here.

**5. [Mixup — uncalibrated cutoff is the only material risk, and it's mitigated] 65% is evidence-informed but untested on this exact net.**
`idea-02.md:30,79` concede the 65% boundary is uncalibrated. Under a 300s single-shot with no tuning budget, a wrong cutoff (tail too short → soft-target underfit; too long → mixup benefit decays) could land under 93.48%. But the mitigation is sound: the long ~105s / ~50-epoch hard-label tail at cosine LR falling 0.061→0.002 (`idea-02.md:7`) is specifically designed to recover margins, and mixup overhead is genuinely negligible (one permutation + one lerp, no extra forward pass), so throughput ≈ EXP-001 and ~146 passes are preserved. The mixup loss form (`idea-02.md:25`) is implemented correctly.
*Path to improve:* pre-register the interpretation table already in `idea-02.md:38-40` (if flat with preserved throughput → shorten to 50%, don't add smoothing) so a null result yields a clean next step rather than ambiguity.

**6. [All three — shared] The +0.10 pp bar is near the run-to-run noise floor.** None of the proposals estimate baseline seed/step jitter, yet all treat 93.48% as a clean line. Mixup has the largest expected effect size (CIFAR mixup gains are typically several tenths to >1 pp on comparable nets), so it has the most margin above noise; EMA has the least. Worth logging that mixup is the only candidate whose *expected* effect comfortably clears the threshold.

## Scored verdict

| Idea | Evidence & Reasoning | Potential Impact |
|---|---|---|
| **Early-Only Mild Mixup + hard tail** | **8/10** — strongest mechanism-to-limiter match (attacks memorization directly); backed by mixup + critical-period regularization papers; only weakness is uncalibrated cutoff, which is mitigated. | **8/10** — highest ceiling of the three; CIFAR mixup gains routinely exceed the +0.10 bar with real margin; near-zero overhead preserves data exposure. |
| **Late-Phase EMA** | **6/10** — clean, low-risk engineering and cited support, but its own stability evidence (0.04 pp gap) undercuts the available headroom; cosine-to-0.002 already does implicit averaging. | **4/10** — low ceiling; expected effect near noise floor, and EMA-only eval introduces a genuine downside (lagging a still-improving late trajectory). |
| **Throughput-Balanced WRN-16-3** | **3/10** — contradicts the stated generalization diagnosis; adds capacity with zero regularization; LR 0.3 + fewer passes add compounding risk. | **5/10** — wide nets have a real ceiling in principle, but under this limiter + short budget the expected sign is as likely negative as positive. |

**Pick: Early-Only Mild Mixup With a Hard-Label Cosine Tail (idea-02).**

It is the only candidate whose mechanism directly attacks the diagnosed bottleneck — near-zero training loss with a persistent test gap is the textbook signature mixup addresses, by softening sample/label pairs to suppress memorization while the hard-label tail restores peak class margins. It pairs the best mechanistic argument with the highest, best-supported ceiling and negligible compute cost, so it preserves EXP-001's image exposure and clean low-LR convergence tail. EMA is the safe runner-up but its expected effect sits at or below the +0.10 pp noise floor — and its own "tight 0.04 pp cluster" justification is the strongest argument *against* it — so it is better held as a cheap follow-on than spent as the primary EXP-002 shot. WRN-16-3 is the weakest: it targets capacity when the evidence says generalization, adds no regularizer to a net already memorizing, and sacrifices data exposure to do it. Refine the winner by pre-committing to the 65%→50% fallback and logging steps/epochs so any near-flat result is attributable to the cutoff rather than throughput.
