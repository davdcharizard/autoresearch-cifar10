**Prioritized Feedback**

1. **Idea-01 and idea-02 are probably too small for this bar.** Both proposals honestly place expected gains in the noise band: decoupled WD likely `+0.02–0.08pp` (`idea-01.md:100-107`), EMA retune best-case `+0.05–0.15pp` (`idea-02.md:65-71`). With a required `≥96.48%` and fixed-seed host-throughput noise around `0.1pp`, these are weak one-shot candidates. Use them as riders on a larger training-side win, not as standalone proof attempts.

2. **Idea-03’s biggest technical flaw: its “coupled weight decay” is not actually equivalent to SGD after Newton-Schulz.** The sketch adds `wd * p` to the gradient, then orthogonalizes/normalizes the whole update (`idea-03.md:105-119`). That can distort or erase the radial L2 penalty, so the experiment is not merely “optimizer only”; it changes regularization semantics in a regularization-bound recipe. Fix by applying decoupled weight decay outside the orthogonalized update, or explicitly pre-register this as a Muon+changed-WD-semantics experiment.

3. **Idea-02 assumes the rising EMA tail proves raw weights are better, but the run only observes EMA accuracy.** `train.py:343-349` evaluates `ema_model` once EMA starts, so EXP-008’s ep147→150 rise is a rise in the already-smoothed model, not evidence that the raw final iterate beats the EMA. A shorter EMA may reduce lag, but may also reduce denoising. If run later, prefer a tail-only EMA schedule or add non-metric diagnostics; do not overclaim from per-epoch EMA prints.

4. **Idea-03’s LR is the load-bearing unverified assumption.** The proposal admits `0.02` may be off by `2–3x` and first run may be calibration (`idea-03.md:201-206`, `279-282`). That matters under one-shot evaluation. Still, unlike ideas 1/2, the upside can exceed the noise floor. Tighten implementation smoke tests and treat early trajectory as diagnostic, but do not spend the official run unless the code path is clean.

5. **Idea-03’s throughput estimate is optimistic.** The FLOP estimate ignores launch overhead from many small bf16 matmuls per step (`idea-03.md:230-244`). Because the LR schedule is time-based (`train.py:286-292`), slower steps still anneal by time, but fewer epochs can hurt tail quality. Verify `num_epochs` remains near 142–150; if it drops materially, the run is confounded.

6. **Idea-01’s BN/alpha argument is sound but overstated.** The partition is code-correct (`train.py:101-153`, `244-250`), but only `5,505` of ~`7.78M` learnable params change decay, and the single ReZero alpha decay force is tiny unless alpha grows (`idea-01.md:102-105`). Add final alpha logging if tested; otherwise treat it as a low-cost rider.

7. **No fatal hard-constraint violation found.** All three can be implemented inside `train.py`, preserve the frozen evaluator, keep one eval per epoch, and avoid new dependencies if implemented as written.

**Scored Verdict**

Scores are `/10`.

| Idea | Evidence & Reasoning | Potential Impact |
|---|---:|---:|
| **01 Decoupled WD** | **7/10**: Standard trick, code partition is clean, but direct evidence in this heavily regularized recipe is weak. | **3/10**: Most likely sub-noise; unlikely to clearly clear `96.48%` alone. |
| **02 EMA 0.995** | **7/10**: Strong code link to the metric and good half-life reasoning, but raw-iterate/noise assumption is unobserved. | **4/10**: Plausible small gain, but bounded by a tiny observed tail and not clearly above noise. |
| **03 Muon** | **6.5/10**: Best external evidence and mechanism, but LR, WD semantics, and throughput are real unresolved risks. | **9/10**: Only finalist with credible upside well beyond `+0.10pp` if it lands. |

**Pick: Idea-03, Muon optimizer.**

It wins because the other two are honest null-band probes: even if they work, one fixed-seed run may not prove anything. Muon is risky and needs implementation tightening, especially around weight decay, but it is the only proposal with a plausible path to a clearly above-noise improvement over `96.38%`.
