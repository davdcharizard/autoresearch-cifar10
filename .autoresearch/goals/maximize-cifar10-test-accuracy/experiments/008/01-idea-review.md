# Adversarial Idea Review - EXP-008

The accepted baseline is 94.07%; one fixed-seed run must reach at least 94.17%.
All three candidates are `train.py`-only, add no RNG draws, preserve the frozen
evaluator and evaluation cadence, and have no fatal scope or reward-hacking issue.

## Prioritized Feedback

1. **Late momentum taper lacks evidence for its specific limiter and conflates
   velocity damping with a large reduction in effective late gradient gain.**
   EXP-002 finished at its best while continuing to improve through the hard-label
   tail (`experiments/002/04-analysis.md`), so there is no observed peak erosion or
   oscillation. In PyTorch SGD, lowering each group's `momentum` also reduces the
   contribution of the existing buffer and the steady-gradient amplification; with
   LR still nonzero, this is not simply a cleaner endpoint settle. The likely failure
   is under-updating during the last roughly 2,800 useful steps. If selected, log
   momentum and LR at 90%, 95%, and the endpoint, taper smoothly without resetting
   buffers, and interpret failure as rejection of this taper, not of late damping in
   general. A measured oscillation or gradient/velocity diagnostic would strengthen
   the premise before spending the scored run.

2. **Evaluator-consistent channel standardization targets a weakly exposed stem
   conditioning problem.** `train.py:98-99` sends the stem convolution directly
   into `layer1`, whose first operation is BatchNorm (`train.py:43,58`); that BN
   cancels most common input-scale change. CIFAR channel standard deviations differ
   only modestly from one another, so after BN the remaining channel-relative
   reparameterization is much smaller than the roughly fourfold raw scaling suggests.
   The proposal is valid because `prepare.py:13-19` applies the same mean-only test
   transform and an in-model buffer reaches both paths, but the claim that an
   unnormalized stem is a demonstrated limiter is unsupported. Improve the preflight
   by verifying exact locally sourced standard deviations, buffer dtype/device and
   state-dict behavior, train/eval forward equivalence, and matched throughput. Do
   not combine it with a stem architecture change.

3. **The decoupled zero floor is the cleanest attribution test, but its settling
   diagnosis and upside remain weak.** The accepted run's final-equals-best trajectory
   is contrary evidence that residual LR 0.002 is harmful. Still, the revised design
   directly fixes the coupling identified in `experiments/006/01-idea-review.md`:
   `WARMUP_START_LR = 0.002` must drive optimizer initialization and the first 5%
   warmup, while only post-warmup `MIN_LR` becomes zero. This preserves the validated
   early path and changes no operations, exposure, RNG order, regularization, or
   optimizer family. Require schedule assertions at 0%, 5%, 65%, 90%, 95%, and 100%,
   and keep continuous `5e-4` matrix decay after EXP-007. Its 7.9% hard-tail and
   52.2% final-10% LR-area reductions make the treatment meaningful, but a lower
   loss below 94.17% must remain a strict no-improvement rather than prompting a
   nearby floor sweep.

4. **All candidates have limited headroom relative to the 0.10-point gate.** The
   history now contains five normal-exposure regressions after EXP-002
   (`04-results.tsv`), and no candidate is supported by a measured remaining
   bottleneck. Run only the single preregistered treatment, do not reroll a near miss,
   and compare steps/passes before assigning a mechanism.

## Scored Verdict

| Candidate | Strength of evidence and reasoning | Potential impact |
|---|---|---|
| **Decoupled cosine-to-zero floor** | **7/10** - exact schedule arithmetic, prior adversarial refinement, and clean isolation are strong, although the stable accepted endpoint contradicts the limiter diagnosis. | **4.5/10** - several thousand late updates change materially at zero cost, but only 7.9% of hard-tail LR area is removed and the gain may miss the acceptance bar. |
| **Evaluator-consistent channel standardization** | **5.5/10** - canonical conditioning is plausible and evaluator placement is correct, but the immediately following BatchNorm largely cancels the raw scale and no local result identifies stem conditioning as limiting. | **5/10** - a persistent trajectory-wide reparameterization could clear 0.10 points, yet the effective channel-relative change is small and may be neutral. |
| **Late momentum taper** | **4.5/10** - the implementation mechanism is coherent, but no trajectory evidence shows momentum overshoot and dynamic momentum changes both buffer memory and effective gradient scale. | **4/10** - it can alter roughly 2,800 endpoint updates without throughput cost, but slowing a still-improving hard-label tail is the leading outcome. |

**Strongest idea: Decoupled Cosine-to-Zero Floor.** It wins over channel
standardization because it isolates a genuinely untested schedule parameter while
preserving the accepted warmup, and over momentum taper because its changed update
amplitude is exactly quantified rather than entangled with momentum-buffer dynamics.
Its ceiling is modest, so selection is justified by attribution quality and existing
review support, not by evidence of visible endpoint instability.
