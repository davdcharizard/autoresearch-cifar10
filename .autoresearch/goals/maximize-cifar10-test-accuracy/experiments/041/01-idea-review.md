# Idea Review EXP-041

Offline fallback adversarial review of the randomized candidates.

## Prioritized Feedback

1. **The direct-path proposal's motivating gap is overstated, but its complete
   objective is still the strongest test.** In accepted `train.py`, the main
   feature is `z + 0.1 * h(z)`, so main CE already sends an identity-path
   gradient directly to `z`; the proposal has not diagnosed loss of raw-feature
   usefulness. Reframe the treatment as a deliberately different coupled
   gradient objective, not as restoration of absent supervision. Require the
   preflight to prove that direct logits and combined gradients differ
   nontrivially from main-only behavior, but do not tune from those diagnostics.

2. **The 90/10 blend does not preserve accepted optimization strength for the
   successful pooled head.** As `proposals/idea-02.md` correctly derives, the
   pooled-head data gradient becomes `0.9 * g_main`, while its coupled `5e-4`
   decay is unchanged. Its decay-to-data-gradient ratio therefore rises by
   `1 / 0.9 - 1 = 11.1%`; this is a concrete confound given EXP036 is the only
   recent gain and EXP037/038 show sensitivity to decay allocation. Do not
   "fix" this with loss rescaling or group-specific decay, which would add a
   second intervention. Interpret the sole score as evidence for or against the
   complete 90/10 objective only.

3. **Shared-classifier conflict is the most likely direct-path failure mode.**
   `fc(z)` and `fc(z + 0.1 h(z))` need not prefer the same class vectors, and
   EXP040 shows the accepted affine boundary uses freedoms that a clean-looking
   constraint can damage. The proposed gradient-cosine diagnostics are useful
   semantic evidence, but they cannot justify a coefficient, detach, or separate
   classifier after inspection. Success would not establish that raw `z` had
   collapsed; failure would not reject auxiliary supervision generally.

4. **The learned gain is principally an optimizer/regularization
   reparameterization, not new expressive capacity.** In
   `proposals/idea-01.md`, scaling the second pooled-head matrix can already
   change branch amplitude. Adding an unconstrained zero-decay scalar creates a
   route to inflate the gain while the equivalent matrix remains decayed,
   despite the history showing that matrix decay throughout training is useful.
   Exact initial-function and RNG preservation make this test clean, but there
   is no evidence that fixed `0.1` is limiting. If ever run, close only the exact
   full-LR, zero-decay scalar factorization on a miss.

5. **Cutout is mechanically distinct but opposed by the strongest local
   evidence.** `proposals/idea-03.md` improves on EXP003 by using a small,
   label-preserving, per-example 6.25% hole and a full clean tail. Nevertheless,
   EXP003, EXP006, and EXP030 all show normal-exposure losses from additive
   region or residual masking, while accepted early mixup and RandAugment are
   already calibrated. The exact private-RNG and post-mixup contract is sound;
   the likely failure is compounded information removal, not implementation
   contamination. It should be a closure experiment only, not the lead.

6. **Cached-feature refinement is not score-worthy in its current form.**
   `proposals/idea-04.md` identifies its fatal ownership problem itself: two
   optimizers create competing momentum buffers, while partitioning ownership
   and stepping the head optimizer twice doubles head decay and changes update
   ordering. The accepted hard tail already nearly interpolates, so head
   underfitting is also undiagnosed. A result could not isolate cached-feature
   refinement from doubled decay and altered Nesterov dynamics. Reject until a
   single derived objective with one state owner and unambiguous decay exposure
   exists.

## Scored Verdict

| Candidate | Evidence and reasoning | Potential impact | Verdict |
|---|---:|---:|---|
| Training-only direct-path auxiliary CE | 3/5 - EXP036 supplies a relevant representation interaction and the gradient contract is rigorous, but no raw-feature failure is diagnosed and the identity path already supervises `z`. | 3/5 - a complementary post-pooling gradient could improve boundary quality at normal exposure, although shared-boundary conflict and 10% head suppression cap confidence. | **Select** |
| Learn only the pooled-residual gain | 2/5 - exact initial equivalence is strong experimental hygiene, but the mechanism is mostly redundant factorization with unsupported zero-decay norm reallocation. | 2/5 - it is virtually free and adaptive, yet adds no function class and is unlikely to clear the 0.10-point margin. | Reject |
| Early post-mixup 8x8 Cutout | 2/5 - established occlusion reasoning and a disciplined treatment are outweighed by three locally relevant masking failures. | 2/5 - localized invariance has some upside, but stacking a third early regularizer is more likely to erase useful signal. | Reject |
| Cached-feature head refinement | 1/5 - head underfitting is undiagnosed and no proposed optimizer formulation preserves one momentum owner and accepted decay semantics. | 2/5 - cheap extra head decisions could matter in principle, but the executable treatment conflates them with double decay and changed ordering. | Reject as non-score-worthy |

## Selection

**Pick: Training-Only Direct-Path Auxiliary Cross-Entropy.**

It wins because it is the only candidate that tests a genuinely different
representation-training signal after the measured spatial bottleneck while
preserving inference state, initialization, data, and the accepted classifier's
geometric freedom. Its evidence is moderate rather than strong; selection means
one controlled score is justified, not that improvement is expected. Learned
gain mostly reparameterizes the successful branch, Cutout repeats a locally
disfavored regularization family, and cached refinement has a fatal causal
confound.

## Tightened Hypothesis and Closure

**Hypothesis:** If the raw and refined pooled representations admit sufficiently
compatible class boundaries, then the exact always-on objective
`0.9 * CE(fc(z + 0.1 h(z))) + 0.1 * CE(fc(z))`, using the same accepted mixup
targets early and hard targets late, will add a nonredundant boundary-shaping
gradient without materially suppressing the accepted residual head. With
unchanged default inference, it will retain at least 127 realized passes and
raise fixed-seed `best_test_acc` from 94.48% to at least 94.58%.

**Closure:** A valid normal-exposure score below 94.58% rejects only this exact
always-on, shared-classifier, 90/10 objective as an improvement to `a7c42dc`.
It should also stop immediate result-conditioned coefficient, cutoff, detach,
separate-head, distillation, or head-scale rescues, but it does not formally
falsify independently motivated intermediate supervision or a different loss
family. Success supports the complete coupled objective; it does not prove raw
`z` was insufficiently supervised or distinguish direct-path regularization
from reduced pooled-head gradient and changed shared-classifier/backbone
updates. Low realized exposure remains operationally inconclusive and must not
be rerun.
