# Adversarial Idea Review — EXP-007

The accepted baseline is 94.07%, and a valid EXP-007 must reach at least
94.17% in one fixed-seed, 300-second run. The history now argues against adding
regularization: CutMix, a shorter mixup window, stronger mixup, and block
dropout all regressed at normal exposure (`04-results.tsv`; EXP-003–006
analyses). None of the three new candidates repeats those treatments unchanged,
and none has a hard-constraint or reward-hacking failure.

## Prioritized Feedback

**1. [Channel standardization] The proposed conditioning mechanism is mostly
neutralized by the actual WRN topology, and one supporting claim is technically
misstated.** The stem convolution at `train.py:69,98` feeds directly into the
first block's `BatchNorm2d` at `train.py:43,58`. That BatchNorm removes most of
the common approximately fourfold activation-scale change caused by dividing
the centered CIFAR channels by their standard deviations; only the relatively
small *between-channel* scale differences can survive through the mixed stem
filters. Moreover, the convolution initializer uses `mode="fan_out"`
(`train.py:80`), so the brainstorm's claim that unit-variance inputs are better
matched to its forward-activation assumptions is not valid: `fan_out` targets
backward variance, not forward variance. The likely result is therefore a
mostly reparameterized stem, with any change coming indirectly through stem
gradient scale and coupled weight decay rather than the claimed network-wide
conditioning improvement. **Improve it** by specifying the exact standard
deviations and their training-set-only provenance, registering a
`(1, 3, 1, 1)` buffer, and narrowing the hypothesis to channel balancing in the
stem. A smoke check should verify identical preprocessing in train/eval mode and
finite stem statistics, without using test accuracy. This is not fatal, but the
current evidence does not justify calling it canonical conditioning of the
whole model.

**2. [Late weight-decay removal] The literature-aligned mechanism is plausible,
but the 65% cutoff is borrowed from mixup without evidence that weight decay has
the same critical period.** The local `Time Matters` note supports the broad
claim that early regularization can retain its benefit after removal; it does
not establish 65% for selective `5e-4` decay on this WRN. EXP-002 validates that
boundary only for alpha-0.2 mixup. In addition, PyTorch SGD applies coupled L2
decay to every matrix parameter in the first optimizer group
(`train.py:177-187`), so disabling it changes both regularization and the
effective optimization dynamics of the many BatchNorm-scale-invariant
convolutions. With near-zero late training loss and an accepted endpoint that
equals its best (EXP-002 analysis), the concrete failure mode is unchecked norm
growth or sharper logits that worsen test loss without changing decisions.
**Improve it** by presenting 65% as a deliberately isolated first probe, not a
validated weight-decay boundary; flip only the decay group's value once at the
existing mixup transition, verify the no-decay group stays zero, and log
decayed-parameter norms before and after the switch. Do not rescue a negative
result by tuning another cutoff on the same run.

**3. [Cosine-to-zero] Decoupling the warmup start fixes EXP-006's attribution
flaw, but the observed trajectory points against residual late motion as the
limiter.** Keeping a separate `WARMUP_START_LR = 0.002` preserves optimizer
initialization and the first 5%, so this version is substantially cleaner than
`experiments/006/proposals/idea-04.md`. Its arithmetic is also explicit: only
about 7.9% of hard-tail LR area is removed. However, EXP-002 finished at its
best after continued gains through the hard-label tail, with test loss 0.2432;
there is no peak erosion or oscillation for a zero floor to cure. The leading
failure is premature freezing of useful margin refinement during the final
roughly 2,800 steps. **Improve it** by retaining the proposed decoupling and
preflight schedule-point assertions, and preregister a strict one-run verdict:
lower loss without 94.17% is still no improvement, and no intermediate floor
should be inferred as a retry from this run.

**4. [Cross-cutting] The +0.10-point gate is close to single-run and
max-over-evaluations noise, so ceiling matters.** The fixed seed and frozen
evaluation cadence prevent rerolling, but they do not provide a variance
estimate. A low-ceiling endpoint change can be directionally correct yet fail
the goal. This favors a candidate with a credible multi-tenth mechanism, while
still requiring the prescribed strict threshold and no repeat of a near miss.

**5. [Constraints] All three ideas remain admissible if implemented as stated.**
They can be confined to `train.py`, add no dependency or extra evaluation,
preserve seed 42 and the 300-second counted budget, and have negligible expected
throughput cost. For channel scaling, the fixed statistics must come only from
the training distribution; deriving them from the frozen test loader would be
a fatal evaluation leak.

## Scored Verdict

| Candidate | Strength of evidence and reasoning | Potential impact |
|---|---|---|
| **Disable weight decay for the hard-label tail** | **6.5/10** — the local temporal-regularization note supports early-only decay and the implementation is isolated, but neither the 65% cutoff nor norm shrinkage as the current limiter is measured. | **5.5/10** — changing coupled decay throughout a 35% clean-label tail could move several tenths, though near-zero training loss and possible norm/calibration degradation cap confidence. |
| **Decoupled cosine-to-zero floor** | **7/10** — it is quantitatively specified, RNG/throughput clean, and resolves the prior warmup coupling, but final-equals-best and continued tail gains are contrary evidence for its settling diagnosis. | **4/10** — removing 7.9% of hard-tail LR area may help marginally, but the expected effect is close to or below the required 0.10 points and can freeze useful late updates. |
| **Evaluator-consistent in-model channel standardization** | **4.5/10** — the train/test transform symmetry is real, but exact statistics are unspecified, the `fan_out` Kaiming argument is wrong, and immediate stem-output BatchNorm weakens the claimed conditioning mechanism. | **5/10** — a full-trajectory stem reparameterization could move the score, but most uniform scaling is normalized away and the remaining channel imbalance is modest. |

**Strongest idea: Disable Weight Decay for the Hard-Label Tail.** It wins
because it is the only candidate backed by directly relevant local evidence
for temporal removal of the exact regularizer being changed, while still
offering more plausible headroom than the narrowly late LR-floor probe. It is
also directionally distinct from the four failed attempts that added, changed,
or prematurely removed input/feature regularization: early decay remains intact
and only clean-label refinement is altered. The zero-floor idea is the cleanest
control but targets an endpoint instability the accepted run does not show;
channel standardization has a potentially broader effect but its stated
mechanism is substantially canceled by the current pre-activation stem and is
the least well evidenced. Select weight-decay removal only with the refinement
that 65% is an explicit falsifiable first cutoff, not a boundary already
validated for weight decay.
