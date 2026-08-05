# Proposal: Cached-Feature Head Refinement Step

## Summary

After each accepted whole-model SGD update, reuse the same already-computed pooled
feature tensor in detached form and perform one additional loss/update affecting
only `pooled_head` and `fc`. The second update repeats the accepted target semantics:
the same two targets and scalar lambda during mixup, or the same hard target during
the tail. It adds no spatial forward or backward and no inference state, but doubles
the number of optimization decisions for the 17,674 post-pooling parameters.

The proposed test is intentionally exact: one auxiliary step after every primary
step, accepted LR curve, momentum 0.9, Nesterov, and existing decay allocation, with
separate head/classifier optimizer state. No coefficient, replay buffer, extra data,
or alternate target is added. The hypothesis is that the successful pooled head is
optimization-limited because its residual scale attenuates its gradients while the
counted budget is dominated by spatial backpropagation.

## Evidence and Counterevidence

EXP036 established that the `128 -> 64 -> 128` pooled residual head improves both
top-1 and loss at small cost, while the system profile attributes about 98% of an
update to model forward/backward and only 1.4% of forward time to the entire head.
This makes cached-feature optimization materially cheaper than another whole-model
update. It also preserves the accepted spatial gradient count and all input
exposure.

The diagnosis is weak: accepted smoothed training loss is already near zero, so
additional head fitting may worsen rather than improve generalization. The detached
second step does not reproduce joint end-to-end optimization, and maintaining a
second optimizer or restricting the accepted optimizer step would complicate exact
Nesterov/decay semantics. EXP037/038 also show that simple classifier optimizer
changes readily worsen loss. This proposal should rank below mechanisms that add a
clear generalization constraint.

## Exact Treatment

Refactor the model forward only enough to expose the raw pooled vector and accepted
refined logits without changing its ordinary return. In the training loop, retain
the pooled tensor from the primary forward or recompute only the tiny accepted head
on `pooled.detach()`. After the accepted optimizer step, zero a dedicated SGD
optimizer containing `pooled_head` and `fc`, compute the same loss on detached
features, backpropagate, and step once at the same current LR.

This exact form has a serious ownership issue: the same head/classifier tensors
cannot safely belong to two live momentum optimizers because each maintains a
different buffer. An alternative single optimizer with a second step would update
all parameters through coupled decay even when their data gradients are absent.
Therefore an executable plan must instead partition the accepted optimizer so
backbone parameters use one optimizer and head/classifier parameters use another,
then step the head optimizer twice. That preserves exactly one momentum buffer per
parameter but changes update ordering and applies head/classifier decay twice per
batch. Suppressing decay on the second step would introduce another exception.

## Verification and Verdict

Any plan must prove byte-identical initialization/RNG and first primary gradients,
one optimizer-state owner per parameter, exact same-batch target reuse without a new
RNG draw, finite preseeded Nesterov oracles for both steps, and unchanged evaluator
cadence/transitions. Counterbalanced complete-body timing must project at least 127
passes and remain below 2,048 MiB. The sole score would use the exact fixed command,
with success only at `best_test_acc >= 94.58%` versus accepted 94.48%.

## Recommendation

Reject for EXP041 unless stronger proposals fail. There is no clean way to add the
second update while preserving accepted decay exposure and optimizer semantics; the
most literal implementation doubles both head data updates and head decay. A miss
would be hard to interpret, while a success would conflate extra head fitting,
double decay, and split update ordering. A future version needs a derived continuous
objective or diagnosed head underfitting before it is score-worthy.
