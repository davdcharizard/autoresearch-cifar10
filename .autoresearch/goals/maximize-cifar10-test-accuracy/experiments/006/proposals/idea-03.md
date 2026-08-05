# Idea 03: Early WRN Block Dropout at p=0.10

## Proposal

Add dropout with probability **0.10** to every `PreActBlock`, after the second
batch-normalization/ReLU and before `conv2`. Enable it only during the first 65%
of counted training time, then disable it at the existing mixup transition. Keep
the accepted alpha-0.2 mixup, WRN-16-2, optimizer, time-cosine schedule, batch
size, seed, loader, augmentation, and evaluation cadence unchanged.

## Diagnosis and Rationale

The accepted WRN-16-2 reaches 94.07% with alpha-0.2 mixup through 65% of the
budget and a 35% hard-label tail. The local experiments already bracket the
input-level regularization strength: ending mixup at 50% loses 0.16 points,
while alpha 0.4 loses 0.50 points and raises test loss. Therefore this proposal
does not further tune mixup. It tests a distinct WRN-native mechanism: weakly
breaking feature co-adaptation inside residual branches while leaving the
identity paths intact.

The dropout belongs between the two convolutions, specifically on
`relu(bn2(conv1_output))` before `conv2`. This follows the intended residual
branch placement: the shortcut remains deterministic, and dropped activations
are transformed by `conv2` before residual addition. Applying dropout after the
addition would corrupt both the learned branch and the identity information and
would be a materially different, less conservative experiment.

## Strength and Timing

Use **p=0.10**, not the commonly explored stronger WRN rates such as 0.3. This
model is only WRN-16-2 (691,674 parameters), already uses mixup, and stronger
mixup has just demonstrated over-regularization. At p=0.10 the residual branch
retains 90% of its activations on average and PyTorch's inverted dropout keeps
the branch expectation unchanged.

The dropout should be **time-limited, not constant**. Disable it at 65% counted
time in the same one-shot transition that disables mixup. This preserves the
validated late regime: real images, hard labels, low learning rates, and no
stochastic feature masking for margin refinement. Constant dropout would
confound the test by changing the 35% tail that EXP-002 identified as useful;
it would also conflict with the local `Time Matters` prior that regularization
has most value early. Aligning both regularizers at 65% makes the hypothesis
sharper: modest input- and feature-level noise during representation formation,
followed by an unchanged clean convergence phase.

## Falsifiable Hypothesis

Early p=0.10 branch dropout will reduce feature co-adaptation enough to improve
`best_test_acc` from 94.07% to at least the required **94.17%**, without reducing
training exposure by more than 5%. The expected signature is equal or lower
final test loss than 0.2432 and continued accuracy improvement after dropout and
mixup are disabled.

The mechanism is falsified if any of the following occurs in a valid run:

- `best_test_acc` is below 94.17%, especially if final test loss rises;
- the post-65% hard-label tail fails to recover from a lower pre-transition
  accuracy trajectory;
- realized exposure falls below 95% of EXP-002's 141.9 passes (about 134.8
  passes), making mask-generation cost a material fixed-budget penalty.

A lower score with normal exposure and higher test loss would specifically
indicate that mixup plus block dropout over-regularizes this small WRN. A lower
score accompanied by materially reduced exposure would not cleanly test the
generalization mechanism and should be classified as a throughput failure.

## Exact Code Scope

Only `train.py` changes:

1. Add `BLOCK_DROPOUT = 0.10` beside the existing hyperparameters.
2. Thread the dropout probability through `WideResNet._make_layer` into each
   `PreActBlock` and store it as a mutable `dropout_p` attribute.
3. Split the current second-convolution expression into:
   `out = F.relu(self.bn2(out))`, then
   `out = F.dropout(out, p=self.dropout_p, training=self.training)`, then
   `out = self.conv2(out)`. Guard the call when `dropout_p == 0.0` so the clean
   tail does not generate masks or consume CUDA RNG.
4. Add a small `WideResNet.set_block_dropout(p)` method that updates only
   `PreActBlock.dropout_p` values.
5. In the existing one-shot 65% mixup-disable branch, call
   `model.set_block_dropout(0.0)` and log that both early regularizers were
   disabled. Do not traverse modules on every optimizer step.

This adds no parameters, no dependency, no extra forward pass, and no evaluator
call. It should add only dropout-mask bandwidth and a small activation buffer;
the expected VRAM increase is negligible relative to the H20's capacity.

## Preflight and Verification

- Confirm one NVIDIA H20 and an exact diff limited to the scope above.
- Run a short matched-path timing preflight after CUDA warmup. Require projected
  exposure of at least 134.8 passes (95% of EXP-002); do not tune the probability
  from this preflight.
- Verify dropout is active in `model.train()` before 65%, inactive in
  `model.eval()`, and set to zero exactly once at the transition.
- Execute one preregistered seeded run with the mandated local command and
  timeout. Do not reroll the seed or retry a completed run.
- Confirm 300 counted seconds, total runtime below 600 seconds, at most one
  evaluation per epoch, unchanged parameter count, and a final summary.

## Interaction Risks

The main risk is additive regularization: mixup already perturbs targets and
inputs, while dropout perturbs intermediate features. Although p=0.10 is mild,
the combination could reproduce the alpha-0.4 failure by weakening early
fitting too much. A second risk is that WRN block dropout is generally more
useful when width creates redundant features; WRN-16-2 may have too little
redundancy to benefit. Finally, CUDA dropout changes the random-number stream,
so later mixup coefficients and data-order RNG may not be bit-identical to
EXP-002 even at the same seed. The result must be interpreted as the fixed-seed
dropout-enabled training process, not as paired identical minibatches.

## EXP-006 Recommendation

This is a **medium-confidence EXP-006 candidate** and is preferable to another
mixup-strength or cutoff trial: it is orthogonal, native to the selected WRN,
cheap, and has a precise failure signature. It should nevertheless rank behind
a deconfounded width increase if that candidate's timing preflight preserves at
least 95% of current data exposure, because the latest evidence points more
strongly toward capacity than additional regularization. If wider WRN throughput
is projected to fall substantially, early p=0.10 block dropout is the cleaner
EXP-006 choice. Constant dropout or p>=0.2 should not be EXP-006 because both
needlessly disturb the validated clean tail and carry a high over-regularization
risk.
