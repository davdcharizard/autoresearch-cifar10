# Proposal: Conservative Strong-Phase Residual Activation Dropout

## Summary

Add low-rate, parameter-free activation dropout **inside only the seven
same-shape residual blocks** of the accepted width-2 postactivation ResNet-20.
During the N1/M7 plus 50% CutMix phase, apply ordinary elementwise inverted
dropout with `p=0.05` after `relu(bn1(conv1(x)))` and immediately before
`conv2`. Exclude the two stride-2/channel-expansion blocks, keep every identity
and Option-A shortcut untouched, and disable the dropout flag at the existing
80% strong-to-hard-weak transition. Evaluation always uses the deterministic
full model because `model.eval()` disables dropout.

This is deliberately **not stochastic depth**. The accepted model has only nine
residual blocks, and the stochastic-depth paper says a sufficiently deep model
is needed for its largest gains while shorter networks should skip less
aggressively. Dropping whole transformations in ResNet-20 would compound the
local strong-phase underfit risk. Low-rate within-branch dropout is the narrower
test: it perturbs feature co-adaptation without deleting complete learned paths.

## Exact Intervention

- Add `RESIDUAL_DROPOUT_P = 0.05`.
- Give `BasicBlock` a fixed boolean identifying an ordinary same-shape block:
  `dropout_eligible = stride == 1 and in_channels == out_channels`.
- Give each block a mutable `residual_dropout_enabled` flag, initialized `True`.
  In `forward`, after the existing first Conv-BN-ReLU, execute
  `F.dropout(out, p=RESIDUAL_DROPOUT_P, training=True)` only when all three are
  true: `self.training`, `self.residual_dropout_enabled`, and
  `self.dropout_eligible`. Then run the unchanged `conv2 -> bn2 -> add -> ReLU`.
  Use PyTorch's ordinary elementwise dropout, not `Dropout2d`, channel dropout,
  spatial masking, or a residual-branch Bernoulli mask.
- The eligible sites are exactly `layer1[0:3]`, `layer2[1:3]`, and
  `layer3[1:3]`: seven sites. `layer2[0]` and `layer3[0]` remain byte-for-byte
  equivalent to the accepted transition computation. The stem, classifier,
  all shortcuts, BN order/state, residual-add scale, and post-add ReLU are
  unchanged.
- Add `ResNet.set_residual_dropout_enabled(enabled)` to update all blocks. At
  the existing augmentation-switch condition, call it with `False` before the
  weak loader can produce its first training batch. Do not introduce another
  time threshold or loader transition. Print the rate/site count at startup and
  print the enabled-to-disabled transition once in the existing switch record.
- Keep `NUM_BLOCKS=3`, `WIDTH_MULTIPLIER=2`, batch 128, FP32, the complete
  accepted N1/M7 plus alpha-1 `p=0.5` CutMix strong phase, hard crop/flip weak
  tail, SGD (`lr=0.1`, momentum `0.9`, all-parameter decay `1e-4`), 80% LR
  hold, `0.01 -> 1e-4` cosine tail, workers, timer, evaluator, and seed 42
  unchanged. The parameter count must remain exactly **1,073,962**, and the
  optimizer must still contain one group with precisely all model parameters.

The `p=0.05` operating point is pre-registered rather than tuned. It is one
sixth of the CIFAR rate selected by Wide Residual Networks (`p=0.3`) and is
restricted away from transitions and the hard refinement phase. This reduction
is necessary because EXP-010's accepted recipe already combines RandAugment and
CutMix, EXP-011 showed that more strong-phase regularization can cross the
87.08% underfit marker, and EXP-012/015 showed that healthy compute does not
rescue representation changes that depress strong-phase fit.

## Evidence and Rationale

The [Wide Residual Networks paper](https://www.bmva-archive.org.uk/bmvc/2016/papers/paper087/paper087.pdf)
places dropout between the two convolutions of a residual block, after the
intermediate ReLU, rather than on the identity path. That location regularizes
the learned residual function while preserving shortcut information. The paper
used `p=0.3` on CIFAR, but its CIFAR-10 table is cautionary rather than uniformly
positive: dropout slightly worsened both reported wide points while helping
CIFAR-100 and SVHN. Therefore the paper supports the **site and mechanism**, not
a claim that its rate or gain transfers to this much shorter, more strongly
augmented run.

The [stochastic-depth paper](https://arxiv.org/abs/1603.09382) provides the
related residual-regularization rationale: random residual-path removal acts as
an implicit ensemble and can strengthen gradient flow. It also reports that the
best survival rule depends on depth and explicitly notes that a sufficiently
deep network is needed for clear gains. That limitation, plus this project's
transition-shortcut failures in EXP-017/021, is why this proposal does not drop
whole blocks and never perturbs the two Option-A transitions.

Locally, width 2 was the highest-value capacity change (EXP-007), and plateau
CutMix raised it to 94.15% while retaining nearly all exposure (EXP-010). The
weak hard-label tail converted a 89.73% strong checkpoint to 94.15%, suggesting
that limited strong-phase noise can be useful if it is removed before
refinement. The proposed dropout targets co-adaptation among residual features
during composite-view training, then exposes the deterministic full network to
the entire hard weak tail for BN/classifier calibration. The primary contrary
evidence is strong: raising CutMix to 0.75 lost 0.15 points (EXP-011), and
identity-oriented residual changes lost strong fit (EXP-012/015). This makes the
idea medium-low evidence and medium-high risk despite its small diff.

## RNG Contract

Use PyTorch's native CUDA dropout RNG under the existing production
`torch.cuda.manual_seed(42)`. Do not create a seed knob, reseed per block,
derive masks from labels or batch indices, or retry a valid run. Dropout
intentionally consumes CUDA RNG only during eligible strong-phase training
forwards. It must not touch CPU RNG, forkserver worker RNG, or the existing
`cutmix_collate` CPU save/restore contract, so the data-policy stream remains
isolated from the new model noise.

When the phase flag is false, bypass the dropout call entirely rather than call
it with `p=0`; this must consume no CUDA random numbers in the weak tail. In
evaluation, `self.training == False` likewise bypasses dropout. Safety and
timing controllers run in disposable fresh processes with seed 42 and cannot
advance the production run's RNG. The only production evidence is one fresh
seed-42 process; changing dropout's random stream relative to the baseline is
part of the pre-registered method, not grounds for a rerun.

## Hypothesis

Low-rate noise between residual convolutions during the composite strong phase
will reduce co-adaptation without weakening shortcuts or transitions, and the
fully deterministic hard-label weak tail will recover any small fit penalty and
calibrate all features. The net prediction is `best_test_acc >=94.25%` while
retaining at least 96.7% of EXP-010's optimizer exposure (at least 26,000 of
26,898 updates). A switch checkpoint below 87.08%, weak-tail failure to recover
past 93.16%, or worse terminal NLL than 0.1934 would instead support excessive
regularization.

## Preflight Safety Gates

Run all diagnostics before the one production job and persist their raw
results. They are launch vetoes for implementation/safety defects, not alternate
accuracy criteria.

1. **Static scope and structure.** Require only `train.py` in the tracked diff;
   compile/lint it; confirm 1,073,962 parameters, one unchanged optimizer group,
   seven eligible blocks, two ineligible transition blocks, unchanged Option-A
   slicing/padding, and unchanged data/schedule/evaluator code.
2. **Tail-identity oracle.** Clone the accepted and candidate models from one
   state dict. In training mode with dropout explicitly disabled, feed the same
   hard and soft-target batches and require exact logits, loss, parameter
   gradients, BN buffers, and CUDA RNG state. Confirm the first weak training
   forward does not execute a dropout site and does not advance CUDA RNG.
3. **Mask semantics.** With dropout enabled, instrument without changing the
   production graph. Across the immutable production-distribution corpus,
   require masks only at the seven declared sites, aggregate observed zero rate
   in `[0.045, 0.055]`, inverted scale `1/0.95`, no masked transition tensor,
   no parameters/buffers added, and deterministic evaluation logits.
4. **Exact-corpus optimization safety.** Reuse one materialized SHA-256 corpus
   drawn through the imported production loader: 100 strong hard-target and 100
   strong CutMix soft-target batches, with row sums and shapes audited. Start
   control and candidate from identical weights and ordinary SGD state. Require
   finite logits/loss/gradients/parameters/BN buffers throughout; no
   candidate-only `>95%` one-class concentration on two consecutive audited
   checkpoints; candidate terminal loss EMA no more than `1.10x` control in
   either hard or soft strata. Record, but do not launch-tune from, per-stratum
   loss EMA and class histograms.
5. **Phase lifecycle.** In a shortened integrated loop, require one and only one
   `True -> False` dropout transition at the same boundary as
   RandAugment+CutMix-to-base, all eight old workers stopped, integer weak
   targets, and no re-enabling in subsequent epochs.

Any failure vetoes the production launch and yields an invalid experiment; do
not alter the rate, scope, threshold, or implementation as an in-experiment
rescue.

## Timing Gate

After conditioning both paths, run five alternating fresh-process H20 pairs on
the actual strong-phase forward/loss/backward/SGD step with batch 128 and
dropout active. Exclude data loading exactly as prior architecture timings do,
but do not time a dropout-disabled surrogate. Record per-pair medians, p95,
weighted ratio, CV, projected fixed-time steps, and peak memory.

Launch only if all hold:

- one idle NVIDIA H20 with approximately 98 GB VRAM;
- candidate/control weighted median step ratio `<=1.03`;
- maximum paired ratio `<=1.05` and candidate-p95/control-mean `<=1.06`;
- control and candidate CV each `<=0.03`;
- at least 26,000 projected optimizer steps from EXP-010's 26,898-step anchor;
- projected end-to-end runtime below 600 seconds and no abnormal memory growth.

The hard gates protect the fixed-time comparison. A faster result is plausible
only from measurement noise because activation dropout does not skip convolution
work; do not claim the stochastic-depth paper's compute saving for this method.

## Production Verification

1. Remove stale completed logs. Verify the working branch, only-`train.py`
   tracked diff, clean static checks, and exactly one idle H20.
2. Run exactly once with
   `timeout 600s uv run train.py > run.log 2>&1`. Do not use `tee`, reroll,
   repeat a valid result, or adjust `p=0.05` after observing any trajectory.
3. Require exit zero and one finite value for all ten standard summary fields.
   Counted training must reach the fixed 300-second budget with only normal
   final-step overshoot, and total wall time must remain below 600 seconds.
4. Require 1,073,962 parameters, one dropout-enabled strong phase, one dropout
   disable at the existing near-80% augmentation switch, eight clean worker
   exits, 45-55% realized CutMix, hard labels throughout the weak tail, and no
   more than one evaluation per epoch.
5. The formal improvement condition is `best_test_acc >=94.25%` versus the
   94.15% moving baseline. A complete lower result is valid no-improvement and
   is never rerun. Preflight/timing diagnostics cannot upgrade or downgrade the
   formal verdict after a valid production run.
6. Preserve for analysis: actual steps versus 26,898; switch accuracy versus
   89.73% and the 87.08% underfit marker; first weak accuracy versus 93.16%;
   every weak-tail accuracy; peak/final gap; final NLL versus 0.1934; CutMix
   fraction; evaluation count; memory; and phase-toggle evidence.

## Risk Assessment and Interpretation

- **Compounded underfit is the dominant risk.** RandAugment plus CutMix already
  makes the plateau difficult. Even `p=0.05` may lower switch fit enough that the
  short tail cannot recover. A lower switch, lower first weak result, and worse
  NLL together reject this regularization point.
- **Published CIFAR-10 evidence is mixed.** WRN's placement is principled, but
  its reported CIFAR-10 dropout rows slightly regressed. Success here depends on
  the distinct phase-limited design; a null result should retire this exact
  strong-only `p=0.05` point rather than imply dropout is universally harmful.
- **BN distribution changes remain possible.** Dropout is before `conv2/bn2`,
  so the branch's learned statistics reflect noisy strong-phase activations.
  The entire deterministic weak tail is intended to adapt them, but final NLL
  and early-tail recovery must be inspected.
- **Single-seed noise.** The acceptance margin is only ten CIFAR-10 examples.
  `94.25-94.35%` is a formal improvement but weak causal evidence unless the
  NLL and trajectory also improve. This caveat never authorizes a rerun.
- **Attribution boundary.** The experiment tests the complete combination of
  seven-site activation dropout, `p=0.05`, and strong-phase-only duration. It
  cannot isolate rate, site, or phase duration, and none may be tuned after the
  run.

## Estimated Effort

Small implementation, moderate verification. The production diff should add
only a constant, a parameter-free block flag, one guarded functional dropout
site, one model toggle helper, and switch logging. Exact RNG/phase tests and the
paired timing controller are the main effort. No dependency, checkpoint format,
optimizer, transition, or evaluation change is required.
