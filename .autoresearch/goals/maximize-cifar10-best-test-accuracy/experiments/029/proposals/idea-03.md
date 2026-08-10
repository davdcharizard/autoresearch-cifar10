# Proposal: Conservative Linear Stochastic Depth

## Intervention and hypothesis

Apply batchwise stochastic depth to the nine residual branches of the accepted
width-2 postactivation ResNet-20. Number blocks globally `i=0..8` and use the
single preregistered linear survival schedule

```text
p_i = 1 - 0.10 * i / 8
[1.0000, 0.9875, 0.9750, 0.9625, 0.9500,
 0.9375, 0.9250, 0.9125, 0.9000]
```

During training, one Bernoulli draw per block and optimizer batch either executes
the whole residual branch unscaled or bypasses it. During evaluation every branch
executes and its residual is multiplied by `p_i` before the unchanged shortcut
addition and ReLU. This is the paper's non-inverted convention: no `1/p_i`
training amplification.

Preserve all other accepted EXP-010 choices: 32/64/128 ResNet-20, Option-A
shortcuts, initialization, FP32/default-TF32, batch 128, ordinary momentum SGD,
all-parameter decay `1e-4`, 80% elapsed-time LR hold, N1/M7 plus probability-0.5
alpha-1 CutMix, hard weak tail, seed 42, workers, timer, and evaluator.

**Hypothesis:** mild depth sampling regularizes the width-2 model while skipping
about 5% of residual-branch work, preserving enough strong-phase capacity and
increasing fixed-budget updates to raise `best_test_acc` from 94.15% to at least
94.25%. Point prediction is 94.25-94.35%, switch accuracy at least 88.5%, and at
least 1% more optimizer steps than EXP-010. A valid miss rejects this exact
schedule; no survival-rate rescue is allowed.

## Rationale and expected compute

Huang et al. report CIFAR gains from training ensembles of shorter residual
networks and evaluating the survival-weighted full network. The local transfer is
weak: their strongest models are far deeper, whereas dropping one of only nine
blocks is substantial and this recipe has repeatedly suffered strong-phase
underfit. Final survival 0.9 is therefore intentionally much milder than the
paper's common deep-network setting.

The schedule averages 0.95 survival, so 8.55 of nine residual branches execute per
batch and 0.45 are skipped. Because CIFAR residual blocks have similar convolution
cost, expected residual-branch convolution work falls about 5%; stem, shortcut,
pooling, classifier, loss, and SGD overhead remain. Actual full-step savings will
be smaller and variable. Unlike ordinary dropout, a dropped branch avoids both
convolutions, both BNs, their saved activations, and their backward work.

## Exact implementation and RNG semantics

Give each `BasicBlock` its fixed global index and survival probability. Construct
one dedicated **CPU** `torch.Generator` after model initialization, seed it from
the already fixed `torch.initial_seed()` (42), and share it across blocks. It is a
method RNG stream, not a tunable second seed. On every training forward, every
block consumes exactly one scalar draw in global block order:

```python
survive = torch.rand((), generator=sd_generator).item() < survival_probability
```

The scalar applies to all 128 examples. Draw even for block 0 (`p=1`) so every
step consumes exactly nine values. Do not use global CPU/CUDA RNG, per-example
masks, data-dependent decisions, or GPU `.item()` synchronization.

For each block, form the existing shortcut first. If training and dropped, return
`F.relu(shortcut)` without calling either residual Conv/BN. Consequently residual
parameters have `grad is None`; ordinary PyTorch SGD skips momentum and coupled
decay for those parameters on that step, and the two BN counters/buffers do not
update. If training and survived, execute the accepted branch unscaled. If
evaluating, consume no RNG, execute the branch, add `p_i * residual` to the
shortcut, and apply ReLU.

Both transition blocks (indices 3 and 6, survivals 0.9625 and 0.925) remain
eligible. On a drop their existing stride-2 slice/zero-pad Option-A shortcut alone
produces the correct 64- or 128-channel output; padded new channels are zero for
that batch. They are not exempted or assigned another rate, because that would
break the registered linear schedule. Forced-transition-drop tests are mandatory
because these are the highest-risk masks.

Add non-parameter training counters for seen/survived batches per block and print
their totals in the final summary. Evaluation must neither increment counters nor
advance the generator. Do not add learned gates, inverted scaling, warmup,
phase-specific rates, per-sample masks, extra blocks, or altered optimizer groups.

## Structural and exact-corpus safety gates

Before timing, require all of the following:

1. Assert the exact nine probabilities, block indices/order, transition indices,
   nine blocks/19 convolutions/19 BNs, unchanged state-dict keys and parameter
   order, exactly 1,073,962 parameters, and bitwise accepted initialization.
2. Prove the dedicated stream leaves global CPU and CUDA RNG states unchanged,
   two seed-42 constructions emit identical mask matrices, each training step
   consumes exactly nine draws regardless of hard/soft targets or earlier drops,
   and evaluation consumes none. Over 100,000 synthetic steps require each
   observed survival within 0.003 of `p_i` and mean active blocks within 0.003 of
   8.55.
3. Forced-all-survive training must be bitwise equal to accepted output, loss,
   gradients, BN updates, and one SGD update. Forced individual drops must match a
   direct `ReLU(shortcut)` reference, leave that branch's gradients absent and BN
   state unchanged, and keep all other blocks ordinary. Test indices 3 and 6 on
   both hard and CutMix targets. Eval must match the direct
   `ReLU(shortcut + p_i*branch)` full-network reference, be repeatable, and leave
   RNG/model state immutable.
4. Persist an immutable 200-batch production corpus—80 strong-hard, 80 strong
   CutMix, 40 weak-hard—with tensors, targets, kind, and SHA-256. Restore accepted
   control and candidate model/SGD states independently; both arms consume exact
   tensors in the same order. Persist the candidate's 200x9 mask matrix before
   training and require total/per-block drops inside a preregistered four-sigma
   binomial interval (clipped to `[0,200]`).
5. Through all 200 aligned steps require finite logits, loss, gradients,
   parameters, momentum, and BN buffers; positive running variances; no
   candidate-only >95% class concentration; candidate/control loss and logit-RMS
   ratios `<=1.5`; global update-norm ratio `<=1.5`; and candidate terminal loss
   EMA `<=1.30x` control. Serialize mask, histogram, and norms before any veto.

Lower candidate updates in dropped branches are intended, so per-tensor parity is
not required. Catastrophic output/update geometry remains a veto given EXP-020,
022, 024, 025, and 028. Any failed semantic, RNG, transition, or trajectory gate
retires this exact schedule.

## Paired H20 timing gate

On one otherwise-idle 97,871-MiB H20, run five alternating fresh-process
control/candidate pairs. Replay the same accepted tensors and the same registered
candidate mask sequence in every candidate arm. After 100 warmups, measure at
least 1,000 synchronized full steps per arm, separately covering strong-hard,
strong-CutMix, and weak-hard paths and combining means 40/40/20. Include CPU draws,
branching, transfer, forward, loss, backward, sparse SGD participation, and
synchronization; record candidate time by active-block count and transition-drop
status.

Proceed only if:

- observed mean active blocks is `8.55 +/- 0.08` and all block frequencies match
  the registered mask sequence;
- weighted candidate/control mean step time `<=0.99`, every pair `<=1.01`, both
  trial-mean CVs `<=2%`, and candidate p95 `<=1.03x` control mean;
- projected exposure
  `floor(26,898 * control_mean / candidate_mean) >=27,170` updates, while retaining
  the accepted 80/20 time allocation;
- candidate peak allocation is below 600 MiB and no higher than control by more
  than 16 MiB; all state stays finite; projected total runtime including unchanged
  evaluations and loader switch is below 540 seconds.

The mean gate requires a real compute benefit but allows full-depth candidate
steps to populate the p95. Do not replace control-flow skipping with post-compute
multiplication, compile, use channels-last/autocast, or alter the survival sequence
to rescue timing.

## Production verification and interpretation

Only after all gates pass, run the exact candidate once at seed 42 with
`uv run train.py > run.log 2>&1` on the sole idle H20. Require exit zero, finite
standard summary, 300.0 counted training seconds, total below 600 seconds, at
least 27,000 optimizer steps, exactly 1,073,962 parameters, one 80% switch with
eight workers stopped, approximately 50% strong CutMix, hard weak-tail targets,
and no duplicate evaluation epoch. Evaluation always uses all nine
survival-scaled branches and must consume zero stochastic-depth draws.

Require each realized block survival to be within five binomial standard
deviations of its fixed probability and the global mean within 0.02 of 0.95.
Record switch accuracy versus 89.73%, first weak accuracy versus 93.16%, best/final
accuracy, final NLL versus 0.1934, steps, epochs, evaluation count, runtime, VRAM,
per-block survival, transition-drop counts, and timing by active depth.

Acceptance requires every integrity condition and `best_test_acc >=94.25%`.
A switch below 87.08% diagnoses excessive shallow-network regularization. Healthy
training fit but weak evaluation implicates stochastic-train/full-network BN or
survival-scaling mismatch. More updates with lower top-1 rejects the claim that
this sampled-depth ensemble improves generalization. A safety/timing failure is
invalid; a valid lower score is no-improvement. Neither permits rate, seed,
transition exemption, scaling convention, or schedule tuning inside EXP-029.

## Risks

- Nine blocks are far shallower than the literature's successful regime; even a
  10% final drop rate can remove a material fraction of representation depth.
- Dropped transition branches temporarily supply no learned features in newly
  padded channels and may amplify the Option-A weakness seen in transition tests.
- BN statistics update only when a branch survives, while evaluation activates
  every scaled branch. The canonical rule can still create train/eval mismatch.
- Skipped parameters also skip coupled decay and momentum on that batch; this is
  canonical whole-branch omission but changes effective regularization by depth.
- Batchwise masks add gradient variance on top of RandAugment and CutMix, which
  may worsen the already protected strong-phase fit.
- A 0.10-point pass is ten examples at one seed and should not be overstated.

## Sources

- Huang et al., *Deep Networks with Stochastic Depth*, ECCV 2016 / arXiv
  1603.09382: https://arxiv.org/abs/1603.09382.
- `knowledge/papers/stochastic-depth.md` — local literature distillation and
  transfer cautions.
- `experiments/010/04-analysis.md` — accepted 94.15% trajectory, exposure, and
  integrity anchors.
- `02-system-understanding.md`, `03-experiment-learnings.md`, `04-results.tsv`,
  and `experiments/029/01-brainstorm.md` — bottleneck, local failures, current
  frontier, and candidate comparison.
