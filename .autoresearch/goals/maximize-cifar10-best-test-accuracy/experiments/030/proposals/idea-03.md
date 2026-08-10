# Proposal: Reduce CutMix Probability from 0.50 to 0.40

## Intervention and falsifiable hypothesis

Change exactly one constant in `train.py`:

```python
CUTMIX_PROBABILITY = 0.4  # accepted: 0.5
```

Keep `CUTMIX_ALPHA=1.0` and every other accepted EXP-010 choice unchanged:
width-2 postactivation ResNet-20, batch 128, FP32/default-TF32, standard momentum
SGD, all-parameter decay `1e-4`, N1/M7 strong views, the full 80% high-LR strong
phase, simultaneous switch to low-LR weak hard-label refinement, seed 42, worker
lifecycle, timer, evaluator, and evaluation cadence.

The candidate makes 60% rather than 50% of strong batches hard while retaining
CutMix throughout the entire eligible phase. It therefore preserves 80% of the
accepted mixed-batch exposure but slightly reduces regional/soft-target pressure.
The hypothesis is that this restores roughly 0.1-0.3 point of strong-phase fit
without giving up the regional invariance that made p=0.50 gain 0.60 point,
raising `best_test_acc` from 94.15% to at least 94.25%. Point prediction:
switch accuracy 89.8-90.1%, first weak accuracy at least 93.16%, final NLL no worse
than 0.1934, and best accuracy 94.25-94.35%.

This is one preregistered point, not a sweep. A valid miss retires p=0.40; do not
try 0.35/0.45 after seeing its result inside EXP-030.

## Mechanism and evidence

CutMix replaces a rectangle with class-bearing donor pixels and weights targets by
visible area. EXP-010 established that alpha-1 CutMix on 49.77% of strong batches
improved the width-2 recipe from 93.55% to 94.15% with nearly unchanged exposure.
Its switch accuracy was 89.73%, only 0.35 below the no-CutMix width-2 run, and the
weak tail finished at its best with 0.1934 NLL.

EXP-011 supplies a directional warning: p=0.75 preserved steps and NLL but lowered
the switch by 2.91 points and best accuracy by 0.15, so excessive mixed frequency
over-regularizes the short strong phase. EXP-012 and EXP-026 each reached 94.22%
after larger strong-fit deficits and rapid hard-tail recovery, suggesting that a
small fit/regularization rebalance could clear the ten-example gate. Conversely,
EXP-027 showed that removing CutMix for a contiguous high-LR interval is harmful.
p=0.40 does not create such a bridge: mixed and hard batches remain interleaved
until the accepted LR/data transition.

There is no local evidence that accuracy is monotone below p=0.50. The accepted
point may already be optimal, and fewer mixed examples may simply discard part of
its 0.60-point gain. The experiment tests that narrow trade rather than assuming
the p=0.75 response extrapolates through p=0.50.

## Exact data and RNG semantics

Retain the existing `cutmix_collate` implementation and its
`torch.random.fork_rng(devices=[])` scope. For each strong batch, the same scalar
gate draw `u` now selects CutMix iff `u < 0.40`; weak-loader batches never use the
collator and remain one-dimensional hard targets.

Relative to control on identical pre-policy inputs and gate draws:

- `u < 0.40`: both arms execute identical alpha-1 CutMix;
- `0.40 <= u < 0.50`: control executes CutMix, candidate returns the unchanged
  hard batch; this 10% band is the entire intervention;
- `u >= 0.50`: both arms return identical hard batches.

Because the gate and CutMix randomness stay inside `fork_rng`, threshold choice
must not advance persistent CPU or CUDA RNG state or change later source
augmentation/shuffle draws. Do not alter comparison direction (`<`), introduce a
new generator, redraw failed gates, schedule probability by time, or change alpha.
Parameter initialization, state dict, model outputs for a fixed collated batch,
and parameter count remain bitwise accepted.

## Semantic and immutable-corpus gates

Before production, require:

1. Static checks prove the one-line constant diff, alpha exactly 1.0, unchanged
   model/optimizer/schedule/evaluator, 1,073,962 parameters, and no tracked change
   beyond `train.py`.
2. On explicit gate values around 0.40 and 0.50, verify all three branch classes.
   Shared hard outputs must be bitwise equal; shared CutMix images/targets must be
   bitwise equal; disagreement-band candidate images/labels must equal the
   pre-policy hard batch while control targets are valid `[128,10]` probabilities
   summing to one and matching pasted area. Global RNG state must be unchanged.
3. Persist 200 exact pre-policy N1/M7 batches, associated gate draws, CutMix
   pairing/box randomness, targets, branch class, and a SHA-256 manifest. Stratify
   the corpus as 80 shared CutMix (`u<0.4`), 20 disagreement-band, and 100 shared
   hard (`u>=0.5`) batches. Independently restored accepted/candidate arms must
   consume byte-identical source tensors and the registered policies.
4. Through all 200 paired updates require finite logits, loss, gradients,
   parameters, momentum, and BN buffers; no candidate-only >95% one-class
   prediction concentration; candidate/control loss, logit-RMS, gradient-norm,
   and update-norm ratios no greater than 1.5; and candidate terminal loss EMA no
   more than 1.25x control. Serialize branch class and histogram before vetoes.
5. Exercise 20,000 live strong-loader deliveries with eight forkserver workers.
   Require 39-41% CutMix, 59-61% hard, valid target shapes/dtypes, no corrupted
   area-weight targets, at least one delivery from every worker, clean shutdown,
   and a rebuilt weak loader with only hard targets.

Fresh seed-only worker replay is not causal enough after EXP-019/021; persisted
post-transform/pre-policy tensors and gate metadata are mandatory. A semantic,
RNG, concentration, or lifecycle failure invalidates this exact point and cannot
be repaired by changing probability.

## Timing and exposure gate

The candidate adds no GPU operator and executes fewer worker-side CutMix calls.
EXP-010 measured hard/soft median steps at 10.823/10.829 ms, so the accuracy
mechanism is regularization balance, not extra exposure. Still, nominally cheap
changes require paired measurement after EXP-003/029.

On one idle 97,871-MiB H20, run five alternating fresh-process control/candidate
pairs after conditioning, using production loaders and at least 1,000 synchronized
steps per arm. Measure hard and soft GPU paths separately and combine them at each
arm's registered 50/50 and 60/40 proportions; include iterator wait, transfer,
forward, loss, backward, SGD, and synchronization. Proceed only if:

- candidate/control weighted mean `<=1.01`, every pair `<=1.025`, both trial-mean
  CVs `<=2%`, and candidate p95 `<=1.04x` control mean;
- projected exposure
  `floor(26,898 * control_mean / candidate_mean) >=26,629` updates (99% retention);
- peak allocation remains below 650 MiB, all state is finite, and projected total
  runtime including the unchanged loader switch/evaluations is below 540 seconds.

Do not claim a benefit from small loader savings or add another GPU/data change to
fund the candidate. No channels-last, precision, loss, worker-count, or evaluation
fallback is allowed.

## Production verification and diagnostics

If all gates pass, run the exact candidate once at seed 42 via
`uv run train.py > run.log 2>&1` on the sole idle H20. Require exit zero, finite
standard summary fields, 300.0 counted seconds, total below 600 seconds, at least
26,629 updates, 1,073,962 parameters, one 80% switch with eight workers stopped,
hard weak-tail targets, and no duplicate evaluation epoch. Realized strong-phase
CutMix must be 38.5-41.5%; no rerun is allowed for an unusual but in-bound draw.

Record:

- 20%, 40%, 60%, 70%, and final-strong accuracy/loss;
- switch accuracy versus EXP-010 89.73% and EXP-011 86.82%;
- first weak accuracy versus 93.16%, peak/final accuracy and epoch, final NLL
  versus 0.1934, steps, epochs, evaluations, runtime, and VRAM;
- hard/CutMix counts and realized target-area statistics.

Expected support is a modestly higher switch without losing immediate weak-tail
conversion or NLL. A higher switch but lower peak/NLL quality means p=0.50's
regional regularization was more valuable than the extra hard fit. An unchanged
switch and lower accuracy means the probability perturbation only removed useful
mixed examples. A lower switch would falsify the assumed direction and should be
reported without rescue.

## Verdict and risks

Accept only if every integrity/runtime condition passes and
`best_test_acc >=94.25%`. A complete result from 94.16-94.24 is still formally
`no-improvement`, not a near-pass warranting a rerun. A lower valid score is also
`no-improvement`; missing safety/timing/protocol conditions is `invalid`; a process
failure is `crash`.

Main risks are:

- p=0.50 may already sit at the useful balance; 20% fewer mixed batches can weaken
  occlusion/localization invariance more than they improve fit;
- interspersed hard N1/M7 batches are not the same as EXP-027's contiguous bridge,
  but they still ask hard labels to explain distorted views at LR 0.1;
- a one-constant policy changes both images and targets in the 10% disagreement
  band, so it cannot isolate visual from soft-target regularization;
- the likely effect is near the one-seed resolution: 0.10 point is ten test
  examples, so a bare pass is protocol-valid but weak causal evidence;
- post-result testing of adjacent probabilities would become parameter chasing.

## Sources

- Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers with
  Localizable Features*, ICCV 2019; distilled in `knowledge/papers/cutmix.md`.
- `experiments/010/04-analysis.md` — accepted p=0.50 gain and trajectory.
- `experiments/011/04-analysis.md` — p=0.75 strong-underfit failure.
- `experiments/002/04-analysis.md`, `experiments/012/04-analysis.md`, and
  `experiments/026/04-analysis.md` — tail behavior and near-gate underfit results.
- `experiments/027/04-analysis.md` — why CutMix must remain interleaved until 80%.
- `02-system-understanding.md`, `03-experiment-learnings.md`, `04-results.tsv`,
  and `experiments/030/01-brainstorm.md` — local constraints and candidate context.
