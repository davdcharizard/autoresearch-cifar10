# Idea: Three-Percent Batchwise Stochastic Depth

## Verdict and Exact Operating Point

This is a plausible but weak-transfer candidate, not a literature-backed favorite.
Use exactly one stochastic-depth policy during the accepted strong phase: keep all
three stage-entry blocks active and independently drop each of the six remaining
same-width residual branches with probability `p_drop = 0.03`, one Boolean per
block per mini-batch. In model order, the survival vector is
`[1.00, 0.97, 0.97, 1.00, 0.97, 0.97, 1.00, 0.97, 0.97]` for
`layer1`, `layer2`, and `layer3`. The two stride-2 Option-A transitions and the
first layer-1 block are never dropped.

On a surviving strong-phase branch, multiply only the residual output after
`bn2` by `1 / 0.97 = 1.030927835051546`; never scale the shortcut. On a dropped
branch, return its input tensor directly and execute neither convolution nor BN.
At the 80% curriculum boundary, disable stochastic depth permanently before the
first weak optimizer step. Weak-tail training and every evaluation execute the
full nine-block accepted network with no scaling.

There is no depth-dependent interpolation, probability ramp, transition drop,
per-example mask, output-only mask, width compensation, or alternative rate in
EXP034. In particular, `0.02`, `0.04`, and the previously brainstormed `0.05`
are not rescue settings.

## Evidence, Local Fit, and Skepticism

Huang et al., *Deep Networks with Stochastic Depth* (ECCV 2016), show that
mini-batch residual bypass can regularize an ensemble of effective depths and
shorten the training graph. The local distillation is
`knowledge/papers/stochastic-depth.md`. Its strongest result is a 1,202-layer
ResNet, so it does not establish a useful rate for this nine-block ResNet-20.
Dropping one local block removes 11.1% of the residual-block depth for that
batch, and the accepted model already shows a recurring underfit response to
identity-oriented residual changes: full preactivation and selective zero-gamma
lowered switch accuracy by 2.85 and 3.25 points in EXP012 and EXP015.

The same family reached the EXP020 finalist review at `p_drop=0.05` but was not
executed. It was rejected because only `0.95^6 = 73.5%` of strong batches would
run all six selectable branches and the compute gain looked marginal relative
to shallow-model/BN risk. The present point is materially more conservative:
`0.97^6 = 83.30%` of strong batches run the complete graph, 16.70% skip at
least one branch, and the expected effective residual depth is 8.82 blocks.
Three percent is still heuristic. It earns a production run only if it proves
both actual speed and safe optimization on immutable production batches.

The accepted data curriculum is otherwise untouched: batch 128, worker-side
crop/flip plus N1/M7, alpha-1 CutMix on 50% of strong batches, the 80% simultaneous
switch to weak hard-label views and low-LR cosine refinement, ordinary momentum
SGD, all-parameter decay `1e-4`, seed 42, and the existing evaluator.

## Exact Implementation and RNG Semantics

Selected `BasicBlock` instances receive a shared, nonpersistent CPU
`torch.Generator` created after model construction and seeded exactly `42`.
The generator is neither a parameter nor a buffer and is excluded from the
state dict and optimizer. It must not consume or replace the global CPU/CUDA RNG
used by model initialization, sampling, workers, or CutMix.

During each strong training forward, sample exactly once for each selected block
in fixed model order with
`torch.rand((), generator=drop_generator).item() < 0.03`. Thus every strong
optimizer step consumes exactly six dedicated draws regardless of which earlier
branches were dropped. The stream is continuous across epochs and evaluations;
it is never reseeded or reset. Evaluation and weak-tail forwards consume zero
draws. A CPU scalar is intentional because the decision controls Python graph
execution; a CUDA scalar would introduce a device-to-host synchronization.
Per-sample masking is excluded because it would still execute every convolution
and therefore would not attack the measured backward bottleneck.

The selected-block path is logically:

```python
if self.training and self.stochastic_depth_enabled:
    dropped = torch.rand((), generator=self.drop_generator).item() < 0.03
    if dropped:
        self.strong_drop_count += 1
        return x

out = F.relu(self.bn1(self.conv1(x)))
out = self.bn2(self.conv2(out))
if self.training and self.stochastic_depth_enabled:
    self.strong_survive_count += 1
    out = out / 0.97
out += x
return F.relu(out)
```

Assert at construction that a selectable block has `stride == 1`, equal input
and output channels, and `need_pad is False`. Returning `x` is valid because
every selected input follows the stem or a post-add ReLU and is nonnegative.
Stage-entry/transition blocks retain their current code exactly. Use
`optimizer.zero_grad(set_to_none=True)` explicitly: on a drop, branch parameters
must have `grad is None`, so PyTorch SGD skips their momentum, coupled decay, and
weight update rather than applying a zero-gradient optimizer step.

Each selected block keeps Python integer survive/drop counters. For its two BNs,
`num_batches_tracked` at run end must equal `strong_survive_count +
weak_batch_count`. A dropped block freezes its BN statistics and optimizer state
for that batch. That state sparsity is part of the declared method, not an
implementation nuisance to compensate. Parameter count and state-dict tensor
keys must remain exactly those of the accepted 1,073,962-parameter model.

## Compute and Branch-Frequency Prior

The six selectable blocks each contribute approximately 18.87M forward MACs;
together they are about 113.25M, or 70.2% of the accepted 161.3M forward path.
At `p_drop=0.03`, the strong-phase expected saving is 3.40M MACs per image,
2.11% of forward work. Weighted over the fixed 80/20 strong/weak schedule, the
forward-MAC prior is a 1.69% reduction. Their convolution/BN backward and saved
activations are also absent on dropped batches, but Python branching, scaling,
and unchanged blocks make 1.69% an upper-bound-like prior, not a measured speed
claim.

At EXP010's 21,446 strong and 5,452 weak updates, each selected branch would
receive about `0.97*21,446 + 5,452 = 26,255` updates before any speed gain, or
97.61% of accepted branch exposure. A measured 1% global step gain projects
about 27,170 total updates and roughly 26,500 selected-branch updates. Production
telemetry must show aggregate strong drop frequency in `[0.028, 0.032]`, every
selected block in `[0.025, 0.035]`, and both dropped and survived cases. These
are integrity checks for the fixed seed, not knobs; failure invalidates the run.

## Immutable-Corpus Safety and Function Gates

Before timing or production, materialize and serialize one exact sequence of 400
accepted strong batches after N1/M7 and the existing CutMix collator, preserving
both hard and probability targets. Record the tensor shapes/dtypes, target-rank
sequence, CutMix count, and a cryptographic file hash. Fresh paired control and
candidate processes must load those same tensors, identical initial parameters,
and identical optimizer state. Seed-only replay across forkserver workers is not
accepted, per EXP019/021/026.

Run these fail-closed gates, serializing evidence before assertions:

1. With aligned weights in eval mode, candidate and control logits are bitwise
   identical on weak and strong tensors, all nine branches execute, scaling is
   absent, BN counters and the mask-generator state do not advance, and repeated
   evaluation leaves the next training mask trace unchanged.
2. A disposable forced-drop case returns the input exactly, fires no residual
   Conv/BN hooks, leaves both BN counters/statistics and branch weights/momentum
   unchanged, gives every branch parameter `grad is None`, and passes the input
   gradient through the identity shortcut.
3. A disposable forced-survival case executes both Conv/BN pairs once, produces
   finite gradients and BN updates, scales the residual by exactly `1/0.97`, and
   leaves the shortcut unscaled. Disabling the feature restores the accepted
   block formula bitwise for aligned tensors.
4. The dedicated seed-42 generator replays the full 400-step, six-draw mask trace
   bitwise from a saved state. It consumes exactly 2,400 draws; model/data global
   CPU and CUDA RNG states remain unchanged by mask generation. Each block must
   drop between 4 and 22 times and aggregate survival must lie in `[0.96, 0.98]`.
5. Train paired models over all 400 immutable batches. Losses, logits, gradients,
   updates, parameters, and BN state must remain finite. Reject candidate-only
   `>95%` one-class predictions, candidate terminal loss EMA above `1.25x`
   control, any per-step whole-model update-norm ratio above `1.50x`, or logit-RMS
   ratio outside `[0.50, 2.00]`. The control is subject to identical reporting;
   a shared transient is not mislabeled candidate-specific.
6. After the simulated 80% boundary, disable masks without resetting the
   generator. Over 64 immutable weak hard-target batches, all selected branches
   execute, no mask draw occurs, scaling is absent, and each selected BN counter
   advances exactly 64 times.

A failed function, RNG, trajectory, or branch-frequency gate blocks production.
Observed preflight results cannot be used to change the probability, selected
blocks, scaling convention, generator seed, or phase scope.

## Paired H20 Timing and Exposure Gates

Use the sole idle NVIDIA H20 and five alternating fresh-process control/candidate
pairs. Warm CUDA kernels on disposable model/optimizer instances, then construct
fresh aligned timed instances so warmup does not consume the measured mask stream
or alter optimization state. Each arm cycles immutable production-shaped pinned
batches through 1,000 full optimizer steps: 800 strong steps (candidate masks
enabled) followed by 200 weak steps (full graph), with the accepted mixture of
hard/probability targets. Include H2D transfer, `zero_grad`, conditional forward,
cross-entropy, backward, SGD, and synchronization exactly as the counted region
does. Loader iteration is unchanged and outside the production timer, but worker
liveness and wall-time projections remain required.

Alternate pair order (`control/candidate`, then `candidate/control`) and record
per-arm mean/median/p95 step time, pair ratios, CV, mask counts, effective branch
updates, peak allocation, and full-graph eval latency. Production is allowed only
if all of the following hold:

- the median of five schedule-weighted candidate/control mean ratios is `<=0.99`,
  every pair is `<=1.01`, and both arm CVs are below 3%;
- the projected candidate has at least 27,169 global optimizer steps and at
  least 26,400 effective updates for every selected branch;
- candidate p95 is `<=1.03x` control, peak allocation is no more than 16 MiB
  above control, and all timed states are finite;
- the fixed timing trace has aggregate survival in `[0.96, 0.98]`, every block
  survives in `[0.94, 0.995]`, and weak steps execute all branches;
- aligned eval-mode logits are bitwise identical, candidate/control inference
  latency is `<=1.01`, projected total wall time is below 540 seconds, and the
  unchanged epoch/evaluator logic projects no more than the accepted 19 looks.

True conditional skipping is mandatory. If the 3% policy cannot prove at least
1% measured schedule speedup, do not replace it with output multiplication or
raise the rate. More than 19 production evaluations is invalid because faster
epochs must not grant extra opportunities to the max-over-checkpoints metric.

## Falsifiable Hypothesis, Production Contract, and No Rescue

**Hypothesis:** Compared with the moving 94.15% accepted baseline, 3% batchwise
skipping of the six identity residual branches during only the strong phase will
reduce schedule-weighted full-step time by at least 1%, preserve at least 26,400
effective updates per selected branch and the accepted data curriculum, avoid
the 87.08% strong-underfit marker, and use mild effective-depth noise to reach
`best_test_acc >= 94.25%`. The point prediction is 94.28%; a plausible positive
range is 94.25-94.40%.

The main falsifiers are no measurable speedup, candidate-only early class
concentration, switch accuracy below 87.08%, sparse-BN calibration damage, fewer
than 26,400 effective branch updates, final NLL worse than EXP010's 0.1934, or a
valid `best_test_acc < 94.25%`. Faster global exposure cannot excuse failure of
the primary metric, and favorable loss or switch diagnostics cannot override it.

After all gates pass, query the moving baseline from `04-results.tsv` and run
exactly once at seed 42 with
`CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1`. Require one
H20, exit zero, a finite summary, 300 counted seconds, total below 600 seconds,
the accepted model/optimizer/data/LR lifecycle, one 80% switch with eight stopped
workers, 45-55% strong CutMix, hard weak targets, at most one evaluation per
epoch and exactly the accepted 19 total looks, 1,073,962 parameters, and valid
mask/BN/update counters. Accept only a gain of at least 0.10 percentage points
over that queried baseline.

A mechanical bug may be repaired only if it leaves the declared block set,
`p_drop=0.03`, scaling, generator seed/order, phase scope, data, optimizer,
timer, and evaluator semantics unchanged. Any semantic failure or lower valid
accuracy ends EXP034 as invalid or no-improvement. Do not rerun, tune, reroll,
combine with channels-last, or substitute another stochastic-depth variant.
