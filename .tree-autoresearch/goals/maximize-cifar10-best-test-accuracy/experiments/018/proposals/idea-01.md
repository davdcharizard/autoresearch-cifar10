# Proposal: Clean Retry of Reference-Ordered Full-Run Gradient Centralization

## Summary

Retry EXP017's scientific intervention from the same EXP002 parent: apply
coefficient-free Gradient Centralization (GC) after every backward to all and
only the 16 convolution weights and final classifier weight. Materialize the
parent's coupled L2 term first, centralize each eligible regularized direction
per output row/filter, then pass it through the parent's unchanged momentum and
Nesterov update. BatchNorm affine parameters and biases retain their ordinary
coupled-decay directions. No model, data, stochastic, schedule, phase, or
evaluation setting changes.

EXP018 differs from EXP017 only in its temporary, accuracy-blind feasibility
harness. EXP017 never produced a complete timing vector or metric because the
harness first miscounted the fixed audit cadence and then retained 1,024 CUDA
loss scalars while asserting allocation stability. Its deterministic mechanism
checks passed, and Claude's result audit classified the leaf as `crash/NaN`,
not evidence against GC. The retry therefore preserves the research hypothesis
while preregistering scalar-only diagnostic accumulation, correct cadence
arithmetic, matched allocation checkpoints, and explicit temporary release.

All commands expose physical GPU 0 only with `CUDA_VISIBLE_DEVICES=0`. Only
`train.py` may change, seed 42 and the 300-second charged budget remain fixed,
validation remains at most once per epoch, and the single metric launch retains
the 600-second outer timeout.

## Scientific Rationale

EXP002's WRN-16-4 plus front-loaded CutMix reached 95.23%, with 95.19% final
accuracy and 0.2044 final cross-entropy. A CutMix/drop-path retuning child did
not yield robust evidence, while the later SAM success shows that optimizer
geometry can matter but pays for a second forward. GC is a one-backward,
coefficient-free projection that targets correlated drift within each output
filter without reducing image diversity or adding persistent training state.

The ECCV 2020 Gradient Centralization paper describes per-output zero-mean
weight-gradient projection and reports optimization/generalization benefits in
vision models. The official CIFAR optimizer forms coupled weight decay before
centralization and momentum, which fixes the ordering used here. See
`knowledge/papers/gradient-centralization.md`, the primary paper at
<https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2471_ECCV_2020_paper.php>,
and the official optimizer at
<https://github.com/Yonghongwei/Gradient-Centralization/blob/master/GC_code/CIFAR100/algorithm/SGD.py>.
This evidence motivates the mechanism but does not guarantee a gain under this
small BatchNorm WRN, CutMix, drop path, Nesterov, and wall-clock schedule.

## Fixed Scientific Mechanism

### Eligibility and axes

Construct the eligible inventory once by iterating `model.named_modules()` and
selecting the `weight` object from every `nn.Conv2d` and `nn.Linear`. Assert the
frozen EXP002 inventory before training:

- 16 convolution weights and one classifier weight;
- 17 eligible tensors, 2,745,264 eligible elements, and 2,266 output rows;
- 44 total parameter tensors and 2,748,890 total trainable elements;
- 27 excluded BN-affine/bias tensors containing 3,626 elements.

For each eligible regularized direction `d`, subtract its mean across every
dimension except the output dimension:

```python
reduce_dims = tuple(range(1, d.ndim))
row_mean = d.mean(dim=reduce_dims, keepdim=True)
torch._foreach_sub_(eligible_directions, row_means)
```

Thus each convolution filter is centered across input-channel and spatial
dimensions, and each classifier row is centered across its 256 inputs. Never
mix output rows, centralize BN/bias tensors, normalize magnitude, clip, cast
down, or introduce a tunable coefficient or epsilon. The parent has BF16
autocast but no `GradScaler`; parameters and gradients remain FP32. GC uses no
RNG and is applied once on every early-CutMix, early-clean, and late-clean
optimizer step.

### Official coupled-decay ordering

Preserve the parent's loss and sole backward. Immediately afterward, under
`torch.no_grad()`, require all gradients and materialize coupled L2 for all 44
parameter tensors:

```python
loss.backward()
torch._foreach_add_(all_gradients, all_parameters, alpha=WEIGHT_DECAY)
centralize_eligible_directions(eligible_gradients)
optimizer.step()
```

Use the existing PyTorch SGD with the same parameter order, learning-rate
schedule, `momentum=0.9`, dampening, `nesterov=True`, and state initialization,
but set its internal `weight_decay=0` to prevent double decay. This implements
the official order exactly:

1. form the data-loss gradient `g`;
2. form `d = g + 1e-4 * parameter` for every parameter;
3. replace eligible `d` with `d - row_mean(d)`;
4. apply the unchanged momentum, Nesterov look-ahead, and parameter update.

For excluded tensors, deterministic two-step checks must establish parameter
and momentum-buffer equality with the parent's internal-decay SGD. For eligible
tensors, checks must establish equality with an explicit
`decay -> centralize -> momentum -> Nesterov` reference. Use in-place gradient
operations without `.data`; do not subclass or replace SGD.

### Parent-preservation contract

Keep EXP002's architecture and initialization order, 2,748,890 parameters,
normalization, crop/flip transform, shuffled loader, batch size 256, dedicated
CutMix RNG streams and constants, drop-path schedule and RNG, BF16/channels-last
execution, time-cosine LR, timing boundaries, 195-step epoch definition, and
evaluation byte-for-byte unchanged. CutMix still constructs one weighted CE
and performs one backward; GC transforms that combined direction. Evaluation
still occurs at most once per epoch and `best_test_acc` remains the maximum of
the inherited evaluator outputs. No compile, CUDA graph, extra forward, model
state, target change, or phase-specific GC gate is allowed.

All L2 and GC work is after the parent's charged-step start and before its
post-step synchronization. Any throughput cost therefore reduces steps within
the same fixed 300-second budget rather than being hidden outside the timer.

## Hypothesis and Falsification

**Hypothesis.** Per-output projection of the regularized directions will
improve conditioning and suppress correlated filter drift across mixed and
clean training while preserving nearly all EXP002 exposure. The primary
prediction is a valid `best_test_acc >= 95.33%`, at least 0.10 points above the
95.23 parent. A result at or above 95.53 is stronger but still noise-limited
single-seed evidence; matching 95.61 or reaching 95.71 is separate global-tree
context, never a revised local threshold.

**Counter-hypothesis.** BatchNorm already removes much of the functionally
relevant common-mode filter direction, GC can discard useful scale adaptation,
or its reductions can reduce wall-clock optimizer exposure enough to offset
any geometric benefit. A valid result below 95.33 falsifies this isolated
configuration for the tree; do not tune axes, eligible layers, cadence, or
strength after observing accuracy.

Sparse FP64 audits at one-based step 1 and every 512 steps report regularized,
removed, and centralized squared energy, convolution/classifier splits,
decomposition error, post-GC row-mean residual, and nonfinite counts. Removed
energy at most 1% supports practical redundancy; 1-5% is ambiguous moderate
action; at least 5% shows substantial projection. These diagnostics explain
the mechanism but never alter training or the formal accuracy verdict.

## Corrected Harness: Measurement, Not Mechanism

The harness is temporary and accuracy-blind; it is not part of `train.py`, the
metric run, or the scientific intervention. It guards the evaluator and test
loader before any trace and asserts zero evaluator calls, test-batch iterations,
or accuracy values. It imports the repository through an explicit fixed path,
uses the mathematically correct audit count
`1 + floor((steps - 1) / 512)` (three samples for 1,024 steps), and emits one
complete machine-readable result vector.

The production-order trace must not retain any per-step CUDA tensor. Allocate
one fixed device scalar before the measured region for nonfinite status and
reduce each detached loss into it in place; read it only after the trace. Do not
append losses, logits, gradients, row means, or audit values to Python lists.
Delete step-local outputs/losses and audit temporaries after use. Compare live
allocation only at matched post-audit points after synchronization and Python
garbage collection: establish the baseline immediately after step 512's audit
and compare immediately after step 1,024's audit, when optimizer state and the
fixed diagnostic scalar already exist in both snapshots. Record both allocated
and reserved memory, but only unexpected growth in live allocated bytes is an
integrity failure; allocator reserve growth is informational.

This redesign is frozen before any numeric vector. It corrects bookkeeping
identified by source inspection and does not use accuracy, latency, energy, or
memory results to favor GC. A malformed/exceptional harness may receive at most
one documented implementation repair before it emits a complete vector; a
complete numeric gate failure is decisive and cannot be rerun.

## Verification Protocol

1. **Scope and GPU identity.** Confirm EXP002 at 95.23, only `train.py` differs
   from commit `a36dc09`, syntax/diff checks pass, physical GPU 0 is the
   approximately 98 GB NVIDIA H20, and `CUDA_VISIBLE_DEVICES=0` exposes exactly
   one CUDA device.
2. **Deterministic mechanism checks.** On CPU and GPU, verify exact inventory,
   2-D/4-D axes, FP32 gradients under BF16 autocast, reconstruction and
   orthogonality, maximum post-row-mean residual at most `1e-6`, FP64 energy
   decomposition error at most `1e-5`, foreach/loop subtraction equality, RNG
   neutrality, excluded-update parity, and explicit two-step Nesterov ordering.
3. **One corrected accuracy-blind preflight.** Run a 1,024-step full-model
   production-order candidate trace and verify the fixed three audit samples,
   scalar-only finiteness, exact path/dose counters, finite state, positive
   removed energy, matched-checkpoint live-allocation stability, and zero test
   access. Then run five alternating-order paired parent/candidate latency
   rounds on the same real CIFAR batches, covering 44 early CutMix, 45 early
   clean, and 31 late clean steps per round.
4. **Frozen feasibility gates.** Require parent drift at most 4%, paired-ratio
   MAD/median at most 1.5%, median candidate/parent charged latency at most
   1.03, every ratio at most 1.06, projected steps at least 27,000, projected
   epochs at least 138, projected total below 600 seconds, and all structural
   integrity checks. Memory capacity is informational except for OOM or
   persistent live-allocation growth.
5. **Exactly one metric launch.** Only after preflight passes, run
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
   Require a complete summary, 299.5-301.0 charged seconds, total below 600,
   exact GC/path reconciliation, sparse/final finiteness, unchanged evaluation
   semantics, and no more than one validation per epoch. Transcribe evidence
   before protocol cleanup; never rerun or modify the mechanism after accuracy.

The projected and realized 27,000-step/138-epoch floors describe scientific
dose. The preflight may use them to prevent an obviously underexposed launch,
but a completed valid metric run is classified solely by the goal's frozen
accuracy and integrity conditions. Final-16 evaluation mean/range, final
accuracy/loss, dose, energy, and VRAM are interpretation context only.

## Why This Is Not Repeating a Research Failure

EXP017 did not observe GC's latency, training trajectory, or test accuracy. Its
only failures were assertions over temporary diagnostics: one expected-count
formula and one list of retained device scalars. The candidate implementation
had already passed exact mechanism, ordering, excluded-parameter, RNG, and
numerical-residual checks. No evidence exists that the official-order GC
hypothesis failed.

EXP018 therefore repeats the unanswered scientific question while replacing
the faulty measuring instrument under a preregistered, accuracy-blind contract.
It does not soften the parent's 95.33 acceptance threshold, alter GC to fit a
prior result, reuse a test observation, or consume EXP017's repair allowance.
If the corrected complete vector fails a frozen feasibility gate, or the sole
metric run finishes below 95.33, this new leaf becomes actual negative evidence
and the isolated mechanism should not be retried again without a materially new
scientific rationale.

## Risk Assessment

- **Scientific risk: medium-high.** BatchNorm may make GC redundant, and a
  0.10-point max-selected gain is below the goal's roughly 0.30-point
  noise-resolution context.
- **Throughput risk: low-medium.** Seventeen reductions and one foreach
  subtraction are small relative to the backward but launch overhead is
  hardware-dependent; the paired preflight decides this before metric access.
- **Implementation risk: low-medium.** External coupled decay must be exactly
  once and before GC. Deterministic excluded and eligible optimizer references
  directly test the dangerous ordering and double-decay cases.
- **Protocol risk: low after correction.** Fixed-size device diagnostics,
  matched audit snapshots, correct cadence arithmetic, and explicit temporary
  release remove both known EXP017 harness defects without changing the
  experiment itself.

## Estimated Effort

Low-to-medium. The scientific code is a compact optimizer-direction transform
plus sparse reporting in `train.py`; most effort is deterministic verification
and one corrected preflight. Expected wall time is under five minutes for
smokes/preflight plus roughly eight minutes for the sole metric run, always on
physical GPU 0.
