# Proposal: Three-Point Terminal Checkpoint Average

## Summary

Preserve the accepted EXP-027 model, augmentation, optimizer, schedule, seed,
and live training trajectory. Capture detached FP32 copies of every trainable
parameter at two predetermined late points: the first completed optimizer
updates whose pre-step counted times are at least 95% and 97.5% of the
300-second budget. At budget exhaustion, form one uniform arithmetic average of
those two snapshots and the terminal live parameters. Use that averaged
parameter view for the existing terminal evaluator call only, while retaining
the terminal live BatchNorm buffers. Restore the terminal live parameters in a
`finally` block.

This is a short, finite checkpoint average, not another version of EXP-013's
late whole-state EMA. It has exactly three equally weighted states, spans only
the final 5% of counted time, never averages BatchNorm buffers, and does not
alter any nonterminal evaluation. It requires no extra model forward, data
traversal, evaluator call, dependency, or source file beyond `train.py`.

## Limiter Diagnosis and Local Evidence

The accepted EXP-027 run reached 94.32% best and 94.22% final accuracy at
133.00736 data passes. Its final loss was 0.2523 even though the clean-tail
training loss approached zero. The remaining limiter is generalization and
top-1 boundary quality, not memory, data delivery, or wall time. The accepted
trajectory also has a 0.10-point best-to-final gap, so a low-cost treatment of
terminal iterate noise remains plausible on the new, deeper-plus-invariance
baseline.

EXP-013 tested a materially different averaging rule on the shallower model:
it initialized a complete inference-state EMA at the 65% transition, updated
all floating parameters and BatchNorm buffers after every later step with
decay 0.999, and evaluated only EMA states after initialization. It retained
140.97 passes and peaked at 94.10%, but its terminal EMA fell to 93.79%. The
analysis identified trajectory lag and approximate averaged BatchNorm moments
as the leading failure mechanisms. It explicitly left a shorter averaging
window and parameter-only semantics unexplored.

The local distillation of Ajroldi, Orvieto, and Geiping, *When, Where and Why to
Average Weights?*, supports carefully windowed checkpoint averaging and warns
against including under-trained iterates. The proposed last-5% window applies
that principle narrowly. It begins long after mixup and RandAugment have ended,
when all three states are produced by the same clean-label data policy and the
cosine schedule is close to its nonzero 0.002 floor.

## Exact Intervention

### Predetermined snapshots

Define exactly:

- `AVERAGE_FRACTIONS = (0.95, 0.975)`;
- implicit third fraction `1.0`, represented by the terminal live state;
- uniform coefficients `(1/3, 1/3, 1/3)`;
- snapshot contents: all and only tensors returned by
  `model.named_parameters()`, including convolution/linear weights and
  BatchNorm affine parameters.

At the beginning of each training step, compare the existing pre-step
`total_training_time` with the next threshold. After `optimizer.step()`, if the
threshold was already reached, clone each parameter under `torch.no_grad()` and
before the existing `torch.cuda.synchronize()`. Thus snapshot-copy GPU work is
charged to that step's counted `dt`. Each threshold fires exactly once. The
95% snapshot is the first post-SGD state whose pre-step counted time is at
least 285.0 seconds; the 97.5% snapshot is defined analogously at 292.5
seconds. A step that crosses a threshold cannot be retroactively copied; using
the next step makes the convention executable and auditable.

Snapshots are external plain dictionaries or lists of detached FP32 tensors.
They must not be registered on the model, inserted into the optimizer, or
included in the production model's `state_dict`. The accepted live gradients,
optimizer momentum, learning rates, and subsequent SGD updates never consume
an averaged value. Log each snapshot's fraction, step, and pre-step counted
time exactly once.

### Terminal arithmetic and restoration

At the budget-exhausted evaluation branch, require both snapshots to exist and
all snapshot and live parameter tensors to be finite. Clone the terminal live
parameters as both the third checkpoint and the restoration backup. Enter
`try/finally` before copying any averaged value. For every original parameter
object, copy in

```text
(snapshot_95 + snapshot_97_5 + terminal_live) / 3
```

under `torch.no_grad()`, invoke the frozen evaluator exactly once, then restore
the terminal values into the same parameter objects in `finally`. Never replace
a `Parameter`, load a replacement module, mutate optimizer state, or keep the
averaged values installed after evaluation.

All ordinary `epoch % EVAL_EVERY == 0` evaluations before budget exhaustion
continue to evaluate the live model exactly as EXP-027. If the terminal partial
epoch also satisfies the normal cadence condition, the existing combined
boolean still causes one terminal call, not two. The intervention therefore
respects the no-more-than-once-per-epoch rule. The snapshot times, average
coefficients, and terminal-only selection are independent of all evaluator
outputs.

## BatchNorm and Data Semantics

Average BatchNorm's trainable affine scale and bias because they are ordinary
optimized parameters, but do not snapshot or average `running_mean`,
`running_var`, or `num_batches_tracked`. The terminal live buffers remain in
place for the averaged-parameter evaluation. Averaging variances is not a
valid pooled-variance calculation, while EXP-013's whole-state averaging
already exposed approximate-buffer mismatch as a credible failure mode.

Do not recalibrate BatchNorm. In particular, do not run any post-budget train
loader pass, reuse prefetched training batches, set BatchNorm layers to train
mode during evaluation, or derive moments from test/evaluator inputs. This
avoids unbudgeted train-data use and evaluator leakage. Terminal live buffers
are an imperfect but preregistered low-cost approximation; all three parameter
states are nearby and come from the same clean tail, making that approximation
more defensible than it was for the 65%-start whole-state EMA.

## Why This Window Is Distinct and Defensible

The two stored states plus endpoint cover only 15 counted seconds, about 5% of
the run and roughly 6.7 accepted data passes. Their center of mass is 97.5% of
training rather than the much earlier effective center of an EMA maintained
through the full 35% hard-label tail. The three-point rule also gives the
terminal state one explicit third of the mass; it cannot develop a long hidden
tail of stale contributions. Meanwhile, 7.5 seconds between samples is long
relative to minibatch noise, so the checkpoints are more meaningfully
separated than adjacent SGD iterates.

The choice is fixed before scoring. Do not choose snapshots from high-accuracy
epochs, weight them by evaluator loss, retain only favorable checkpoints, or
change the fractions after observing any metric. A shorter span than 5% risks
averaging nearly identical models; a longer span recreates EXP-013's lag risk.
Three points are the smallest symmetric endpoint-inclusive rule that samples
an interior trajectory rather than merely interpolating two endpoints.

## Feasibility and Preflight Validation

Use an evaluator-free preflight before the sole score:

1. Require the accepted `(2,2,3)` topology, 987,098 trainable parameters,
   unchanged original `state_dict` keys, seed 42, and frozen `prepare.py`.
2. On a deterministic toy model and on the real model, prove that each
   parameter equals the exact float32 three-point arithmetic oracle while all
   BatchNorm buffers remain bitwise equal to terminal live buffers.
3. Prove complete parameter coverage with no buffer or optimizer-state
   inclusion; detached snapshots; unchanged parameter object IDs, optimizer
   references, gradients, momentum, param groups, and live values after normal
   restoration.
4. Inject an exception after a partial average-to-live copy and prove complete
   bitwise restoration. Verify finite-value guards fail closed before calling a
   fake evaluator.
5. Drive synthetic pre-step times around 285.0 and 292.5 seconds and require
   exactly one correctly ordered snapshot at each threshold, no earlier copy,
   and no result-dependent branch. Verify one fake evaluator call at a terminal
   epoch whether or not it is divisible by five.
6. CUDA-time two full-model parameter clones after warmup and emit raw windows
   before assertions. Charge both clone costs conservatively to the accepted
   300-second path. Require CV at most 5%, at least 99% projected image-throughput
   retention, at least 131.67 projected passes from EXP-027's 133.00736, and an
   explicit total-wall projection below 500 seconds. The terminal averaging and
   restoration happen in the already-excluded evaluation interval but must
   still fit the 600-second hard wall.
7. Require peak-memory projection to remain far below H20 capacity and final
   averaged logits to be finite with shape `[256, 10]`. No real evaluator or
   CIFAR-10 test examples may be used in preflight.

Abort before scoring on any semantic, restoration, threshold-count, finite,
throughput, wall, topology, or source-integrity failure. A stable preflight
miss is not rerun or repaired within EXP-033.

## One-Shot Scoring and Audit

After preflight passes, run exactly once on one idle NVIDIA H20:

```bash
rm -f run.log
timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, one complete finite summary, 300.0-300.5 counted seconds, less
than 600 seconds total, at least 131.67 realized passes, exactly two ordered
snapshot markers, 987,098 parameters, the accepted mixup and exhausted-epoch
RandAugment transitions, and unique evaluation epochs. The terminal evaluation
must be marked as the three-point average and occur once; all earlier
evaluations must be marked live. Never evaluate live and averaged terminal
states in the same epoch, reroll seed 42, repeat a valid score, or adjust the
window, coefficients, parameter set, or BN policy from interim/final results.

The goal-level improvement threshold is `best_test_acc >= 94.42%`. For direct
support of the averaging mechanism, also preregister the stronger endpoint
condition `final_test_acc >= 94.42%`, because the final value is the only
averaged evaluation. If best accuracy reaches 94.42% only at an earlier live
evaluation while the averaged endpoint does not, report the primary metric
faithfully but classify the checkpoint-averaging mechanism as unsupported.
Final loss is corroborating evidence only and cannot override top-1 accuracy.

## Falsifiable Hypothesis

On the accepted EXP-027 learner, uniform averaging of the predetermined 95%,
97.5%, and terminal trainable-parameter checkpoints will reduce terminal
iterate noise without meaningful lag or exposure loss. In one fixed-seed run it
will retain at least 131.67 data passes and produce a terminal averaged accuracy
of at least 94.42%, thereby also raising `best_test_acc` by the required 0.10
points over 94.32%.

This hypothesis is falsified by any valid run whose averaged terminal accuracy
is below 94.42%, even if loss improves. A materially worse endpoint with normal
exposure implicates parameter-space curvature or terminal-BN mismatch; a low
pass count implicates implementation cost rather than the generalization
mechanism.

## Closure and Risks

- A valid normal-exposure miss closes the exact uniform
  `[95%, 97.5%, 100%]` trainable-parameter average with terminal live BN buffers
  on the EXP-027 baseline. It does not prove all checkpoint averaging invalid,
  but no adjacent window, checkpoint count, coefficient, or BN-policy rescue is
  allowed without new independent evidence in a later loop.
- A semantic or feasibility failure closes only this exact implementation and
  is recorded as a crash without a score; do not simplify it after seeing the
  failed gate.
- **Correlated checkpoints:** three nearby low-LR states may be too similar to
  remove meaningful variance. The fixed spacing makes that a clean empirical
  result rather than a tunable degree of freedom.
- **Nonlinear parameter geometry:** an arithmetic midpoint can lie in a worse
  region even when every endpoint is good. A worse averaged endpoint with
  unchanged live evaluations directly exposes this risk.
- **BatchNorm mismatch:** terminal buffers are not exactly calibrated to the
  averaged weights. The short clean-tail window limits but cannot eliminate the
  mismatch; post-hoc recalibration is forbidden.
- **Selection ambiguity:** earlier live evaluations still contribute to the
  mandated `best_test_acc`. The separate terminal criterion prevents an
  unrelated live peak from being misattributed to averaging.
- **Snapshot overhead:** device clones are rare but synchronization makes their
  cost real. Counting them inside `dt` and enforcing the exposure gate prevents
  hidden compute from being treated as free.

## Evidence

- `experiments/027/03-execute.md` and `experiments/027/04-analysis.md`: accepted
  deeper-plus-early-RandAugment run, 94.32% best, 94.22% final, 0.2523 loss,
  133.00736 passes, and a valid 345.3-second wall time.
- `experiments/013/03-execute.md` and `experiments/013/04-analysis.md`: the
  distinct 65%-start, decay-0.999 whole-state EMA retained exposure but lagged
  terminal top-1 and left shorter parameter-only averaging unexplored.
- `knowledge/papers/weight-averaging.md`: local offline distillation supporting
  carefully windowed checkpoint averaging while excluding under-trained
  iterates.
- `02-system-understanding.md`: generalization and accuracy per backward pass,
  rather than I/O, memory, or wall time, are the current limiting factors.
