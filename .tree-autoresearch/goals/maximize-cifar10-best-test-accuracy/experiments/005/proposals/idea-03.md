# Proposal: Clean-Gated Last-Mini-Batch Self-Distillation

## Summary

Add Self-Distillation from the Last Mini-Batch (DLB) to the EXP-004 WRN/CutMix/period-two-SAM stack. Rearrange each epoch into 256-example batches whose first 128 raw image identities repeat the prior batch's second 128 identities, but let the dataset transform each occurrence independently so the repeated image receives a fresh crop/flip. Cache detached primary-forward logits for the outgoing second half and, on the next aligned clean batch, add the published last-batch KL consistency loss on the repeated first half.

Use the fixed paper settings:

- `DLB_TAU = 3.0`
- `DLB_ALPHA = 1.0`
- `DLB_HALF = BATCH_SIZE // 2 = 128`
- `DLB_SEED = 42`

DLB is enabled only when the current batch and the teacher-producing prior batch are both clean. A CutMix batch receives the unchanged CutMix supervised loss, contributes no DLB KL, and invalidates the outgoing teacher cache. This conservative gate preserves one-to-one sample identity rather than distilling from spatially mixed logits. No hyperparameter is adjusted from test accuracy.

## Evidence and Expected Effect

The CVPR 2022 DLB method uses half-overlapping consecutive batches and detached logits from the immediately prior parameter state, adding one KL term without another model forward. Its CIFAR-10 results improve error by 0.37-1.01 points across several backbones. It is explicitly compatible with CutMix; reported CutMix+DLB gains over CutMix alone range from 0.09 to 1.48 points, including a 0.60-point error reduction on WRN-20-8. Those are three-run averages and do not establish the effect on this stronger WRN-16-4/CutMix/SAM stack.

EXP-004 is the 95.40% parent and requires at least 95.50%. DLB is orthogonal to its validated late SAM tail: it targets prediction consistency across fresh views and one optimizer update, whereas SAM targets local parameter sharpness. The proposal predicts `best_test_acc >= 95.50%`, with a secondary expectation of final test loss no worse than EXP-004's 0.1654.

Sources:

- `experiments/005/papers/dlb.md`
- `experiments/004/04-analysis.md`
- `experiments/005/00-navigate.md`
- CVPR paper: https://openaccess.thecvf.com/content/CVPR2022/html/Shen_Self-Distillation_From_the_Last_Mini-Batch_for_Consistency_Regularization_CVPR_2022_paper.html

## Overlapping Batch Stream and Fresh Augmentation

Implement a deterministic batch sampler in `train.py`, plus a thin dataset wrapper that returns `(image, target, index)` for identity audits. For each epoch:

1. Draw one seed-controlled permutation of the 50,000 training indices using a dedicated CPU generator; do not consume the global generator, CutMix generators, or CUDA stochastic-depth stream.
2. Split the first 49,920 indices into 390 full chunks of 128 and drop the final 80, matching the parent's 49,920-example `drop_last=True` coverage.
3. Emit 389 batches: `[chunk_0, chunk_1]`, `[chunk_1, chunk_2]`, ..., `[chunk_388, chunk_389]`.
4. Reset the DLB cache at the epoch boundary. The first emitted batch has no historical teacher; an interior chunk is fetched twice on consecutive iterations.

The sampler repeats indices, not transformed tensors. `CIFAR10.__getitem__` therefore reruns random crop, horizontal flip, tensor conversion, and normalization for each occurrence through the existing workers. This creates the intended consistency target across fresh augmentations. Passing dedicated, fixed generators for sampler ordering and DataLoader worker seeding makes the stream reproducible while leaving global model/drop-path and existing CutMix RNG streams isolated.

This stream has 389 optimizer steps per full data permutation rather than the parent's 195, because interior images appear twice. Under the wall-clock budget it does not increase batch size or forward count per step; it halves the rate of new unique-image introductions and roughly halves the number of completed epochs/evaluations. That is an inherent risk of the paper's overlap mechanism and must be reported, not hidden by redefining epoch boundaries.

## Exact Cache and Loss Semantics

Maintain `cached_logits` and `cached_indices`, initially `None` at every epoch start.

For the primary forward at step `t`, compute the existing supervised loss over all 256 examples. If the batch is clean, `cached_logits` exists, and `indices[:128]` exactly equals `cached_indices`, compute in FP32:

```python
teacher_prob = softmax(cached_logits / DLB_TAU, dim=1)
student_log_prob = log_softmax(logits[:128].float() / DLB_TAU, dim=1)
dlb_loss = DLB_TAU**2 * kl_div(
    student_log_prob, teacher_prob, reduction="batchmean"
)
loss = supervised_loss + DLB_ALPHA * dlb_loss
```

`cached_logits` is a detached FP32 clone of the prior primary forward's outgoing `logits[128:]`; it is never a probability tensor and never receives gradients. The KL direction is teacher-to-student, matching `KL(teacher || student)`. Cross-entropy retains its current mean over all 256 examples, while KL is the mean over exactly 128 repeated examples; do not divide the KL by 256 again.

After a successful optimizer step on a clean batch, publish `primary_logits[128:].detach().float().clone()` and `indices[128:].clone()` as the next teacher. Assigning the cache after `optimizer.step()` does not change its origin: the saved logits were produced immediately before that update. If the current batch used CutMix, set both cache fields to `None` after the step. Consequently, one mixed step suppresses DLB on itself and prevents its mixed outgoing logits from supervising the next step. The next clean step rebuilds the cache for the following batch.

With CutMix probability 0.5 during the first 75%, early DLB is expected mainly on consecutive clean-clean transitions, approximately 25% of early steps. After CutMix ends, DLB becomes active on nearly every aligned step except epoch bootstrap/recovery steps.

## Reconciliation with Period-Two SAM

Preserve the EXP-004 schedule: SAM starts at charged progress 0.75 and applies on every even upcoming one-based step. Since CutMix ends at the same boundary, all SAM batches are clean and can use DLB.

On a SAM step:

1. The unperturbed primary forward computes `CE + DLB` and produces the only candidate outgoing teacher logits.
2. The first backward of that complete objective defines the SAM perturbation.
3. The perturbed second forward recomputes the same `CE + DLB` against the same detached incoming teacher cache. Preserve EXP-004's CUDA RNG replay and second-pass BatchNorm-stat suppression.
4. The perturbed second-forward logits must never update the DLB cache.
5. Restore parameters and BatchNorm flags, perform exactly one Nesterov update, then publish the saved unperturbed primary outgoing logits.

Thus SAM optimizes the combined objective consistently on both passes, while DLB retains the paper's teacher semantics. The incoming cache is read-only throughout both passes. Any exception or nonfinite loss leaves the cache unpublished and aborts the run.

## Compute and Memory Cost

DLB adds FP32 softmax/log-softmax/KL over only `128 x 10` logits on active steps and caches about 5 KiB of logits plus 1 KiB of indices. It adds no model forward and no new dependency. The overlapping sampler still presents 256 transformed images per model step, so GPU model work and host transform count per step remain essentially unchanged.

The expected charged overhead is under 2%; relative to EXP-004's 25,560 steps, expect approximately 25,000-25,560 steps and require at least 24,500. SAM second passes remain the dominant overhead. All sampler handling, cache checks, KL work, both SAM passes, and cache publication bookkeeping must remain within the existing `t0`-to-CUDA-synchronize charged interval. Validation remains excluded and occurs at most once per longer DLB epoch.

## Implementation Scope

Modify only `train.py`:

- Add fixed DLB constants, deterministic overlapping batch sampler, and index-returning dataset wrapper.
- Replace `shuffle=True` batching with the overlapping batch sampler while preserving batch size 256, workers, pinning, transforms, and dropped remainder semantics.
- Add cache/KL logic to the existing primary loss and SAM second loss.
- Add counters for `dlb_active_batches`, `dlb_active_examples`, `dlb_clean_batches`, `dlb_cache_invalidations`, `dlb_cache_resets`, and `dlb_overlap_mismatches`; print them before the unchanged final summary.

Do not alter architecture, optimizer hyperparameters, CutMix probability/geometry/generators, SAM rho/start/period/restoration, LR/drop-path schedules, evaluator, metric accumulation, required summary keys, global seed, or time budget.

## Verification

Before the one full run:

1. Compile/lint/format and confirm the tracked diff contains only `train.py`; structurally audit the evaluator, summary, timing boundary, CutMix constants, and SAM constants.
2. On a toy indexed dataset, require every emitted batch after bootstrap to satisfy `batch_t[:128] == batch_(t-1)[128:]`, every epoch to contain 390 unique chunks/49,920 unique indices, and two calls for an overlapping index to produce independently transformed views.
3. Run the sampler twice from seed 42 and require identical index batches; require unchanged global CPU/CUDA RNG states after sampler-only tests.
4. Verify the KL numerically against a manual teacher-to-student computation, is zero for identical logits, affects only the student first-half gradient, and never backpropagates into cached logits.
5. Exercise clean-clean, clean-mixed, mixed-clean, and epoch-reset transitions. Require DLB active only for clean-clean aligned transitions, cache invalidation on mixed batches, no stale cache reuse, and zero overlap mismatches.
6. Instrument one ordinary DLB step and one SAM+DLB step. Require one/two forwards respectively, one optimizer update, primary-only cache publication, identical incoming teacher use on both SAM passes, exact SAM parameter/RNG/BatchNorm restoration, and finite BF16 gradients on GPU 0.

Run exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Require the H20 on physical GPU 0, exit 0, 299.5-301.0 charged seconds, total time below 600 seconds, at least 24,500 steps, unchanged 2,748,890 parameters, complete summary, one evaluation per logged epoch, zero overlap mismatches, nonzero DLB activity, expected SAM period-two counters, and no NaN/Inf, traceback, CUDA error, or timeout. Success requires `best_test_acc >= 95.50%` versus parent 95.40%.

## Failure Modes

- **Unique-sample throughput is halved.** Repetition may reduce data diversity enough to outweigh consistency gains despite unchanged step throughput.
- **CutMix gates too much early DLB.** A mixed step invalidates the next teacher, so early active coverage is far below the paper's always-on protocol; this is the cost of preserving identity correctness.
- **Fresh views may be too different.** Crop/flip plus one optimizer update can make cached targets noisy. Fixed `tau=3`, detached soft targets, and the clean-only gate bound this risk.
- **DLB can over-regularize the SAM tail.** Both methods favor consistency/flatness and may impede low-LR fitting. The experiment keeps published DLB weight/temperature fixed and must not tune them after observing test accuracy.
- **Sample/logit misalignment silently corrupts KL.** Runtime index equality, mismatch counters, transition tests, and epoch cache resets are mandatory.
- **Wrong SAM cache source changes the method.** Caching perturbed second-pass logits or publishing before a failed update is prohibited; only successful primary-forward outgoing logits become teachers.
- **Fewer epoch evaluations reduce best-checkpoint opportunities.** This can lower `best_test_acc`; final accuracy/loss and evaluation count must be reported alongside the primary metric.

## Testable Hypothesis

Clean-gated DLB with `tau=3` and `alpha=1` will preserve at least 24,500 charged optimizer steps, retain exact EXP-004 CutMix/SAM semantics, and achieve `best_test_acc >= 95.50%`. The prediction is that fresh-view, one-update temporal consistency supplies a low-cost generalization signal complementary to CutMix and late SAM. A result below 95.50%, any overlap/cache mismatch, or a scope/timing failure falsifies the proposal without a seed rerun or post-hoc parameter change.
