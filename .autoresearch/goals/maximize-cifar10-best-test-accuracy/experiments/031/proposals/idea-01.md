# Proposal: Reset SGD Momentum Once at the 80% Objective Boundary

## Decision and falsifiable hypothesis

Preserve the complete accepted training recipe through its existing 80% boundary. After the final LR-0.1 N1/M7+CutMix update and the existing switch evaluation/loader rebuild, but before the first LR-about-0.01 weak hard-label update, zero every live PyTorch SGD `momentum_buffer` exactly once. Parameters, BN state, optimizer groups/scalars, coupled decay, LR curve, data, evaluator, timer, precision, seed, and all subsequent ordinary SGD semantics remain unchanged.

**Hypothesis:** the momentum vector accumulated under high-LR distorted views and mixed targets contains stale objective-specific direction. Deleting it when the view, target, and LR regimes all change will reduce inherited weak-tail motion without raising the accepted 0.01 tail LR, improve refinement generalization, and lift seed-42 `best_test_acc` from 94.15% to at least **94.25%**. Point prediction: **94.27%**, final NLL at or below EXP010's 0.1934, and approximately 26.9k updates. A valid lower result falsifies this one-time reset; it does not authorize a partial, delayed, or repeated reset.

## Mechanism and distinction from failed optimizer paths

For ordinary momentum SGD with coupled decay, let `d_t = g_t + lambda*w_t`, `mu=0.9`, and let `b_s` be the buffer after the last strong update. Accepted training begins the first weak update with:

```text
b_1(control) = mu*b_s + d_1
w_1(control) = w_0 - lr_1*b_1(control)
```

The candidate first zeroes the existing buffer and therefore performs:

```text
b_0(reset) = 0
b_1(reset) = d_1
w_1(reset) = w_0 - lr_1*d_1
```

All later candidate updates are unmodified installed PyTorch SGD. The omitted inherited contribution is `mu^k*b_s`; at momentum 0.9 it halves in about 6.6 weak steps, falls below 10% after about 22, and below 1% after about 44. Its cumulative early-tail displacement is still material: with nearly constant LR 0.01, avoiding the decaying inherited term is approximately `0.01 * 9 * b_s = 0.09*b_s`, comparable to one final strong update of `0.1*b_s`. The intervention is thus a bounded one-time path change whose parameter-location consequence can persist after the velocity difference decays.

This is not a retry of the recurring global interventions:

- EXP020 Nesterov amplified the newest direction on every LR-0.1 step and crossed the class-concentration veto by step 11.
- EXP022 Lookahead repeatedly moved parameters while retaining velocity from the old fast location, producing location/velocity mismatch and concentration after the first pullback. EXP031 never relocates parameters at reset and deletes, rather than preserves, inherited velocity.
- EXP028 replaced every update with alternating PNM state; changing-gradient spikes reached 12.35x and collapse began at step 3. EXP031 uses ordinary SGD for the full strong trajectory and every weak step after a single zeroing event.

The global failures still matter: they show finite state and plausible scalar algebra do not ensure healthy class geometry, so EXP031 requires a copied-state weak-corpus comparison before production. But their high-LR repeated mechanisms do not directly reject a low-LR boundary-only reset. EXP030 is the closest motivation: doubling weak-tail LR achieved lower train loss but worse NLL/top-1. A reset seeks *less inherited transition motion* while retaining the accepted 0.01 quench, rather than adding tail amplitude.

## Exact implementation and ordering

Add one helper to `train.py`:

```python
@torch.no_grad()
def reset_sgd_momentum(optimizer):
    reset_count = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if state is None or "momentum_buffer" not in state:
                raise RuntimeError("missing SGD momentum buffer at transition")
            state["momentum_buffer"].zero_()
            reset_count += 1
    return reset_count
```

Call it in the existing transition block only after the strong model has been evaluated and the weak loader has been constructed successfully, immediately before flipping `randaugment_enabled` and logging the switch:

```python
if randaugment_enabled and progress >= LR_HOLD_FRACTION:
    worker_pids = shutdown_train_loader(train_loader)
    del train_loader
    gc.collect()
    train_loader = make_train_loader(weak_train_tf)
    momentum_buffers_reset = reset_sgd_momentum(optimizer)
    randaugment_enabled = False
    print(
        f"augmentation_switch: randaugment+cutmix->base | epoch: {epoch} | "
        f"progress: {100 * progress:.1f}% | workers_stopped: {len(worker_pids)} | "
        f"cutmix_batches: {cutmix_batch_count}/{strong_batch_count} | "
        f"momentum_buffers_reset: {momentum_buffers_reset}"
    )
```

This establishes the only acceptable order:

1. Complete and synchronize the final timed strong SGD step at LR 0.1.
2. Run the existing switch evaluation on unchanged parameters/BN state and old optimizer state.
3. Stop all strong workers and construct the weak loader through the accepted lifecycle.
4. Zero all momentum buffers once, without changing parameters, gradients, optimizer groups, or RNG.
5. Begin the next epoch; its first batch is weak/hard and its LR is approximately 0.01.

The accepted model has **59 parameter tensors**, all active in every ordinary training step, so exactly 59 buffers must exist and be reset. Zero the tensor in place; do not delete/recreate state, replace the optimizer, clear the whole `optimizer.state`, reset `step` metadata, touch gradients, or zero only Conv/classifier/BN subsets. Do not add a phase flag: the existing one-way `randaugment_enabled` transition guarantees one call. The returned count may extend the existing switch line, but no per-step diagnostics or evaluation may be added.

The 59 asynchronous zero operations are launched outside the per-batch timer, as are the existing loader rebuild and `gc.collect()`. Do not add a candidate-only CUDA synchronization to hide their execution; same-stream ordering guarantees completion before the first weak optimizer step, and any pending kernel time is conservatively absorbed by that step's existing synchronization. This one-time cost cannot meaningfully change 300-second exposure.

## Static, algebraic, and copied-state safety gates

Before production, use an ignored controller and serialize/fsync its report before any veto assertion.

First verify static and algebraic semantics:

- Exactly one helper and one call exist, nested in the existing 80% transition after the evaluation and weak-loader construction and before the first weak iterator.
- All 59 optimizer parameters have one FP32 momentum buffer with matching shape/device after a strong step; no extra optimizer key/group/scalar changes.
- On deterministic analytic tensors with nonzero buffers and coupled decay, the reset candidate's first update matches `w-lr*(g+lambda*w)` and subsequent updates match installed PyTorch momentum within `1e-7` absolute / `1e-6` relative tolerance.
- Immediately after reset, every buffer is exactly zero while every parameter, gradient, model buffer, optimizer group, CPU RNG state, and CUDA RNG state is bitwise unchanged; a reset-state save/load roundtrip preserves this state.
- A second call is treated as a controller failure, proving production call count is exactly one rather than demonstrating idempotence as permission for repeats.

Then perform a copied-state production-distribution comparison. Reuse the immutable EXP022 strong corpus, file SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`, and EXP028 weak corpus, file SHA-256 `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`. Recompute the registered file/tensor digests, validate 200 strong and 64 weak records, and never regenerate, reorder, filter, or mutate them.

Train **one** accepted source model/optimizer through all 200 strong records at LR 0.1. Capture its model, BN, optimizer, RNG, and corpus hashes once, then instantiate control and candidate arms from independent exact copies of that same boundary state. This avoids the separate-model CUDA drift that complicated EXP020 and attributes all post-boundary divergence to the reset. Leave control buffers unchanged; zero all 59 candidate buffers; confirm pre-reset logits are bitwise identical and remain identical immediately after reset because no model tensor changed.

Replay all 64 exact weak hard-label records at LR 0.01 in both arms, recording each loss, prediction histogram, update norm, parameter norm, gradient norm, momentum norm, BN counter/state, and finite-state check. Required gates:

- exact copied parameters/BN/optimizer groups/RNG before reset, and exact candidate/control logits/loss before either first weak update;
- candidate post-reset buffer count 59 and aggregate/max buffer norms exactly zero, with control buffers equal to the captured source;
- first and multi-step candidate recurrences match a manual ordinary-SGD oracle; coupled all-parameter decay remains `1e-4` and no gradient is mutated;
- all logits, losses, gradients, parameters, BN state, and optimizer state finite, with exact BN counters and hard targets throughout;
- no weak step where candidate maximum predicted-class share exceeds 95% while control is at or below 95%; lower loss cannot waive this gate;
- candidate/control terminal debiased loss-EMA ratio at most 1.5, no candidate whole-model update above 25% of pre-update parameter norm, no candidate/control update ratio above 5, and no candidate update above 5 times its own preceding 16-step median;
- momentum-difference norm decays consistently with the 0.9 recurrence absent gradient-driven trajectory divergence, and both corpus/source-state hashes remain unchanged.

The 200-step copied boundary is a safety/semantics proxy, not evidence that the mature 80% production state will improve. A safety failure retires the exact full-buffer reset. It does not authorize clipping, a partial reset coefficient, deleting buffers, resetting selected layers, changing LR, delaying the reset, or relaxing a gate.

## Expected production trajectory and diagnostics

The switch evaluation occurs before reset, so EXP031's pre-switch accuracy and NLL cannot be caused by the candidate mechanism. Historical differences from EXP010's 89.73% switch fit are trajectory context, not scope failure. The intervention first affects the next weak optimizer update. Expected signatures are:

- candidate first weak update excludes `0.9*b_s`; aggregate displacement may be smaller or larger than a hypothetical control because `b_s` can align with or cancel the new weak gradient, so direction/cosine diagnostics are more informative than size alone;
- buffer norm restarts from zero, rises under weak gradients, and loses meaningful dependence on the old strong buffer within roughly 44 steps;
- first weak evaluation should match or exceed EXP010's 93.16% without EXP030's doubled-LR overfit signature;
- a favorable run retains approximately 26,898 steps, final NLL at or below 0.1934, little best-final regression, and reaches at least 94.25%;
- a slower first weak recovery indicates inherited velocity was useful or reset caused a short optimization stall; worse final NLL/top-1 indicates that strong-phase momentum carried beneficial invariant direction.

Production logging should add only `momentum_buffers_reset: 59` to the switch line. Preserve preflight-only pre/post buffer norms, first-update cosine against the inherited buffer, candidate/control update ratios, and 64-step decay evidence in the ignored report. From `run.log`, report early checkpoints, switch and first-weak accuracy, best/final accuracy and epoch, final NLL, best-final gap, train-loss EMA, strong/CutMix counts, evaluation count, epochs/steps, VRAM, and counted/total time. These explain the mechanism but do not override the metric verdict.

## Production verification

Only after safety passes, reconfirm the 94.15 baseline at `7c1e7d8`; only `train.py` is tracked-modified; no stale run log exists; syntax, Ruff/format/diff/scope checks pass; exactly one idle 97,871-MiB H20 is visible; and model/data/schedule/evaluator/seed/precision contracts are unchanged. Run seed 42 exactly once:

```bash
timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
```

Require exit zero; one finite ten-field summary; approximately 300.0-301.0 counted seconds; total below 600 seconds; 1,073,962 parameters; one transition near 80% with eight workers stopped and exactly 59 buffers reset; 45-55% CutMix among eligible strong batches; no soft target after the switch; first weak LR approximately 0.01; no duplicate evaluation epoch and no more than one evaluation per epoch; and no retry. At least 26,629 steps is an exposure hypothesis/consistency expectation, not a post-hoc reason to discard an otherwise protocol-valid fixed-budget result.

## Risks, verdict, and no-rescue boundary

- **Impact risk — high:** the inherited contribution decays within dozens of roughly 5,400 weak steps; its lasting effect may be too small to clear a 0.10-point gate.
- **Optimization risk — medium:** removing momentum can stall early weak adaptation, and reset updates can exceed control updates when inherited velocity had canceled the new gradient.
- **Generalization risk — medium-high:** accumulated strong/CutMix velocity may encode useful invariant descent direction rather than stale bias; deleting it could mirror the harmful loss of strong-phase information seen in other boundary changes.
- **Evidence-transfer risk — medium:** restart-momentum literature concerns scheduled accelerated methods, not a literal one-time PyTorch SGD buffer zero at this curriculum boundary.
- **Implementation risk — low:** the operation is small, but placement, full 59-buffer coverage, state preservation, and exactly-once execution are load-bearing.
- **Runtime risk — negligible:** one set of zero operations adds no recurring per-step work or meaningful memory.

Formal verdict:

- **Improvement:** every protocol condition passes and `best_test_acc >=94.25%`. A 94.25-94.35 result is formally positive but remains weak single-seed evidence.
- **No improvement:** the run is valid but `best_test_acc <94.25%`, even if NLL, first-weak accuracy, or exposure improves.
- **Invalid/crash:** copied-state safety veto, wrong reset count/order/cadence, state/RNG/model mutation, malformed transition/targets, hardware/scope/evaluator/timer violation, nonfinite/incomplete summary, nonzero exit, or runtime at least 600 seconds.

Do not rerun a valid completion. Do not rescue with partial buffer scaling, buffer deletion, per-layer selection, repeated resets, a later/earlier boundary, warmup, clipping, LR changes, extra evaluations, another seed, or relaxed gates. Each is a new intervention requiring a new experiment ID.
