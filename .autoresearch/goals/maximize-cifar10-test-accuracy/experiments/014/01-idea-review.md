**Prioritized Feedback**

1. **Idea-01 is technically viable, but the warmup plan needs fixes.** If compile + warmup is placed before `t_start_training` in `train.py:268`, its cost is excluded from `training_seconds`; the timed budget only accumulates per-batch `dt` inside `train.py:280-314`. But any missed graph compile inside the loop will be charged because `torch.cuda.synchronize()` is inside the timed step. Use exact train shape, dtype/autocast, train mode, channels-last, and backward warmup.

2. **Idea-01 has two nonfatal correctness traps.** Warmup forward passes mutate BatchNorm running stats on dummy data; save/restore BN buffers or disable BN stat updates during warmup. Also, early pre-EMA eval would call `Eval.evaluate(compiled_model)` at batch 256 plus last batch 16, likely compiling eval graphs. Eval is off training-budget but not off the 10-minute wall cap. Keep `raw_model` before `torch.compile` and use it for raw eval; use compiled model only for training.

3. **Idea-01 optimizer/EMA aliasing is sound.** In local Torch 2.9.1, `torch.compile` returns an `OptimizedModule` whose parameters are the same tensor objects, in the same order, under `_orig_mod.*`. Constructing optimizer/EMA before compile is okay; `optimizer.step()` updates the tensors used by compiled forward, and `AveragedModel.update_parameters(compiled_model)` zips the same params/buffers. Still smoke-test this invariant.

4. **Do not run a 3-cell harness as the official EXP-014 script.** Goal verification kills runs over 10 minutes, and current single runs already take ~447s wall. A back-to-back cell-0/A/B run will likely violate the wall constraint. Use smokes/diagnostic cells separately; the official run should be one selected configuration.

5. **Idea-01’s value is not compile-only.** Compile-only buys maybe 7-15% throughput, but this model is already near a ~96.4 ceiling, so extra epochs past ~150 may be worth <0.1pp. The real high-upside version is compile funding `layer2_width=320`, because it attacks the prior under-anneal failure mode while adding capacity.

6. **Idea-02 is simple but its “identity-init” argument is overstated.** ReZero makes only the internal residual branch identity. Widening `conv_bn(128,320)` and `conv_bn(320,512)` changes the main representation path, so this is not bit-equivalent to the baseline at init and “no LR retune needed” is weaker than claimed.

7. **Idea-02’s dominant risk is repeating EXP-007 softly.** The 320 width may land around 120-135 epochs, but if cuDNN shape efficiency or host load pushes it below ~120, it likely under-anneals. If `best==final`, treat it as the same failure signature as EXP-007. Best fix: pair it with throughput; standalone 320 is lower EV.

8. **Idea-03 has the shakiest implementation.** The proposed Ghost BN variance folding is not equivalent to full-batch running variance; averaging per-ghost variances misses between-ghost variance. Since frozen eval uses `model.eval()` running stats, this can miscalibrate eval. Fix by updating running buffers with pooled/full-batch stats while using ghost stats for training normalization.

9. **Idea-03 may not be throughput-free.** `channels_last` tensors will not safely support the proposed `.view`; `.reshape` may copy at every BN layer, and there are 10 BN layers. If throughput drops, the result is confounded by under-anneal. Also use real `nn.BatchNorm2d` for the `num_splits=1` control, not the custom module.

**Scored Verdict**

- **Idea-01: torch.compile throughput, preferably compile + 320 width**
  - Impact: **7/10**. Compile-only is probably sub-noise, but compile-funded 320 is the only finalist with a credible path to >0.1pp by changing the epoch/capacity tradeoff.
  - Confidence/Soundness: **6/10**. Mechanism is strong and param aliasing is sound, but the proposal needs BN-buffer restore, raw uncompiled eval, identical warmup controls, and wall-cap discipline.

- **Idea-02: standalone layer2 256→320**
  - Impact: **4/10**. Plausible thin win, but it spends epochs in a regime where prior capacity/compute additions usually lost.
  - Confidence/Soundness: **7/10**. Easy and constraint-clean, but expected value is capped by under-anneal risk and the post-EXP008 regularization ceiling.

- **Idea-03: Ghost BatchNorm**
  - Impact: **5/10**. Mechanistically distinct and a real missing DavidNet component, but the modal outcome is tie/over-regularization on an already strongly regularized net.
  - Confidence/Soundness: **4/10**. The implementation has serious BN running-stat and layout hazards; “throughput-free” is unproven.

**Pick: Idea-01**, specifically the **compile-funded 320-width cell**, not compile-only. It wins because it is the only finalist that attacks the diagnosed limiter directly: fixed 300s training time makes throughput the scarce resource, and throughput is what lets useful capacity anneal instead of repeating EXP-007. Run it only after the fixes above; if compile throughput is negligible in smoke, standalone 320 should not be promoted as the fallback.
