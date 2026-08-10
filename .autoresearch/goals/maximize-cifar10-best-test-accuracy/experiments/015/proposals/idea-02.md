# Proposal 02: H20 BF16 Autocast Exposure

## Proposal

Run only the accepted EXP-010 training forward and cross-entropy under CUDA BF16 autocast:

```python
optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
loss.backward()
optimizer.step()
```

Keep the model, parameter storage, gradients, BatchNorm persistent state, SGD momentum, coupled weight decay, and fixed evaluator in FP32. Do not use `GradScaler`. The intervention targets the measured convolution/normalization forward-backward cost while preserving batch-128 optimization noise and every accepted statistical choice.

The causal hypothesis is exposure, not capacity or a new regularizer: BF16 Tensor Core kernels reduce counted step time enough to execute at least 15% more updates and images under the same elapsed-time phases, and that additional accepted training raises `best_test_acc` from 94.15% to at least 94.25%. BF16 rounding is part of the net candidate and may help or hurt, but a speed claim is valid only if CUDA-event model forward/backward time falls materially.

## Accepted Baseline and Local Bottleneck

The experiment starts from accepted commit `7c1e7d8` and changes no architecture or recipe component:

- width-2 postactivation ResNet-20, 1,073,962 parameters;
- batch 128, standard momentum 0.9, LR 0.1, and coupled all-parameter decay `1e-4`;
- N1/M7 RandAugment and p=0.5 alpha-1 CutMix through 80% of counted time;
- crop/flip-only, hard-label weak tail with the accepted step-to-0.01 then cosine-to-`1e-4` LR;
- seed 42, persistent worker lifecycle, evaluator, checkpoint policy, and 300-second timer.

EXP-010 reached 94.15% in 26,898 updates / 3,442,944 images (68.9 dataset passes), with 89.73% at the strong-phase switch, 93.16% at the first weak checkpoint, final equal to best, 0.1934 final NLL, 598.7 MB peak allocation, and 330.7 seconds total wall time.

EXP-013 decomposed a warm accepted step on the H20: forward 2.408 ms (22.11%), backward 8.220 ms (75.46%), transfer 0.067 ms, cross-entropy 0.016 ms, and gradient reset plus update 0.182 ms. Model forward plus backward is 10.628 ms, 97.57% of the 10.893 ms CUDA-stage total. The synchronized wall step exceeds those stages by only 0.034 ms. Therefore BF16 is attractive only if it accelerates model GPU kernels, especially the 8.220 ms backward; autocast context, Python, memory-capacity, or launch-overhead arguments are insufficient.

EXP-013 also showed that stable serial timing can overstate a candidate and established five alternating fresh-process pairs as the local standard. EXP-014's chance collapse came from a new unbounded max-readout branch and does not motivate a rescue or combined change here. This proposal retains the accepted model exactly.

## Precision Semantics

PyTorch 2.9 documents that autocast should wrap forward and loss, while backward should execute after leaving the context; backward ops use the dtype selected for their corresponding forward ops. It lists CUDA `conv2d` and `linear` as lower-precision-eligible and `cross_entropy` as FP32-policy. It also warns not to cast the model or inputs manually when using autocast. Source: [PyTorch 2.9 AMP and CUDA op reference](https://docs.pytorch.org/docs/2.9/amp.html).

The exact dtype contract is:

- Construct the model normally and move it with `.to(device)`, never `.bfloat16()`. All leaf parameters remain `torch.float32` master values.
- Inputs arrive on CUDA as FP32 exactly as accepted. Within autocast, eligible convolutions and the final linear receive autocast-selected BF16 operands and are expected to return BF16 tensors.
- BatchNorm is not on PyTorch's explicit CUDA force-BF16 or force-FP32 lists. It runs in the dtype implied by its inputs; in this downstream BF16 graph its visible outputs are expected to be BF16, while affine parameters, `running_mean`, and `running_var` remain FP32 and `num_batches_tracked` remains integer. Do not claim or force a particular hidden reduction-accumulation dtype; record observed module input/output and persistent-state dtypes locally.
- ReLU, pooling, slicing, padding, and residual addition inherit/promote from their inputs under ordinary operator rules. Record representative stage outputs rather than adding manual casts.
- Both target paths remain unchanged: hard targets are int64; CutMix probability targets remain FP32. CUDA autocast policy runs `F.cross_entropy` in FP32 for both paths, so the scalar loss must be FP32 and finite.
- Exit autocast before `loss.backward()`. The backward kernels follow forward-selected dtypes, but gradients accumulated into FP32 leaf parameters must be FP32.
- Ordinary unfused SGD remains outside autocast. Momentum buffers must be FP32; coupled decay is computed against FP32 parameters; parameter updates are FP32. Keep the existing `optimizer.zero_grad()` semantics.
- Every call to `evaluator.evaluate(model, device)` occurs outside autocast. The untouched evaluator sees the FP32 model and performs the same FP32 inference/loss path used to define the 94.15% baseline.

Do not use `GradScaler`. Gradient scaling primarily protects FP16 values from underflow; BF16 has FP32-like exponent range, and this model accumulates leaf gradients into FP32 parameters. A scaler would add mutable scale state, non-finite checks, possible skipped optimizer steps, and overhead, changing both the mechanism and update count. If unscaled BF16 produces a non-finite loss or gradient, this candidate fails; do not add a scaler or retry in FP16.

## Exact `train.py` Change

1. After device selection, fail unless the run is CUDA and `torch.cuda.is_bf16_supported()` is true.
2. Add the autocast context exactly around `outputs = model(inputs)` and `loss = F.cross_entropy(outputs, targets)` in the timed training step.
3. Leave `loss.backward()`, `optimizer.step()`, synchronization, timing, logging, and loss extraction where they are.
4. Optionally print one startup provenance line, `training_autocast: cuda/bfloat16 | grad_scaler: false`; add no per-step dtype probes to production.

Do not change batch size, width, architecture, initialization, memory format, TF32 flags, optimizer flags, zero-grad mode, loss, CutMix, augmentation, workers, seed, LR values or phase boundaries, evaluation schedule, or summary schema. Do not add channels-last, compilation, fused SGD, width changes, gradient scaling, or any fallback. Only `train.py` may differ; no dependency changes are permitted.

## Numerical Feasibility Gate

Run diagnostics from disposable scripts without modifying tracked files. Confirm one idle H20, compute capability 9.0, approximately 97,871 MiB, installed PyTorch/CUDA versions, and `torch.cuda.is_bf16_supported() == True`.

Use cloned, state-identical FP32 and BF16 candidate models and optimizers. For each of 20 fixed-seed batch-128 inputs, alternate hard int64 targets and valid FP32 CutMix probability targets. Reset both arms to the same model and optimizer state before each comparison so BatchNorm mutation does not compound across unequal paths. Require:

- all model parameters, parameter gradients, floating BatchNorm buffers, and SGD momentum buffers are FP32; BN counters remain integer;
- every convolution and linear output observed by temporary hooks is BF16 under autocast; representative BN outputs and final logits are BF16; loss is FP32 for both target formats;
- both losses are finite and BF16-versus-FP32 relative loss error is at most 2% on every batch;
- FP32-cast logit cosine similarity is at least 0.995 on every batch, with mean absolute and maximum absolute logit error reported;
- flattened parameter-gradient cosine similarity is at least 0.99 and gradient-norm ratio lies in `[0.90, 1.10]` on every batch;
- BF16 does not increase the fraction of exactly-zero parameter-gradient elements by more than 1 percentage point relative to FP32;
- after one ordinary SGD step, all parameters and momentum buffers are finite/FP32, update-vector cosine similarity is at least 0.99, and BF16/FP32 update-norm ratio lies in `[0.90, 1.10]`;
- BN `num_batches_tracked` values match exactly, and normalized L2 differences for running mean/variance are reported and remain at most 2% after the paired step.

Then run 500 BF16 training steps on a deterministic alternating hard/soft batch stream with the accepted optimizer. Require finite loss, logits, gradients, parameters, BN buffers, and momentum throughout; no skipped update semantics exist. Compare periodic losses against a paired FP32 control and reject any BF16 loss above twice the same-step FP32 loss or any gross prediction collapse. These checks reject unsafe numerics but do not claim trajectory equivalence or guarantee generalization.

## Fresh-Process H20 Timing Gate

Use five alternating fresh-process control/candidate pairs, following the EXP-013 correction. Each trial constructs a new state-aligned model and SGD optimizer, uses batch 128, performs 100 unmeasured warm steps for cuDNN/autocast caches, then measures at least 500 synchronized steps. Reverse order across trials (`FP32,BF16`; `BF16,FP32`; and so on). The H20 must be uncontended for every pair.

Measure the exact accepted timed interval, including pinned H2D copy, zero-grad, forward, cross-entropy, backward, optimizer step, and final synchronize. Also use CUDA events around transfer, forward, loss, backward, and optimizer separately. Alternate valid hard and CutMix-probability targets during every arm, matching the two production loss paths. Report each trial's mean, median, p95, coefficient of variation, images/s, and peak memory, plus paired ratios.

Advance to a full run only if every condition passes:

- median paired synchronized full-step speedup is at least `1.15x` (BF16/FP32 time at most `0.86957`), and BF16 is at least 1.12x faster in all five pairs;
- BF16 median CUDA-event `(forward + backward)` time is at most `0.85x` FP32, and backward alone is at most `0.90x` FP32;
- at least 90% of absolute CUDA-stage time saved comes from forward plus backward, preventing an autocast/host bookkeeping artifact from supporting the claim;
- BF16 p95 full-step time is below the FP32 median full-step time;
- trial-mean CV is below 3% in each arm and paired speedup CV is below 2%; no thermal/order trend reverses the result;
- ratio-projected exposure is at least `floor(26,898 * 1.15) = 30,932` updates and 3,959,296 images, about 79.2 dataset passes;
- peak allocation stays below 1 GiB and has no monotonic per-step growth.

The full-step and GPU model-stage gates are conjunctive. A wall-time gain without forward/backward CUDA-event savings fails. If the gate misses, do not run the accuracy experiment and do not add channels-last, compilation, larger batches, fused optimizer, FP16/scaling, or width changes as a rescue.

## Loader and Wall-Time Gate

Because DataLoader wait was only 0.145 ms at accepted speed, BF16 should retain headroom, but the faster consumer must be tested with real N1/M7 plus CutMix production batches. In a fresh process, warm persistent workers and measure at least 1,000 real strong batches. Require median iterator wait below 10% of projected BF16 full-step time, p95 below 20%, and delivered batch rate at least 1.20x the projected BF16 step rate. Exercise the exact strong-to-weak loader shutdown/rebuild and require all eight prior workers stopped and a successful hard-label weak batch.

Project total time from cold startup, 300 counted training seconds, the observed switch, and the increased number of once-per-epoch weak-tail evaluations. Require a conservative projection below 540 seconds, leaving 60 seconds before the absolute 600-second kill threshold. Autocast warmups belong only to disposable timing scripts; do not move a production warmup outside the counted timer.

## Fixed-Time and Exposure Hypothesis

The accepted timer already includes transfers, autocast entry/exit, all mixed/FP32 kernels, backward, SGD, and synchronization. Therefore additional updates are legitimate accelerator work completed inside the same 300 seconds. Evaluation and loader wait remain outside the counted timer but inside the total wall limit exactly as before.

At the minimum 1.15x gate, projected exposure rises from 26,898 to 30,932 updates and from 3.44M to 3.96M presented images. The elapsed-time 80/20 split means roughly 24,746 high-LR strong steps and 6,186 weak annealing steps, versus about 21,518 and 5,380 at accepted exposure. More steps also mean more applications of LR, momentum, and coupled decay per phase. That is intentional but creates a statistical risk: the schedule is time-aligned, not update-aligned, and extra exposure is not known to improve the already-flat terminal trajectory.

**Primary hypothesis:** BF16 passes the numerical and stage-timing gates, completes at least 30,932 updates in 300 counted seconds, preserves a healthy strong checkpoint, and reaches `best_test_acc >= 94.25%` under unchanged FP32 evaluation.

Diagnostic expectations, fixed before the run:

- switch accuracy should remain above the recurring underfit marker of 87.08%; a lower value indicates mixed-precision/extra-update damage but cannot trigger tuning;
- realized CutMix should remain near the accepted 50% probability, weak targets must stay one-dimensional, and the switch should remain at approximately 80% elapsed time;
- actual full-run speedup should broadly agree with the fresh-pair projection; less than 30,932 steps means the throughput mechanism failed even if accuracy fluctuates upward.

## Accuracy and Systems Risks

- **FP32 may already be efficient.** cuDNN can use TF32 or optimized convolution kernels, while small CIFAR maps leave BN, residual adds, and launches material. BF16 may miss the 15% gate.
- **BF16 precision changes the gradient field.** BF16 retains dynamic range but has fewer mantissa bits; repeated residual operations and convolution backward can round small corrections.
- **BatchNorm statistics see quantized activations.** Persistent buffers stay FP32, but their observed batches arise from the mixed path. A dtype assertion cannot guarantee identical running statistics.
- **Cross-entropy is protected, not the whole model.** CE is FP32-policy, but its gradients flow through BF16-selected logits/backward kernels. Hard and soft targets both need direct parity checks.
- **No scaler means hard failure on instability.** This keeps BF16 isolated and update semantics exact; any non-finite is a no-go, not permission to add scaling.
- **More updates may over-optimize.** The accepted run finished at its best, but its late trajectory was nearly flat. More high-LR steps and cumulative coupled decay can worsen generalization even with perfect numerical stability.
- **FP32 evaluation differs from mixed training.** This is required to preserve the metric harness, but training may adapt to BF16 rounding in ways that do not transfer favorably to FP32 inference.
- **Extra epochs add test passes and wall time.** Validation remains at most once per epoch; more tail epochs can add excluded-from-training but wall-counted evaluations. The 540-second projection gate controls this.
- **Seeded streams diverge by exposure.** Seed remains 42 and no reroll occurs, but faster training consumes more shuffled/augmented batches. The result is the net fixed-time method, not a batch-for-batch ablation after the shared prefix.
- **Autocast policy is version/backend dependent.** Record PyTorch, CUDA, cuDNN, capability, and observed dtypes. Do not infer hidden BN accumulation or kernel selection from documentation alone.

## Full-Run Verification

If and only if all preflight gates pass:

1. Confirm the moving baseline is 94.15% at `7c1e7d8`; formal improvement requires at least 94.25%.
2. Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM and BF16 support.
3. Verify `git diff` changes only `train.py` and exactly the capability/provenance plus training autocast scope. Run syntax, lint/pre-commit, target-format, lifecycle, and parameter/state-dtype checks.
4. Assert 1,073,962 parameters, all FP32, batch 128, unchanged optimizer groups/hyperparameters, and unchanged accepted transforms, CutMix, LR/time boundaries, evaluator, seed, and evaluation checkpoints.
5. Remove stale `run.log`; run once as `uv run train.py > run.log 2>&1` under a 600-second supervisor. No reroll or alternate precision run is allowed.
6. Require exit 0, 300.0-ish counted seconds, total below 600 seconds, a complete finite ten-field summary, unchanged parameter count, and no non-finite training output.
7. Require exactly one strong-to-weak switch near 80%, eight stopped workers, p=0.5 CutMix provenance during strong training, no probability target in the weak tail, and no more than one evaluation in any epoch.
8. Require at least 30,932 optimizer steps / 3,959,296 presented images for the systems mechanism. Record actual step speedup, epochs, strong/tail step split, peak VRAM, startup, and total wall time.
9. Compare switch accuracy, first weak checkpoint, tail slope, final/best gap, and final NLL to EXP-010. Require `best_test_acc >=94.25%` for improvement.
10. Remove `run.log` after analysis and restore the accepted branch on any no-go/no-improvement outcome.

## Decision Rules

- **Preflight no-go:** any capability, dtype/state, numerical, fresh-pair timing, GPU-stage attribution, loader, lifecycle, or projected-wall gate failure blocks the full run.
- **Accept:** all integrity/mechanism checks pass, actual steps are at least 30,932, and `best_test_acc >=94.25%`.
- **Throughput success, accuracy miss:** valid no-improvement. BF16/extra exposure did not improve this recipe; do not tune precision or combine another lever post hoc.
- **Accuracy pass, exposure miss:** report the metric result but do not attribute it to the proposed throughput mechanism; adversarial review must decide whether numerical BF16 alone is acceptable evidence.
- **Non-finite or dtype-state failure:** invalid/no-go. Revert to FP32; no scaler, FP16, retry, or fallback.
- **Scope, evaluation, timing, wall, or seed violation:** invalid regardless of accuracy.

## Recommendation

BF16 autocast is a clean, high-relevance systems candidate because it preserves batch-128 optimization and targets 97.57% of measured CUDA-stage time without changing capacity. It should advance only if fresh H20 pairs prove at least 15% synchronized acceleration rooted in forward/backward kernels and numerical checks show the FP32 persistent state receives aligned finite updates. Its ceiling is uncertain: more exposure is not itself a new generalization mechanism, and the accepted trajectory may already be saturated. The strict gate is therefore essential, and no combined width/layout/optimizer change belongs in this experiment.
