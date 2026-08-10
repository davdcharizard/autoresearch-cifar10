# EXP-004 Proposal: Noise-Scale-Matched Batch 256 for H20 Throughput

## Summary

Double the training batch from 128 to 256 and linearly scale the full EXP-002 learning-rate schedule by 2x: hold `lr=0.2` through 80% of counted training time, then step to `0.02` and cosine-decay to `2e-4`. Keep momentum 0.9, weight decay `1e-4`, the accepted 269,722-parameter ResNet-20, augmentation, loader, synchronized timing, evaluation cadence, and seed unchanged.

This is a coherent throughput-plus-optimizer-scaling intervention. A batch-only change at fixed `lr=0.1` would isolate a source-code constant, but it would not isolate H20 throughput: it would also halve the approximate SGD noise scale and reduce parameter movement per processed example. Exact linear LR scaling compensates for those predictable optimization changes. At batch 256 the scale increase is modest enough to avoid adding warm-up in the first run, preserving EXP-002's validated long high-LR exploration pattern.

## Local Diagnosis

The moving baseline is EXP-002 at 91.83% best accuracy. It completed 38,629 optimizer steps and 100 reported epochs in 300.0 counted seconds, with 336.0 seconds total, 269,722 parameters, and 330.1 MB peak VRAM. Its effective step time was approximately:

```text
300.0 s / 38,629 = 7.77 ms/step
```

At batch 128 those steps presented:

```text
38,629 * 128 = 4,944,512 examples
```

The current DataLoader has 390 full batches per epoch, so this is about 99.0 full-data equivalents plus a partial final epoch. The H20 has 97,871 MiB, making memory irrelevant at this operating point. The opportunity is to amortize fixed Python, kernel-launch, synchronization, and transfer costs over twice as many examples per optimizer step.

EXP-003 is the cautionary result: adding nominally lightweight label smoothing reduced fixed-budget steps by 6.7%, from 38,629 to 36,039, and failed to improve accuracy. In this benchmark, synchronized step cost changes the number of optimization opportunities enough to matter. A larger batch deliberately accepts fewer updates, so it must earn enough additional examples per second and retain comparable SGD dynamics.

## Proposed Configuration

Change only these constants in `train.py`:

```python
BATCH_SIZE = 256
LR = 0.2
ANNEAL_START_LR = 0.02
MIN_LR = 2e-4
```

Keep all of the following exactly as accepted in EXP-002:

- `NUM_BLOCKS = 3`, widths 16/32/64, and 269,722 parameters.
- SGD momentum 0.9, standard momentum rather than Nesterov.
- Weight decay `1e-4`.
- `LR_HOLD_FRACTION = 0.8` and the existing elapsed-time schedule logic.
- BatchNorm, initialization, hard-label cross-entropy, and all model code.
- Random crop, horizontal flip, normalization, and the CIFAR-10 train set.
- Eight persistent DataLoader workers, pinned memory, shuffle, and `drop_last=True`.
- Per-step `torch.cuda.synchronize()` and the existing counted-time accounting.
- Evaluation checkpoints `(0.2, 0.4, 0.6, 0.7)`, dense once-per-epoch evaluation after 80%, and terminal evaluation.
- Seed 42, evaluator, output format, and one-H20 execution protocol.

At batch 256, the 50,000-example training set yields 195 full batches and drops 80 examples per epoch. This presents exactly 49,920 examples per full epoch, identical to batch 128 (`390 * 128 = 195 * 256 = 49,920`). Epoch counts therefore remain directly comparable despite the different number of optimizer updates.

## Why Couple Batch Size and Learning Rate

For a mean-reduced minibatch gradient, doubling batch size halves the number of updates per epoch. Keeping `lr=0.1` would therefore reduce cumulative parameter movement per epoch and reduce stochastic gradient noise. Under the common diffusion approximation, the noise scale is proportional to `lr * N / batch`. The proposed values preserve the leading ratio exactly:

```text
baseline: 0.1 / 128 = 0.00078125
proposal: 0.2 / 256 = 0.00078125
```

This is why the LR change is not an unrelated second idea. It preserves the approximate exploration temperature that EXP-002 showed to be valuable. It also preserves first-order cumulative update magnitude at equal examples: in the pessimistic case where batch-256 steps take exactly twice as long, the run performs half as many steps at twice the LR.

The entire schedule is scaled, not only the plateau. The `0.2 -> 0.02` discontinuity retains EXP-002's 10x transition, and `2e-4` retains the same LR-to-batch ratio at the terminal point. The 80/20 elapsed-time allocation remains unchanged.

Momentum stays at 0.9 because changing it would alter the effective smoothing horizon and noise properties. Weight decay stays `1e-4`: PyTorch SGD applies it inside the gradient, so at equal examples, half as many updates at twice the LR gives approximately the same cumulative shrinkage. If H20 throughput permits more examples, proportionally more total regularization is consistent with the longer effective data exposure.

## Why No Warm-Up

Goyal et al. pair linear LR scaling with warm-up for very large distributed ImageNet batches, but this experiment increases batch by only 2x on one GPU and a shallow BatchNorm network. Adding warm-up would shorten the high-LR exploration phase that local evidence identifies as important and would introduce another schedule variable. The first run should therefore start at `lr=0.2` and use the same 80% plateau shape.

If loss becomes non-finite or early checkpoint accuracy is grossly abnormal, the run is a valid optimization failure; do not patch or retry it in place. A later experiment could test a short warm-up or the more conservative square-root-scaled `lr≈0.141`, but those should not be selected adaptively after seeing the same run.

## Throughput and Work Projections

The accepted model's measured batch-128 rate is about 16,482 examples/s. Three regimes bound batch 256:

| Regime | Step time | Steps in 300s | Presented examples | Full-data equivalents |
|---|---:|---:|---:|---:|
| Fully compute-bound pessimistic | 15.5 ms | ~19,300 | ~4.94M | ~99 |
| Conservative expected | 12 ms | ~25,000 | 6.40M | ~128 |
| Strong amortization | 9 ms | ~33,300 | 8.53M | ~171 |

The pessimistic regime performs no more example work than EXP-002 and only half as many optimizer updates; linear LR scaling is essential there. The expected useful range is 25,000-33,000 steps, 128-171 full-data equivalents, and 21k-28k examples/s. This is 29-73% more image exposure than EXP-002 while retaining 65-86% of its optimizer-step count.

The proposal should be considered throughput-successful only if it exceeds EXP-002's 4.94M examples materially. A practical diagnostic threshold is at least 23,000 steps, or 5.89M examples and about 118 full-data equivalents. Below that, batch 256 did not amortize enough fixed cost to justify its lower update count.

Peak VRAM should rise because activations and input buffers roughly double, but is likely to remain below 600 MB and is immaterial relative to 97,871 MiB. `MAX_STEPS=64,000` can remain unchanged; reaching it would require less than 4.7 ms per batch-256 step and is not credible given the measured batch-128 step time.

## Mechanism for Accuracy Improvement

The larger batch exposes more parallel work per synchronized step, which should let the H20 process more augmented views during the same 300 active seconds. Each additional full-data equivalent supplies a fresh crop/flip realization, so the model sees more useful input variation, not merely duplicate tensors. Linear LR scaling preserves approximately the same SGD noise scale per epoch and compensates for fewer updates. The expected benefit is therefore more data/augmentation exposure at a comparable optimization temperature, followed by the already validated 20% low-LR refinement.

Relevant primary evidence is directional rather than benchmark-specific:

- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, use linear LR scaling to compensate for larger minibatches and show that optimization, rather than an unavoidable generalization loss, is central to successful scaling: <https://arxiv.org/abs/1706.02677>.
- Smith and Le, *A Bayesian Perspective on Generalization and Stochastic Gradient Descent*, derive an approximate SGD noise scale proportional to `lr * N / batch`, directly motivating the matched `lr/batch` ratio: <https://arxiv.org/abs/1710.06451>.
- Keskar et al., *On Large-Batch Training for Deep Learning*, document the opposing risk that reduced stochasticity can lead to poorer generalization: <https://openreview.net/forum?id=H1oyRlYgg>. Batch 256 is deliberately modest, and noise-scale matching is the mitigation.

These studies do not guarantee a CIFAR-10 gain under a 300-second single-H20 budget. The local experiment is needed because the hardware utilization and fixed-time objective are specific to this repository.

## Hypothesis and Expected Benefit

**Hypothesis:** batch 256 with exact 2x LR scaling will process at least 5.89M examples in 300 counted seconds while preserving the effective SGD exploration/refinement regime, increasing `best_test_acc` from 91.83% to at least 91.93%.

The expected accuracy range is 91.95-92.25%. The gain is likely modest because the model and augmentation are unchanged, but 29-73% more augmented example exposure can improve convergence and late-tail refinement. A much larger gain should be treated cautiously and checked against schedule, evaluator, and timing integrity.

## Generalization and Noise-Scale Risks

- **Generalization gap from reduced gradient noise.** Batch 256 averages twice as many examples. Exact linear LR scaling preserves the leading-order noise scale, but finite-batch, momentum, BatchNorm, and nonstationary-loss effects mean it is not exact. The modest 2x change bounds this risk.
- **Instability at `lr=0.2`.** Linear scaling can overshoot early in training. Do not add unplanned warm-up. Record early smoothed loss and the 20% checkpoint; non-finite loss or catastrophic accuracy is evidence against this operating point.
- **Fewer parameter updates despite more examples.** At 9-12 ms, the run retains only 65-86% of EXP-002's update count. Additional examples may not compensate if this small ResNet is update-limited. That outcome would reject batch 256 under the current eager synchronized loop.
- **The 80% plateau becomes many more data epochs.** Elapsed-time scheduling preserves wall-clock fractions, not epoch counts. With 128-171 epochs, high LR lasts roughly 102-137 epochs versus about 80 in EXP-002. This is intended: it uses the throughput dividend for more high-noise exploration before the same 60-second refinement window.
- **Stronger cumulative weight decay when throughput rises.** More scaled-LR steps increase integrated shrinkage. Keeping weight decay fixed is the conventional clean control and preserves equal-example behavior at the compute-bound limit. If train loss remains high, weight decay is a follow-up variable, not an in-run adjustment.
- **BatchNorm statistical behavior changes.** Larger batches reduce noise in batch statistics, which can improve stability but also removes another source of regularization. No BatchNorm hyperparameter change is proposed.
- **Input pipeline or total-time overhead.** DataLoader work happens before the synchronized training timer starts for each batch but still contributes to total wall time. Persistent workers are retained. Batch 256 halves batches per epoch and should reduce Python iterator overhead per example; total time must still remain below 600 seconds.
- **More dense-tail evaluations.** More epochs mean more once-per-epoch evaluations in the final 20%. EXP-002's 26 evaluations still completed in 336 seconds, leaving 264 seconds of margin, and its best-final gap was only 0.01 point. Retain the policy for comparability; if the run approaches the supervisor limit, terminate it as required rather than changing evaluation cadence mid-run.

## Isolation Decision and Confound Controls

This should **not** be run as batch 256 at unchanged LR. That would be a clean code ablation but an ambiguous scientific test because throughput and SGD temperature would change in opposite directions. Batch 256 plus exact linear scaling is one mechanistically coupled intervention whose invariant is `lr/batch` across the entire schedule.

Everything not mechanically tied to batch remains fixed:

- Same accepted architecture and exact parameter count.
- Same optimizer type, momentum, and weight decay.
- Same elapsed-time phase boundaries and 10x plateau-to-tail step.
- Same hard-label loss, augmentation, data order seed, evaluator, and validation policy.
- No AMP, channels-last, compilation, GPU-resident data, mixup, smoothing, EMA, or model widening.
- One fixed seed and one valid run; no reroll or adaptive retry.

If causal attribution is needed after an improvement, the appropriate follow-up is a new experiment at batch 256 and `lr=0.1`, or batch 128 and `lr=0.2`. Those ablations should not be bundled into EXP-004 and are lower priority than the goal metric.

## Implementation Sketch

No training-loop rewrite is required. Update the four constants and add informative startup logging:

```python
print(
    f"Batch {BATCH_SIZE} | plateau lr {LR:g} | "
    f"tail {ANNEAL_START_LR:g}->{MIN_LR:g}"
)
```

The existing loss is mean-reduced cross-entropy, which is required for linear scaling. Do not change it to a summed loss. Preserve `drop_last=True`; batch 256 divides the 49,920 retained examples per epoch exactly.

Optionally add a summary field for total presented examples only if the harness permits extra fields without affecting required parsing. It can also be derived exactly as `num_steps * BATCH_SIZE`, so avoiding a new summary field is safer.

## Verification Plan

1. Read the moving baseline from `04-results.tsv`: 91.83%, so improvement requires at least 91.93%.
2. Confirm the sole visible GPU is an idle NVIDIA H20 with approximately 98 GB VRAM.
3. Verify the diff changes only `train.py`, with only batch/LR constants and optional descriptive logging changed.
4. Verify architecture count remains 269,722 and `len(train_loader) == 195`.
5. Run syntax compilation, Ruff, pre-commit, and scope checks without editing `prepare.py` or dependencies.
6. Remove any stale log and execute once as `uv run train.py > run.log 2>&1` under the 600-second supervisor.
7. Require a zero exit code, complete finite summary, `training_seconds` near 300, `total_seconds < 600`, and one H20.
8. Require `best_test_acc >= 91.93%` for an improvement verdict.
9. Compute `presented_examples = num_steps * 256` and `full_data_equivalents = num_steps / 195`. Compare with EXP-002's 4,944,512 examples, 38,629 steps, and about 99 full-data equivalents.
10. Record average synchronized step time as `training_seconds / num_steps`, image throughput as `presented_examples / training_seconds`, peak VRAM, best-final gap, and early/tail accuracy trajectory.
11. Confirm every evaluation epoch is unique and the terminal evaluation matches the summary epoch.
12. Remove `run.log` after analysis as required.

## Decision Rules

- **Accept:** `best_test_acc >= 91.93%` with all integrity checks passing. Batch 256 and its scaled LR schedule become the new moving baseline.
- **Accuracy failure with throughput success:** below 91.93% but at least 23,000 steps. Reject the operating point; the result indicates reduced update count or large-batch generalization outweighed extra image exposure. Do not rerun with a different seed.
- **Throughput failure:** fewer than 23,000 steps or fewer than 5.89M examples. Reject batch 256 under the eager synchronized pipeline; consider an isolated systems optimization before another batch increase.
- **Optimization failure:** non-finite loss or severely depressed early accuracy. Reject `lr=0.2`; a future predeclared run may test batch 256 with `lr≈0.141` or a short warm-up.
- **Do not escalate to batch 512 from this run unless batch 256 improves accuracy and demonstrates clear example-throughput headroom.**
