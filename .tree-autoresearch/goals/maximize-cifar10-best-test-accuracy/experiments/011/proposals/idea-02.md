# Proposal: Epoch-End Full-State SWA over the Clean SAM Tail

## Summary

Add uniform stochastic weight averaging to EXP-004 without changing its online optimization. At complete epoch boundaries in the clean final quarter, uniformly average the restored post-optimizer model's parameters and floating BatchNorm buffers. After five samples, evaluate only that averaged state once per epoch by swapping it into the live model and restoring the complete online state exactly afterward. Preserve the independent-image stream, full early CutMix dose, period-two late SAM, fixed seed, one optimizer update per batch, 300-second charged budget, and frozen evaluator.

The proposal intentionally does not add the constant/cyclic LR tail used by canonical SWA. That would bundle an optimizer-schedule intervention with averaging and discard EXP-004's validated cosine endpoint. The test is whether the existing late SAM trajectory is diverse enough for uniform epoch-spaced averaging to improve its central solution.

## Exact Start, Sampling, and Evaluation Rule

Use the fixed constants:

```text
SWA_START_PROGRESS = 0.75
SWA_MIN_SAMPLES = 5
SWA_SAMPLE_CADENCE = one sample per completed natural epoch
SWA_EVAL_AFTER_MIN = true
```

At the end of an epoch, after all optimizer updates and before evaluation, compute `post_epoch_progress = total_training_time / TIME_BUDGET_S`. Add a sample only when all conditions hold:

1. the DataLoader epoch completed all 195 batches rather than ending early on budget exhaustion;
2. `post_epoch_progress >= 0.75`;
3. `total_training_time < TIME_BUDGET_S` before the averaging update starts.

The first qualifying sample is copied exactly. For sample number `n >= 2`, update every averaged floating tensor with the numerically stable uniform recurrence

```text
swa_n = swa_(n-1) + (online_n - swa_(n-1)) / n
```

Copy non-floating buffers from the latest online sample rather than averaging them. The averaging work is explicitly timed, synchronized, and added to `total_training_time`. If an epoch-end update itself crosses 300 seconds, retain that charged sample, stop training, and evaluate. If a training batch already exhausted the budget, skip the final partial-epoch sample and record `final_sample_skipped_budget=1`; do not perform uncharged catch-up averaging.

At each epoch evaluation:

- if `swa_num_samples < 5`, evaluate the online model exactly as EXP-004;
- if `swa_num_samples >= 5`, evaluate only the SWA state;
- never evaluate online and SWA in the same epoch;
- final accuracy/loss must come from SWA whenever the minimum was reached.

The best accumulator can therefore contain preregistered early online evaluations and late SWA evaluations, but no metric-driven model choice. Five samples require roughly five natural epochs and prevent a one- or two-checkpoint object from being mislabeled as an average. Do not change the start or minimum after seeing accuracy.

## Expected Sample Window and Diversity

EXP-004 completed 132 epochs and entered the clean/SAM phase at step 20,664 of 25,560. At 195 batches per epoch, the 75-second tail spans about 25.1 natural epochs, not 30-35: `4,898 / 195 = 25.12`. Depending on where progress 0.75 falls inside an epoch and whether the final epoch is partial, this rule should collect approximately 24-25 complete epoch-end samples. The fifth sample should activate SWA evaluation near progress 0.79-0.80.

The samples are separated by about 195 optimizer updates and roughly three charged seconds. At progress 0.75 the time-cosine LR is approximately 0.034; it decays monotonically to 0.002, while drop path decays to zero and half of eligible updates use SAM gradients. Early samples should therefore be more separated than late samples, but uniform SWA gives each sample equal weight. Unlike canonical SWA, no constant or cyclic LR maintains late trajectory breadth, so the final samples may be strongly correlated and the early high-LR samples may bias the mean behind the final solution.

Audit consecutive-sample FP32 parameter L2 distance, distance from current online weights to the running mean, and LR/progress for every sample. These diagnostics test whether useful diversity existed; they are not gates for selecting a different window.

## Full-State and BatchNorm Treatment

Do not instantiate a second model, which could perturb initialization RNG and is unnecessary. Preallocate three name-aligned state collections with `torch.empty_like`:

- `swa_floating`: running uniform means for every floating parameter and floating buffer;
- `swa_nonfloating`: latest copies of every non-floating buffer;
- `online_restore`: exact temporary snapshots of every model state tensor used around evaluation.

Parameters, BatchNorm `running_mean`, and BatchNorm `running_var` all enter the same uniform average at each eligible epoch end. BatchNorm `num_batches_tracked` is integer and is copied from the latest sampled online state. Apply the same dtype rule to any future registered buffer and assert complete state-key coverage.

This full-state average is the fixed substitute for canonical SWA BatchNorm recalibration. No training-loader pass, augmented-example replay, or buffer update occurs outside charged training. Averaging running statistics is only an approximation to the statistics of the averaged weights, but copying live BN buffers would pair averaged parameters with one endpoint's statistics, and recalibration would be uncharged training. The proposal chooses one self-contained averaged state and treats BN approximation as a central risk.

The online model's BN behavior remains exact EXP-004: primary passes update buffers once and SAM second passes disable tracking. Epoch-end samples occur after all temporary SAM perturbations have been restored and the sole optimizer update has completed.

## Exact Swap and Restore Semantics

When SWA evaluation is active:

1. Assert no SAM perturbation is live and optimizer parameters still reference the online tensors.
2. Snapshot every online parameter and buffer into `online_restore` with `torch._foreach_copy_` where compatible. Snapshot every module's `training` flag.
3. Copy averaged floating tensors and latest non-floating tensors into the existing model state tensors. Do not replace `Parameter` objects, mutate optimizer state, or construct a new state dictionary with new ownership.
4. Call the frozen `evaluator.evaluate(model, device)` exactly once. It sets eval mode and performs no BN updates.
5. In a `finally` block, copy `online_restore` back into every state tensor and restore all module training flags, even if evaluation raises.
6. Verify exact restoration by deterministic state checksums on the first and final SWA evaluations and by full tensor equality in smokes.

The reported `test_loss` and `test_acc` remain the SWA result captured before restoration. The next epoch starts from the exact online weights, BN buffers, optimizer momentum, and stochastic state that would have existed without SWA. State copy/evaluation consumes no RNG. Swapping and restoration are evaluation preparation and occur in the existing excluded evaluation interval; they do not train the model or compute new statistics.

## Integration with EXP-004

Modify only `train.py` and leave these paths unchanged:

- PreAct WRN-16-4 architecture, initialization, BF16/channels-last execution;
- batch-256 DataLoader and independent random crop/flip views;
- seed 42 and dedicated seed-42 CutMix CPU/CUDA generators;
- CutMix probability 0.5, alpha 1.0, and strict progress cutoff 0.75;
- time-normalized LR, weight decay, momentum, Nesterov, and drop-path decay;
- SAM rho 0.05, start 0.75, every-even-step cadence, CUDA RNG replay, second-pass BN suppression, exact parameter restoration, and one optimizer update;
- one evaluator call per epoch and all required final summary fields.

At an eligible epoch end, place the SWA averaging timer before the evaluation branch. Synchronize before reading its duration, add the duration to `total_training_time`, and recompute `budget_exhausted`. This makes state averaging part of the charged training mechanism. Evaluation swap/restore must not update `total_training_time`, matching the protocol's exclusion of evaluation work.

## Compute and Memory Overhead

The 2,748,890 FP32 parameters occupy about 10.5 MiB; BN buffers are small. SWA averages only once per roughly 195 updates and should produce about 25 samples. Each sample reads online and SWA state and writes the mean, roughly 30-35 MiB of GPU memory traffic. Total charged averaging traffic is below 1 GiB across the run, negligible on H20 compared with training.

The persistent SWA mean adds about 11 MiB. The preallocated evaluation restore state adds another approximately 11 MiB, though it could be allocated only before the first SWA evaluation without RNG. Combined with EXP-004's SAM snapshots, expected peak allocation is around 1.21-1.23 GiB versus 1,190.5 MiB. No extra training forward/backward, activation, random transform, or optimizer state is introduced.

Warm-smoke the epoch-end average and swap/restore on physical GPU 0. The fixed feasibility conditions are projected total updates at least 25,000, candidate/parent ordinary-step latency ratio at most 1.02 when averaging cost is amortized over 195 steps, and projected total runtime below 570 seconds. The parent is measured in the same harness, following the parent-relative gate learning. A valid overhead failure rejects this implementation; it does not permit a sparser sample cadence.

## Evidence and Mechanism

SWA averages late SGD trajectory points toward a wider central solution and improves CIFAR residual-network generalization with little overhead (`knowledge/papers/stochastic-weight-averaging.md`). EXP-004 already validates a late flatness-aware optimizer phase, so averaging epoch-spaced post-SAM iterates could reduce checkpoint variance without changing how those iterates are generated. EXP-006's final four evaluations varied by 0.15 points, showing that late solutions can move enough for averaging to be relevant (`experiments/006/04-analysis.md`).

The evidence against a large gain is substantial. EXP-004's final accuracy equaled its best 95.40%; its own trajectory did not expose a best-to-final drop. SWA literature benefits from constant/cyclic LR diversity, while this parent deliberately collapses LR. Child variation under EXP-006's different mixed-sample mechanism does not prove parent variation. The EMA literature further emphasizes horizon/cadence dependence; this proposal avoids an arbitrary exponential decay but still makes a fixed uniform-window choice (`knowledge/papers/how-to-scale-your-ema.md`).

## Attribution

Online training is additive and parent-compatible. A gain is attributable to the full package of uniform epoch-end averaging, full floating-state BN averaging, the five-sample evaluation switch, and the 0.75-to-end window. It cannot isolate weight averaging from the BN approximation or the loss of late online evaluation visibility.

Because the averaging update is charged, its tiny overhead may alter final step/SAM counts. Record exact exposures and avoid causal claims for differences below the protocol's noise scale. Do not evaluate online late as a hidden control; the once-per-epoch constraint allows only the preregistered SWA path after activation.

## Expected Effect and Falsification

The formal improvement gate is `best_test_acc >= 95.50%` over EXP-004's 95.40%. The meaningful mechanism target is `>=95.70%`, a +0.30-point gain large enough to exceed observed local variation.

A realistic expectation is only **+0.05 to +0.20 points**. Clearing 95.70 is possible if SAM maintains a broad trajectory and the uniform center corrects endpoint variance, but it is not likely under the collapsing cosine LR and parent final-equals-best result. This proposal should rank below candidates with direct evidence for a >=0.30-point effect; its advantages are low compute cost, preservation of all validated training mechanisms, and direct targeting of late variance.

The accuracy hypothesis is that SWA reaches at least 95.50% while collecting at least 20 samples and preserving at least 25,000 updates. Falsification conditions are:

- `best_test_acc < 95.50%`: no accuracy improvement, regardless of loss or apparent smoothness;
- fewer than 5 samples: SWA evaluation never validly activates, so the experiment is infeasible;
- fewer than 20 samples or fewer than 25,000 steps: planned dose/exposure failed even if the formal metric happens to pass;
- nonfinite mean, BN/state mismatch, imperfect restore, second evaluator call, or uncharged averaging: invalid implementation;
- online-to-SWA distance near zero with no metric gain: the cosine trajectory lacked useful diversity for this fixed window.

Results from 95.50 to 95.69 are formal improvements but below the preregistered meaningful target. Do not move the start, change the minimum, add a flat LR tail, select online versus SWA, or rerun a seed after observing results.

## Audit Contract

Startup output must report `swa_start=0.75`, `swa_min_samples=5`, `swa_cadence=epoch_end`, `swa_state=all_floating`, `swa_bn=recurrent_full_state`, floating/non-floating tensor and element counts, and restore inventory parity.

For every sample, log or aggregate sample number, epoch, step, progress, LR, charged update milliseconds, consecutive-online-sample L2 distance, and online-to-SWA L2/relative distance. Final audit must report:

- sample count, first/last sample epoch/step/progress, and final-partial skip count;
- cumulative charged SWA update time and min/mean/max update latency;
- online and SWA evaluation counts, first SWA evaluation epoch, and exactly one total call per epoch;
- averaged parameter/floating-buffer counts and latest integer-buffer copy counts;
- first/final pre-swap and post-restore checksums, restoration failures, nonfinite failures, inventory failures;
- online-to-SWA final L2 and relative distance;
- unchanged CutMix and SAM eligible/applied counts, ratios, and first SAM progress;
- final optimizer steps, epochs, peak VRAM, and complete metric summary copied durably before log deletion.

## Correctness Smokes and Run Verification

Before launch:

1. **Uniform recurrence**: compare fixed scalar/tensor sequences against an FP64 arithmetic mean at samples 1, 2, 5, and 25.
2. **Eligibility boundaries**: simulate complete/partial epochs just below, at, and above progress 0.75 and at budget exhaustion; verify the exact sample/skip rule.
3. **State inventory**: reconcile every parameter/buffer key, shape, dtype, device, memory format, and floating/non-floating classification.
4. **BatchNorm semantics**: verify means/variances enter uniform averaging, integer counters copy latest, evaluator changes no averaged/live buffers, and no data recalibration occurs.
5. **Swap/restore**: use distinct online/SWA states, force both successful and failing evaluations, and prove tensor-bitwise restoration, module-mode restoration, unchanged optimizer ownership/state, and captured SWA metrics.
6. **SAM ordering**: verify epoch samples see only restored post-optimizer weights, never temporary perturbed parameters or second-pass BN state.
7. **Parent parity**: before the first sample, verify CutMix/SAM loss, RNG, weights, buffers, and counters match EXP-004. After activation, verify online updates remain parent-identical apart from charged-time exposure.
8. **Cadence**: simulate evaluation routing around samples 4/5 and assert one evaluator call per epoch and final SWA evaluation after activation.
9. **GPU overhead**: measure charged averaging and swap/restore latency/memory on GPU 0; apply the fixed parent-relative exposure/runtime gates without accuracy.
10. **Static checks**: compile/lint `train.py`, confirm only it differs from EXP-004, and validate all config/audit fields.

After passing smokes, confirm GPU 0 is the approximately 97,871 MiB H20 and launch exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Verify exit 0, 300-second charged budget, total runtime below 600 seconds, no nonfinite/CUDA errors, one evaluation per epoch, `num_params=2,748,890`, `num_steps>=25,000`, at least 20 SWA samples, exact CutMix/SAM preservation, zero restore/inventory failures, complete durable audit, and `best_test_acc>=95.50%`. Remove `run.log` after analysis. No repeat or metric-driven adjustment is permitted.

## Effort

**Medium.** Uniform arithmetic is simple and cheap; exact full-state classification, charged epoch-end updates, exception-safe swap/restore, BatchNorm semantics, and single-path evaluation require careful implementation and tests.
