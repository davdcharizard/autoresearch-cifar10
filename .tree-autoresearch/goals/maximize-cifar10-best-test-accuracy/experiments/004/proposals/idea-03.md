# Proposal: Clean-Finish Periodic SAM

## Summary

Keep the complete EXP-002 recipe and apply plain Sharpness-Aware Minimization (SAM) only during the clean final quarter of charged training, on one of every four optimizer steps. Use a fixed global perturbation radius `rho=0.05`, no adaptive parameter scaling, the same batch and loss for both SAM passes, and exactly one existing Nesterov-SGD update per batch. The fixed configuration is:

- `SAM_RHO = 0.05`
- `SAM_START = 0.75`
- `SAM_PERIOD = 4`
- SAM condition: `progress >= SAM_START and step % SAM_PERIOD == 0`

CutMix is active only while `progress < 0.75`, so the SAM and CutMix branches are mutually exclusive. This keeps SAM's extra pass away from mixed-target and patch-RNG interactions and applies it while the model is refining a clean, low-learning-rate solution. `rho=0.05`, the 75% start, and period 4 are preregistered hypotheses rather than values selected from this run's test accuracy.

## Motivation and Evidence

EXP-002 improved the parent from 94.62% to 95.23% by adding front-loaded CutMix, but it still ends with a nonzero generalization gap and only a 0.04-point best-to-final gap. EXP-003 found no confirmed improvement from a narrow CutMix-probability/drop-path grid; its best confirmation reached only 95.28%, below the 95.33% gate. A qualitatively different optimizer-level generalization mechanism is therefore more informative than another scalar regularization adjustment.

SAM explicitly minimizes loss in a local parameter neighborhood by constructing a normalized adversarial weight perturbation and taking the update gradient at the perturbed weights. The source reports CIFAR generalization gains, but standard SAM requires two sequential gradient calculations. Full-run SAM is not credible under this goal's 300-second charged budget because it would approximately halve optimizer exposure. The proposed late periodic schedule is the least expensive variant that still supplies a sustained block of SAM updates in the final basin: about 1,400 perturbation-aware updates rather than a few isolated pulses.

Sources:

- `experiments/004/papers/sharpness-aware-minimization.md`
- `experiments/002/04-analysis.md`
- `experiments/003/04-analysis.md`
- `experiments/004/00-navigate.md`

## Compute Model

EXP-002 completed 27,950 steps in 300 charged seconds, or 10.73 ms per ordinary step on average. A SAM batch adds a second forward, loss, and backward plus a small gradient-norm/perturb/restore cost, but it does not repeat host-to-device transfer, CutMix, or the optimizer update. Conservatively model the added pass as 0.8-1.0 ordinary-step equivalents.

The first 225 charged seconds remain entirely ordinary. During the final 75 seconds, one quarter of batches receives the extra pass, making the late-phase average cost `1 + (0.8..1.0)/4 = 1.20..1.25` ordinary steps. Relative to EXP-002, this predicts:

- 20,963 ordinary-equivalent steps in the first three quarters;
- approximately 5,590-5,823 total batches in the final quarter;
- approximately 1,398-1,456 actual SAM updates;
- approximately 26,550-26,786 total steps, a 4.2-5.0% throughput loss versus 27,950.

Allowing for perturbation bookkeeping and run-to-run system throughput, the operational expectation is 26,300-26,800 total steps. This is materially cheaper than full SAM (roughly 14,000-16,000 expected steps) or period-4 SAM throughout training (roughly 22,400-23,300 expected steps). All added work must remain between the existing per-batch `t0` and CUDA synchronization, so SAM consumes the fixed budget rather than receiving uncharged compute.

## Algorithm

Ordinary batches retain the exact EXP-002 code path. On a scheduled SAM batch:

1. Transfer the clean batch and set the existing time-based learning rate and drop-path scale exactly as EXP-002 does.
2. Clear gradients, save the global CUDA RNG state immediately before the first model forward, run the existing clean-label loss under BF16 autocast, and backpropagate once.
3. Compute one global L2 norm over all non-`None` parameter gradients in FP32. Require it to be finite and greater than a small epsilon. Under `torch.no_grad()`, clone each affected parameter's unperturbed value and apply `parameter += rho * gradient / (global_norm + eps)`. Do not perturb BatchNorm buffers or optimizer state.
4. Clear the first gradients with `optimizer.zero_grad(set_to_none=True)`. Restore the saved CUDA RNG state before the second forward so stochastic-depth masks exactly replay the first pass. This leaves the post-second-forward global RNG state equal to the state after one parent forward, preventing SAM from shifting all future drop-path draws.
5. Disable BatchNorm running-stat tracking only for the second forward, while leaving modules in training mode so the second pass still normalizes with current batch statistics. Run the identical batch and clean-label loss under the same autocast settings and backpropagate the perturbed loss. Restore every BatchNorm module's tracking flag immediately afterward.
6. In a `finally`-protected restoration block, copy the saved unperturbed parameters back exactly after the second backward and before the optimizer step. Exact copies avoid accumulating float32 roundoff from repeated add/sub restoration.
7. Call the existing `optimizer.step()` exactly once. Nesterov momentum and weight decay therefore update exactly once from the second-pass gradient at the restored parameters; the first gradient only defines the SAM perturbation and never enters optimizer state.

The second forward must reuse the already prepared `inputs` and `targets`; it must not redraw augmentation, CutMix, data, or labels. Because SAM starts where CutMix ends, every SAM pulse uses ordinary hard-label cross-entropy. The loss recorded in progress output on a SAM step should be the second-pass loss because that gradient drives the optimizer update.

## BatchNorm and Optimizer Correctness

Each scheduled batch should update running means, running variances, and `num_batches_tracked` exactly once, during the unperturbed first pass. Temporarily setting `track_running_stats=False` on all `nn.modules.batchnorm._BatchNorm` modules for only the second forward prevents a second buffer update while retaining batch-statistic normalization. Restore the original per-module flags even if the second pass fails. Evaluation behavior remains unchanged.

The base `optim.SGD` object remains the sole optimizer; no wrapper or new dependency is needed. There is no optimizer step, momentum update, or weight-decay application after the first backward. The first gradients are discarded after the perturbation is formed. The second gradients are computed while weights are perturbed, weights are then restored exactly, and the existing single SGD step applies those gradients to the original weights. A nonfinite gradient norm, loss, or second-pass gradient is a hard failure, not a reason to skip SAM silently.

The parameter snapshots add roughly one FP32 model copy for affected parameters, about 11 MiB for the 2.75M-parameter network. The two autograd graphs are sequential rather than simultaneous, so activation memory should remain close to the existing peak plus this small snapshot and temporary gradient-norm storage, far below the 98 GB device capacity.

## Implementation Scope

Modify only `train.py`:

- Add the three fixed SAM constants and include them in the startup configuration line.
- Add small helpers for collecting non-`None` gradients, computing/applying the perturbation while snapshotting parameters, exact restoration, and toggling/restoring BatchNorm running-stat tracking.
- Branch inside the existing timed training step between the unchanged ordinary path and the scheduled two-pass SAM path.
- Add `sam_eligible_batches` and `sam_applied_batches` counters. Eligibility means a batch whose pre-batch charged-time progress is at least 0.75; applied means the fixed period condition fired. Print a final `sam: applied=X eligible=Y ratio=Z` mechanism-audit line without changing any required summary key.

Do not alter the model, data transforms, CutMix helper or generators, LR/drop-path schedules, global seed, optimizer hyperparameters, evaluator, validation cadence, timing boundary, final metric computation, or summary values. No package or dependency is added.

## Implementation Tests

Before the one full run:

1. Run compilation, lint, format, and diff checks. Confirm only `train.py` is modified and audit the diff to ensure the evaluator, metric accumulation, summary data flow, and timing boundary are unchanged.
2. Exercise the cadence function over synthetic `(progress, step)` pairs. Require no SAM below 0.75, exactly one application per four eligible steps at and above 0.75, and no overlap with `progress < CUTMIX_END`.
3. On a tiny model with BatchNorm and a fixed synthetic batch, instrument forward and optimizer calls. Require two forwards and one optimizer step for a SAM batch, versus one forward and one optimizer step for an ordinary batch.
4. Snapshot parameters and optimizer state around a SAM pulse. Require perturbed weights during the second forward, bitwise restoration before `optimizer.step`, and exactly one momentum-state update. Verify the global perturbation L2 norm equals `rho` within float tolerance when the gradient norm is nonzero.
5. Snapshot all BatchNorm buffers. Require the scheduled SAM batch to produce the same `num_batches_tracked` increment as one ordinary batch and verify no second-pass running-stat update occurs.
6. Compare CUDA RNG state after a two-pass SAM smoke step with a one-forward reference from the same initial state. Require equality, demonstrating that stochastic-depth replay consumes only one forward's RNG sequence.
7. Inject an exception during the second pass in a helper-level test and require exact parameter and BatchNorm-flag restoration. A failed full run must never leave perturbed weights available for evaluation or summary generation.
8. Run one BF16 channels-last GPU batch on physical GPU 0 and require finite first/second losses, finite gradient norm, finite gradients, and no device/layout error. This is a smoke test, not a training or accuracy trial.

## Full-Run Verification

Launch exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Require physical GPU 0 to be the NVIDIA H20, exit status 0, 299.5-301.0 charged training seconds, total time below 600 seconds, all existing summary keys exactly once, unchanged 2,748,890 parameters, and no more than one evaluation for each epoch.

Mechanism checks should require:

- no SAM application before 75% charged progress and no CutMix application at or after it;
- a nonzero eligible and applied count;
- `applied / eligible` close to 0.25, with exact cadence confirmed structurally rather than inferred only from the printed counter;
- approximately 1,300-1,550 SAM updates and at least 26,300 total optimizer steps;
- no nonfinite loss/gradient, traceback, CUDA error, or timeout.

Do not rerun, change `rho`, alter the start/cadence, or select a checkpoint based on intermediate test accuracy.

## Risks and Mitigations

- **Sparse late SAM may be too weak.** The original method applies SAM much more broadly; a final-quarter period-4 approximation is not directly validated by the cited paper. The schedule deliberately trades mechanism fidelity for the hard time budget, but still supplies roughly 1,400 coherent SAM updates.
- **Lost optimizer exposure may outweigh flatness gains.** The proposal sacrifices about 4-5% of steps. Starting only after 75% preserves the high-LR representation-learning and successful CutMix phase; the `>=26,300` step audit prevents an unexpectedly expensive implementation from being interpreted as a clean SAM test.
- **A fixed radius may be miscalibrated.** `rho=0.05` is a fixed small plain-SAM hypothesis, not tuned for this WRN/BF16 recipe. A single run can falsify it; metric-driven radius adjustment or retry is prohibited.
- **BatchNorm can be updated twice.** Explicit second-pass tracking suppression and buffer-counter tests prevent this known two-forward failure mode.
- **Stochastic depth can decorrelate the two gradients.** CUDA RNG replay makes both passes use identical masks and prevents an extra forward from perturbing future masks.
- **Perturbed weights can leak into the optimizer or evaluation.** Exact snapshots, `finally` restoration, and an injected-failure test protect the base optimizer and model state.
- **Observed improvements can be evaluation variance.** EXP-003 demonstrated substantial selected-run variance. This proposal is a one-shot mechanism test against the fixed 95.23% parent and must clear the preregistered 0.10-point margin without reruns.

## Testable Hypothesis

Clean-finish period-4 SAM will complete within the 600-second outer limit and the unchanged 300-second charged budget, retain at least 26,300 optimizer steps, execute approximately 1,300-1,550 SAM updates, and achieve `best_test_acc >= 95.33%` versus the EXP-002 parent at 95.23%. The predicted gain comes from flattening the final clean solution after CutMix has built the representation, while the restricted cadence limits the exposure loss that makes full SAM infeasible.

Failure to reach 95.33%, any timing/scope/cadence violation, or substantially fewer than 26,300 steps falsifies the proposal for this experiment; it must not trigger a seed rerun or post-hoc SAM hyperparameter change.
