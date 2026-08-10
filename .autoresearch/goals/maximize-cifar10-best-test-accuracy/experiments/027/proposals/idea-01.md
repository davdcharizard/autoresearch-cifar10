# Proposal: CutMix-Off, RandAugment-On Refinement Window

## Decision and falsifiable hypothesis

Keep the accepted width-2 ResNet-20 recipe and its first 70% of counted training unchanged: crop/flip, RandAugment N1/M7, and alpha-1 CutMix on 50% of batches at `lr=0.1`. At the first completed optimizer step at or beyond 70% counted time, switch **only CutMix eligibility off** while retaining the same persistent strong loader and N1/M7 transforms. Continue hard-label N1/M7 training at `lr=0.1` until the existing 80% boundary, then perform the accepted one-time loader rebuild to the hard crop/flip weak tail and unchanged cosine refinement.

The implementation must not rebuild or reseed the strong loader at 70%. A forkserver-safe shared policy flag is read once by each worker collator and returned as explicit batch provenance. This preserves the source-image transform stream and makes the intended change narrowly temporal: after a bounded prefetched-batch drain, the same N1/M7 batches that would have been eligible for CutMix instead retain their hard labels and unpasted pixels. The accepted 80% loader lifecycle is unchanged.

**Primary hypothesis:** approximately 10% of the budget (about 2,650-2,750 updates) of hard-label N1/M7 refinement will reduce the soft-target/classifier adaptation debt without giving up the broad-view invariances that EXP005 showed are still useful, raising `best_test_acc` from `94.15%` to at least `94.25%`. The point prediction is **94.32%**, with at least **26,629 optimizer steps** and a valid one-run completion. This is falsified by any valid result below 94.25%; no switch checkpoint, loss, or trajectory diagnostic can rescue a miss.

Secondary mechanism predictions, registered for interpretation rather than acceptance, are:

- the final strong checkpoint near 80% will be at least EXP010's `89.73%` and preferably at least `90.00%`;
- the first weak-tail checkpoint will be at least EXP010's `93.16%` without the severe switch underfit seen in EXP011/EXP026;
- final NLL will be no worse than `0.1984` (EXP010's `0.1934` plus 0.0050), and the best/final gap will remain at most 0.20 points;
- the production transition request will occur in `[69.9,70.2]%`, all policy-enabled prefetched batches will drain by `70.5%`, and no later strong batch will have a soft target.

## Mechanism and workload fit

The current systems bottleneck is width-2 convolution/BN backward, which accounts for 75.46% of GPU-stage time; loader wait, transfer, cross-entropy, optimizer, and host dispatch are collectively small. The candidate does not add a model path, activation, loss, or device operation. It replaces some probability-target CutMix steps with ordinary hard-target steps, so it should preserve or slightly improve counted-step throughput and the accepted 598.7 MiB allocation. A shared one-byte policy read happens in prefetched workers and should remain hidden by the measured loader headroom.

The accuracy bottleneck is the balance between useful composite regularization and recoverable strong-phase fit. In EXP010, 50% CutMix improved the frontier by 0.60 points, but its role was deliberately limited to the first 80%; the hard weak tail converted a `89.73%` strong checkpoint into `94.15%`. EXP011's 75% CutMix caused a `2.91`-point stronger underfit deficit and then recovered immediately to a better first weak checkpoint before plateauing. EXP026 similarly entered the tail `1.60` points below EXP010, recovered to a first weak checkpoint `0.21` points above it, but missed the gate with worse NLL. Those trajectories show that removing soft targets can rapidly repair classifier/representation mismatch, while excessive or less suitable mixing leaves too much debt for the fixed tail.

The proposed middle phase changes about half of the batches in the 70-80% interval. Relative to EXP010's roughly 10,673 mixed strong batches, it should remove about 1,300 CutMix events (roughly 12.5% of accepted CutMix exposure), not half or all of the successful mechanism. The first 70% still contains about 9,300 natural CutMix batches, preserving most regional supervision. The final 10% of high-LR training retains N1/M7, so hard labels sharpen the objective without prematurely dropping the strong-view distribution.

This is specifically different from lowering `CUTMIX_PROBABILITY` globally. A constant probability such as 0.4375 would distribute the same expected number of mixed events throughout the plateau and dilute early regional learning. The candidate instead preserves the accepted early regularizer and creates an ordered refinement curriculum after representations have already absorbed it.

## Evidence and counterevidence

### Supporting local evidence

- **EXP010:** 50% alpha-1 CutMix composed with N1/M7 raised accuracy from 93.55% to 94.15%, retained 99.10% of optimizer exposure, lowered NLL to 0.1934, and improved continuously through the hard tail. This establishes both the regional mechanism and hard-label conversion stage.
- **EXP011:** increasing CutMix to 75% left exposure unchanged but lowered the strong checkpoint to 86.82%; the first hard checkpoint nevertheless jumped to 93.40%. This is direct evidence that target hardening can repair a soft-target deficit quickly, while also warning against simply strengthening mixing.
- **EXP026:** replacing half of CutMix with Mixup increased steps to 27,268 yet lowered the switch checkpoint to 88.13%; the first hard checkpoint recovered to 93.37%. Throughput was not the limiter, and the response to hard labels was again immediate.
- **EXP005:** switching the entire strong transform to weak crop/flip at 75% reduced accuracy by 0.18 points. The proposal incorporates that negative result rather than repeating it: N1/M7 remains active until exactly the accepted 80% boundary.
- **Protocol history:** persistent workers and worker-side augmentation preserve exposure; the candidate keeps both. The only worker shutdown/rebuild remains the accepted transition at 80%.

### Literature grounding

CutMix replaces a region with class-bearing pixels and adjusts the target by realized area; its reported benefits include spatially distributed evidence, robustness, and localization. This supports substantial early CutMix exposure, but does not require that soft regional targets remain active until the end of a short high-LR plateau. Source: `knowledge/papers/cutmix.md`; Yun et al., ICCV 2019.

RandAugment applies a small fixed number of randomly selected image operations with shared magnitude and was designed as a practical strong augmentation policy. Keeping N1/M7 active during the hard-label window preserves this broader input invariance mechanism while changing only the target/pasting geometry. Source: `knowledge/papers/randaugment.md`; Cubuk et al., NeurIPS 2020.

Neither paper validates a 70% CutMix-off boundary, a three-phase 300-second schedule, or this exact width-2 ResNet-20. The boundary is a locally motivated curriculum hypothesis, not a literature-derived optimum.

### Counterevidence and limits

- EXP010's strong deficit was only 0.35 points versus the pre-CutMix width-2 run, its first weak checkpoint was already better, and it finished at its best. The accepted recipe may not have meaningful adaptation debt; ending CutMix early could simply discard useful late regional examples.
- EXP005 implies that hard high-LR adaptation is not automatically beneficial. The candidate avoids its weak-view shift, but hard N1/M7 may still be too distorted to provide clean classifier refinement.
- EXP011 and EXP026 demonstrate recovery after **all** strong augmentation and mixing are removed. They do not prove that removing only CutMix while N1/M7 remains will produce the same recovery.
- The 70% choice is one temporal point with no local sweep. A miss does not prove all CutMix-off schedules fail, but no boundary may be changed or retried inside EXP027.
- A shared flag is observed at worker-collation time while counted progress is observed in the parent after GPU updates. Up to the bounded prefetch window can therefore retain the old policy after the request. Provenance and a strict drain deadline make this smear observable; they do not make the switch mathematically instantaneous.

## Exact `train.py` production implementation

Modify only tracked `train.py`. Preserve `CUTMIX_ALPHA=1.0`, `CUTMIX_PROBABILITY=0.5`, `LR_HOLD_FRACTION=0.8`, N1/M7, crop/flip ordering, batch size, model, optimizer, seed, evaluator, timer, and evaluation checkpoints. Add only:

```python
import multiprocessing as mp

CUTMIX_OFF_FRACTION = 0.70
POLICY_ON = 1
POLICY_OFF = 0


class PhaseCutMixCollator:
    def __init__(self, cutmix_enabled):
        self.cutmix_enabled = cutmix_enabled

    def __call__(self, batch):
        inputs, targets = default_collate(batch)
        policy_enabled = bool(self.cutmix_enabled.value)
        with torch.random.fork_rng(devices=[]):
            if policy_enabled and torch.rand(()).item() < CUTMIX_PROBABILITY:
                inputs, targets = cutmix(inputs, targets)
        return inputs, targets, int(policy_enabled)
```

Inside guarded `main()`, create the shared flag from the exact loader context before constructing the initial loader:

```python
forkserver_context = mp.get_context("forkserver")
cutmix_enabled = forkserver_context.Value("b", True, lock=True)
strong_collator = PhaseCutMixCollator(cutmix_enabled)
train_loader = make_train_loader(strong_train_tf, collate_fn=strong_collator)
cutmix_switch_requested = False
```

The collator class must remain module-level so forkserver workers can import it. It reads the synchronized value exactly once per batch and returns that snapshot. The policy gate and CutMix transform remain inside `torch.random.fork_rng(devices=[])`, as in the accepted code, so policy RNG consumption does not perturb later crop/flip/RandAugment randomness. When policy is off, it performs no policy draw and returns the unmodified post-N1/M7 FP32 batch with int64 hard targets. It never uses Python, NumPy, or CUDA RNG; it never mutates the shared flag; and it never changes workers, prefetch factor, or sampler.

The strong loop unpacks triples and validates the returned policy snapshot rather than assuming the parent's current request state. Count four disjoint quantities: policy-on hard, policy-on CutMix, policy-off hard N1/M7, and total strong. A policy-off batch with two-dimensional targets, or a policy-on batch with a target rank other than one or two, is fatal. The weak loop continues to unpack the accepted two-item batches and requires one-dimensional int64 targets.

Immediately after completing and timing the first optimizer step satisfying
`total_training_time >= CUTMIX_OFF_FRACTION * TIME_BUDGET_S`, acquire the shared lock, set the flag false once, release it, and print a compact request marker containing step, epoch, progress, and all current counts. Do not break the iterator, evaluate, shut down workers, rebuild a loader, collect garbage, reset a seed, or change LR at this boundary. Already-prefetched batches retain the explicit snapshot they observed; track the last policy-on batch's parent-side progress and require all subsequent strong batches to be policy-off by 70.5%.

The existing inner-loop break remains keyed only to `LR_HOLD_FRACTION=0.8`. At 80%, shut down the same eight strong workers, delete the loader, collect garbage, create the same default-collated weak loader, and log both the CutMix-off request and realized drain statistics in the existing augmentation-switch record. No new evaluation is added: the existing 70% checkpoint remains subject to its existing epoch-boundary cadence, and dense evaluation still begins only at 80%.

The implementation must remain safe if the 70% request occurs mid-epoch. It must not infer policy state from target rank alone because an eligible batch can naturally select the hard branch. The explicit returned snapshot distinguishes “hard because CutMix was not drawn” from “hard because mixing was disabled.” At final summary, explicitly stop the active loader's workers and assert no live loader child remains; this strengthens lifecycle cleanup without changing counted work.

## Semantic, RNG, and lifecycle preflight

Before timing or production, use experiment-local ignored diagnostics with a guarded `if __name__ == "__main__"` entry point. Serialize and fsync evidence before applying any veto.

### Direct semantic gate

Use controlled constant-image/distinct-label fixtures and natural CIFAR-10 batches to require:

- policy-on forced-hard returns FP32 `[128,3,32,32]` inputs and unchanged int64 `[128]` targets;
- policy-on forced-CutMix returns finite FP32 inputs and finite nonnegative `[128,10]` targets whose rows sum to one and whose target mass matches pasted area within `1/1024`;
- policy-off never draws the gate, never calls CutMix, and returns inputs/targets bitwise equal to `default_collate` output with snapshot `POLICY_OFF`;
- policy-on and policy-off calls leave surrounding CPU RNG byte-identical, consume no CUDA RNG, and do not mutate source tensors;
- the synchronized flag and collator survive actual forkserver pickling, are visible to every worker, and cannot change within one collator call because the value is snapshotted once.

### Immutable-source comparison

Persist the first 200 unfiltered post-N1/M7, pre-policy source batches and their exact worker CPU policy-boundary states using the EXP026 corpus protocol: mirror production seed/model/iterator ordering, hash all tensors/states, atomically write and reload the corpus, and stop all workers. From each same cloned source/state, apply the accepted policy-on control and candidate policy-off branch independently. Require:

- when the control gate is hard, control and candidate outputs are bitwise identical;
- when the control gate is CutMix, candidate remains bitwise equal to the hard source while control has valid area-adjusted inputs/targets;
- all source/state hashes remain unchanged and surrounding RNG is restored;
- all 200 candidate targets are hard and all accepted-control policy decisions naturally contain 40-60% CutMix; no filtering or corpus rematerialization is allowed.

From one seed-42 initial model/SGD state, run an identical accepted-policy common prefix, clone exact parameters, buffers, optimizer state, and backend flags, then train explicit control/candidate continuations over all 200 immutable records. The control retains accepted 50% CutMix; the candidate uses hard N1/M7 sources. Require finite losses/logits/gradients/parameters/BN buffers/momentum, exact BN counters, complete optimizer state, candidate terminal debiased loss EMA no greater than 1.5 times control, and no candidate-only step above 95% one-class predictions when control is at or below 95%. This is a gross integrity gate, not evidence that a step-200 model reproduces the 70% production state.

### Real-loader transition and cleanup gate

Run at least 20,000 real strong collations. Keep policy enabled long enough to observe 47.5-52.5% CutMix, then flip the shared flag from the parent while workers are active. Record every returned snapshot and require:

- the request is seen by all eight worker IDs;
- any prefetched policy-on batches are bounded by `2 * NUM_WORKERS + 8` delivered batches after the request (24 with the accepted eight workers and default prefetch factor two);
- after that drain allowance, at least 5,000 consecutive strong batches are policy-off, hard-target, valid, and still N1/M7-transformed;
- no worker is added or replaced, PID set is constant through the CutMix-only switch, and process/RSS/pinned-allocation counts show no monotonic growth;
- at the simulated 80% transition, exactly those eight strong workers stop, weak rebuild plus first batch takes less than five seconds, the weak batch is a hard two-item batch, weak workers then stop, and there are zero live loader children.

If the natural scheduler exceeds the 24-batch prefetch allowance, if a policy-on tag reappears after the drain is declared complete, or if the synchronized object fails under forkserver, abort rather than replacing the design with a 70% loader rebuild or a manager process.

## Timing and exposure gates

Only after semantic/safety/lifecycle evidence passes, confirm exactly one idle NVIDIA H20 near 97,871 MiB. Run one unscored device conditioner followed by five alternating fresh-process accepted/candidate pairs. Every process must record identical CUDA/cuDNN/TF32/determinism flags, use the actual width-2 model/SGD and eight-worker production loader, include at least 100 warmup steps, and explicitly shut down every loader.

Each measured arm runs a scaled 1,000-step production curriculum with no evaluator:

- accepted control: 800 policy-on N1/M7+50% CutMix steps, then rebuild to 200 weak steps;
- candidate: request policy-off after 700 policy-on steps, retain the same strong loader through step 800 while recording the prefetch drain and hard N1/M7 segment, then perform the same 200-step weak rebuild.

Record synchronized GPU counted time, iterator wait, full segment wall time, transition wall time, target/snapshot counts, worker PIDs, RSS, CUDA allocation, and complete process exit. Charge the candidate's shared-flag transition and any prefetch drain honestly; do not add a control loader rebuild at 70% because the accepted production path has none.

Require:

- aggregate candidate/control synchronized counted-step ratio at most `1.01`, every pair at most `1.04`, and per-arm CV below 3%;
- conservative projected exposure `floor(26898 * control_mean / candidate_mean) >= 26629`;
- candidate warmed loader delivery at least 1.20 times GPU consumption, median iterator wait below 10% and p95 below 20% of synchronized step time;
- integrated non-evaluation wall/count ratio at most 1.07 and no more than 0.02 above control;
- natural policy-on CutMix proportion 47.5-52.5%, zero soft targets after the registered drain, propagation within 24 delivered batches, and no weak soft target;
- peak CUDA allocation below 650 MiB, no worker/allocation growth, weak rebuild below five seconds, complete cleanup, and conservative projected production total below 540 seconds.

The scaled timing is for feasibility only. It cannot tune the 70% boundary, drain allowance, probability, or acceptance threshold after observation.

## Production run and ordered verification

After all gates pass:

1. Confirm the moving baseline remains `94.15%` at `7c1e7d8`, the acceptance threshold is `94.25%`, only tracked `train.py` differs, no stale `run.log` variant exists, and exactly one idle H20 is visible. Pass `py_compile`, Ruff, formatting, `git diff --check`, source/AST invariants, and exact 1,073,962-parameter model/optimizer/schedule/evaluator checks.
2. Run seed 42 exactly once as `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`. Never rerun a completed finite candidate, regardless of trajectory or margin.
3. First require process exit zero and exactly one complete ten-field finite summary. A timeout, missing summary, worker failure, or malformed target is invalid/crash, not a numeric miss.
4. Compare `best_test_acc` with `94.25%`. A lower finite result is `no-improvement`; continue parsing informational diagnostics but do not alter or retry the schedule.
5. Require `299.9 <= training_seconds <= 300.2`, `total_seconds < 600`, `num_params=1,073,962`, `num_steps>=26,629`, unique at-most-once-per-epoch evaluations, and unchanged evaluator cadence.
6. Require exactly one CutMix-off request in `[69.9,70.2]%`, no policy-on batch after 70.5%, exactly one strong-to-weak transition in `[79.5,80.5]%`, exactly eight workers stopped there, hard weak targets, no live loader children at exit, and all count identities.
7. Report pre-70 policy-on hard/CutMix counts and fraction, drain length, hard N1/M7 count, weak count if instrumented, switch/first-weak/peak/final accuracy, final NLL, best/final epochs, steps, VRAM, corpus/report hashes, and timing projections.

The formal verdict is improvement only at `best_test_acc >=94.25%` with all integrity conditions satisfied. A bare 0.10-point gain meets the registered rule but must be described as weak single-seed causal evidence. The result identifies the net temporal CutMix-off curriculum; it does not identify whether any effect came specifically from harder targets, removal of pasted pixels, or their interaction.

## Abort criteria

Abort before production for any tracked modification outside `train.py`; seed, model, optimizer, decay, LR, timer, evaluator, N1/M7, weak transform, CutMix alpha/probability, batch size, worker count, or 80% boundary drift; a loader rebuild/reseed at 70%; use of Python/NumPy/CUDA policy RNG; a policy flag read more than once per collator call; missing provenance; or a new evaluation.

Also abort on shared-flag/forkserver failure; policy-on/off semantic mismatch; surrounding RNG mutation; corpus/hash mutation; forced or filtered evidence; non-finite or incomplete safety state; candidate-only concentration; loss-EMA ratio above 1.5; CutMix proportion outside range; transition propagation beyond 24 batches or 70.5%; any soft target after drain; worker replacement/growth/leak; weak rebuild failure; timing, exposure, memory, loader-headroom, or projected-wall gate miss; GPU contention; or stale logs.

Do not rescue an abort or miss by rebuilding the strong loader at 70%, moving mixing to the parent/GPU, draining selected batches, changing prefetch, moving the boundary to 65/75%, changing probability/alpha, adding label smoothing, extending the weak tail, altering a threshold, or rerolling the seed. Such changes are new experiments.

## Risk assessment

- **Scientific risk — medium-high:** the accepted recipe already ends at its best; adaptation debt is inferred from related over-regularized trajectories rather than demonstrated at p=0.5. Late CutMix may be beneficial, and the 70% boundary is not externally validated.
- **Attribution risk — medium:** the intervention jointly removes pasted pixels and soft labels. It isolates timing from RandAugment and avoids a source-loader reset, but cannot distinguish those two CutMix components.
- **Implementation risk — medium:** forkserver-safe synchronized state and bounded prefetch lag require careful provenance. The top-level callable, single snapshot, real-loader propagation gate, and fixed abort rule bound this risk.
- **Optimization risk — low-medium:** hard N1/M7 is less regularized than accepted CutMix and is already used indirectly whenever the 50% gate is hard; gross collapse is unlikely, but exact-corpus continuation still gates unexpected transients.
- **Runtime risk — low:** there is no new GPU work or 70% loader rebuild. The shared worker read should be prefetched, but alternating full-path timing protects fixed-budget exposure.
- **Estimated effort — medium:** production code is compact; most effort is in proving forkserver propagation, exact policy semantics, lifecycle cleanup, and full-path timing.

## Interpretation table

| Outcome | Interpretation | Required action |
|---|---|---|
| `best_test_acc >=94.25%`, all gates pass | Temporal CutMix removal is a valid fixed-budget improvement | Accept the commit; report trajectory and weak single-run causality honestly |
| Valid result below 94.25%, switch fit rises | Hard N1/M7 repairs fit but sacrifices useful late CutMix generalization | No-improvement; retire this 70% boundary without in-place tuning |
| Valid result below 94.25%, switch fit does not rise | Hard N1/M7 does not provide the predicted adaptation mechanism | No-improvement; discredit the mechanism more broadly |
| Gate/protocol/lifecycle failure | The proposed execution is not trustworthy | Invalid; do not run or rerun production with relaxed gates |
| Timeout/crash/incomplete summary | No valid metric exists | Crash/invalid according to failure mode; no rescue run |
