# Proposal: Fixed RandAugment Before CutMix

## Summary

Add torchvision's canonical fixed RandAugment transform to every EXP-004 training image while preserving the full parent crop/flip, CutMix, model, optimizer, and clean-tail period-two SAM recipe. Use one preregistered policy with no held-out search, scalar sweep, per-phase tuning, or metric-driven retry:

```python
transforms.RandAugment(
    num_ops=2,
    magnitude=9,
    num_magnitude_bins=31,
    interpolation=transforms.InterpolationMode.NEAREST,
    fill=0,
)
```

These are the installed torchvision 0.24.1 defaults made explicit. Each image receives two uniformly sampled operations with replacement from torchvision's standard RandAugment space, with the shared magnitude index 9 and the implementation's random sign handling. The policy is fixed before execution and does not use the CIFAR-10 test set or a held-out policy search.

## Evidence and Effect Prior

RandAugment reduces automated augmentation to operation count and shared magnitude and reports competitive CIFAR-10 results across WRN-28-2, WRN-28-10, Shake-Shake, and PyramidNet. Its direct CIFAR/WRN evidence gives it a larger prior than recent optimizer substitutions, but the paper selected policy scalars on held-out data and its default pipelines include crop, flip, and Cutout. This proposal transfers only the frozen torchvision `(N=2, M=9)` recipe; it does not claim that those values are optimal for EXP-004.

The system needs a plausible effect near 0.3 points because selected-run and late-tail variation has reached 0.14-0.29 points. EXP-005 lost accuracy after halving new-image introduction, EXP-006 gained only 0.01 by substituting manifold mixup for validated CutMix, and EXP-007's literature-scale ASAM package regressed 0.06. RandAugment retains all 256 independent identities per step and adds image-level invariances rather than exchanging an accepted mechanism.

After discounting longer-schedule and weaker-baseline paper results, the preregistered effect prior is +0.30 to +0.60 points, predicting 95.70-96.00% from the 95.40% parent. Formal success still requires only `best_test_acc >= 95.50%`; the 95.70 lower effect target distinguishes a mechanism-sized gain from threshold-scale noise.

Sources:

- `experiments/008/papers/randaugment.md`
- `02-system-understanding.md`
- `03-experiment-learnings.md`
- `experiments/004/04-analysis.md`
- `experiments/005/04-analysis.md`
- `experiments/006/04-analysis.md`
- `experiments/007/04-analysis.md`
- https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html

## Exact Transform Order

Change only the training transform in `train.py`:

```python
train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        IsolatedRandAugment(...fixed arguments above...),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
```

RandAugment receives the already cropped/flipped 32x32 PIL image and performs color/geometric operations in raw pixel space. `fill=0` matches the existing crop-padding fill. Tensor conversion and the parent's `(0.4914, 0.4822, 0.4465)` mean / unit-std normalization occur afterward. The test transform in read-only `prepare.py` remains unchanged.

CutMix remains a GPU-side operation after transfer, so selected early batches mix independently RandAugmented, normalized images with the exact existing rectangle, permutation, target pairing, and clipped-area label weight. RandAugment never sees a CutMix composite. During the final quarter, CutMix remains off and period-two SAM uses hard labels; both SAM forwards reuse the same already RandAugmented batch, so no operation or magnitude is redrawn between the unperturbed and perturbed passes.

## Dedicated RNG and Reproducibility

Torchvision `RandAugment` samples from the process-global CPU torch generator and has no generator argument. Calling it naively would consume each DataLoader worker's crop/flip stream and shift subsequent parent augmentations. Wrap the unmodified torchvision transform in `IsolatedRandAugment`:

1. Lazily create one private CPU `torch.Generator` in each worker process.
2. Seed it deterministically from that epoch's `torch.utils.data.get_worker_info().seed` combined with fixed namespace `RAND_AUGMENT_SEED=42`; for `num_workers=0`, use the fixed namespace directly.
3. Before calling the torchvision transform, save the worker-global CPU RNG state and replace it with the private generator state.
4. In a `finally` block, save the advanced RandAugment state back into the private generator and restore the worker-global state exactly.

Workers are recreated by the current non-persistent DataLoader each epoch. Their deterministic parent seeds therefore produce deterministic but epoch-varying private RandAugment streams. The wrapper must not touch Python `random`, NumPy, main-process CPU RNG, or CUDA RNG. Crop/flip and shuffle streams remain aligned with EXP-004 over a shared step prefix; dedicated CutMix CPU/CUDA generators and the global CUDA drop-path/SAM-replay stream remain unchanged.

The seed namespace is isolation, not seed selection. Do not try alternate namespace values or use the result to choose one.

## Additive Versus Substitutive Interpretation

This is meaningfully additive in exposure and mechanism:

- no crop/flip, clean batch, CutMix-selected batch, or SAM pulse is removed;
- the DataLoader still introduces 49,920 independent identities per dropped-last epoch;
- CutMix's gate and area-weighted mixed-label objective remain at full dose;
- SAM's clean-label, final-quarter, period-two two-pass objective remains at full dose;
- RandAugment adds per-image photometric/geometric invariance without another model forward.

It is not an isolation of RandAugment on otherwise identical pixels: all parent paths now consume policy-transformed images, including the source images CutMix combines and the late images SAM sharpens around. "Clean" in the parent means no CutMix, not unaugmented; the late tail already retained crop/flip, and this proposal keeps RandAugment active there as well. A result therefore evaluates the full additive RandAugment+CutMix+SAM package, including possible over-regularization, rather than a standalone RandAugment effect.

## CPU Cost and Charged Exposure

RandAugment runs in eight DataLoader workers on PIL images before `ToTensor`. It adds no GPU model work, parameters, activations, or forward pass. The current timer starts inside the loop body after the next batch has been yielded, so worker wait is mostly outside `training_seconds`; prefetch can overlap it with GPU work. A slow policy may preserve the nominal 300 charged GPU seconds and step count while increasing real end-to-end time. That must be measured and reported rather than treated as free compute.

EXP-004 processed 25,560 steps, about 21,800 image appearances per charged second after accounting for SAM. The fixed policy is expected to preserve approximately 24,500-25,560 steps if eight workers keep the queue fed, but PIL geometric operations may become the bottleneck. Before the metric run, compare parent and candidate DataLoaders in separate fixed-seed processes:

- warm at least 20 batches, then time at least five full 195-batch epochs;
- report median/p90 batch inter-arrival, images/second, and epoch wall time;
- run an integrated 500-step GPU-0 smoke with the production ordinary/SAM cadence and report charged versus end-to-end seconds;
- confirm identical sample count, shapes, labels, pinning, worker count, and no worker crash.

Proceed only if candidate loader throughput is at least 22,000 images/second, integrated total/charged overhead projects total runtime below 600 seconds, and projected optimizer exposure is at least 24,000 steps. Do not change worker count, `N`, `M`, interpolation, or operation set if the fixed policy fails this feasibility gate.

## Parent Preservation

- Keep `BATCH_SIZE=256`, `NUM_WORKERS=8`, shuffle, `drop_last=True`, global seed 42, model architecture/initialization, BF16, and channels-last.
- Keep `CUTMIX_PROB=0.5`, `CUTMIX_ALPHA=1.0`, `CUTMIX_END=0.75`, `CUTMIX_SEED=42`, helper geometry, and dedicated generator draw order.
- Keep `MAX_DROP_PATH=0.08`, LR warmup/cosine, SGD/Nesterov, weight decay, and all time-based boundaries.
- Keep `SAM_RHO=0.05`, `SAM_START=0.75`, `SAM_PERIOD=2`, CUDA RNG replay, second-pass BatchNorm suppression, exact snapshots/restoration, and one optimizer update.
- Keep evaluator, once-per-epoch validation, metric accumulation, timer boundary, parameter count, and required final summary keys unchanged.

Add only fixed policy constants, the isolated wrapper, the transform entry, and a startup line identifying torchvision version and the complete policy. No package or dependency is added.

## Smokes and Audit

1. **Policy lock:** Assert the production object reports `num_ops=2`, `magnitude=9`, `num_magnitude_bins=31`, nearest interpolation, and zero fill under installed torchvision 0.24.1; reject incomplete/default-only config logging.
2. **Ordering/type:** Instrument sentinel transforms to require RandAugment receives a 32x32 PIL image after crop/flip and returns a PIL image before `ToTensor`; verify normalization follows it and test transforms are untouched.
3. **Private-stream replay:** From fixed worker seeds, require byte-identical output sequences across two runs. Different successive samples should not all receive identical operations.
4. **RNG isolation:** Snapshot worker-global torch, Python, and NumPy RNG states around the wrapper and require equality afterward while the private state advances. Compare parent/candidate crop/flip outputs from matched states to prove RandAugment does not shift them. Require unchanged main CPU and CUDA states.
5. **Transform validity:** Exercise representative RGB images through many calls; require size/mode preservation, finite tensor values, nontrivial changed pixels, and no invalid magnitude/fill behavior at image boundaries.
6. **CutMix integration:** On source-coded policy outputs, apply fixed CutMix lambda/center/permutation and verify pristine paired patch orientation, target pairing, clipped-area lambda, and no additional RandAugment call after mixing.
7. **SAM integration:** Run one late BF16/channels-last batch and require both passes use the same tensor, finite losses/gradients, replayed drop masks, one BatchNorm update, exact parameter restoration, and one Nesterov update.
8. **Parent invariants:** Confirm `num_params=2,748,890`, identical model state from seed 42, unchanged CutMix/SAM static code, 195 batches/epoch, and only `train.py` differs from EXP-004.
9. **CPU feasibility:** Pass the loader and integrated cost gate above before any metric run.

Production audit should log the complete policy, fixed seed namespace, worker count, measured preflight loader throughput, and unchanged CutMix/SAM counters. No test-set-dependent policy choice or rerun is allowed.

## Full Verification and Falsification

Run exactly once after confirming physical GPU 0 is the 97,871 MiB NVIDIA H20:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Require exit 0, 299.5-301.0 charged seconds, total time below 600 seconds, at least 24,000 steps, unchanged 2,748,890 parameters, all required summary keys exactly once, one evaluation per completed epoch, CutMix ratio near 0.5 only below progress 0.75, period-two SAM only at/above 0.75, and no worker exception, NaN/Inf, traceback, CUDA error, OOM, or timeout.

The proposal is falsified if the CPU feasibility gate fails, reproducibility/isolation cannot be demonstrated, total runtime exceeds 600 seconds, exposure falls below 24,000, or `best_test_acc < 95.50%`. A valid 95.50-95.69 result is a formal improvement but below the preregistered 95.70 mechanism-sized target. No alternate magnitude, operation count, interpolation, policy phase, seed, or worker count may be selected from the outcome.

## Risks

- **Over-regularization:** Crop/flip, RandAugment, early CutMix, drop path, and late SAM may be redundant. RandAugment remains active in the nominally clean tail and may prevent low-LR fitting.
- **Policy mismatch:** The paper searched `N` and `M`; torchvision defaults are fixed for no-search integrity, not validated as optimal for this exact WRN or unit-std normalization.
- **CutMix composition:** Geometric/color transforms can make both source images harder before spatial mixing. Full CutMix exposure is preserved, but the combined image may be unnaturally strong.
- **CPU starvation hidden by timing:** PIL operations occur before the current charged boundary. They can inflate total runtime without reducing reported training seconds; the integrated preflight and 600-second cap are mandatory.
- **Worker RNG mistakes:** A naive transform would shift crop/flip streams; identical per-worker private seeds could repeat policies across workers/epochs. Worker-seed derivation and state-isolation smokes are required.
- **Run noise:** A threshold-scale gain can sit inside known 0.14-0.29-point variation. The 95.70 effect target and single fixed run prevent post-hoc magnification.

## Testable Hypothesis

The fixed torchvision `(N=2, M=9)` RandAugment policy will add useful photometric/geometric invariance while retaining full independent-image, CutMix, and SAM exposure, complete at least 24,000 optimizer steps within the 600-second outer limit, and improve EXP-004 by 0.30-0.60 points to 95.70-96.00%. Any result below 95.50% or any cost/reproducibility/protocol failure rejects this package without a search or retry.
