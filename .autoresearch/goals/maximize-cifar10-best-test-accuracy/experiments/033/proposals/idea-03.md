# Proposal: Conservative Small-Area Random Erasing Composition

## Intervention and falsifiable hypothesis

Compose parameter-free Random Erasing with the accepted strong-view policy using
one preregistered point:

```text
probability: 0.25 per source image
area scale:  0.02 to 0.10 of the 32x32 image
aspect ratio: 0.3 to 3.3 (torchvision default, log-uniform sampling)
fill: CIFAR channel mean
inplace: false
```

Keep accepted N1/M7 RandAugment and probability-0.5 alpha-1 CutMix through the
entire 80% high-LR strong phase. Keep the weak crop/flip hard-label tail free of
both erasing and CutMix. Preserve width-2 postactivation ResNet-20,
FP32/default-TF32, batch 128, ordinary momentum SGD, all-parameter decay `1e-4`,
LR schedule, seed 42, timer, evaluator, and worker lifecycle.

At p=0.25 with mean requested area 6%, only about 1.5% of source pixels are erased
on average. That is 16.7x less deletion than EXP-006's 25%-area mask on every
image. The hypothesis is that sparse mean-filled local absence complements
RandAugment's broad invariance and CutMix's class-bearing regions without crossing
the local strong-underfit boundary, raising `best_test_acc` from 94.15% to at
least 94.25%. Point prediction: switch accuracy at least 89.0%, first weak at
least 93.16%, final NLL no worse than 0.195, and at least 99% accepted exposure.

This is one fixed policy, not a sweep. A valid miss retires the exact
p=0.25/2-10% composition; no probability, area, ratio, fill, or placement rescue
is allowed inside EXP-033.

## Mechanism and local evidence

Random Erasing trains robustness to missing local evidence and is reported as
complementary to crop/flip augmentation. The local case is narrower:

- EXP-004 showed worker-side N1/M7 plus a weak tail improved accuracy without
  costing updates.
- EXP-006 replaced N1/M7 with an every-view 16x16 mean patch and lost 0.67 point;
  its 25% average deletion and replacement of broad augmentation do not test this
  much milder composition.
- EXP-010 showed CutMix adds useful labeled regional occlusion, gaining 0.60 point
  at p=0.50. Random Erasing is different: its mean-filled region has no donor
  label, so it encourages tolerance to absence rather than localization of a
  second class.
- EXP-011 and EXP-026 warn that extra strong-phase regularization can lower switch
  fit, while EXP-027 shows the complete accepted N1/M7+CutMix phase must remain
  intact until 80%. The candidate therefore adds only a sparse per-image mask and
  changes no phase boundary or target frequency.

The main scientific risk is redundancy: CutMix already supplies regional
occlusion with class-bearing pixels, and an unlabeled erased patch can only add
target-content mismatch. The width-2 model may not have enough short-horizon fit
to absorb even 1.5% average deletion.

## Exact placement and pixel semantics

Random Erasing requires a tensor, so the strong transform order is exactly:

```python
RandomCrop(32, padding=4)
RandomHorizontalFlip()
RandAugment(num_ops=1, magnitude=7)
ToTensor()
RNGNeutralRandomErasing(
    p=0.25,
    scale=(0.02, 0.10),
    ratio=(0.3, 3.3),
    value=mean,
    inplace=False,
)
Normalize(mean, std)
```

The accepted `std=(1,1,1)` means filling with `mean` **before** normalization
creates exact normalized zeros, matching EXP-006's benign mean-fill semantics
rather than an out-of-distribution black patch. Erasing follows N1/M7 so its
requested 2-10% refers to final crop coordinates. It precedes CutMix because
CutMix remains the batch collator after normalization: CutMix may paste over an
erased recipient region or paste an erased donor region, but retains its ordinary
area-weighted targets. Random Erasing itself never changes a label.

Implement the wrapper as a top-level forkserver-picklable callable. Execute the
torchvision `RandomErasing` call inside
`torch.random.fork_rng(devices=[])`. The erasing draws therefore vary with each
worker's current seed/state but are restored afterward, so they do not shift later
crop/flip/RandAugment or CutMix source streams. Draw no CUDA RNG and introduce no
secondary tunable seed.

For production provenance, extend only the strong collator to inspect normalized
inputs **before** CutMix. A pixel whose three channels are exactly zero is an
erased pixel; the declared mean cannot occur exactly in ordinary 8-bit source
pixels. Return two CPU scalar metadata fields with each strong batch:
`erased_examples` and `erased_pixels`. The main loop aggregates them before `t0`
and prints totals at the switch. Metadata never reaches CUDA or changes targets.
Weak batches retain the accepted two-field form and must report no erasing.

Do not place erasing after CutMix, after normalization with a black fill, in the
weak tail, or before RandAugment; do not use random-pixel fill, per-batch gating,
in-place mutation, or target-area correction.

## Policy, RNG, and immutable-corpus gates

Before any scored run, require:

1. Static checks prove the exact policy/order, unchanged weak/eval transforms,
   unchanged CutMix alpha/probability, one top-level picklable wrapper, no new
   model/optimizer state, and exactly 1,073,962 parameters.
2. On controlled tensors and RNG states, require no-op outputs when the probability
   gate fails; outside-mask pixels bitwise unchanged; inside-mask raw pixels equal
   `mean` and normalized pixels equal zero; labels unchanged; sampled requested
   area in `[0.02,0.10]`; sampled aspect ratio in `[0.3,3.3]`; and identical global
   CPU/CUDA RNG state before/after. A repeated saved state must reproduce the same
   mask, and subsequent accepted transforms must remain bitwise aligned.
3. Materialize and hash 200 exact post-N1/M7, post-ToTensor, pre-erasing source
   batches plus 64 weak batches. Register every candidate gate/mask and the same
   100 hard/100 CutMix decisions, permutations, boxes, and targets for both arms.
   Control normalizes the original sources; candidate erases then normalizes;
   CutMix is applied afterward with identical registered geometry. Persist source,
   output, target, provenance mask, and SHA-256 before training.
4. Across the registered strong corpus require 23-27% erased source examples,
   at least 99.5% placement success among selected examples, requested areas and
   ratios in range, conditional achieved area in `[0.015,0.105]` after integer
   rounding, conditional mean area 4.5-7.5%, and unconditional mean erased area
   1.1-1.9%. Propagate masks through CutMix and require final effective erased area
   no greater than 20% for any image. Hard labels remain integers; CutMix targets
   sum to one and depend only on CutMix box area.
5. Replay all 264 batches from independently restored accepted/candidate
   model/SGD state. Require finite logits, losses, gradients, parameters, momentum,
   and BN buffers; no candidate-only >95% class concentration; candidate/control
   loss, logit-RMS, gradient-norm, and update-norm ratios `<=1.5`; positive running
   variances; terminal strong and weak loss EMAs `<=1.25x` control; and no corpus,
   target, RNG, or BN-counter divergence. Serialize the offending mask/histogram
   before any assertion.

Exact post-transform source persistence is mandatory after EXP-019/021. A safety
failure invalidates this exact policy; lower loss cannot override concentration,
geometry, or update gates.

## Live worker throughput and lifecycle gates

Run an instrumented strong loader for at least 5,000 delivered batches using all
eight production forkserver workers, followed by the ordinary weak-loader rebuild.
Require:

- 23.5-26.5% erased examples, conditional/unconditional area statistics inside
  the corpus bounds, 48-52% CutMix batches, valid metadata/targets, and deliveries
  from every worker;
- candidate warmed throughput at least 80% of paired accepted throughput and at
  least 140 batches/s, comfortably above roughly 90-batch/s GPU consumption;
- median iterator wait `<=0.5 ms`, p95 `<=1.5 ms`, no repeated starvation spike,
  clean shutdown of all eight workers, and weak-loader rebuild below 5 seconds;
- weak batches remain hard, contain no normalized-zero rectangle attributable to
  erasing, and use the accepted crop/flip transform only.

The per-image RNG fork and two provenance scalars are real host work. Do not remove
them after benchmarking or move erasing to GPU as a throughput rescue.

## Paired full-step timing and exposure gate

On one idle 97,871-MiB H20, run five counterbalanced fresh-process
control/candidate pairs with real production loaders. After 100 warmups, measure
at least 1,000 synchronized steps per arm, separating strong-hard, strong-CutMix,
and weak-hard paths and weighting them 40/40/20. Include iterator wait,
nonblocking transfer, forward, loss, backward, SGD, metadata handling, and final
synchronization. Separately charge the one-time loader transition over the full
production horizon rather than concentrating it into the short probe.

Proceed only if:

- weighted candidate/control mean step time `<=1.01`, every pair `<=1.03`, both
  trial-mean CVs `<=2%`, and candidate p95 `<=1.04x` control mean;
- `floor(26,898 * control_mean / candidate_mean) >=26,629` projected updates
  (99% retention) with the accepted 80/20 time split;
- peak allocation remains below 650 MiB, all state stays finite, and projected
  total runtime including unchanged evaluations/transition remains below 540s.

Random Erasing has no GPU-speed mechanism; it must be effectively exposure-neutral.
Do not change workers, prefetching, pinning, batch size, precision, memory format,
or evaluation cadence to rescue timing.

## Production verification and verdict

If every gate passes, run the exact candidate once at seed 42 with
`uv run train.py > run.log 2>&1` on the sole idle H20. Require exit zero, finite
standard summary fields, 300.0 counted seconds, total below 600 seconds, at least
26,629 updates, exactly 1,073,962 parameters, one 80% switch with eight workers
stopped, 45-55% CutMix, hard weak-tail targets, and no duplicate evaluation epoch.

Production provenance must show 23.5-26.5% erased strong source examples,
conditional mean erased area 4.5-7.5%, unconditional area 1.1-1.9%, and zero weak
erased examples. Record 20/40/60/70% and switch accuracy/loss, switch accuracy
versus 89.73%, first weak versus 93.16%, peak/final accuracy, final NLL versus
0.1934, steps, epochs, evaluations, runtime, VRAM, erase counts/area, and CutMix
counts. A switch below 87.08% is the registered compounded-underfit diagnosis but
cannot stop or rerun production.

Accept only if all integrity/runtime conditions pass and
`best_test_acc >=94.25%`. A complete lower result is `no-improvement`; a safety,
policy, throughput, or timing failure is `invalid`; a failed process is `crash`.
A higher switch with worse peak/NLL means extra hard-view fit cost useful
occlusion invariance. A lower switch implicates compounded distortion. Healthy
switch/tail with unchanged accuracy means the mild erasing effect is below this
protocol's resolution.

No p/area/ratio/fill tuning, weak-tail extension, CutMix adjustment, reroll, or
same-experiment combination is allowed.

## Risks and evidence limits

- CutMix already supplies labeled regional occlusion; mean-filled absence may be
  redundant and only deepen strong-phase underfit.
- Erasing before CutMix can be partly overwritten or transferred from donors, and
  its label is intentionally not area-adjusted. Provenance masks describe the
  compound geometry but cannot isolate which component drove accuracy.
- RNG-neutral per-image forking preserves later source draws but adds host cost;
  worker headroom must be demonstrated rather than assumed.
- Integer 32x32 rectangles make achieved area/aspect ratios approximate even when
  sampled requests are in range.
- The likely effect is near one-seed noise: 0.10 point is ten test examples, so a
  bare pass is protocol-valid but weak causal evidence.

## Sources

- Zhong et al., *Random Erasing Data Augmentation*, AAAI 2020 / arXiv 1708.04896:
  https://arxiv.org/abs/1708.04896.
- Torchvision `RandomErasing` documentation:
  https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html.
- `experiments/004/04-analysis.md`, `experiments/006/04-analysis.md`, and
  `experiments/010/04-analysis.md` — augmentation, Cutout, and accepted CutMix
  anchors.
- `experiments/011/04-analysis.md`, `experiments/026/04-analysis.md`, and
  `experiments/027/04-analysis.md` — compounded strong-fit risks.
- `experiments/029/04-analysis.md` — small helper overhead precedent.
- `experiments/031/04-analysis.md` — per-example geometry safety precedent.
- `experiments/033/01-brainstorm.md`, `02-system-understanding.md`,
  `03-experiment-learnings.md`, and `04-results.tsv` — current context.
