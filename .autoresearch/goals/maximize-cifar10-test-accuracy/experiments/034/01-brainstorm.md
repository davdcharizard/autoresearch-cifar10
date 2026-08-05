# Brainstorm EXP-034
**Created**: 2026-07-26

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): moderate depth and width can improve CIFAR compute efficiency, but the new batch operating point must be measured directly on the H20.
- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): symmetric input/target interpolation remains the strongest low-cost local regularizer; alpha changes its endpoint concentration.

No network or remote source was consulted. This offline quick pass used the existing knowledge base and the fully developed EXP033 proposals.

## Experimental History Review

- EXP027 remains the 94.32% accepted learner: `(2,2,3)` depth plus early worker-safe N1/M5 RandAugment and alpha-0.2 batch-shared mixup retained 133.007 passes and converted two standalone near-misses into a +0.25-point improvement.
- Generalization and top-1 boundary quality remain limiting. Exposure-only reallocations have repeatedly failed despite 159-172 passes, and EXP033 now closes immediate late averaging after its short parameter-only endpoint scored 93.87 at 132.47 passes.
- Batch 128 was infeasible because it retained under 90.22% image throughput, but batch 512 is a distinct utilization direction with 98.9% H20 memory headroom. It must earn its roughly 45% reduction in optimizer/BN/mixup decisions through at least 1.10x measured image rate and 146.308 projected passes.
- Alpha 0.1 is the only clean unmeasured side of batch-shared mixup strength on the accepted composition. Its attribution is strong, but alpha 0.4, 50%/75% duration, and per-example coefficient changes all regressed, so its prior and upside are weaker.
- Exact masking, attention, late SAM, averaging, padding, channels-last, BF16, freezing, decay/schedule changes, rank bottlenecks, and nearby capacity allocations are closed or infeasible in their tested forms. The remaining search is narrow enough to reuse the two adversarially developed EXP033 proposals rather than repeat broad ideation.

## Collected Ideas

## Combinations

## Candidate Ideas

### Weaker Alpha-0.1 Batch-Shared Mixup
**Summary**: Change only `MIXUP_ALPHA=0.2` to `0.1`, retaining the batch-shared coefficient, 65% cutoff, model, early RandAugment, optimizer, schedule, seed, and hard tail. This is a one-shot closure of the weaker strength bracket, not a monotonic inference from alpha 0.4.

**What it targets**: Generalization and boundary quality at identical graph shape and near-identical exposure, testing whether the accepted deeper-plus-RandAugment learner needs less interpolation severity.

**Reasoning**: Alpha 0.1 increases near-endpoint batches while preserving mean coefficient and batch-level coherence. It is the cleanest remaining one-line treatment, but the accepted composition shows no over-regularization symptom and every neighboring mixup perturbation has been negative. The full protocol is already developed in `experiments/033/proposals/idea-01.md`.

**Sources**: `knowledge/papers/mixup.md`; EXP002, EXP004, EXP005, EXP015, EXP020, EXP027; `experiments/033/proposals/idea-01.md`.

**Estimated Effort**: low

**Risk Assessment**: About 81% of alpha-0.1 draws lie near endpoints, so the intervention may under-regularize. Concentration-dependent Beta rejection also changes the later fixed-seed CUDA coefficient/permutation trajectory by design, and a single run cannot isolate that trajectory from average treatment effect.

### Batch 512 With a Fully Scaled LR Curve
**Summary**: Change exactly `BATCH_SIZE=512`, `LR=0.4`, `MIN_LR=0.004`, and image-equivalent `MAX_STEPS=32000`, preserving every other accepted component. Treat it as one indivisible large-batch optimizer operating point and score only if balanced complete-body timing demonstrates at least 1.10x image rate and 146.308096 projected passes.

**What it targets**: The 98% forward/backward counted-time bottleneck and the H20's 98.9% memory headroom, seeking higher full-model image exposure without removing capacity or high-resolution gradients.

**Reasoning**: Linear LR scaling approximately preserves cumulative first-order displacement per image, while larger convolution batches may improve accelerator utilization. The treatment intentionally changes update, BN, mixup-decision, dropped-example, and epoch cadence; local evidence that extra exposure alone fails justifies the unusually strict feasibility gate. The full protocol is already developed in `experiments/033/proposals/idea-02.md`.

**Sources**: `02-system-understanding.md`; `project-notes/project-insights.md`; EXP009, EXP016, EXP028, EXP029; `experiments/033/proposals/idea-02.md`.

**Estimated Effort**: medium

**Risk Assessment**: Even at the gate the learner makes roughly 14.3k updates versus 26k accepted, each batch-shared coefficient covers twice as many examples, BN statistics change, and late hard-label refinement is coarser. A speed pass is feasibility evidence, not accuracy evidence.

## Review

The offline adversarial reviewer selected batch 512 for its larger objective-facing upside and direct test of unused H20 utilization. I adopted its material refinements: the throughput gate proves feasibility rather than optimizer equivalence; memory headroom and passes are not accuracy evidence; active/inactive loader timing must balance fresh accepted and candidate controls in the same session; and extra evaluation opportunities, dropped examples, final accuracy/loss, and best-final gap must be reported explicitly. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. Batch 512 scored 3/5 for evidence and 3.5/5 for impact, ahead of alpha 0.1's 2/5 and 2/5. Its strict `>=1.10x` image-rate gate makes the systems upside meaningfully testable before accepting the large change in optimizer, BN, and mixup decision cadence. The experiment remains an indivisible fixed-seed operating-point test rather than a claim of large-batch equivalence.

## Chosen Idea
**Selected**: Batch 512 With a Fully Scaled LR Curve

**Why this idea**:
Change exactly batch size to 512, peak/floor LR to `0.4/0.004`, and the image-equivalent safety cap to 32,000 while preserving the accepted model, full gradients, augmentation, schedule shape, decay, momentum, seed, and evaluator. This uses the H20's idle memory to test the largest remaining utilization lever, but proceeds to a score only if balanced complete-body timing earns at least 1.10x image rate and 146.308096 projected passes with stable contemporaneous loader delivery.

**Hypothesis**:
If batch 512 improves complete-body H20 image rate by at least 10% and the linearly doubled LR curve preserves useful optimization despite roughly 45% fewer Nesterov/BN/mixup decisions, then the sole seed-42 run will realize at least 146.308096 passes and raise `best_test_acc` from 94.32% to at least 94.42%. A stable feasibility miss or a valid score below 94.42% closes this exact `512, 0.4 -> 0.004, 32000-cap` point without adjacent repair.
