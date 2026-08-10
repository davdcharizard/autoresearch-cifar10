# Proposal: Small Final-Classifier Initialization Only

## Decision and hypothesis

Change only the initialization of the existing final `Linear(128,10)` classifier: draw its weight from `Normal(0,0.01)` and set its bias to exactly zero. Keep all 19 Conv2d initializations on the accepted default fan-in Kaiming-normal rule and preserve every other parameter, buffer, graph, optimizer, data, schedule, evaluator, seed, and runtime mechanic.

```python
@staticmethod
def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, mean=0.0, std=0.01)
        init.zeros_(m.bias)
```

The accepted classifier's Kaiming fan-in standard deviation is `sqrt(2/128)=0.125`; the candidate weight scale is therefore 0.08x accepted, with no random per-class bias. The hypothesis is that smaller, more symmetric initial logits reduce early confident class geometry and briefly temper backbone gradients while the classifier learns from the accepted N1/M7/CutMix objective. This may act as an initialization-only representation warmup without changing the later optimizer or consuming fixed-budget steps. Point prediction is 94.25%; formal success is `best_test_acc >=94.25%` versus EXP010's 94.15% frontier.

## Mechanism and limits

On fixed pooled features `h`, the candidate begins with `z_c=W_c h` rather than accepted `z_a=W_a h+b_a`. If both normal initializers consume the same standard-normal stream as expected, `W_c` is 0.08x `W_a`; removing accepted random bias further centers the class logits. Cross-entropy probabilities should begin closer to uniform and initial NLL closer to `log(10)`, reducing accidental class confidence.

The effect on gradients is asymmetric. The classifier gradient `(p-y)^T h/B` need not shrink with the classifier weights, while the backbone signal `W^T(p-y)` should shrink sharply at the first step. At global LR 0.1, the first classifier update can consequently be several times larger than the candidate's small initial weight norm even if its absolute update is ordinary. The classifier may rapidly grow toward the accepted scale; useful impact, if any, is a short early change in representation learning, not a permanent low-logit constraint.

This is also a two-part classifier initialization package: weight scale and zero bias change together exactly as specified. It does not isolate bias zero. Coupled all-parameter decay remains `1e-4`, so the smaller classifier has proportionally smaller initial absolute decay displacement; that is part of the net initialization effect.

## Evidence

The initialization sources summarized in the EXP033 brainstorm support variance-aware initialization but do not directly establish this classifier point. He-style Kaiming initialization motivates the accepted Conv rule; PyTorch documents fan-in as forward-variance preserving, while torchvision's ResNet precedent concerns Conv fan-out rather than a `std=0.01` CIFAR classifier. The candidate is therefore a plausible common small-logit convention with weak direct transfer evidence, not a paper-backed expected win.

Local evidence makes the safety burden unusually high:

- EXP010 proves the full accepted training recipe and its 89.73% switch fit; it is the only baseline to displace.
- EXP014 began with accepted logits/backbone gradients but its new branch had a 4.10x classifier gradient, and one update caused complete class collapse. Initial output continuity alone was insufficient.
- EXP024 concentrated 98.44% of predictions at step 2 despite lower loss; low loss cannot waive an output-geometry veto.
- EXP025 passed first-update continuity but its gates saturated by step 19; multi-step recruitment matters more than a single benign update.
- EXP031's initialization-scale bound drifted within three steps to class concentration and 1.58x updates. A fixed initialization statistic does not control the trained trajectory.

Unlike those candidates, this proposal adds no branch, gate, pooling statistic, parameter, or operator. Its risk is concentrated in early classifier/backbone update geometry and can be tested on exact production-distribution batches.

## Exact scope and RNG invariants

Split the accepted combined `Conv2d/Linear` initializer only as shown. There is exactly one Linear module. Do not change the `nn.Linear` constructor, module registration order, Conv branch, BN defaults, Kaiming Conv arguments, classifier shape, forward, loss, or optimizer group.

The default Linear constructor still consumes its ordinary weight/bias draws before `_weights_init`; zeroing bias afterward consumes no RNG. `kaiming_normal_` on accepted FC and `normal_(std=0.01)` on candidate FC both normally fill the same 1,280 elements from a normal stream. Verify rather than assume:

1. Reset seed 42 and construct accepted/candidate CPU models independently.
2. Require every Conv weight, every BN tensor/buffer, and all non-FC state to be bitwise identical.
3. Require candidate FC weight shape `[10,128]`, empirical finite values, exact zero bias, and total parameters unchanged at 1,073,962.
4. Require identical post-construction CPU/CUDA RNG-state hashes. If draw alignment holds, require `W_candidate` to equal `0.08 * W_accepted` within FP32 rounding tolerance; otherwise use a saved pre-FC RNG-state oracle to prove both rules consumed the same draw count without changing the intended distributions.
5. Require every parameter appears exactly once in the unchanged SGD group with LR 0.1, momentum 0.9, and coupled decay `1e-4`.

The identical post-construction RNG state is mandatory because workers are spawned only when the loader is iterated after model construction; RNG drift could change shuffling/augmentation and confound the initialization effect. Do not wrap candidate FC initialization in an RNG fork, redraw Conv weights, restore an earlier state, or compensate with a seed offset.

## Early logit, gradient, and update geometry

Use disposable diagnostics on persisted production-distribution training batches; do not add per-step production instrumentation.

On identical initial hard and probability-target batches, record centered-logit RMS, max absolute logit, entropy, loss, predicted-class histogram, FC/backbone gradient norms, per-layer gradients, parameter norms, and installed-SGD updates. Require:

- finite state and logits `[128,10]` on both target paths;
- candidate centered-logit RMS between 0.04x and 0.15x accepted weight-only centered-logit RMS, consistent with the intended 0.08 scale rather than a draw/scope error;
- candidate mean prediction entropy greater than accepted and no >95% one-class concentration;
- nonzero finite classifier and backbone gradients, with first candidate/control backbone-gradient norm in `[0.02,0.50]` and classifier-gradient norm in `[0.50,1.50]`;
- candidate first total-update norm no more than 1.25x control, FC update norm no more than 1.50x control, and `||delta W_fc||/||W_fc,0|| <=5.0`;
- after the first update, finite same-batch loss no more than 2x its pre-update value and no candidate-only >95% class share.

The backbone-gradient lower bound rejects a nearly disconnected representation, while the update bounds explicitly reject an EXP014-style empty-head overwrite. These are mechanism/safety gates, not accuracy proxies. If a bound fails, do not enlarge weights, lower classifier LR, clip gradients, or add warmup as a rescue.

## Exact-corpus trajectory gate

Materialize and hash one immutable corpus before either arm: 200 exact post-N1/M7 strong batches balanced near 50/50 hard and resolved alpha-1 CutMix targets, followed by 64 exact weak hard batches. Shut down all eight workers. Fresh accepted/candidate processes must begin from their reviewed model states, identical optimizer/backend/RNG states after construction, and replay byte-identical tensors in byte-identical order. No CIFAR-10 test evaluation is allowed.

Serialize every step's loss, class histogram, FC weight/bias norm, centered-logit RMS/entropy, FC/backbone gradient norm, total update norm, BN counters, and finiteness before assertions. Require:

- no corpus mutation, skips, RNG drift, BN-counter mismatch, or nonfinite parameter/buffer/momentum;
- zero candidate-only prediction shares above 95%;
- candidate/control total-update median <=1.20, p95 <=1.30, and maximum <=1.50;
- candidate/control FC-update p95 <=1.50 and maximum <=2.0;
- strong and weak terminal loss-EMA ratios <=1.15;
- candidate FC norm after step 1 <=0.50x control and finite/nonzero thereafter, proving the small initialization is not overwritten into a dominating head immediately;
- candidate backbone gradients remain nonzero through the corpus and no layer remains exactly frozen for ten consecutive steps.

Passing the first step does not waive this gate; EXP025 and EXP031 show that failures can emerge after apparently controlled initialization. A candidate that becomes numerically close to control later may still have executed the intended early warmup and is not invalid for that reason.

## Zero-overhead feasibility

No fresh paired timing gate is scientifically necessary. Candidate and control have identical parameter shapes/count, forward/backward operators, memory accesses after construction, loss, optimizer, and logging. Only starting tensor values differ; initialization runs before the counted timer. The method therefore adds zero recurring kernels, allocations, loader work, or evaluator work, unlike EXP029's gradient helper.

Verify this claim statically by comparing FX/operator traces or profiler operator names/counts for one hard and one soft step, and require identical peak tensor shapes and optimizer state size. Do not use tiny runtime fluctuations to select or reject the candidate. Production must nevertheless retain at least 26,629 steps (99% of EXP010's 26,898), stay near 598.7 MiB, and finish below 600 seconds; a material exposure loss indicates environment or unintended code drift, not a legitimate initialization cost.

## Expected production diagnostics

After every invariant and exact-corpus gate passes, confirm one idle 97,871-MiB H20 and run exactly once at seed 42 with `uv run train.py > run.log 2>&1`; no reroll.

Require exit zero, ten unique finite summary fields, 300.0 counted seconds, total below 600 seconds, 1,073,962 parameters, at least 26,629 steps, one 80% transition, all eight strong workers stopped, 45-55% CutMix among strong batches, hard weak targets, and 18-19 unique evaluations including terminal and never exceeding EXP010's 19.

Interpret trajectory against EXP010:

- **Early/strong fit:** smaller initial backbone gradients may delay optimization. The candidate should recover enough to keep the switch above the preregistered 87.08% underfit marker and preferably near 89.73%. A low switch is evidence that the short horizon could not repay the classifier warmup.
- **Weak conversion:** compare first weak accuracy to 93.16%. A representation benefit should survive the unchanged data/LR quench rather than only lower initial loss.
- **Generalization:** compare final/best accuracy to 94.15%, final NLL to 0.1934, best epoch, and best-final gap. Lower training loss or higher entropy cannot override worse test performance.
- **Mechanism:** report initialization norms/std, zero bias, initial logit/entropy/gradient/update ratios, exact-corpus FC growth, and production exposure. Do not inspect test performance to revise any initialization value.

The likely positive mechanism is reduced early class-confidence bias with gradual backbone recruitment. Principal negative outcomes are early classifier overwrite/class collapse, prolonged backbone undertraining, or an effect too transient to clear a ten-image gate. BatchNorm does not protect the final unnormalized classifier, and a 0.01 standard deviation is not variance-matched to the 128-wide ReLU descriptor.

## Verdict and no-rescue rules

- **Improvement:** all protocol/integrity conditions pass and `best_test_acc >=94.25%`. Accept the exact `Normal(0,0.01)`/zero-bias classifier package. Healthy geometry and NLL strengthen attribution but do not redefine the metric.
- **No improvement:** a valid production run is below 94.25%. Revert without reroll even if early loss, switch accuracy, NLL, or final accuracy individually improve.
- **Invalid/no-go:** RNG/Conv identity, geometry, exact-corpus, scope, hardware, summary, lifecycle, evaluator-count, finiteness, exposure, or wall-limit failure. Fix only an independently demonstrated controller or literal implementation defect while retaining the exact package.

Do not rescue with std 0.005/0.02, bias-only zeroing, a separate classifier LR, warmup, frozen backbone/head, gradient clipping, changed decay, additional evaluation, another seed-42 run, or combination with Conv fan-out. Those are new hypotheses.

## Evidence consulted

- PyTorch initialization documentation, torchvision ResNet source, He et al., and *How to Start Training*, as summarized in EXP033 brainstorming.
- `experiments/010/04-analysis.md`, `experiments/014/04-analysis.md`, `experiments/024/04-analysis.md`, `experiments/025/04-analysis.md`, and `experiments/031/04-analysis.md`.
- Goal definition, system understanding, learnings/results through EXP032, current `train.py`, and EXP033 brainstorm.
