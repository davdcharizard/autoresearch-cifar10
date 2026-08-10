# Brainstorm EXP-031
**Created**: 2026-08-06

## Web Search & Literature Review

- **Scheduled Restart Momentum** (arXiv 2002.10583)
  Restarting accelerated momentum on a schedule can improve convergence on CIFAR/ImageNet, supporting the general mechanism that stale velocity can be harmful across optimization regimes. The paper studies a NAG-style method rather than literal PyTorch SGD-buffer clearing, so it is directional evidence only.
- **SGDR** (`knowledge/papers/sgdr.md`; ICLR 2017)
  Warm restarts improve anytime performance on CIFAR, but the accepted recipe already uses one abrupt curriculum/LR transition. EXP030 shows more tail LR increases fit while worsening generalization, motivating a state reset at the existing boundary rather than more amplitude.
- **Generalizing Pooling Functions in CNNs** (`knowledge/papers/mixed-pooling.md`; AISTATS 2016)
  Mixed max/average pooling can improve invariance and CIFAR performance. Local EXP014 invalidated an independent raw-max classifier branch with 4.10x first-gradient scale, but did not test a bounded, scale-controlled perturbation of the accepted average descriptor.
- **PyTorch channels-last** (`knowledge/references/pytorch-channels-last.md`; official PyTorch memory-format tutorial)
  Conv2d/BatchNorm can propagate NHWC layout, though official GPU gains emphasize reduced precision. This remains the cleanest direct probe of the measured 75.46% backward bottleneck, provided exact evaluation count and fresh FP32 timing gates are enforced.

## Experimental History Review

- EXP010 remains the 94.15% frontier: width-2 postactivation ResNet-20, ordinary momentum, all-parameter decay `1e-4`, N1/M7 plus p=0.5 alpha-1 CutMix through 80%, then LR-0.01 weak hard-label refinement.
- EXP030 retained 99.48% exposure and drove weak train-loss EMA to 0.0264, but worsened NLL to 0.2083 and peaked at 93.90. More tail amplitude is counterproductive; preserve the 0.01 quench and target inherited state or representation instead.
- Global-LR Nesterov, Lookahead, and PNM repeatedly caused early class-collapse transients. A boundary-only momentum reset is materially narrower: it leaves the entire high-LR strong trajectory ordinary and deletes velocity just as LR falls tenfold, but still needs a copied-state update/concentration gate.
- Raw max readout collapsed after a 4.10x first gradient, while mixed-pooling literature suggests localized features can complement averages. Any revisit must be a fixed bounded perturbation with explicit feature/update scale, not an independent unnormalized max branch.
- EXP029 proves even tiny per-step helpers can lose the 99% exposure gate. Favor one-time boundary operations, worker-side rules, or kernel/layout changes with paired measurement.

## Collected Ideas

- **Boundary-only momentum-buffer reset** — At the accepted 80% data/LR switch, clear every SGD momentum buffer once before the first weak hard-label update. This preserves all strong learning and the proven 0.01 quench while preventing composite-target/high-LR velocity from leaking into the new objective; the intervention is one-time and negligible in the counted budget.
- **Scale-controlled 10% max residual in global pooling** — Replace the final average descriptor with `avg + 0.10 * s * (max-avg)`, where a fixed preregistered scale `s` is derived from accepted initialization/corpus RMS before production. This injects bounded localized evidence without EXP014's independent raw-max classifier or uncontrolled gradient scale, at the cost of an extra reduction/backward path.
- **FP32 channels-last with fixed evaluation opportunities** — Convert model Conv weights and training/eval inputs to channels-last while preserving FP32/default TF32 and logical NCHW shapes. It directly targets convolution/BN backward; proceed only after multiple fresh alternating trials show a meaningful full-step gain and layout propagation.
- **Tail-only sparse EMA** — Beginning at 80%, update an EMA shadow only every 16 optimizer steps and evaluate it at the same 19 opportunities with online BN buffers. Sparse updates may reduce late parameter noise without EXP018's uniform historical bias or significant per-step cost, but EXP010 ends at its best and evaluator choice must be preregistered.
- **Freeze BN affine parameters only in the weak tail** — Set BN scale/shift gradients off at the existing transition while continuing running-stat adaptation. This simplification could prevent the low-LR hard objective from over-specializing normalization, but removes useful degrees of freedom and changes optimizer parameter activity.
- **Small-probability Random Erasing composed after N1/M7** — Add worker-side erasing with conservative probability/area while preserving CutMix. It adds an occlusion prior without deleting class evidence on every view as EXP006 did, but risks compounding already strong regularization and host throughput.
- **TrivialAugmentWide strong policy** — Replace N1/M7 with one randomly selected wide-magnitude operation. It imports a low-tuning augmentation family with no GPU work, but abandons the locally validated magnitude and may change worker throughput/data severity substantially.
- **Tail-only classifier weight normalization** — Normalize classifier rows and use a fixed temperature only after the 80% switch to constrain confidence sharpening implicated by EXP030's low train loss/worse NLL. The abrupt parameterization change is high-risk and could disrupt logits or momentum state.
- **Moonshot: stochastic-depth compute reinvestment** — Apply conservative block survival through the strong phase and spend measured savings on one extra residual block, then use the full model in the tail. Literature supports deep stochastic depth, but the local nine-block network and transition fragility make this a compound, attribution-poor bet.

## Combinations

- **Channels-last + boundary momentum reset**: Layout could add exposure while the one-time reset improves the objective transition. Each mechanism is separable and should be tested alone first; combining now would confound systems and optimization effects.
- **Scale-controlled mixed pooling + sparse tail EMA**: Localized descriptors could raise representation quality while EMA tempers their late variance. Both touch inference selection/geometry, so a miss would be uninterpretable without isolated results.
- **Momentum reset + BN-affine freeze**: Both reduce inherited/continued state at the curriculum boundary and could create a cleaner refinement regime, but jointly removing velocity and affine adaptation may underfit and obscures which state mattered.

## Candidate Ideas

### Scale-Controlled 10% Max-Residual Global Pooling
**Summary**: Use `avg + 0.10*s*(max-avg)` before the unchanged classifier, with `s<=1` fixed once from a preregistered training-only initialization corpus so the added residual RMS is at most 10% of average-descriptor RMS. Full specification: `proposals/idea-02.md`.

**What it targets**: Representation. It preserves extent-sensitive average evidence for CutMix while introducing a bounded amount of localized salience that pure averaging discards.

**Reasoning**: Mixed-pooling literature reports improved invariance, and CutMix's local regions make spatial evidence relevant. This is distinct from EXP014's independent raw-max classifier: no new parameters/state exist, the max coefficient is capped, and the shared classifier sees a convex descriptor. Calibration, exact-corpus gradient/update safety, and fresh paired timing are load-bearing because hard-max gradients are sparse and EXP029 exposed small-operation overhead.

**Sources**: `knowledge/papers/mixed-pooling.md`; EXP010, EXP014, EXP029, EXP030.

**Estimated Effort**: high.

**Risk Assessment**: Medium-high. Initialization RMS control does not bound later per-example peaks, max may amplify augmentation/CutMix artifacts, and the extra reduction/backward path may miss the 99% exposure gate.

### Reset SGD Momentum Once at the 80% Objective Boundary
**Summary**: Preserve ordinary SGD through the entire accepted strong phase, then zero all 59 momentum buffers exactly once after the switch evaluation/loader rebuild and before the first LR-0.01 weak hard-label update. All later steps use unchanged PyTorch SGD. Full specification: `proposals/idea-01.md`.

**What it targets**: Objective-transition state. It removes the decaying high-LR composite-target velocity when views, labels, and LR change together, while retaining the accepted 0.01 quench that EXP030 supports.

**Reasoning**: EXP030 shows extra tail motion lowers train loss but harms NLL, so deleting stale inherited motion is more coherent than raising tail LR. Unlike EXP020/022/028, the intervention cannot affect high-LR strong geometry, never relocates parameters, and uses ordinary SGD after one low-LR boundary event. A copied mature-boundary state and exact weak corpus must still verify update geometry and concentration.

**Sources**: Scheduled Restart Momentum (arXiv 2002.10583); SGDR; EXP010, EXP020/022/028, EXP030; `02-system-understanding.md`.

**Estimated Effort**: medium.

**Risk Assessment**: Medium. Inherited momentum may carry useful invariant descent rather than stale bias, and its direct velocity effect decays within roughly 44 weak steps. Implementation/runtime risk is low; scientific effect size may be small.

### FP32 Channels-Last with Exactly 19 Evaluations
**Summary**: Restride initialized Conv weights and all image inputs to channels-last while retaining FP32/default TF32 and the accepted recipe. Cap nonterminal evaluations so production still has exactly 19 unique looks including terminal. Full specification: `proposals/idea-03.md`.

**What it targets**: The measured systems bottleneck: convolution/BN backward consumes 75.46% of counted step time. A ≥3% full-step gain would increase accepted-recipe updates without changing batch noise or capacity.

**Reasoning**: Official PyTorch guidance supports channels-last Conv/BN propagation, but its strongest GPU results use reduced precision and larger images. Seven fresh paired trials, profiler conversion checks, exact-corpus numerical safety, and a 27,705-step projection are required before production; otherwise the idea is invalidated without consuming a scored run.

**Sources**: `knowledge/references/pytorch-channels-last.md`; official PyTorch memory-format tutorial; EXP010, EXP013, EXP016, EXP023, EXP029.

**Estimated Effort**: high.

**Risk Assessment**: High feasibility/impact risk. Tiny FP32 CIFAR kernels may be neutral or slower, Option-A operations may force repairs, and even verified extra exposure has no established accuracy benefit.

## Review

Claude's independent adversarial review (`01-idea-review.md`) selected **Scale-Controlled 10% Max-Residual Global Pooling**. It scored the candidate 6.5/10 on evidence/reasoning and 7/10 on impact because it uniquely targets the diagnosed representation/generalization limiter and a named open question. Boundary momentum reset scored 7/10 and 4/10: its causal story is clean and safe, but its inherited-velocity effect decays below 1% in roughly 44 steps and its cumulative displacement is only about one strong update with ambiguous sign. Channels-last scored 5/10 and 3.5/10 because it requires both an uncertain >=3% FP32 speedup and an unproven exposure-to-accuracy link.

I adopt the representation-first selection. The reviewer agrees that the design genuinely differs from EXP014: the candidate has no independent classifier or optimizer state, shares the accepted FC, uses a convex descriptor with max coefficient at most 0.10, and explicitly gates gradient/update scale on production-distribution batches. It is also unrelated to the retired pool-first transition shortcuts.

Three concerns become load-bearing. First, max is area-insensitive and can overvalue a small CutMix donor or augmentation artifact, so exact-corpus strong fit, class concentration, descriptor scale, and update ratios must pass before timing. Second, EXP029 makes the <=1% full-step overhead gate non-negotiable; missing it invalidates this exact implementation without coefficient rescue. Third, initialization calibration does not guarantee later descriptor scale. The exact-corpus controller must therefore record residual/average RMS throughout strong and weak replay, and production must expose diagnostic-only eval-mode ratios at the 80% switch and terminal evaluation without changing logits, model tensors, evaluation count, or the frozen coefficient. These diagnostics interpret drift but cannot retune `s` or override the primary metric.

## Idea Evaluation

- **Scale-controlled 10% max-residual global pooling** — Advance. It has the highest ceiling and directly tests whether localized class-bearing evidence survives the final aggregation, with EXP014's uncontrolled branch mechanism removed by construction.
- **Boundary-only momentum reset** — Defer as the clean fallback. It is validly distinct from recurring high-LR optimizer failures and almost free, but its direct effect is brief, bounded, and directionally ambiguous.
- **FP32 channels-last with exactly 19 evaluations** — Reject as a scored accuracy experiment. It faces an uncertain timing gate and then relies on extra exposure improving generalization, while changed kernel numerics confound that attribution.

## Chosen Idea
**Selected**: Scale-Controlled 10% Max-Residual Global Pooling

**Why this idea**:
This is the only finalist that directly attacks the current accuracy limiter rather than transition state or raw exposure. A fixed, training-only calibrated coefficient gives the accepted classifier at least 90% average evidence plus a bounded localized residual, testing the standing spatial-aggregation question without new parameters, gates, or independent max logits. Strict semantic, exact-corpus, scale-drift, and paired timing gates contain the known EXP014/029 risks.

**Hypothesis**:
Replacing pure global average pooling with `avg + 0.10*s*(max-avg)`, where `s<=1` is frozen from a preregistered training-only initialization corpus, will preserve at least 99% of accepted exposure and healthy strong fit while retaining useful localized CutMix evidence, raising seed-42 `best_test_acc` from 94.15% to at least 94.25%. Point prediction is 94.30%. Any safety/timing veto invalidates this exact point; a valid lower production result rejects it without coefficient tuning.
