# Brainstorm EXP-013
**Created**: 2026-08-06

## Web Search & Literature Review

- **PyTorch 2.9 compiler documentation** (https://docs.pytorch.org/docs/2.9/torch.compiler.html; https://docs.pytorch.org/docs/2.9/torch.compiler_get_started.html)
  TorchInductor captures forward and AOTAutograd backward graphs and can fuse operations, while CUDA graphs mainly remove host launch overhead. EXP-013 measurement shows fusion must reduce GPU backward rather than merely Python overhead.
- **PyTorch `torch.compile` API** (https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
  `reduce-overhead` targets small-batch CUDA-graph overhead but is not guaranteed to apply; `default` balances compile/runtime cost and `max-autotune` can use Triton convolution templates. Every mode needs local H20 measurement and exact state restoration after lazy compile warmup.
- **Averaging Weights Leads to Wider Optima and Better Generalization** (https://arxiv.org/abs/1803.05407; `goals/maximize-cifar10-best-test-accuracy/knowledge/papers/weight-averaging.md`)
  SWA improved CIFAR residual-family models by averaging trajectory points, but the accepted rapid terminal anneal and BatchNorm buffers make the local averaging window and state policy decisive.
- **Accurate, Large Minibatch SGD** (https://arxiv.org/abs/1706.02677; EXP-012 `proposals/idea-03.md`)
  Large batches can preserve accuracy only when optimization effects are addressed; the same work reports a modest gain from zero-initializing each postactivation residual branch's last BN scale.
- **PyTorch 2.9 SGD API** (https://docs.pytorch.org/docs/2.9/generated/torch.optim.SGD.html)
  The installed optimizer exposes `fused`, but the measured optimizer stage is under 0.8% of step time, so fused SGD alone has too little systems ceiling.

## Experimental History Review

- The accepted frontier remains EXP-010 at 94.15%: width-2 postactivation ResNet-20, all-parameter decay `1e-4`, p=0.5 alpha-1 CutMix on N1/M7 views through 80%, then a hard weak tail. It completed 26,898 steps with a healthy 89.73% switch checkpoint and final/best equality.
- Width was the largest gain (+1.25) despite fewer updates, and p=0.5 CutMix added +0.60 with equal exposure. Representation capacity plus complementary regional regularization remain validated.
- Decay changes are exhausted at this width; p=0.75 CutMix and canonical full preactivation both crossed the 87.08 strong-underfit marker. New candidates should preserve postactivation and avoid adding strong-phase suppression.
- EXP-012's preactivation package was compute-neutral and reached 94.22%, but its switch checkpoint fell 2.85 points below EXP-010. It is a valid near miss, not evidence to retry a deeper identity-path reorder.
- The new system decomposition measures 10.927 ms/step: backward 8.220 ms (75.46%), forward 2.408 ms (22.11%), iterator wait 0.145 ms, and visible host gap 0.034 ms. Peak allocation is only 598.7 MB of 97,871 MiB. Backward fusion and batch scaling have headroom; loader and optimizer-only tuning do not.
- TorchInductor is a measured no-go rather than a finalist: PyTorch 2.9.1 raises `torch.compile is not supported on Python 3.14+` before capture. The frozen dependency constraint rules out changing interpreter or framework, and no fallback backend will be substituted.
- Untried gaps include fixed-time batch/image scaling, compiled backward kernels, late trajectory averaging, postactivation zero-gamma, spatial aggregation beyond average pooling, and input standardization. None has consumed a full experiment.

## Collected Ideas

- **Larger-batch fixed-time training** — raise batch size to 256 (with LR 0.1 initially preserved) so the H20 processes more images per counted second, while retaining every accepted recipe component. This attacks the large unused memory/parallelism envelope and could increase effective epochs, but fewer noisy optimizer updates and more dense-tail evaluations may hurt generalization or wall time.
- **State-restored TorchInductor training (retired at feasibility)** — the proposed reversible warmup targeted the measured 97.6% forward/backward GPU share, but installed PyTorch rejects compilation on Python 3.14 before graph capture. With dependencies frozen, this seed cannot become an experiment candidate.
- **Late weak-tail weight averaging** — maintain an averaged shadow of epoch-end model parameters during a pre-registered late annealing window, copy a declared BatchNorm-buffer policy, and evaluate only the averaged model once per epoch. This targets noisy terminal iterates without weakening the already-sensitive strong phase; averaging work must be counted and no extra evaluation is allowed.
- **Zero-gamma postactivation initialization** — initialize all nine `bn2.weight` tensors to zero on the accepted block so every residual branch begins as identity, then learns its scale through the post-conv BN. It is exactly compute-neutral and literature-backed, but may recreate EXP-012's strong-phase underfit by delaying branch recruitment.
- **Average-plus-max spatial aggregation** — replace final global average pooling with a learned or fixed combination of global average and maximum summaries, optionally with a 256-wide classifier input. It targets localized CutMix features that average pooling may dilute; added pooling kernels are small but must pass a launch-overhead gate after SE's failure.
- **Per-channel CIFAR standardization** — use standard train/test-compatible channel standard deviations rather than `(1,1,1)` to condition input gradients. Because `prepare.py` fixes evaluation normalization, applying different train scaling would create a train/eval distribution mismatch unless the model explicitly compensates; this makes the naive version infeasible and only a model-internal fixed rescale defensible.
- **Cosine-normalized classifier** — normalize pooled features and classifier weights with a fixed or learned scale, making logits depend on angular separation after mixed-label training. It directly changes representation geometry with few parameters, but adds sequential normalization kernels and an unvalidated scale operating point.
- **Moonshot: dual-path multi-scale final stage** — replace the 128-channel final stage with parallel standard and dilated residual paths under a matched timing budget, then merge before pooling. It could capture both regional CutMix parts and global shape, but is a broad high-risk architecture change after simple capacity already delivered most of the gain.

## Combinations

- **Compilation + batch 256**: compiler fusion and better H20 occupancy may produce more images per second than either alone, potentially turning backward kernel savings into materially more data exposure. The combination is higher upside but confounds systems and optimization effects, so isolated measurements should precede it.
- **Compilation + late weight averaging**: faster accepted training could generate more terminal trajectory samples while averaging smooths their evaluation. This crosses exposure and generalization levers, but only makes sense after each isolated mechanism clears its own correctness/cost gate.
- **Zero-gamma + accepted CutMix**: identity initialization changes only branch recruitment while preserving the proven regional regularizer. It is cleaner than full preactivation, yet the compounded-underfit history makes the combination's early-fit risk central.
- **Average/max pooling + accepted CutMix**: regional mixing creates spatially localized class evidence and max pooling preserves its strongest response while average pooling retains extent. The paired summary may be more expressive than either statistic, but it must justify extra classifier degrees of freedom and kernel cost.

## Candidate Ideas

### Uniform Late Weak-Tail Endpoint Averaging
**Summary**: Preserve online EXP-010 training exactly. From 90% counted progress onward, uniformly average all trainable parameters at eligible weak-tail epoch endpoints, charge the average update to training time, and use the average for the one existing evaluation. Continue training the restored online model; evaluate averaged parameters with current online BatchNorm buffers and never evaluate both models in one epoch.

**What it targets**: The accepted tail remains productive but contains correlated terminal-iterate noise. Endpoint averaging adds no strong-phase suppression and uses about seven annealed weak-tail snapshots to seek a wider, better-generalizing center while retaining batch 128 exposure.

**Reasoning**: Weight-averaging literature reports CIFAR residual-family gains, and the local knowledge base finds averaging complements annealing. Starting at 90% excludes the abrupt 80% data-distribution transition while leaving enough endpoints. A shadow parameter/backup state costs about 8.2 MiB and endpoint-only updates should have negligible charged overhead.

**Sources**: `proposals/idea-02.md`; `knowledge/papers/weight-averaging.md`; Izmailov et al. UAI 2018; EXP-010/012.

**Estimated Effort**: medium-high

**Risk Assessment**: EXP-010 finished at its best, so a uniform mean can lag an improving online iterate and replacement evaluation can hide an online peak. Current BN moments are only approximate for averaged weights; epoch samples may be too correlated; average update/swap restoration errors could contaminate optimization. The 90% window and buffer policy cannot be changed after observation.

### Batch-256 Fixed-Time Training with Linear LR Scaling
**Summary**: Change only batch size 128 to 256 and scale the full LR curve exactly 2x (`0.1/0.01/1e-4` to `0.2/0.02/2e-4`). Keep the accepted model, momentum, decay, augmentation, CutMix, phase timing, seed, loader, and evaluator unchanged. Linear scaling approximately preserves LR and coupled-decay displacement per dataset pass while using otherwise idle H20 parallelism.

**What it targets**: The system decomposition finds only 0.61% of H20 memory used and a backward-dominated GPU step. A paired preliminary diagnostic measured batch 256 at 14,012 images/s versus 10,909 for batch 128: 28.44% more projected image exposure, about 84-89 dataset passes, at the cost of retaining only about 64% of optimizer updates.

**Reasoning**: Batch 256 is the measured knee; batch 512 adds only 5.1% more images while halving batch-256 updates. The 2x LR rule preserves 39 LR-units per complete 49,920-image pass and avoids silently testing an under-optimized large batch. EXP-010 finished at its best and was still improving at the budget boundary, while EXP-007 gained 1.25 points despite 29.2% fewer updates; together these are affirmative evidence that effective capacity/exposure can beat raw update count at this frontier.

**Sources**: `proposals/idea-01.md`; `00-batch-timing.md`; `02-system-understanding.md`; Goyal et al. large-minibatch SGD; EXP-010/011/012.

**Estimated Effort**: medium

**Risk Assessment**: LR 0.2 can overshoot without warmup, and 16.4k-17.3k updates may be too few even with 28% more images. Larger batches reduce beneficial gradient/BN noise, halve CutMix decisions per dataset pass, span twice as many images per momentum horizon, and trigger more excluded-time weak-tail evaluations. A healthy 87.08%+ switch checkpoint is diagnostic, not a rescue trigger.

### Fixed Average-Max Final Spatial Aggregation
**Summary**: Keep the exact 128-wide classifier and replace pure global average features with `torch.lerp(global_avg, global_max, 0.5)`. The symmetric parameter-free blend preserves every model tensor, initialization draw, optimizer group, and data/training mechanic while adding one endpoint max reduction and pointwise blend.

**What it targets**: EXP-010 validates class-bearing spatial regions, but global average pooling dilutes compact donor/object evidence across the final 8x8 map. The max half preserves peak salience and the average half retains area/extent, testing a representation readout without another strong-phase regularizer.

**Reasoning**: A fixed blend gives dense gradient to all 64 locations plus an additional max gradient to one location per channel. Unlike concatenation, it adds no classifier capacity or RNG shift. Its single endpoint reduction is much narrower than the nine-gate SE design that failed timing, but still requires paired H20 measurement.

**Sources**: `proposals/idea-05.md`; `02-system-understanding.md`; EXP-010 CutMix result; EXP-012 SE timing failure.

**Estimated Effort**: medium

**Risk Assessment**: Max evidence is area-insensitive and can conflict with CutMix area targets, amplify RandAugment artifacts, concentrate gradients 65x at selected locations, and shift feature scale upward. Rank discontinuities and one extra reduction/backward can hurt stability or exposure; a 50/50 miss cannot be rescued by tuning the coefficient.

## Review

The mandatory Claude idea review completed successfully and is preserved in `01-idea-review.md`; no fallback reviewer was used. It selected batch-256 linear scaling on 7/10 evidence and 6/10 potential impact because it is the only live candidate grounded in measured headroom and a measured operating knee. The review required foregrounding EXP-010's final-equals-best trajectory and EXP-007's gain despite fewer updates as evidence that more effective image exposure can move the limiter. It also required using a switch checkpoint below 87.08% to distinguish failure of the batch/LR operating point from failure of the exposure premise.

The review judged uniform endpoint averaging near-fatally mismatched: EXP-010's monotone final-best tail, low decaying LR, and correlated endpoints predict lag, while averaged-only evaluation can hide the very online peak that established the frontier. It judged average-max pooling plausible but under-evidenced, with an inseparable upward feature-scale shift and weak alignment between area-proportional CutMix labels and area-insensitive maxima. TorchInductor remains infeasible on Python 3.14/PyTorch 2.9.1, and zero-gamma remains structurally invalid with Option-A dead channels.

The later mandatory Claude plan review identified evaluation multiplicity as a validity confound: 195-step epochs would otherwise create 22-23 test-set looks versus EXP-010's 19, biasing a max-based metric upward. The execution plan therefore adds a protocol-control change to use exactly 19 fixed elapsed-progress evaluations, preserving the accepted early checkpoints and evenly covering the 80-100% tail. This is not part of the accuracy mechanism; it removes an observation-count advantage created by batch scaling.

## Idea Evaluation

- **Batch-256 with 2x LR**: selected. It provides 28.44% more measured image exposure at the H20 throughput knee and uses a pre-registered scaling rule. The main risk is fewer, lower-noise updates and LR-0.2 instability, but these are the experiment rather than an implementation ambiguity.
- **Fixed average-max pooling**: not selected. It preserves strong-phase recipe and has moderate upside, but its localized-evidence mechanism is only plausible at an 8x8 map and feature-scale shift confounds the clean readout claim.
- **Uniform late averaging**: not selected. Careful implementation does not repair its trajectory mismatch or the loss of online peak evaluation opportunities.
- **Retired seeds**: compilation and zero-gamma cannot legally or structurally reach execution, respectively; neither is a fallback candidate.

## Chosen Idea
**Selected**: Batch-256 Fixed-Time Training with Linear LR Scaling

**Why this idea**:
It is the only finalist with a quantified systems mechanism and a locally measured operating point: batch 256 processes 28.44% more images in the fixed training budget, while batch 512 is already past the throughput knee. The accepted frontier still improved at its final endpoint, and width previously demonstrated that useful capacity/data processing can outweigh fewer updates. Scaling the complete LR curve by two preserves first-order update and decay displacement per dataset pass, making the candidate a coherent large-batch method rather than an unreviewed LR choice.

**Hypothesis**:
Batch 256 with the exactly doubled `0.2 -> 0.02 -> 2e-4` LR curve will process at least 20% more image slots while preserving the accepted architecture, augmentation, CutMix, optimizer, and elapsed-time phases, raising `best_test_acc` from 94.15% to at least 94.25%. A final strong checkpoint below 87.08% will diagnose that the linear-scaled batch operating point failed to fit the composite phase; it cannot trigger warmup, LR adjustment, batch fallback, or a rerun.
