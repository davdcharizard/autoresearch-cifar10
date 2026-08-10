# Brainstorm EXP-010
**Created**: 2026-08-05

## Web Search & Literature Review

- **When, Where and Why to Average Weights?** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/weight-averaging.md`): late averaging can complement annealing at low memory cost, but BatchNorm state must be handled explicitly.
- **mixup: Beyond Empirical Risk Minimization** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/mixup.md`): input/target interpolation improves CIFAR generalization, though soft-target work and slower convergence matter in 300 seconds.
- **CutMix** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/cutmix.md`): donor pixels and area-adjusted targets directly address Cutout's information deletion, but composition with N1/M7 may over-regularize.
- **SGDR** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/sgdr.md`): late cosine trajectory geometry can support checkpoint averaging without changing the validated elapsed-time schedule.

## Experimental History Review

- The accepted frontier is 93.55% from EXP-007: width 2 overcame a 29.2% step loss and strongly improved representation learning under N1/M7.
- The 80% high-LR/N1-M7 plateau and deterministic weak hard-label tail are validated composition points; early weak switching, Cutout replacement, label smoothing, and short cosine exploration failed.
- EXP-008 and EXP-009 bracketed the accepted all-parameter `1e-4` decay. Stronger decay underfit; removing BN/bias decay fit harder but worsened NLL and did not improve top-1. Decay tuning is now a recurring low-value direction.
- EXP-007 ended nearly flat around 93.5% after approximately 27.1k steps, so remaining error is not simply an unfinished terminal ascent. Candidate mechanisms should improve the late solution, increase useful fixed-time exposure, or change representation/target geometry without breaking the strong/weak lifecycle.
- No tracked file other than `train.py` may change; one H20, one fixed seed, one evaluation per epoch, 300 counted seconds, and 600 total seconds remain fixed.

## Collected Ideas

- **Tail checkpoint averaging** — maintain a uniform parameter average of weak-tail epoch endpoints and evaluate the averaged parameters once per epoch using current-model BN buffers. This targets noisy late trajectory location rather than stronger regularization and is supported by the observed flat EXP-007 tail plus weight-averaging literature.
- **BF16 autocast throughput** — execute forward/loss under CUDA BF16 autocast while retaining FP32 master parameters and optimizer state. H20 tensor cores could increase the 27.1k-step exposure at nearly unchanged memory, directly attacking fixed-time optimization exposure; numerical stability and small-CNN launch overhead are the main risks.
- **Plateau CutMix composition** — apply batch CutMix with conservative probability/alpha during the N1/M7 high-LR phase, then preserve the weak hard-label tail. Donor pixels and area targets address EXP-006's lossy Cutout failure while adding spatially meaningful regularization.
- **Plateau mixup with hard tail** — use low-alpha input interpolation only during high-LR exploration and restore unmixed hard labels at 80%. This imports a proven CIFAR geometry bias while retaining the validated refinement stage, but may repeat soft-target convergence costs seen with label smoothing.
- **Preactivation width-2 blocks** — convert the residual units to BN-ReLU-Conv preactivation while retaining width, depth, Option-A projection shape, and training recipe. It may improve gradient flow and feature reuse, but is a broad architecture change whose benefit at only 20 layers is uncertain.
- **Zero-initialize residual branches** — initialize each block's final BN scale to zero so the network starts near identity. This is a small representation/optimization intervention that can stabilize residual learning, though it may slow early learning inside the short horizon.
- **Larger batch for exposure** — test batch 256 to improve H20 utilization and reduce Python/loader overhead, then preserve LR initially for isolation. It could increase images processed but cuts optimizer updates and changes optimization noise, so a paired throughput/update diagnostic is mandatory.
- **Moonshot: stochastic depth on width 2** — randomly skip residual branches during the strong phase and disable dropping in the weak tail. It could regularize an ensemble of effective depths, but the shallow three-block-per-stage model has little depth redundancy and branch control may add overhead.

## Combinations

- **BF16 + tail checkpoint averaging**: faster mixed-precision steps create a denser annealed trajectory, while averaging reduces late iterate noise. The cross could turn added exposure into a better terminal solution, but it combines two mechanisms and should follow isolated feasibility evidence.
- **CutMix + hard weak tail**: use class-bearing regional mixing only during the accepted exploration phase, then exactly recover the successful un-mixed refinement tail. This is stronger than CutMix throughout because it limits soft targets to high LR and stronger than fixed Cutout because no image region becomes information-free.
- **Preactivation + zero residual initialization**: identity-like initialization is structurally aligned with preactivation blocks and could make the architecture transition train smoothly. The combination has higher upside than either initialization or ordering alone, but its attribution and short-horizon risk are worse.

## Candidate Ideas

### Conservative Plateau CutMix Composition
**Summary**: In worker collation, apply torchvision CutMix with alpha 1.0 to a fixed 50% of N1/M7 plateau batches, then preserve the exact 80% loader shutdown and entirely hard-label weak tail. Require target/RNG correctness, worker throughput, and at least 97% projected step retention before execution.

**What it targets**: EXP-006 showed that information-deleting Cutout harms strong-view representation learning. CutMix keeps donor pixels class-bearing and makes target mass follow visible area, adding regional supervision rather than blank occlusion.

**Reasoning**: The primary CutMix evidence is strong on CIFAR, while the 0.5 probability and hard tail limit compounded regularization. Worker-side collation retains GPU exposure; installed probability-target cross-entropy avoids a custom loss.

**Sources**: `proposals/idea-03.md`; `knowledge/papers/cutmix.md`; EXP-004, EXP-006, and EXP-007.

**Estimated Effort**: medium

**Risk Assessment**: N1/M7 plus CutMix may still over-regularize the short strong phase, dense targets may reduce steps, worker RNG isolation is subtle, and 14-15 hard-tail epochs may not erase composite-distribution mismatch.

### Late Weak-Tail Checkpoint Averaging
**Summary**: From 90% elapsed progress, maintain a uniform FP32 average of trainable parameters at eligible weak-tail epoch endpoints. Evaluate only the averaged parameters once per epoch using current online BN buffers, then restore online parameters bitwise so SGD and momentum continue unchanged.

**What it targets**: EXP-007's near-flat late cosine trajectory may contain correlated iterate noise around a good basin. Averaging seeks a more central evaluated solution without adding strong-phase regularization or optimizer steps.

**Reasoning**: Weight-averaging literature reports mild gains when paired with annealing, and the accepted schedule provides roughly 7-8 late weak endpoints. Per-epoch averaging avoids per-step overhead and preserves the one-evaluation constraint; explicit swap/restore and charged update time keep attribution honest.

**Sources**: `proposals/idea-01.md`; `knowledge/papers/weight-averaging.md`; `knowledge/papers/sgdr.md`; EXP-007.

**Estimated Effort**: medium

**Risk Assessment**: Endpoints may be too correlated for a 0.10-point gain, and current online BN buffers are only approximate statistics for averaged weights. Evaluating the online model too would violate the protocol, so within-run decomposition is unavailable.

### CUDA BF16 Autocast Throughput
**Summary**: Wrap only the training forward pass and cross-entropy in CUDA BF16 autocast while retaining FP32 parameters, gradients, BN buffers, SGD momentum/decay, and FP32 evaluation. Require paired H20 numerical parity and at least 15% synchronized speedup, projecting at least 31,215 fixed-budget steps, before a full run.

**What it targets**: Width 2 is capacity-effective but completes only about 27.1k updates. This candidate increases useful high-LR and weak-tail exposure without changing the accepted statistical recipe.

**Reasoning**: H20 supports BF16 Tensor Cores and the wider 32/64/128 channels are eligible, while EXP-007 established paired synthetic timing as predictive within 2.5%. The gate protects against FP32 TF32 already being competitive and the small CNN remaining launch-bound.

**Sources**: `proposals/idea-02.md`; EXP-007; official PyTorch AMP documentation cited in the proposal.

**Estimated Effort**: medium

**Risk Assessment**: Autocast may fail the 15% speed gate or alter BatchNorm/gradient trajectories despite FP32 persistent state. More updates and cumulative decay need not improve a tail that was already flat.

## Review

Claude completed the mandatory external review with exit code 0; no fallback reviewer was used. It rejected late averaging because replacing online evaluations after 90% forfeits EXP-007's actual peak while averaging highly correlated near-zero-LR points. It judged BF16 rigorous but low-impact and likely preflight-infeasible because FP32 convolutions may already use TF32 and exposure is not the diagnosed primary limiter. It selected conservative CutMix as the only finalist with a genuinely new, literature-backed representation/target mechanism. I accept its request to register a >3-point drop from EXP-007's 90.08% switch checkpoint as evidence of compounded underfit; this is diagnostic only and cannot trigger post-hoc tuning or override the primary metric. Full scores and critique are in `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's CutMix pick. Its p=0.5, plateau-only, compose-not-replace design is materially different from EXP-006's every-view information-deleting Cutout and directly addresses that failure mechanism. The hard weak tail, 97% exposure gate, and fixed operating point bound the known soft-target and compounded-regularization risks better than the alternatives' structurally low ceilings.

## Chosen Idea
**Selected**: Conservative Plateau CutMix Composition

**Why this idea**:
CutMix has the strongest defensible upside because it changes spatial supervision while retaining class-bearing pixels, rather than asking a flat late trajectory or extra iterations to produce a new solution. Claude correctly surfaced the prior warning against stacking more difficulty on N1/M7 and the EXP-003 soft-target precedent; the selected design confronts both with 50% probability, worker-side mixing, a hard-label tail, fixed alpha, and mandatory throughput plus lifecycle gates.

**Hypothesis**:
Applying alpha-1 CutMix to 50% of N1/M7 batches only during the 80% high-LR plateau will improve regional feature use without discarding information or retaining less than 97% of EXP-007's update exposure. After the unchanged hard-label weak tail, `best_test_acc` will reach at least 93.65%. A switch checkpoint below 87.08% will support compounded underfit as the failure mechanism if the primary gate is missed.
