# Report EXP-022: Fixed 288-channel final-stage taper
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the frozen 300-second charged-training protocol by testing whether modest low-resolution semantic capacity can lift the stable global-best plateau. EXP-022 grew from EXP-011 at 95.61%, also the global best, so a metric run would have needed at least 95.71%.

## Idea & Hypothesis

Widen only the final two 8x8 residual blocks from 256 to 288 channels, preserving early/middle processing and the complete EXP-011 CutMix, period-two late SAM, and cadence-31 full-state EMA recipe. The fixed candidate added 511,552 parameters (+18.61%) but only 8.33% Conv/Linear MACs, added no module or kernel launch, and used multiple-of-32 shapes. Based on EXP-014 width 320's measured cost and MAC interpolation, it was hypothesized to cost roughly 1.07-1.10x, retain at least 23,200 steps and 140 EMA samples, then reach at least 95.71%.

## Approach

Only `train.py` changed: block specifications 5-6 became `128->288` and `288->288`, final BatchNorm and classifier input became 288, and architecture metadata was made truthful. No data, seed, loss, optimizer, schedule, SAM, EMA, evaluator, timing, or summary logic changed. Exact parent-source, architecture, parameter/MAC, deterministic construction, CPU/GPU gradients, CutMix/drop-path, SAM replay/restore, full-state EMA, optimizer identity, and memory checks preceded one frozen accuracy-blind paired preflight. Shape-dependent initialization and later DataLoader RNG changes were explicitly treated as part of the one fixed seed-42 package.

## Execution

The first two CPU verifier launches exposed only transient harness setup mistakes before model construction or timing: a Git blob prefix was confused with a raw SHA-256, then the out-of-tree script lacked the repository import path. Both were recorded and corrected without touching production code or observing a gate. The complete CPU and candidate-only GPU verification then passed. The frozen preflight (SHA-256 `7ba77de9...`) ran five alternating rounds on physical GPU 0, produced one complete vector, and failed the preregistered feasibility envelope. The result was not rerun, no fallback width was tried, and no evaluator/test metric or `run.log` was produced. A raw-result audit recomputed every gate and confirmed the `crash/NaN` classification.

## Results

- **Primary metric**: `NaN` (parent: 95.61%; delta vs parent/global best: N/A because no accuracy was measured)
- **Observations**: Candidate correctness passed at 3,260,442 parameters and 425,315,136 MACs/image. Candidate-only peak was 656.104 MiB, SAM epsilon 0.05000001, EMA samples 30 with 15/15 parity, and state/RNG failures zero. The paired preflight's ordinary/CutMix/SAM medians were 1.163564/1.158708/1.160245x; weighted median was 1.160794x with 0.001326 MAD/median. Parent charged-path drift stayed 0.009301-0.012005, so the rejection was not caused by an unstable baseline. Early/late ratios 1.161156/1.161904 projected 22,214.78 steps, 113 epochs, and 137.705 EMA samples. Projected total runtime 446.75 seconds and joint peak 728.967 MiB passed, but weighted latency, charged-path maximum, step, epoch, and EMA gates failed.
- **Analysis**: The capacity hypothesis remains accuracy-unqueried, but its assumed systems premise is falsified at this operating point. Width 288's 1.160794x cost is effectively identical to width 320's 1.160975x even though their incremental MAC fractions are 8.33% and 17.56%. Ordinary, CutMix, SAM, and evaluation paths all show the same roughly 16% penalty, pointing to unfavorable dense 288-channel H20 kernel/tensor efficiency rather than augmentation, SAM, EMA, memory, or one noisy path. Fifth-round CutMix/SAM outliers worsened maximum gates, but the stable median alone rejects the candidate and underprojects dose by about 985 steps.
- **Key Learning**: Dense final-stage widths 288 and 320 both cost about 1.161x on H20, invalidating MAC-linear latency interpolation for this architecture.

## Verification

- **Conditions**: Architecture, numerical integrity, memory, parent stability, dispersion, and total-runtime projection passed. Mandatory weighted latency (`1.160794 >1.10`), charged-path maximum, projected steps (`22214.78 <23200`), epochs (`113 <118`), and EMA samples (`137.705 <140`) failed; metric verification was therefore skipped.
- **Review Notes**: The raw JSON arithmetic is internally consistent, the frozen harness used separate charged-path and uncharged-evaluation accounting, and no accuracy field or evaluator call exists. Rejection does not depend on the two late outlier rounds because all three charged path medians are near 1.16.
- **Verdict**: `crash`, metric `NaN`
- **Verdict Basis**: Mechanical tree encoding for a valid preflight rejection that produced no primary metric. This is not a literal runtime crash, an accuracy no-improvement, or an integrity-invalid result.

## Unexplored Avenues

- A qualitatively different late-capacity geometry using bottlenecked or grouped operations could reduce dense arithmetic, but it needs exact module design and H20 launch benchmarking; FLOP savings alone are now explicitly unreliable.
- Capacity-neutral representation changes that retain existing dense tensor shapes avoid the rejected width kernel regime, though prior attention and auxiliary-loss attempts warn against extra small FP32 launches or forwards.
- Width 272/304 interpolation is not justified: the 288/320 nonlinearity removes the systems basis for selecting another nearby scalar width, and using one after this result would be a post-hoc fallback.

## Next Steps

- **High confidence**: Close scalar dense stage-3 widening on this H20 and navigate toward a different mechanism class with unchanged core tensor shapes.
- **Medium confidence**: Profile one fully specified fused or bottlenecked late transform before proposing it, including launch-level timing rather than parameter/MAC estimates.
- **Medium confidence**: Prefer mechanisms with a credible >=0.3-point ceiling, since the remaining 95.61 plateau and single-run tail noise make narrow scalar tuning poor value.
