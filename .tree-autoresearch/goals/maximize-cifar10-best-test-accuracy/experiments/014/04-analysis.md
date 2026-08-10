# Report EXP-014: Calibrated stage-3 width-5 expansion
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the fixed 300-second charged-training protocol. EXP-014 grew from EXP-011, whose `95.61%` is also the global best; a metric run would have needed at least `95.71%` for improvement.

## Idea & Hypothesis

Widen only the final 8x8 stage from 256 to 320 channels, producing a six-block `64/128/320` taper with 3,827,290 parameters. The idea spent unused memory on late semantic capacity for 17.6% more MACs while preserving early/middle processing and the validated CutMix/SAM/EMA recipe. It was hypothesized to stay within `1.15x` parent latency, retain at least 22,000 projected steps and 130 EMA samples, then lift the stable EMA plateau.

## Approach

Only `train.py` changed. The last two residual block shapes, tail BatchNorm, and classifier input became 320 channels; architecture metadata and outside-charged diagnostics were updated. Exact parent/candidate source, parameter, MAC, RNG, gradient, CutMix, drop-path, SAM, EMA, and memory checks preceded one accuracy-blind paired preflight. No width fallback or training retuning was available after measurement.

## Execution

CPU and candidate-only GPU checks passed. One transient audit corrected an invalid `.view()` on a channels-last diagnostic tensor, and syntax checking used an owned bytecode cache after a shared `/tmp` permission error; both occurred before any numeric gate vector and changed no production code. The first complete preflight then ran five alternating rounds on physical GPU 0. It returned a decisive rejection, so no metric process, evaluator call, test-loader iteration, or `run.log` occurred. Claude Opus independently returned PASS on the evidence and classification.

## Results

- **Primary metric**: `NaN` (parent: `95.61%`; delta vs parent/global best: N/A because no accuracy was measured)
- **Observations**: Parent weighted rounds were `12.6226-12.7207 ms`; candidate rounds were `14.6637-14.7685 ms`. All five paired ratios were `1.15901-1.16353`, with median `1.160975`, above the `1.15` stop gate. Parent drift `0.007757`, ratio MAD/median `0.001694`, max ratio `1.163529`, projected 22,220.98 steps, 113 epochs, 137.82 EMA samples, and 520.00 total seconds passed their respective checks. The candidate-only peak was only 652.138 MiB, confirming that compute rather than memory blocked this configuration.
- **Analysis**: The fixed 320-channel taper was operationally sound but missed the calibrated latency ceiling by about 0.95% relative, unanimously across rounds. Its 16.10% measured training-path cost closely tracks its 17.56% MAC increase and leaves almost no exposure margin over the 22,000-step floor. The finite conditioning trace and 6.58% smaller relative SAM perturbation are diagnostic only; without a metric run they cannot confirm or reject the proposed representation benefit.
- **Key Learning**: Widening the final stage to 320 costs about 16% weighted step latency on H20, tripping the 1.15 gate before any metric run.

## Verification

- **Conditions**: Source, architecture, state, GPU, memory, stability, dispersion, maximum-ratio, and projected-dose checks passed; the mandatory median latency ratio failed (`1.160975 > 1.15`); metric verification was therefore skipped.
- **Review Notes**: Claude recomputed every gate and found no blocker or integrity issue. The weighted ratio covers training paths rather than whole-run evaluation, and absolute projections are approximate, but neither caveat can reverse the unanimous ratio failure.
- **Verdict**: `crash`, metric `NaN`
- **Verdict Basis**: Mechanical tree encoding for a valid preflight rejection that produced no primary metric. This is not a literal runtime crash, `no-improvement` accuracy result, or integrity-invalid run.

## Unexplored Avenues

- A separately preregistered 288-channel final stage could test the same cheap-late-capacity thesis with a likely lower latency ratio; it was intentionally not used as a conditional fallback here.
- A bottlenecked or grouped late-stage expansion could add representation width with fewer dense MACs, but it changes residual geometry and needs independent correctness and H20 kernel-efficiency validation.
- Capacity-neutral late representation changes remain open because this experiment rejected cost feasibility before measuring generalization, not the semantic-capacity hypothesis itself.

## Next Steps

- **High confidence**: Return to EXP-011 and prioritize a low-overhead objective or state-space mechanism, such as the already developed bounded soft-target Poly-1 candidate, to preserve optimizer exposure.
- **Medium confidence**: Refine clean-tail averaging beyond the existing EMA only after correcting the proposed uniform-SWA kernel accounting and BatchNorm-state rationale.
- **Medium confidence**: Consider width 288 only as a fresh fixed experiment, never as a retry or continuation of EXP-014.
