# Brainstorm EXP-022
**Created**: 2026-08-06

## Web Search & Literature Review

- **Deep Pyramidal Residual Networks** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/deep-pyramidal-residual-networks.md`): CIFAR residual networks can benefit from reallocating representation capacity through channel growth, but the paper does not validate this repo's coarse last-stage-only taper and therefore supplies a mechanism prior rather than a promised effect size.
- **Squeeze-and-Excitation Networks** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/squeeze-and-excitation.md`): learned channel interactions can improve representation quality, but compact CIFAR models can be launch-bound; this repo's EXP-009 directly confirms that execution risk.

## Experimental History Review

- EXP-011 is the 95.61% global best. Its time-calibrated EMA of the validated CutMix/SAM recipe produced a stable final-16 mean of 95.493%, so a useful intervention must lift an established generalization plateau rather than merely rescue a noisy endpoint.
- EXP-014 widened only the final 8x8 stage from 256 to 320 channels. All implementation, numerical, memory, SAM, and EMA checks passed, but its first complete paired preflight measured a 1.160975x weighted latency ratio and rejected it before any accuracy query. The report explicitly left a fresh fixed width-288 experiment open.
- EXP-010 showed that removing an early full-resolution block in exchange for late depth scored only 95.04% despite more updates. This argues against purchasing late capacity by deleting early local processing; a taper that preserves the 2-2-2 block allocation remains untested.
- EXP-009's four-path FP32 SE package cost 20.7% latency despite adding less than 1% parameters, demonstrating that tiny dynamic modules can be much less H20-efficient than dense low-resolution convolutions.
- EXP-012, EXP-013, EXP-015, and EXP-021 did not improve the tip through narrower averaging, loss, or auxiliary-supervision interventions. The remaining limiter is credible representational headroom at acceptable exposure cost, not memory: EXP-014 peaked at only 652 MiB in its isolated preflight.
- The developed width-288 proposal computes 3,260,442 parameters (+18.61%) and 425,315,136 Conv/Linear MACs per image (+8.33%) without adding modules, stochastic choices, or kernel launches. Interpolating EXP-014's measured cost gives a 1.076x latency prior, though this must be measured rather than assumed.

## Collected Ideas

## Combinations

## Candidate Ideas

### Bottlenecked Grouped Late-Stage Expansion
**Summary**: Preserve the 256-channel backbone and add a single identity-initialized grouped bottleneck residual transform at 8x8 before the final pool, using a narrow reduction and grouped spatial mixing to add nonlinear late capacity at fewer dense MACs than widening both final blocks.

**What it targets**: The same late semantic-capacity limiter, but tries to increase representational depth and transformation diversity without paying the full dense width-288 arithmetic cost.

**Reasoning**: EXP-014 explicitly names bottlenecked or grouped late expansion as an open avenue, and abundant memory makes state size irrelevant. Identity initialization could preserve the parent function at step zero. However, EXP-009 shows that operation count and launch granularity dominate small-module FLOP estimates on this H20 workload; a new reduction/group/spatial/expansion path has substantially weaker systems evidence than dense existing blocks.

**Sources**: EXP-014 `04-analysis.md`; EXP-009 `04-analysis.md`; `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`.

**Estimated Effort**: high

**Risk Assessment**: Grouped or bottleneck kernels may be launch-bound, identity initialization may delay useful learning inside the short budget, and a new residual geometry adds correctness and optimization uncertainty. Exact dimension, grouping, initialization, and LR treatment are underdetermined enough to make this a less controlled one-shot experiment.

### Final-Block-Only Identity SE
**Summary**: Add one identity-centered squeeze-and-excitation gate only to the final 256-channel block, avoiding EXP-009's four separate FP32 gates and keeping all convolution widths unchanged. Use the existing standardized descriptor and zero-initialized centered gate design, with a fixed isolated parameter initialization and no recipe retuning.

**What it targets**: Channel-selection quality at the final semantic representation rather than raw capacity, while restricting dynamic channel interactions to one low-resolution location.

**Reasoning**: SE provides a direct representation mechanism and EXP-009 established that the implementation was numerically live; that experiment rejected the four-gate execution package before accuracy, not the accuracy mechanism. A single final gate would reduce launch count, but it still inherits the exact class of small FP32 operations that caused 20.7% overhead and has no measured latency point below the project's suggested 7.5% credibility threshold.

**Sources**: EXP-009 `04-analysis.md`; `knowledge/papers/squeeze-and-excitation.md`; EXP-008 `04-analysis.md`.

**Estimated Effort**: medium

**Risk Assessment**: The final-only gate may have too little mechanism reach, or still be disproportionately slow. Its separate initialization and optimizer-group design introduce more degrees of freedom than width 288, and the zero-initialized gate may not become useful quickly enough.

### Fixed 288-Channel Final-Stage Taper
**Summary**: Change EXP-011's six residual widths from `64/64/128/128/256/256` to `64/64/128/128/288/288`, and update only the final BatchNorm and classifier dimensions plus truthful metadata. Preserve the complete CutMix, late period-two SAM, time-calibrated EMA, fixed seed, and evaluation protocol. This is one fixed package with no fallback to 272 or 320 after timing.

**What it targets**: The stable 95.49-95.61 EMA generalization plateau identified in `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`, by adding semantic capacity only at 8x8 while retaining early local processing and most optimizer exposure.

**Reasoning**: This is the only capacity point backed by a measured device-specific cost curve and a prior preregistration. It adds 511,552 parameters but only 8.33% MACs, no extra launches, and uses multiple-of-32 dimensions. EXP-014 proved the same code path and expanded EMA/SAM state are operationally sound at width 320, while rejecting only that larger point's cost. The 1.076x interpolated latency prior leaves substantially more dose than width 320 and plausibly keeps roughly 23.5-24k updates from EXP-011's 25,798. The fixed-seed initialization and downstream data RNG stream will change with tensor shapes, so the result must be attributed to the complete width-288 seed-42 package.

**Sources**: EXP-014 `04-analysis.md`; EXP-021 `proposals/idea-03.md`; `knowledge/papers/deep-pyramidal-residual-networks.md`; EXP-010 `04-analysis.md`; EXP-011 `04-analysis.md`.

**Estimated Effort**: medium

**Risk Assessment**: Eight to ten percent slower updates may erase any capacity benefit, and width alone may worsen overfitting under a fixed-time recipe. Shape-dependent initialization prevents paired causal isolation. A preflight must reject unexpectedly poor H20 kernel efficiency, inadequate projected dose, or insufficient EMA samples before the sole metric run.

## Review

The independent fallback critic selected the fixed width-288 taper. Its strongest concern is that the detailed seed proposal was originally framed against EXP-004: EXP-022 must instead use EXP-011's 95.61% metric, 25,798-step dose, path mix, and cadence-31 full-state EMA. The plan will therefore use a formal 95.71% threshold, expand and audit all EMA shadow/swap/restore state, and weight paired timing with EXP-011's path counts. The review also cautions that EXP-014 supplies direct feasibility evidence but no accuracy evidence, and that PyramidalNet is only a mechanism prior. The hypothesis is consequently framed as a controlled capacity bet. The fixed-seed architecture changes common initialization and the later DataLoader stream; one seed remains mandatory, with no submatrix copying, draw burning, or retry. Best, final, final-16 mean/range, optimizer dose, and EMA samples will all be recorded, but only the primary metric determines the formal verdict.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Width 288 is the only candidate with an exact geometry, parameter/MAC accounting, a directly relevant H20 timing datum, no new kernel topology, and enough plausible reach to clear the 0.10-point bar. The grouped bottleneck is underdetermined and exposed to launch and zero-initialization risks; final-only SE returns to the small-FP32-operation class already measured as inefficient and has weaker reach.

## Chosen Idea
**Selected**: Fixed 288-Channel Final-Stage Taper

**Why this idea**:
It spends unused memory on two existing low-resolution dense blocks, preserves the validated early processing and complete EXP-011 training stack, and has a device-specific latency prior derived from the operationally sound width-320 preflight. Unlike the alternatives, it adds no module or launch topology and requires only a bounded shape change. The intervention is still risky because no widened model has produced an accuracy measurement and fixed-seed shape changes alter the full stochastic package; those limitations are explicit rather than obscured.

**Hypothesis**:
Under one fixed seed-42 run, changing EXP-011 from `64/128/256` to `64/128/288` while preserving CutMix, SAM, and cadence-31 EMA will remain within a preregistered paired-latency and projected-dose envelope and raise `best_test_acc` from 95.61% to at least 95.71%. Strong mechanism support additionally requires at least 23,200 optimizer steps, adequate EMA sampling, a final result within 0.15 points of the best, and a final-16 accuracy mean that is materially above EXP-011's 95.493%; these diagnostics cannot override the formal primary-metric verdict.
