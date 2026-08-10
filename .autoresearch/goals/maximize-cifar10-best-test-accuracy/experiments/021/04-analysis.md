# Report EXP-021: Deterministic Pool-First Option-A Shortcuts
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc` above the 94.15% frontier at `7c1e7d8` under the fixed seed-42, one-H20, 300-counted-second, `train.py`-only protocol. Formal improvement required at least 94.25% with a complete trustworthy run.

## Idea & Hypothesis

Replace only the two transition shortcut `::2` samples with fixed nonoverlapping 2x2 average pooling before the existing Option-A zero channel pad. This isolated deterministic all-position downsampling from EXP-017's learned 1x1 projection and shortcut BN. The registered hypothesis was at least 98% exposure retention and `best_test_acc>=94.25%`; external Claude review correctly reframed no-improvement as the more likely result and asked for a mechanistic comparison with EXP-017.

## Approach

The sole production diff in `train.py` replaced shortcut slicing with explicit `F.avg_pool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)`. Parameters, learned state, residual branches, channel padding, RNG, data, optimizer, LR/phase schedule, timer, evaluator, and logging stayed unchanged. An ignored Python-3.14-safe controller checked exact shapes/values/gradients, then persisted one 200-batch post-transform corpus and replayed it in fresh deterministic accepted/candidate processes.

## Execution

Mandatory external Claude idea and plan reviews both exited 0; no fallback reviewer was used. Static and semantic gates passed. The immutable corpus contained exactly 100 hard and 100 CutMix batches, SHA-256 `9b2801ac6d2d07f9bc5a5204e370db815f1b8bfaca8a1b9848a97091703388ea`, and all eight workers stopped. Candidate and control began from identical model/RNG hashes and used identical backend flags. The registered candidate-only class-concentration gate failed at steps 17 and 18, so timing and production were correctly skipped without retry, corpus regeneration, threshold relaxation, or variant testing. No `run.log` or test metric was produced.

## Results

- **Primary metric**: `NaN` / not run (baseline: `94.15%`, delta: N/A)
- **Observations**: All 200 steps remained finite. At step 17 the candidate predicted class 5 for 123/128 examples (96.09375%) versus control 113/128 (88.28125%); at step 18 it predicted class 5 for 128/128 (100%) versus control 112/128 (87.5%). Candidate terminal loss EMA was nevertheless lower, 2.11687 versus 2.24978 (ratio 0.940923), with smaller maximum gradient/update norms (12.6602/1.70546 versus 14.2152/1.84540). Candidate concentration fell to 90.625% at step 19, so the evidence is a sharp transient, not non-finite divergence.
- **Analysis**: The exact pool-first Option-A point failed its pre-registered safety contract despite clean semantics and complete state/corpus parity. Lower loss does not rescue a candidate-specific one-class transient; it shows that average shortcut filtering changed early logit geometry rather than merely slowing or destabilizing arithmetic. The control was also class-skewed at these early steps, but only the candidate crossed 95%, twice, on identical inputs. Because production was vetoed, EXP-021 cannot adjudicate the planned switch/first-weak/NLL discriminator or claim that every pool-first model loses accuracy. Together with EXP-017, however, pool-first transition shortcuts now have two distinct negative mechanisms: learned projection-BN harmed late NLL/top-1, while deterministic Option-A filtering created an unsafe early transient. This makes further shortcut-pooling variants low priority in the accepted short-horizon recipe.
- **Key Learning**: Average-pooled Option-A transitions produced candidate-only one-class transients on exact shared batches, making this deterministic anti-aliasing point unsafe.

## Verification

- **Conditions**: Exact semantics, parameter/state/RNG alignment, immutable corpus, target coverage, worker shutdown, state finiteness, and loss-EMA gates passed. Candidate-only class concentration failed at steps 17 and 18; timing and fixed-budget accuracy conditions were not reached.
- **Review Notes**: The failure is trustworthy rather than infrastructural. Both fresh arms shared the same corpus hash, starting state hash, RNG hash, deterministic backend, and 100-hard/100-mixed sequence; diagnostics were serialized before veto assertions. No test evaluation, retry, or fallback variant occurred.
- **Verdict**: invalid
- **Verdict Basis**: The registered safety gate prevented production, leaving only partial controller results and no trustworthy primary metric. Metric is therefore `NaN`; this is not a crash or hard-constraint violation.

## Unexplored Avenues

- Filtering both residual and shortcut downsampling could avoid the registered branch-spectrum mismatch, but it changes a larger graph surface and has no evidence that it avoids the early class transient; treat it as a distinct low-priority hypothesis.
- A learned identity-informed projection without shortcut BN differs from both EXP-017 and EXP-021, but two pool-first failures now make transition-path work lower expected value than representation changes outside downsampling.
- Do not revisit this exact 2x2 box-filter Option-A point by relaxing the concentration threshold, regenerating batches, filtering one transition, or adding a rescue inside the same experiment.

## Next Steps

- **High confidence**: Preserve the accepted strided-slice Option-A transitions and move the next representation search outside transition shortcuts.
- **Medium confidence**: Refresh literature around low-cost final-stage feature calibration or aggregation that preserves active postactivation residual branches and passes first-update/production-batch gates.
- **Medium confidence**: Prefer a clean generalization mechanism over another exposure-only systems candidate or scalar CutMix interpolation, consistent with both Claude reviews and the current bottleneck diagnosis.

## Exit Action Results

- None defined.
