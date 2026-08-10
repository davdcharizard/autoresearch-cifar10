# Report EXP-016: 106-State Trailing Uniform Clean-Tail SWA
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the fixed seed-42, 300-charged-second, physical-GPU-0 protocol while modifying only `train.py`. EXP-016 grew from EXP-004 (`95.40%`), so formal local improvement required at least `95.50%`. The goal-wide best remained EXP-011 at `95.61%`.

## Idea & Hypothesis

Claude's adversarial idea review selected uniform full-state clean-tail SWA, conditioned on limiting state age. The refined hypothesis used a fixed 106-state trailing window sampled every 31 clean-tail optimizer steps: predicted mean state age `24.6894s`, ESS 106, balanced ordinary/SAM states, and no 6.30% first-state anchor. It was expected to preserve optimizer exposure and reach at least `95.50%`, with full-window tail behavior reported separately from the selected maximum.

## Approach

`train.py` gained a `TrailingUniformSWA` manager and fixed `SWA_START=0.75`, `SWA_UPDATE_EVERY=31`, and `SWA_WINDOW=106`. Before charged timing it allocates one leading-dimension GPU ring for every parameter and persistent floating buffer. Post-optimizer/post-SAM samples populate the ring; from sample 106 onward, direct reductions materialize the exact uniform mean and integer buffers copy the latest sampled state. Evaluation stays live until readiness and then swaps only the complete 106-state average into the existing model, calls the frozen evaluator once, and restores live state in `finally`.

The plan was hardened after Claude review: cumulative warm-up averages were excluded from evaluation, the median latency gate tightened to `1.005x`, the realized-dose floor increased to 25,400 steps, VRAM became informational, ring allocation moved before charged timing, and incremental sums were replaced with direct reductions. The implementation added reconstructable cadence, kernel, source-at-best, distance, BN, integer, RNG, alias, and restoration audits.

## Execution

Static scope, syntax, and CPU arithmetic/restoration checks passed. The CPU smoke reached two full wraps and reported `window=106 evictions=106 wraps=1 restore_checks=1 rng_failures=0`. A transient `/tmp` import-path issue was fixed before production checks. The first GPU preflight attempt stopped before numeric output because PyTorch-visible usable memory did not exactly equal the `nvidia-smi` physical label; the harness was corrected to the plan's approximate H20 check.

The sole complete numeric preflight then ran five alternating-order 248-step parent/candidate rounds on physical GPU 0. It guarded the evaluator, read zero test batches, exercised exactly two full-window updates per trace, and preserved exact online state. The preregistered ratio-dispersion gate failed, so execution stopped without a metric run, test accuracy, or `run.log`. No numeric gate was changed or retried.

## Results

- **Primary metric**: `NaN` (parent: `95.40%`; delta vs parent: N/A; global best: `95.61%`)
- **Observations**: Parent seconds were `[3.147468, 3.138780, 3.165140, 3.140729, 3.163192]`; candidate seconds were `[3.145776, 3.155481, 3.146648, 3.145871, 3.143007]`. Ratios were `[0.99946230, 1.00532079, 0.99415781, 1.00163726, 0.99361867]`. Parent drift `0.008375`, median ratio `0.999462`, maximum ratio `1.005321`, projected steps `25,573.751`, projected total `457.488s`, and candidate peak `1,738.40 MiB` all passed their applicable checks. MAD/median was `0.00530734`, narrowly above the fixed `0.005` ceiling.
- **Analysis**: The implementation demonstrated correct, inexpensive state management in the measured traces, but the experiment generated no accuracy evidence and cannot validate or reject the SWA hypothesis. The failure is best understood as a conservative measurement-protocol rejection: observed median and maximum overhead were benign, while five-round dispersion exceeded its ceiling by `0.00030734`. Claude Opus independently recomputed the ratios and MAD, confirmed the failure was arithmetically correct, found no hard-constraint or reward-hacking issue, and recommended `crash/NaN` (`03-result-review.md`).
- **Key Learning**: A correct near-zero-overhead boxcar can still be blocked by an under-calibrated five-round dispersion gate; preflight stability criteria need measured noise support.

## Verification

- **Conditions**: Pre-metric paired-ratio MAD/median failed (`0.00530734 > 0.005`); metric verification was not run.
- **Review Notes**: GPU identity, scope, evaluator guard, zero test access, direct arithmetic, exact online-state equality, restoration, median latency, maximum latency, projected dose/runtime, and memory evidence were trustworthy. The dose projection formula was not printed in the JSON and should be made independently auditable in future harnesses. Claude returned `REVIEW: PASS` for the record and classification.
- **Verdict**: crash
- **Verdict Basis**: No primary metric was produced because execution stopped at a preregistered numeric preflight gate; per the tree guide this records metric `NaN`, not `no-improvement` or `invalid`.

## Unexplored Avenues

- Revisit the same 106-state estimator only in a newly preregistered experiment whose paired-noise gate is calibrated from more rounds or a justified tolerance; EXP-016 provides no accuracy evidence against the method.
- Use a fused packed representation or amortized reduction if later measurements expose charged reduction overhead, although EXP-016's `0.999462x` median provides no current evidence that this is necessary.
- Compare a younger uniform suffix against EXP-011 only after the evaluation count and max-selection premium are explicitly powered; suppressing pre-fill SWA leaves roughly eleven mechanism-bearing evaluations.

## Next Steps

- **High confidence**: return to EXP-011 and seek an additive representation or training mechanism large enough to move its stable 95.49 tail, avoiding another estimator micro-correction.
- **Medium confidence**: explore a low-kernel-count capacity intervention that uses H20 memory headroom while preserving at least 25,400 optimizer steps.
- **Low confidence**: revisit age-matched trailing SWA with a calibrated, more auditable preflight; the mechanism remains untested rather than disproven.
