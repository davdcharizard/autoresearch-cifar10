# Report EXP-003: Balance CutMix and Drop Path
- **Created**: 2026-08-05

## Goal

Maximize CIFAR-10 `best_test_acc` under the fixed 300-second charged training budget by testing whether the EXP-002 regularization balance could be improved. The parent and accepted global best were EXP-002 at 95.23%; a valid improvement required at least 95.33%.

## Idea & Hypothesis

Sweep CutMix probability and maximum drop path because both regularize the first 75% of charged training time. The hypothesis was that their parent values might be redundant: less drop path could improve fitting under CutMix, while a different CutMix frequency could improve the mixed-to-clean supervision balance.

## Approach

A deterministic `itertools.product` grid covered `CUTMIX_PROB` in `{0.35, 0.50, 0.65}` and `MAX_DROP_PATH` in `{0.04, 0.08}`. Five new points ran serially on physical GPU 0; the parent point `(0.50, 0.08)` reused its existing 95.23% measurement. Search trials used ephemeral copies of `train.py`, fixed seed 42, identical architecture and optimization, the frozen 300-second budget, and a 600-second outer timeout. Only independently confirmed results were eligible for the verdict.

## Execution

All five new search trials completed normally. Search ranked `(0.65, 0.08)` first at 95.48% and `(0.65, 0.04)` second at 95.42%. The first candidate's single confirmation scored 95.19%; the fallback candidate's single confirmation scored 95.28%. Both were rejected without retry. A required Claude adversarial review rejected confirming the next point because its 95.28% search score was already below the 95.33% gate, exhausting eligible candidates. The EXP-002 parent configuration was restored and all transient logs were removed.

## Results

- **Primary metric**: 95.28% (parent: 95.23%, delta vs parent: +0.05 points, +0.05%; accepted global best: 95.23%)
- **Observations**: Increasing CutMix probability appeared favorable in the selected search runs at both drop-path levels, but the two eligible candidates regressed by 0.29 and 0.14 points on confirmation. The best confirmed run used CutMix 0.65 and drop path 0.04, ending at 95.25% accuracy and 0.1963 loss after 26,596 steps. It used the same 1,178.9 MiB peak allocation and 2,748,890 parameters as the parent.
- **Analysis**: The grid did not support a reliable change to the parent regularization balance. The apparent monotonic CutMix benefit was not stable under confirmation, while reducing drop path produced only a +0.05-point confirmed gain. This discredits simple probability/drop-path scalar tuning in this local range, but not different augmentation mechanisms or schedules.
- **Key Learning**: Higher CutMix looked favorable in selected search runs but failed confirmation, exposing variance rather than a reliable regularization gain.

## Verification

- **Conditions**: Accuracy condition failed; runtime and process conditions passed.
- **Review Notes**: Both confirmation runs used physical GPU 0, fixed seed 42, exactly 300.0 charged training seconds, complete summaries, one evaluation per epoch, and total runtimes below 600 seconds. Results are trustworthy; search maxima were not substituted for confirmation metrics.
- **Verdict**: no-improvement
- **Verdict Basis**: The best confirmed 95.28% result improved only 0.05 points over the parent, below the required 0.10-point margin. No hard constraint was violated.

## Unexplored Avenues

- Change CutMix's alpha or active-time window rather than only its probability; these alter patch geometry or scheduling and could produce a more stable effect.
- Test a different complementary augmentation such as random erasing or policy-based color/geometric transforms instead of adding more strength to the same CutMix/drop-path pair.
- Reduce measurement selection pressure by reserving confirmation for candidates with a larger search margin or by comparing mechanisms rather than narrowly spaced scalar settings.

## Next Steps

- **High confidence**: Explore a qualitatively different generalization mechanism, such as EMA or model averaging, because scalar regularization tuning did not yield a stable gain.
- **Medium confidence**: Test an augmentation mechanism that changes invariances rather than CutMix frequency, while retaining dedicated RNG isolation.
- **Medium confidence**: Revisit capacity or block allocation within the large VRAM headroom if recipe-level changes plateau.
