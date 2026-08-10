# Report EXP-015: Same-Width Residual Identity Initialization
- **Created**: 2026-08-06

## Goal

Raise CIFAR-10 `best_test_acc` above the 94.15% frontier under the fixed seed-42, one-H20, 300-second training protocol while modifying only `train.py`. Formal improvement required at least 94.25%, normal completion, and total runtime below 600 seconds.

## Idea & Hypothesis

Zero the final BatchNorm scale in only the six non-entry same-width residual blocks, leaving every stage-entry block and both padded Option-A transitions active. The selected idea aimed to begin ordinary blocks as exact forward identities, recruit their residual branches after one bounded gamma update, preserve the accepted postactivation graph/exposure, and improve conditioning/generalization enough to reach 94.25%.

## Approach

Added a literal keyword-only `zero_init_residual` flag to `BasicBlock` and enabled it for block indices 1 and 2 in all three stages when stride/channels prove an equal-shape identity. Exactly 448 existing gamma scalars changed from one to zero; parameter count, state shapes, RNG, model operations, optimizer, data, CutMix, schedule, evaluator, timer, workers, and logging remained accepted. Mandatory external Claude idea and plan reviews completed successfully with no fallback. The plan review removed proposed production gamma logging, pinned visible H20 index 0, and separated formal verdict from exposure attribution.

## Execution

Structural and first-update checks proved six exact identities, live padded transitions, max first-step gamma 0.003296, improving replay loss, and complete residual-convolution recruitment on backward two. A 64-batch real strong-view test looked favorable: candidate/control terminal-loss EMA ratio was 0.769176 and candidate terminal concentration was 26.56%. The first timing attempt had one cold control-inference outlier; one unchanged infrastructure retry passed at 0.999219x training, 26,919 projected steps, 0.999527x inference, and identical memory. The sole production run then completed with exit 0 and no retry.

## Results

- **Primary metric**: 93.80% (baseline: 94.15%, delta: -0.35 points, -0.37%)
- **Observations**: The candidate completed 26,983 steps, 85 more than EXP-010, with unchanged 598.7 MiB allocation. Accuracy reached 87.36% at 40% but fell to 86.48% at the 80% switch, 3.25 points below EXP-010 and 0.60 below the underfit marker. The first weak checkpoint recovered to 93.17%, essentially matching EXP-010's 93.16%, then plateaued; final equaled best at 93.80% with NLL 0.2064 versus 0.1934. CutMix was 10,716/21,528 strong batches (49.78%), and all 19 evaluations were on unique epochs.
- **Analysis**: The intervention was numerically safe, compute-neutral, and fully recruited, so neither dead branches nor lost exposure explains the miss. Instead, suppressing six residual branches at initialization changed the long strong-view optimization path enough to peak early and lose fit by the switch. The hard weak tail immediately recovered the clean classifier but could not recreate the accepted representation/generalization ceiling. Most importantly, the favorable 64-step loss test inverted at the full phase: short optimization ease was not evidence of better strong-view representation learning.
- **Key Learning**: Selective zero-gamma can pass bounded first-update and short-fit gates yet still suppress long strong-phase representation learning; full-phase fit remains decisive.

## Verification

- **Conditions**: Primary accuracy failed. Scope, hardware, completion, summary, budget, wall time, parameter count, exposure, lifecycle, target provenance, and evaluation fairness passed.
- **Review Notes**: Results are trustworthy. The run was pinned to the sole idle H20, changed only reviewed initialization plumbing, exceeded accepted exposure, used one fixed seed with no reroll, and produced a coherent 19-checkpoint trajectory.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result completed under every hard constraint but finished 0.35 points below the baseline and 0.45 below the required threshold.

## Unexplored Avenues

- Initialize the same six gammas to a small positive value such as 0.1, retaining residual signal and first-step convolution gradients while reducing initial branch magnitude; this is a separately reviewed operating point, not a rescue.
- Scope identity initialization to fewer late-stage blocks, where residual refinement may benefit without making most within-stage computation initially dormant.
- Use a deterministic gamma ramp only if its counted overhead and additional schedule mechanism are justified; it trades the clean initialization-only attribution for controlled recruitment.

## Next Steps

- **BF16-funded representation change (medium confidence)**: require a measured speed gain and spend it on a reviewed accuracy mechanism rather than assuming extra accepted updates alone help.
- **Isolated Nesterov on the accepted schedule (medium confidence)**: revisit the optimizer component confounded by EXP-001's failed short hold, with no architecture or regularization change.
- **Transition information preservation (medium confidence)**: evaluate a compute-light anti-aliased shortcut or similarly narrow representation change that should not suppress strong fit.
