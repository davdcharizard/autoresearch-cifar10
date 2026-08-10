# Report EXP-012: Canonical Full Preactivation
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc (%)` from the 94.15% moving baseline at `7c1e7d8`. A valid improvement required at least 94.25% under the fixed one-H20, seed-42, 300-second training protocol.

## Idea & Hypothesis

Test a paper-faithful full-preactivation width-2 ResNet-20 on the complete accepted EXP-010 p=0.5 CutMix recipe. The external Claude idea review initially preferred identity-initialized SE, but required a paired timing condition; SE failed decisively at 1.23324x training cost and only 21,810 projected steps. The review explicitly selected preactivation if that gate failed. The hypothesis was that the complete preactivation package would improve representation geometry enough to reach 94.25% while retaining at least 97% of accepted exposure.

## Approach

Changed only the model architecture in `train.py`. All nine blocks became `BN-ReLU-Conv-BN-ReLU-Conv-add` with no post-add ReLU. Six ordinary shortcuts retained raw identity values; the first and two Option-A transition shortcuts used preactivated values. The stem lost BN-ReLU and a terminal BN-ReLU was added before pooling. The model kept 1,073,962 parameters, 19 convolutions, 19 BNs, 19 ReLUs, identical shared Conv/Linear initialization and post-construction CPU RNG, and the complete EXP-010 data, optimizer, schedule, timer, worker, evaluator, seed, and logging mechanics. No zero-gamma was combined because it would deadlock the canonical branch at ReLU zero.

## Execution

Mandatory external Claude idea and plan reviews completed successfully with no fallback reviewer. Structural checks proved exact shortcut semantics, gradients, target compatibility, parameter membership, and RNG/weight alignment. Five alternating fresh-process H20 timing pairs measured a 0.99378 candidate/control training ratio, 27,066 projected steps, and stable CV below 0.51%; inference was 1.00925x and projected total runtime was 330.73 seconds. One fixed-seed run exited 0 without retry in 332.2 seconds total.

## Results

- **Primary metric**: `94.22%` (baseline: `94.15%`, delta: `+0.07` percentage points, `+0.07%` relative)
- **Observations**: The candidate completed 27,029 steps, 100.49% of EXP-010 exposure, with 582.7 MB peak VRAM. Its final strong checkpoint was 86.88%, 2.85 points below EXP-010's 89.73% and 0.20 below the predeclared 87.08 underfit marker. The hard weak tail recovered immediately to 93.48%, 0.32 above EXP-010's first weak checkpoint, then reached 94.20% at epoch 62 and finished at its best of 94.22%. Final NLL was 0.1974 versus EXP-010's 0.1934. Realized CutMix was 10,787/21,665 strong batches (49.79%), and all eight workers stopped correctly.
- **Analysis**: Compute and exposure did not limit the candidate: it ran slightly more steps than the accepted architecture with aligned shared initialization. The full-preactivation package instead fit the strong composite phase substantially less well. Its superior immediate weak checkpoint shows that the learned representation converted effectively to hard clean labels, but the fixed tail recovered only enough to add 0.07 points and left worse NLL. Because reorder, boundary shortcuts, unrectified sums, and terminal conditioning changed together, the run rejects this complete package at the fixed gate rather than isolating any component. The +0.07 result is within the one-seed noise floor and only 0.03 below threshold, so it is weak evidence against preactivation generally; it is nevertheless a valid no-improvement with no reroll.
- **Key Learning**: Full preactivation was compute-neutral and recovered strongly in the weak tail, but deeper strong-phase underfit limited its gain to +0.07 points.

## Verification

- **Conditions**: Hardware, scope, completion, numeric summary, fixed timing, parameter count, worker lifecycle, CutMix provenance, unique evaluation epochs, and exposure passed. Primary accuracy failed: 94.22% <94.25%.
- **Review Notes**: Results are trustworthy. Only `train.py` changed; model boundaries and shared RNG/weights were tested directly; the H20 was idle; the run exited 0 with one seed and no retry; realized exposure exceeded both the plan floor and accepted baseline. The small delta should not be over-interpreted causally.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid run improved by only 0.07 percentage points, below the required +0.10 margin.

## Unexplored Avenues

- Zero-initialize the final residual BN scale on the accepted postactivation architecture. This is compute-neutral and avoids canonical preactivation's dead-branch issue, but may slow early fitting under the short horizon.
- Isolate terminal BN-ReLU feature conditioning on the accepted blocks. It would test the most representation-relevant subcomponent, though changing BN count/parameters makes it a distinct architecture experiment.
- Projection shortcuts remain feasible at about 1.87% measured overhead, but the adversarial idea review judged their ceiling low in a nine-block network.
- A reduced-scope or fused channel-selection mechanism could avoid the measured SE launch cost, but it must be treated as a new reviewed idea rather than a rescue of the rejected all-block SE candidate.

## Next Steps

- **Medium confidence**: adversarially evaluate zero-initialized final BN scales on the accepted postactivation network as a compute-neutral identity-initialization test.
- **Medium confidence**: prioritize representation changes whose strong-phase fit should not compound N1/M7 plus CutMix underfit.
- **Low confidence**: consider the already-timed Option-B projection shortcut only if the next review finds a sharper shallow-CIFAR mechanism.

## Exit Action Results

- None defined.
