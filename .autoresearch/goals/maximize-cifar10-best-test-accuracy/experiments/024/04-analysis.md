# Report EXP-024: Depth-Preserving Final-Stage Widening to 160
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` above the moving 94.15% baseline under the fixed evaluator, one-H20 hardware, 300-second counted training budget, seed-42, and `train.py`-only scope. This experiment tested whether preserving ResNet-20 depth while widening only the final 8x8 stage could reach the current 94.25% acceptance threshold.

## Idea & Hypothesis

Keep the accepted 32/64 early and middle stages and all nine residual blocks, but widen `layer3` from 128 to 160 channels. This directly followed EXP-023's diagnosis that deleting one block per stage outweighed global width-3 capacity. The hypothesis predicted that low-resolution semantic capacity would preserve healthy strong-phase fit, retain at least 24,000 projected updates, and reach 94.25-94.30%.

## Approach

Added one optional final-stage override to `ResNet`, set production to 160 final channels, and left every `BasicBlock`, Option-A transition operation, initializer, optimizer, augmentation, schedule, timer, worker, evaluator, and logging path unchanged. The candidate is an FP32 32/64/160 ResNet-20 with nine blocks, 19 convolutions, Option-A pads 32/96, a 160-to-10 classifier, and exactly 1,507,818 parameters. A reviewed preflight persisted one SHA-256-addressed corpus with 100 strong-hard, 100 strong-soft CutMix, and 100 weak-hard batches. Fresh explicit `ResNet(3,10,2)` and `ResNet(3,10,2,160)` processes replayed the same 200 strong batches before any timing or scored run.

## Execution

Static structure, FP32 state, lint, formatting, diff, and tracked-scope checks passed. The corpus was 472,747,293 bytes with SHA-256 `d4294f5adb2e58e0847366231458b21901c6f01f270d4cd1c9eae14a05b64565`. Both arms ran all 200 safety steps with finite parameters, buffers, momentum, loss, and BN counters. At step 2—the first strong-soft batch—the candidate assigned 126/128 predictions to one class (98.4375%), while the aligned control assigned 78/128 (60.9375%). The plan required no candidate-only share above 95%, so the controller serialized the evidence and vetoed the experiment. Timing and production were not launched, no `run.log` was created, and no retry or alternate width was attempted.

## Results

- **Primary metric**: unavailable (baseline: 94.15%; delta: N/A)
- **Observations**: Candidate/control step-2 losses were 3.8348/5.0240, so the concentration event did not present as a higher-loss or non-finite failure. Over all 200 steps, candidate terminal loss EMA was 1.96966 versus 2.08604 control, a 0.94421 ratio; both trajectories remained finite and completed. Only one candidate-only concentration event occurred.
- **Analysis**: The evidence is sufficient to reject production under the predeclared integrity gate, but it does not establish that the 32/64/160 model would sustain collapse or miss the accuracy target. The transient appeared immediately on the first CutMix-target step and then recovered, while terminal loss was lower. Its net cause cannot be isolated: the intervention jointly changes the 64-to-160 Option-A identity fraction, final-stage convolution/BN initialization, classifier width, parameter count, and subsequent RNG consumption. This is therefore an invalid/no-go experiment rather than an accuracy failure. It strengthens the local warning that lower short-horizon loss does not rule out class-concentrated transients, now for an abrupt asymmetric architecture change as well as prior optimizer-path changes.
- **Key Learning**: Asymmetric 32/64/160 width caused a candidate-only 98.44% class transient at step 2 despite lower terminal loss; this point is unsafe.

## Verification

- **Conditions**: All goal necessary conditions were skipped because the mandatory safety preflight failed before production.
- **Review Notes**: The veto is trustworthy: both arms used explicit constructors, reset seed-42 CPU/CUDA state, and consumed byte-identical serialized batches; the report was written before the assertion. The event is a single transient, so it should not be overinterpreted as sustained model collapse, but the predeclared no-event gate cannot be relaxed after observing it.
- **Verdict**: invalid
- **Verdict Basis**: No scored metric exists after the safety process-integrity gate blocked timing and production; metric is recorded as `NaN`.

## Unexplored Avenues

- A less abrupt final-stage width such as 144 would preserve a larger Option-A identity fraction, but it is a new operating point requiring independent ideation and must not be treated as a rescue of EXP-024.
- Conditional final-stage capacity with an exact accepted initial function, such as identity-scale ECA, avoids changing the shortcut channel ratio and may separate capacity allocation from static width.
- A learned projection is not a direct remedy: EXP-017 already found learned pool-first transition shortcuts unhelpful, and changing both projection and width would lose attribution.

## Next Steps

- **Medium-high confidence**: Evaluate identity-initialized final-stage ECA, subject to exact-corpus recruitment and a tight launch-overhead timing gate.
- **Medium confidence**: Probe channels-last on the unchanged accepted model to seek extra exposure without altering architecture or optimizer geometry.
- **Low confidence**: Reconsider bounded smooth pooling only if its expected effect size can justify the added reduction kernels.
