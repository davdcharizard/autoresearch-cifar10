# Report EXP-010: Conservative Plateau CutMix
- **Created**: 2026-08-06

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the width-2 moving baseline of 93.55% at `8faf0f3`. A valid improvement required at least 93.65% under the fixed one-H20, 300-second, `train.py`-only protocol.

## Idea & Hypothesis

Compose alpha-1 CutMix with a fixed 50% of accepted N1/M7 plateau batches, then preserve the exact 80% transition to the hard-label weak tail. Claude selected this over BF16 and late averaging because it introduced class-bearing regional target geometry rather than adding exposure to or averaging an already-flat solution. The hypothesis predicted at least 93.65% with at least 97% update retention; a switch checkpoint below 87.08% would indicate compounded underfit.

## Approach

Changed only `train.py` at commit `7c1e7d8`. Added a forkserver-safe worker collator using installed torchvision v2 CutMix, alpha 1.0 and probability 0.5. CPU RNG is saved/restored around the new gate and transform; the strong loader alone uses the collator, while the rebuilt weak loader returns ordinary integer labels. Target-format counters run before timed work, weak targets are asserted one-dimensional, and the existing switch log records realized mixing. Model, optimizer, N1/M7 transform, LR/phase schedule, seed, timer, and evaluator remain fixed.

## Execution

Mandatory external Claude idea and plan reviews completed with exit code 0; no fallback reviewer was used. Functional, RNG, target-area, worker, lifecycle, isolated H20 loss-path, and integrated contention gates passed. Hard/soft median steps were 10.823/10.829 ms; the joint 1,000-step test had 1.045x wall/count ratio; warmed workers delivered 179.51-199.89 batches/s. One fixed-seed run exited 0 without retry in 330.7 seconds total. The only preflight issue was a disposable script missing its Python 3.14 forkserver main guard; adding the standard guard fixed the harness without changing candidate code.

## Results

- **Primary metric**: `94.15%` (baseline: `93.55%`, delta: `+0.60` percentage points, `+0.64%` relative)
- **Observations**: The run mixed 10,673 of 21,446 strong batches (49.77%). Its final strong checkpoint was 89.73%, only 0.35 below EXP-007's 90.08% and safely above the 87.08% underfit marker. The first weak checkpoint reached 93.16%, 0.20 above EXP-007; the tail passed the gate at epoch 60, reached 94.06% by epoch 65, and finished at its best, 94.15%. Final NLL was 0.1934 versus EXP-007's 0.2196. It completed 26,898 steps, 99.10% of EXP-007's 27,143, with unchanged 598.7 MB peak VRAM.
- **Analysis**: The hypothesis is strongly supported as a net fixed-time result. Class-bearing regional mixing complemented rather than replaced broad N1/M7 invariances, and width 2 absorbed the extra plateau difficulty without the collapse seen from Cutout or `5e-4` decay. The hard tail converted the slightly lower clean strong checkpoint into an immediate higher weak checkpoint, then continued improving through the final epoch. Near-identical hard/soft step cost and worker headroom rule out extra exposure as the gain; the 0.60-point margin also exceeds Claude's concern about a bare 0.10-point single-run pass. The experiment does not isolate regional mixing from the altered augmentation/RNG trajectory, and one fixed seed is not a precise effect estimate, but its exact scope, large margin, lower NLL, and lifecycle integrity make the protocol improvement trustworthy.
- **Key Learning**: Plateau-only 50% CutMix raised accuracy 0.60 points with 99.10% step retention; regional mixing complements RandAugment and hard-tail refinement.

## Verification

- **Conditions**: All passed: 94.15% >=93.65%, exit 0, ten finite summary fields, 300.0 counted seconds, 330.7 total seconds, one 80.0% switch with eight stopped workers, 19 unique evaluation epochs, 49.77% realized mixing, 26,898 steps, and 1,073,962 parameters.
- **Review Notes**: Results confirmed trustworthy. One idle H20 was used; only the reviewed `train.py` CutMix diff changed; seed, evaluator, architecture, optimizer, timing, schedule, and weak tail remained fixed; no soft target crossed the switch; no reroll occurred. The gain is the net fixed-seed CutMix training-method effect, not a claim that every stream realization gains exactly 0.60 points.
- **Verdict**: improvement
- **Verdict Basis**: Every integrity condition passed, and the primary metric exceeded the moving baseline by 0.60 points and the acceptance threshold by 0.50.

## Unexplored Avenues

- Tune CutMix probability only with a separately reviewed experiment. The successful 0.5 point leaves room in either direction, but lower probability may preserve strong fit while higher probability may improve regional invariance; neither is implied by one point.
- Change the phase relationship rather than alpha: an earlier CutMix-off interval while retaining N1/M7 could provide hard-label adaptation before the existing weak tail, but persistent-worker state and an extra transition make it a distinct mechanism.
- Move the full strong-to-weak boundary earlier only with care. EXP-005 showed 75% hurts without CutMix, while EXP-010's rising final tail suggests composite training may alter that tradeoff.
- Combine CutMix with a representation change only after isolated review; stacking another regularizer immediately would lose the clean successful attribution.

## Next Steps

- **High confidence**: preserve width 2, all-parameter `1e-4`, and the complete successful CutMix/N1-M7-to-hard-tail recipe as the new baseline.
- **Medium confidence**: adversarially compare CutMix probability or CutMix-off timing, prioritizing an isolated change with the same target/lifecycle audits.
- **Medium-low confidence**: revisit a higher-ceiling architecture lever if CutMix tuning lacks a mechanism for clearing the new 94.25% threshold.

## Exit Action Results

- None defined.
