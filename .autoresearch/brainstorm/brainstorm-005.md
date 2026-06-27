# Brainstorm EXP-005
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search this loop — the knowledge base and in-project scaling measurements cover the candidate space:
- **WRN (arXiv 1605.07146)** (knowledge/README.md): width gains on CIFAR-10 continue to 8–12x at this depth *given sufficient training*; the binding question here is epochs under the fixed time budget, which only in-project data answers.
- **TrivialAugment (knowledge/papers/trivialaugment.md)**: validated in-project (EXP-004); notes that augmented runs spend mid-schedule below less-augmented ones and cash in during the anneal — relevant for judging undertraining at fewer epochs.
- **In-project throughput scaling** (exp-reports 001/002): time/epoch at 4x = 2.63s, at 8x = 7.5s (2.85x for 4x FLOPs — sublinear, GPU utilization improves with width). Predicts ~80 epochs at 5x, ~60 at 6x.

## Experimental History Review

- **Trajectory**: 91.97 → 93.16 (recipe) → 95.23 (4x width) → 94.41 (8x, no-improvement) → 96.06 (RandomErasing) → **96.23 (TrivialAugmentWide)**. Baseline: 96.23 @ 1174e0d.
- **Patterns**: time-keyed one-cycle composable (validated 3x); width scales steeply (+2.07pp at 4x) but trades against epochs; augmentation stacking shows steeply diminishing returns at 4x (RE +0.83, then TA only +0.17) — bottleneck has rotated back to capacity/throughput.
- **Failed Approaches**: capacity without throughput — 8x starved at 40 epochs (−0.82pp), final=best is the undertraining signature. That failure predates BOTH regularizers; width itself is not dead, but the epoch floor is real.
- **Protocol Findings**: eval overhead scales with epoch count; heavier models have MORE wall-clock headroom (fewer epochs → fewer evals).
- **Untried gaps**: intermediate width 5–6x on the regularized recipe, torch.compile throughput, third regularizer class (Mixup/CutMix — flagged low-confidence in exp-report-004), ResNet-9 topology, larger batch.

## Candidate Ideas

### 1. Width 5x on the doubly-regularized recipe (stage widths 80/160/320, ~6.7M params)
**Summary**: WIDTH_MULT 4 → 5; everything else (recipe, TA, RE) unchanged. Predicted ~75–85 epochs from in-project scaling.

**Reasoning**: Capacity is the rotated bottleneck (augmentation returns collapsed 0.83 → 0.17pp), and the two regularizers raise effective data complexity, shifting the width-epoch optimum right of where EXP-002 probed it. The risk dimension is epoch starvation, and it is width-graded: 5x keeps ~80 epochs — close enough to 114 that the regularized recipe should still converge (EXP-004's anneal completed with final=best; the time-keyed schedule adapts automatically) — while adding 56% parameters. The +0.1pp pass bar does not require the full width payoff, just a positive slope at 5x. If 5x banks a gain, 6x becomes next loop's question from a higher baseline; if 5x fails, the width direction at current throughput is conclusively dead and torch.compile becomes the unlock.

**Sources**: experiment-indices rows 001/002/003/004; goal-learnings § Patterns (width scaling, diminishing augmentation returns) and § Failed Approaches (epoch starvation); arXiv 1605.07146.

**Estimated Effort**: trivial (one constant)

**Risk Assessment**: Moderate — the width-accuracy curve between 4x and 8x is unmeasured; 5x could land within noise if the no-reg optimum was already near 4x and regularization only partially compensates. Clean failure mode (no-improvement); VRAM ~2.5GB, wall clock SHRINKS (fewer evals).

### 2. Width 6x on the doubly-regularized recipe (stage widths 96/192/384, ~9.6M params)
**Summary**: WIDTH_MULT 4 → 6; predicted ~60 epochs.

**Reasoning**: Bigger capacity step, bigger payoff if the regularized optimum moved well right of 4x. But 60 epochs is only 1.5x the epoch count at which 8x starved badly, and the stacked augmentation makes each epoch HARDER to fit, not easier — the two mechanisms (more signal per epoch vs fewer epochs to fit harder data) push in opposite directions with unknown net sign.

**Sources**: same as Idea 1.

**Estimated Effort**: trivial

**Risk Assessment**: Higher variance than 5x in both directions; a failure is also less informative (could mean "optimum is between 4 and 6" or "width is dead"), whereas 5x failing cleanly kills the direction.

### 3. torch.compile for throughput (epoch multiplier at any width)
**Summary**: `model = torch.compile(model)` — 1.2–1.8x img/s typical for small CNNs; compile cost (~1–2 min) lands in startup/epoch-1, outside the 300s training budget but inside the 600s wall cap (current total 416.5s → ~480–540s, tight but viable).

**Reasoning**: Raises the epoch budget at every width, attacking the starvation mechanism directly and permanently (it compounds with every later experiment). But at 4x, final=best says epochs are not currently binding, so its standalone accuracy gain is likely sub-bar; its real value is as an enabler immediately before a width push.

**Sources**: PyTorch torch.compile docs; goal-learnings § Failed Approaches (starvation mechanism); exp-report-004 § Next Steps.

**Estimated Effort**: low code, fiddly failure modes (graph breaks, recompiles on the eval/train boundary, autocast interaction, wall-clock inflation)

**Risk Assessment**: Crash/wall-clock tail risk; standalone gain likely below the +0.1 bar — would burn a loop to enable a future experiment.

## Idea Evaluation

**Evidence strength**: Ideas 1 and 2 share the same evidence base (in-project width scaling + the regularization-shifts-optimum argument from EXP-003/004); Idea 1 additionally has the EXP-004 convergence evidence (final=best at 114 epochs) suggesting ~80 epochs remains in the safe zone, while Idea 2 sits much closer to the measured starvation regime (40–60 epochs). Idea 3's evidence is about throughput, not accuracy.

**Mechanism clarity**: All three clear. Idea 1: add capacity while staying above the epoch floor. Idea 2: same but possibly below the floor. Idea 3: raise the floor itself, but the floor isn't binding at 4x.

**Expected impact**: Idea 1: +0.2–0.6pp with moderate confidence. Idea 2: −0.5 to +0.8pp, coin-flip. Idea 3: ~0 this loop.

**Risk profile**: Idea 1 safest among the capacity options and maximally informative on failure (kills the direction cleanly). Idea 3 risks crash/wall-clock for sub-bar gain.

**Feasibility**: Ideas 1–2 one-constant changes; Idea 3 fiddly.

## Chosen Idea
**Selected**: Idea 1 — Width 5x on the doubly-regularized recipe

**Why this idea**:
Capacity is the demonstrated next bottleneck, and 5x is the largest width step that stays clearly above the measured epoch-starvation floor (~80 predicted epochs vs the 40 where 8x failed), giving the best probability-weighted gain against the +0.1pp bar while being decisive either way: success re-opens 6x from a higher baseline; failure conclusively kills width at current throughput and promotes torch.compile.

**Hypothesis**:
Raising WIDTH_MULT from 4 to 5 (≈6.7M params) under the TA+RandomErasing recipe will raise best_test_acc from 96.23% to ≥96.4%, because the two regularizers raised effective data complexity beyond what 4.29M params best exploits, and ~80 time-keyed one-cycle epochs remain sufficient for convergence (EXP-004 converged with margin at 114); epoch count is predicted at 75–85 with total wall clock ≤ EXP-004's 416.5s due to fewer evals.
