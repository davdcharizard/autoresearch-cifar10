# Report EXP-017: Per-stage depth redistribution [3,3,3] → [2,3,4] at constant FLOPs
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md
- **Plan**: plans/plan-017.md
- **Log**: logs/exp-log-017.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: with the recipe certified a single-knob local optimum and the schedule axis closed (EXP-016), does the first STRUCTURAL architecture move — reallocating one block from stage 1 to stage 3 at provably-equal FLOPs — beat the baseline?

## Idea & Hypothesis

Chosen idea: change `NUM_BLOCKS` from uniform 3 to per-stage `(2, 3, 4)`. By ResNet's halve-spatial/double-width construction, per-block MACs are equal across stages (36,864 params × 1024 px = 589,824 params × 64 px), so the move adds +1.11M params (+26%, concentrated in stage 3) at zero FLOPs delta — the only capacity move that does not spend epochs, the resource every prior capacity failure (EXP-002/005/007) identified as binding. External support: RegNet (arXiv 2003.13678) found optimized design spaces are third-stage-heavy/first-stage-light at matched compute. Hypothesis: dt within ~5% of 22.4ms, ≥130 epochs, best_test_acc ≥ 96.81. Runners-up: width asymmetry 64/128/320, SE blocks.

## Approach

5-line diff in train.py: `NUM_BLOCKS = (2, 3, 4)`, the three `_make_layer` calls index it, depth print becomes `2 + 2*sum(NUM_BLOCKS)` (still ResNet-20). All training constants byte-identical to baseline. No deviations from plan; params confirmed exactly at the analytic 5,392,714.

## Execution

One run, no retries (task b5k4crkc2, launched 09:52:12 via the composite launcher + inline watchdog into a verified-free GPU 0). Pristine: zero watchdog SLOW events, post-hoc windowed profile 0 of 278 windows >30ms, mean 21.5ms. 144 epochs / 13,950 steps, total 515.1s, startup 23.7s (one-time compile-cache miss for the new graph), VRAM 1427.7MB, params 5,392,714. The separately-armed early-signal Monitor again first-polled near run end (known turn-scheduling pattern; harmless — inline watchdog was the protection).

## Results

- **Primary metric**: best_test_acc = 96.43% (baseline: 96.71, delta: −0.28pp, −0.29%); bar was 96.81
- **Observations**: The throughput half of the hypothesis was confirmed WITH MARGIN — the reallocation was not merely FLOPs-neutral but faster (21.5 vs 22.4ms windowed dt; 144 vs ~139 epochs; VRAM −185MB), exactly as the activation-traffic argument predicted (removed stage-1 block had 16x the activation volume of the added stage-3 block). The accuracy half failed cleanly: the run converged with a proper plateau (final eight evals flat at 96.2–96.43, best at ep 138/144, final ≈ best) yet that plateau sits 0.28pp BELOW baseline. By the EXP-008 diagnostic (plateau-at-end = architecture problem, not starvation), this is a representational deficit, not undertraining: +26% params, +5 epochs, and a faster step could not compensate for removing one of three stage-1 blocks.
- **Analysis**: The two depth experiments now tell one consistent story from different angles. EXP-008 (ResNet-14 @ 6x) showed total depth cannot be traded for width; EXP-017 shows even REDISTRIBUTING depth toward where params are cheap costs more in early-stage representation than the extra late-stage capacity returns. Mechanistically: stage 1 operates at 32×32 where the network builds its low-level vocabulary with the most spatial evidence per filter; at depth 20 (already the shallow extreme of the family) there is no slack to thin it. The RegNet evidence did not transfer because its third-stage-heavy populations were found at much larger total depths (and optimized final-accuracy-at-fixed-iterations, not max-over-evals at fixed seed/time) — at 9 total blocks, [3,3,3] appears to be at or near the allocation optimum. Notably this is ALSO the cleanest refutation of the "params are the binding capacity resource" reading of EXP-001: this run had MORE params, MORE epochs, FASTER steps, and lower VRAM than baseline and still lost — what it lacked was where the capacity sits. Twelve consecutive misses; the certified optimum extends from training constants to (now) depth allocation at this width profile.
- **Key Learning**: Equal-FLOPs depth reallocation toward stage 3 ran faster (144 ep) yet converged 0.28pp below baseline — at depth 20 early-stage blocks carry representation that late-stage parameters cannot replace; uniform [3,3,3] is at/near the allocation optimum.

## Verification

- **Conditions**: pre-condition contention sanity CLEAN (144 epochs on-model at measured 21.5ms; 0/278 windows >30ms); condition 1 FAILED (best_test_acc 96.43 < 96.81); conditions 2–3 skipped per first-failure stop (observed informally: 515.1s ≤ 600 with rc=0, and 144 evals = 144 epochs would have passed)
- **Review Notes**: trustworthy — metric consistent with the eval trail (best 96.43 @ ep 138); params line matched the analytic count exactly, so the intended architecture ran; pristine contention profile means the deficit is attributable to the allocation change alone
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid clean run; primary-metric necessary condition not met)

## Unexplored Avenues

- **Milder reallocation [3,2,4] or [2,4,3]**: keeps stage 1 intact while thinning stage 2 instead — stage 2 (16×16) may have more slack than stage 1. Plausible but the effect size here (−0.28pp from one block) suggests inter-stage moves are all priced in the same currency; low interest.
- **Adding a block at constant stage-1 depth ([3,3,4], depth 22)**: not FLOPs-neutral (+11% FLOPs ⇒ ~130 epochs) but tests "more stage-3 capacity" without robbing stage 1; entangles the closed capacity-vs-epochs trade at small dose. Moderate-low interest.
- **Width asymmetry 64/128/320 (brainstorm-017 runner-up)**: adds stage-3 capacity WITHOUT removing early-stage depth — the failure mechanism here (early-representation loss) does not apply, though the epoch cost (~+19% FLOPs) does. The EXP-017 result partially rehabilitates it: it now isolates "more stage-3 capacity" from "less stage-1 depth". Moderate interest.

## Next Steps

1. **Width asymmetry at constant alignment (64/128/320, [3,3,3])** — the one untried capacity-where-cheap move that PRESERVES early-stage structure; pays ~14 epochs for +2.0M stage-3 params; requires the measured-dt gate (project-insights Medium). Confidence: low.
2. **Optimizer-axis structural probe (e.g. SGD → SGD+nesterov with decoupled/scheduled WD, or momentum 0.9→0.95 as a paired-constant trade)** — the optimizer family is the last major axis with only the baseline data point; but single-knob bracketing logic (goal-learnings Patterns High) predicts small moves lose; only consider as a multi-constant trade. Confidence: low.
3. **Accept-the-optimum check remains a non-experiment** (repeated-seed = forbidden seed hacking); if width asymmetry also brackets out, the honest framing is that 96.71 is the measured optimum of this recipe family under this budget. Confidence: n/a (flagged for completeness).

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
