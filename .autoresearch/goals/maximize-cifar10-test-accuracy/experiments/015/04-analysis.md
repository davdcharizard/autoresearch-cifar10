# Report EXP-015: Mild policy-based augmentation (RandAugment), replacement vs add
- **Created**: 2026-06-30

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within the fixed 300s training budget, editing only train.py. Baseline **96.38** (EXP-008). Improvement bar: ≥ +0.1pp → **≥ 96.48**. This experiment tested whether transform-based policy augmentation — the one augmentation MECHANISM never tried on this goal — can raise the diagnosed ~96.3–96.5 generalization ceiling.

## Idea & Hypothesis
Chosen idea (cross-model review pick, `01-idea-review.md`): mild policy augmentation via torchvision `RandAugment`. Rationale: the net is at a generalization ceiling (8 prior no-improvements; EXP-014 proved it is not epoch/throughput-bound), and the canonical way to raise a generalization ceiling on fixed data is more effective data diversity. Augmentation is the proven productive lane here (EXP-008 Cutout12+RandomErasing, +0.38pp, the largest post-EXP-001 win), and the two aug classes tried — occlusion (EXP-008, won) and mixing (CutMix EXP-011, tied) — exclude the transform-based policy class (geometric+photometric) that is the documented lever pushing CIFAR-10 ResNets ~96→97%+. **Hypothesis**: a MILD RandAugment(1,6), tested both replacing RandomErasing (cA) and added on top (cB), keeping Cutout12, lifts `best_test_acc` to ≥96.48 and clearly above the same-session control, without under-fitting at the ~150-epoch budget.

## Approach
train.py only (45+/10−). Added `import os`, `from torchvision.transforms import RandAugment` (torchvision 0.24.1 — no new dependency), an env-toggle config (`AUG_MODE ∈ {baseline, randaug_replace, randaug_add}`, `RANDAUG_N=1`, `RANDAUG_M=6`), and a module-level `build_train_tf(aug_mode, n, m)` helper that inserts `RandAugment(n,m)` between the PIL geometric augs and `ToTensor`, keeps `Cutout(12)`, and drops `RandomErasing` only in `randaug_replace`. Same-session 3-cell design: c0 baseline (noise control), cA randaug_replace, cB randaug_add. Mild N=1,M=6 chosen ≪ the CIFAR default (N=2,M=14) for the short budget. Design hardened by the plan review: absolute 96.48 bar (not just c0), replace-vs-add disambiguation (review #7), under-fit diagnosis via ep25+final-trend, `build_train_tf` unit-smoke, inline throughput probe (no new file), winner-becomes-default on a win.

## Execution
One clean run, three sequential cells on GPU 1, no retries. Pre-smoke DataLoader probe surfaced the key operational finding: **RandAugment is CPU-bound on this 8-worker harness** — `randaug_add` loader 20,586 img/s vs baseline 37,765, BELOW the ~26k GPU compute rate. Decided to proceed: the budget accumulates per-step COMPUTE time and the DataLoader wait sits outside the timer, so epoch count is protected and only WALL inflates. Confirmed exactly at runtime — all cells fit 149 epochs; wall 450.6/463.2/513.9s (c0/cA/cB), all < 600s cap. Foreign GPU-1 PID 1723342 stayed dormant (0–3% util) across all cells → no contention.

## Results
- **Primary metric**: best policy cell **96.36%** (baseline 96.38, delta **−0.02pp**, −0.02%). Cells: c0 96.36, cA (replace) 96.34, cB (add) 96.36 — all @149 epochs.
- **Observations**: (1) Perfect epoch-matching (149 all) validated the compute-budget protection — CPU-bound aug inflated wall but not epochs, so the same-session comparison is fair. (2) **Not under-fit**: ep25 = 92.27/92.04/92.19 (c0/cA/cB), policy cells within ~0.2pp of control; all fully annealed (best≈final, no still-climbing). The mild aug is fully absorbed. (3) cB (add-on-top) ties c0 *exactly* (96.36) with normal ep25 → not over-regularization either. (4) cA (replace) marginally below (96.34) → swapping RandomErasing→RandAugment is neutral.
- **Analysis**: Hypothesis rejected. The intervention executed correctly (RandAugment active, geometric+photometric transforms applied, no bug) but produced **zero generalization benefit** in either configuration. Crucially the ep25/anneal evidence rules out the pre-registered failure mode (under-fit at 150ep): the net converges fine on the mildly-augmented task and simply lands at the same ~96.36 optimum. This is a genuine null at this strength, not a budget artifact. It is the third distinct augmentation mechanism (after occlusion and mixing) to tie on this net, strongly reinforcing that input-space augmentation is saturated and the ~96.3–96.5 plateau is a generalization ceiling not movable by adding aug diversity at this scale.
- **Key Learning**: A mild transform-based policy aug (RandAugment) adds nothing as either a replacement (96.34) or an addition (96.36) at matched 149 epochs with normal ep25 — input-space augmentation is now saturated across all three mechanisms (occlusion/mixing/transform), confirming the generalization ceiling.

## Verification
- **Conditions**: (a) run valid/within budget/wall<600s — PASS; (b) best policy cell ≥ 96.48 — **NOT MET** (96.36 < 96.48, +0.00pp over same-session c0); (c) integrity (train.py-only, prepare.py byte-unchanged, ≤1 eval/epoch, seed 42 fixed) — PASS.
- **Review Notes**: Results trustworthy. Epoch-matched (149 all), no contention, fully annealed, ep25 normal — the null is clean and not a contention/under-fit/under-anneal artifact. Multiple-comparison caveat moot (neither policy cell even reaches the same-session c0). No reward-hacking surface (eval harness frozen, aug is a legitimate train-side change).
- **Verdict**: no-improvement
- **Verdict Basis**: valid completed run; necessary condition (b) not met — metric did not reach the baseline+0.1pp bar (no hard-constraint violation → not invalid; valid metrics produced → not crash).

## Unexplored Avenues
- **Stronger RandAugment magnitude (N=2, M≈9–12) and/or TrivialAugmentWide**: ep25 showed the mild N=1,M=6 is fully absorbed with ~0 slowdown, so there is headroom to push strength before under-fitting. A stronger setting might engage more — but the prior is poor: it more likely tips into under-fit at 150ep (the canonical recipes use 200–2000ep) than breaks a ceiling that mild aug couldn't dent. Low-medium confidence; the single residual probe inside the augmentation lane.
- **Policy aug REPLACING Cutout (not RandomErasing)**: untested permutation; unlikely to differ given both occlusion augs are individually non-binding (EXP-012 showed the ReZero/occlusion regularizers are not accuracy-limiting). Low confidence.
- These are within-lane variations; the EXP-014 + EXP-015 evidence says the lane itself (input-space augmentation) is saturated.

## Next Steps
- **Pivot OFF the augmentation/within-architecture lane entirely** (high confidence it's needed): 9 straight no-improvements (EXP-006→015) now span optimizer, capacity (width+depth), all 3 aug mechanisms, regularization scalars, loss-geometry, eval-TTA, and epoch-buying. The ceiling is robust. The highest-EV remaining move is a genuinely DIFFERENT backbone/representation (e.g. a wider-but-shallower or differently-structured fast-CIFAR net, or a fundamentally different stem), funded by EXP-014's banked +12% torch.compile headroom for its per-step cost. Medium confidence any single new architecture beats the highly-tuned 96.38, but it is the only un-exhausted axis.
- **Test-time / ensemble-style gains within the frozen eval** (medium-low): the eval harness allows model-side TTA; a small self-ensemble (e.g. averaging EMA + raw, or multi-crop folded onto a training-side win) is cheap but EXP-006 showed eval-side alone is near-noise.
- **Stronger-magnitude policy aug as a quick cheap probe** (low-medium): one same-session cell at N=2,M=10 to close the augmentation lane definitively before pivoting; bounded downside, mostly confirmatory.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
- None (no exit actions defined).
