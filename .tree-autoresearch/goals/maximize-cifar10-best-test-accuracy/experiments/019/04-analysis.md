# Report EXP-019: Full-Run Official-Order Gradient Centralization
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the frozen 300-second charged training budget, with higher being better and only `train.py` changes allowed. EXP019 grew from parent EXP002 at 95.23%, so formal improvement required at least 95.33%. The goal-wide best remains EXP011 at 95.61%.

## Idea & Hypothesis

Retry the scientifically untested EXP017 mechanism with a corrected feasibility harness: after every backward, add coupled L2 to all gradients, centralize exactly the 16 convolution and classifier regularized directions per output row, then apply unchanged Nesterov momentum. ECCV 2020 reports CIFAR-100 SGDM gains across architectures without another forward, while EXP017 had proved the exact update math but crashed before metrics. The hypothesis predicted a valid `best_test_acc >=95.33%` with negligible exposure loss; a complete miss would definitively reject this exact composition.

## Approach

Only `train.py` changed. The implementation inventories 44 parameter tensors and centralizes 17 eligible weights containing 2,745,264 elements and 2,266 rows, excluding 27 BN-affine/bias tensors and 3,626 elements. After the sole inherited backward, one foreach add materializes `1e-4` coupled L2 on all gradients; 17 explicit heterogeneous reductions compute row means; one foreach subtraction applies them; inherited PyTorch SGD then runs with internal decay zero and unchanged momentum/Nesterov. Every operation is inside charged time and active in early-CutMix, early-clean, and late-clean phases.

Cadence-512 fixed FP64 device scalars audit total/convolution/classifier regularized, removed, and centralized squared energy, residual, decomposition, and nonfiniteness. Exact call/path/inventory/final-state reconciliation and final-16 context are appended without changing the evaluator, cadence, or max selection. The corrected preflight reused EXP018's 32+1,024 fixed-scalar trace and five 80/80/55 paired rounds; a 1 MiB live-allocation tolerance replaced brittle byte identity.

## Execution

The deterministic smoke first exposed a harness-only excluded-parity mistake: after one GC update, eligible divergence changed the second loss-derived bias gradient. Injecting identical raw gradients fixed that preflight-independent test. The decisive preflight then used its sole permitted pre-vector repair because its expected gradient used out-of-place addition rather than production's in-place FP32 operation order. After matching the reference arithmetic, the first complete vector passed: median overhead 1.007687x, maximum 1.014301x, MAD/median 0.006098, projected 27,736 steps / 143 epochs, 655,360 bytes live-allocation growth, fixed reserved allocation, and zero evaluator calls.

Exactly one metric run executed on physical GPU 0. It exited 0 after 300.0 charged and 449.8 total seconds, completed 27,976 steps across 144 epochs, and produced a complete integrity-valid summary. No production code, CUDA/OOM, assertion, nonfinite, infrastructure, or metric-driven adjustment occurred.

## Results

- **Primary metric**: 95.07% (parent: 95.23%, delta vs parent: -0.16 points, -0.17%; global best: 95.61%, delta: -0.54 points)
- **Observations**: GC executed 27,976/27,976 times, split 10,261 CutMix, 10,416 early-clean, and 7,299 late-clean steps, with 55 exact audits. It removed 6.4162 of 29.1016 regularized squared-energy units: 22.05% energy fraction and 46.95% norm ratio. Convolution directions lost 41.89% of their norm and the classifier lost 93.21%, so the mechanism was far from redundant with BN. Decomposition error was `4.00e-9`, maximum residual `3.52e-9`, and nonfinite counts were zero. The run slightly exceeded parent exposure (27,976 vs 27,950 steps); final-16 accuracy was 94.966875% mean, 94.82-95.07% range, and 95.02% final. Final CE 0.2046 was essentially parent-like (0.2044), while accuracy stayed lower.
- **Analysis**: The exact hypothesis is rejected. GC was cheap and strongly active, yet its stable tail sat about 0.26 points below the parent's best and 0.22 below the parent's final accuracy, so neither underdose nor selected-checkpoint noise explains the miss. The projection likely removed useful common-mode directions in this already-regularized WRN/CutMix recipe. The classifier is the clearest candidate: 93% removed norm is much stronger than the convolutional 42%, and the primary paper notes that convolution-only GC is sufficient for small-resolution CIFAR. This result definitively closes full eligible conv+classifier official-order GC on EXP002; it does not prove a separately preregistered convolution-only projection would fail.
- **Key Learning**: Full-run official-order GC removed 22% of eligible gradient energy but lowered EXP002 by 0.16 points despite unchanged exposure.

## Verification

- **Conditions**: Execution integrity passed, but the primary metric condition failed: 95.07% was below the 95.33% parent-relative threshold.
- **Review Notes**: Results are trustworthy. Claude independently returned `AUDIT_VERDICT: PASS`, rechecked freshness/scope, all 144 evaluations, exact dose/path/audit arithmetic, energy fractions, tail statistics, raw preflight evidence, and found no reward-hacking path (`04-result-review.md`).
- **Verdict**: no-improvement
- **Verdict Basis**: The run was complete, constraint-compliant, full-dose, and mechanism-valid, but underperformed its parent by 0.16 points. `tree.sh insert` recorded EXP019 as a terminal failed leaf on `br-000`; the global best remained 95.61% at EXP011.

## Unexplored Avenues

- **Convolution-only GC**: the ECCV paper states convolution-only centralization is sufficient on small-resolution CIFAR, while EXP019 removed 93.21% of classifier direction norm. Excluding the classifier is a literature-grounded mechanism change that may preserve useful class-boundary motion, but convolutional removal remains large and the case is not conclusive.
- **Raw-gradient GC with ordinary coupled decay applied afterward**: this would preserve the L2 common-mode direction rather than projecting `data gradient + L2`. It is no longer the exact official-order composition tested here and needs independent justification before a run.
- **Phase-limited GC**: applying projection only during high-LR early training could target conditioning while leaving the low-LR clean endpoint unconstrained, but there is no phase-resolved energy evidence yet and this is low confidence.

## Next Steps

- **Medium-high confidence**: Consider one convolution-only GC experiment from EXP002, justified by the paper's small-image recommendation and EXP019's disproportionate classifier removal; preregister it as a distinct mechanism, not a retry.
- **Medium confidence**: Prefer a low-overhead representation intervention if convolution-only GC is not selected, since both frequent Lookahead feedback and full eligible GC lowered stable tails despite preserved exposure.
- **Medium confidence**: Return to EXP011 only with an orthogonal mechanism expected to lift its 95.49% EMA plateau, not another generic optimizer smoother.

## Exit Action Results

No exit actions were defined for this goal.
