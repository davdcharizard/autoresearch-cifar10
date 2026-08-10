# Report EXP-021: Stage-2 training-only companion classifier
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the frozen 300-second charged training budget, with higher better and only `train.py` mutable. EXP021 grew from EXP004 at 95.40%, so formal improvement required at least 95.50%. The goal-wide best remains EXP011 at 95.61%.

## Idea & Hypothesis

Add a disposable classifier after the second 128-channel residual stage so hidden features receive direct class supervision without another backbone forward or inference-time path. AISTATS Deeply-Supervised Nets and CVPR Auxiliary Training provide CIFAR representation priors, while EXP004's stable final-equals-best result suggests generalization rather than basic fitting is limiting. The hypothesis predicted that fixed full-run companion CE at weight 0.15 would preserve at least 24,000 steps and produce `best_test_acc >=95.50%`; stronger stable criteria governed any later EMA composition.

## Approach

Only `train.py` changed. A dedicated-seed `Linear(128,10)` head taps the `[B,128,16,16]` output after block 3, applies ReLU and global pooling, and adds 1,290 parameters. It is constructed after inherited initialization with global CPU RNG saved/restored, so every parent tensor and future stochastic stream remain unchanged. Default/evaluator forward still returns only main logits and never pools or calls the companion.

Every primary pass used `main CE + 0.15 * companion CE` with identical hard or area-corrected CutMix targets. Scheduled SAM used that same joint objective for perturbation and descent, included the head in optimizer/snapshot state, replayed CUDA RNG, suppressed second-pass BN tracking, and restored exactly. Minimal audits counted training/evaluator calls and sparsely measured the unnormalized pooled feature norm by charged-time quartile.

## Execution

Deterministic smoke passed inherited state/main-logit/RNG parity, isolated head initialization, target formulas, auxiliary-only gradient reach, radius-0.05 SAM replay/restoration, audit routing, and zero evaluator calls. The first preflight stopped before timing because `nvidia-smi` reported a stale `[Not Found]` compute-app PID. Process, utilization, and `pmon` evidence proved no active process; the sole allowed harness repair changed only active-process detection.

The decisive seven-round preflight then passed: median overhead 1.022327x, p90 1.025486x, MAD/median 0.003090, parent drift 0.010438, projected 25,001 steps / 129 epochs, zero live-allocation growth, exact call formulas, and zero evaluator calls. Exactly one metric run executed on physical GPU 0, exited 0 after 300.0 charged and 433.8 total seconds, and completed 25,336 steps across 130 epochs without adjustment, retry, or metric-driven control.

## Results

- **Primary metric**: 95.11% (parent: 95.40%, delta vs parent: -0.29 points, -0.30%; global best: 95.61%, delta: -0.50 points)
- **Observations**: All 25,336 primary losses and 2,402 SAM replays executed the companion, yielding exactly 27,738 training head forwards; 130 evaluations left the counter unchanged. CutMix applied 10,180/20,533 and SAM 2,402/4,803. The 50 sparse norm samples reconciled; mean pooled stage-2 L2 followed a pronounced U-shape across time quartiles: 5.593, 4.347, 4.133, then 5.988. Head displacement reached 11.435 and nonfinite count stayed zero. Final-16 accuracy averaged 94.944375%, ranged 94.66-95.11%, and ended at 95.10%; final loss was 0.1883 versus EXP004's 0.1654.
- **Analysis**: The exact hypothesis is rejected. Exposure fell only 0.88% from EXP004 (25,336 vs 25,560) and cleared the 24,000 dose threshold, while the main endpoint fell 0.30 and loss worsened 0.0229. The head was strongly trained, its shared targets and both SAM passes executed exactly, and evaluation never used it, so underdose, inactivity, or inference leakage cannot explain the miss. The companion likely imposed premature stage-2 class separability/noisy intermediate CutMix semantics and changed the global SAM direction in a way that harmed the representation consumed downstream. The raw pre-activation tap's U-shaped norm confirms conditioning was nonstationary, but magnitude alone cannot assign causality or prove that adding BN would rescue it. The tail recovered late but remained substantially below the parent's final-equals-best solution, so EMA composition is unsupported.
- **Key Learning**: Full-dose stage-2 companion supervision preserved exposure but lowered EXP004 by 0.29 points, with a volatile lower tail and U-shaped feature norms.

## Verification

- **Conditions**: Execution integrity passed, but primary accuracy failed: 95.11% was below parent 95.40% and threshold 95.50%.
- **Review Notes**: Results are trustworthy. Claude independently returned `AUDIT_VERDICT: PASS`, rechecking freshness/scope, 130 evaluations, budget, exact companion/CutMix/SAM/audit arithmetic, tail and composition criteria, isolated initialization, charged work, and absence of reward hacking (`04-result-review.md`).
- **Verdict**: no-improvement
- **Verdict Basis**: The run was complete, in-scope, full-dose, evaluator-isolated, and mechanism-active, but regressed 0.29 points. `tree.sh insert` recorded EXP021 as a terminal failed leaf on `br-000`; global best remained 95.61% at EXP011.

## Unexplored Avenues

- **Normalized companion input**: a dedicated BN before pooling could stabilize the U-shaped feature scale, but it introduces new running-state and SAM second-pass semantics. Norm drift is diagnostic, not evidence that BN would improve accuracy, so this is not an immediate retry.
- **Early-only companion supervision**: turning the head off before clean-tail SAM could avoid objective interaction, but changes dose and phase while retaining uncertain intermediate CutMix targets.
- **Full auxiliary distillation package**: selective normalization, auxiliary-to-primary transfer, or classifier alignment is materially different from minimal companion CE and retains literature support, but costs complexity and several coefficients.
- **Lower coefficient or another tap**: not ruled out scientifically, but post-result coefficient/location search would be weakly identified and vulnerable to benchmark overfitting.

## Next Steps

- **High confidence**: Do not compose minimal stage-2 companion CE with EMA; move to a representation mechanism that does not force early linear separability.
- **Medium confidence**: Retain width-288 as a separately preregistered capacity experiment only if navigation still favors EXP004 despite its growing failed-child pileup.
- **Medium confidence**: Prefer a new base or a mechanism with direct stable-tail leverage over single-view SupCon, whose source-protocol components do not fit the fixed budget.

## Exit Action Results

No exit actions were defined for this goal.
