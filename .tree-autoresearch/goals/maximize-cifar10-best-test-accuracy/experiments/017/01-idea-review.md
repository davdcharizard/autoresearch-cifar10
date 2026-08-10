# Adversarial Review — EXP-017 candidate ideas

## Prioritized feedback

### 1. [Idea 1, GC] The self-imposed ban on `torch._foreach_*` is the single most likely cause of a zero-information outcome — **fix this before running**
`proposals/idea-01.md` ("Fixed Mechanism") forbids "foreach approximation," then declares that if the literal 17-tensor loop is too slow "the experiment is rejected at preflight." At ~10 ms/step (`02-system-understanding.md`), 34 extra kernel launches (17 `mean` + 17 `sub_`) at 5–10 µs each is ~2–3.5% — straddling the 1.03 median-ratio gate exactly. So the most probable single outcome is a preflight reject with no metric run and no evidence either way.
**Fix:** `torch._foreach_sub_` is not an approximation — it is bit-identical elementwise subtraction with one launch instead of 17, it is in the already-declared torch dependency, and the proposal's own FP64/FP32 reconstruction checks can *prove* bitwise equality against the loop reference in preflight. Permit it (and a fused per-tensor mean where shapes allow), with a preflight assertion of bitwise parity to the reference implementation. Keep the timing gate; remove the arbitrary implementation ban.

### 2. [Idea 3, EMA-without-SAM] Fatal for impact: this tests an already-validated mechanism on a node that cannot reach the goal-wide threshold
The proposal concedes it: "a local pass below that level is useful isolation evidence, not a new global result," and EXP-011 already established EMA's +0.21 on this lineage (`02-system-understanding.md`). EXP-002 is 95.23; even a good EMA result lands ~95.4, versus the 95.71 needed to move the goal metric. The genuinely *new* content — the zero-mass bias-corrected kernel that removes EXP-011's 6.30% terminal weight on the copy-in sample, plus the readiness gate — is real and is being spent on the wrong node.
**Fix:** move the corrected kernel + readiness gate to a child of EXP-011, where removing a 6.3% stale-sample bias has direct frontier upside (95.61 → 95.71+). The isolation question "does EMA work without SAM?" has no downstream action attached to either answer.

### 3. [Idea 3] The readiness gate is metric-hostile against a max-selected objective
`best_test_acc` is a maximum over epochs. Forcing exactly one source per epoch, and suppressing EMA evaluation until `m>=0.75` and `ESS>=90` (~18 ready evaluations), deliberately discards checkpoints that could win the max. The proposal admits it "may surrender max-selection premium." You are trading the primary metric for attribution purity on a question whose answer changes nothing.
**Fix:** if this idea is run at all, keep the readiness statistics as *reporting* (source-at-best is fully auditable from the recorded per-epoch source/mass/ESS) rather than as an evaluation *gate*.

### 4. [Idea 3] The 1.005 median-latency / 1% dispersion preflight gate is likely to reject a genuinely free change
EXP-016's observed ratio MAD was 0.005307 — the proposal cites this itself and then sets `median ratio <= 1.005` with `MAD/median <= 0.01`. EXP-011 measured sparse EMA at 0.9996x, i.e. the true effect is smaller than the harness noise floor. This is a coin flip on a dead leaf.
**Fix:** loosen to the ≤1.03 median / ≤1.06 per-round convention used by the other two proposals, which is still far tighter than the mechanism's plausible cost.

### 5. [Idea 2, companion head] The cited evidence comes from a regime whose pathology does not exist here — the mechanism/limiter gap is the core problem
DSN (AISTATS 2015) demonstrates companion objectives on shallow, non-residual, pre-BatchNorm networks where gradient delivery to early layers was genuinely poor. EXP-002 is a 6-block pre-activation WRN with identity shortcuts and BN on nearly every conv input. The proposal states this counter-case honestly ("not demonstrably gradient-starved") but does not resolve it, and `02-system-understanding.md` names the limiter as *stable generalization*, not gradient delivery. This is the classic "improves something, but not the thing the diagnosis says binds."
**Fix:** either (a) re-frame the mechanism as *intermediate-representation regularization* (forcing early linear separability as a capacity constraint) and predict the corresponding signature — the diagnosis-relevant claim is a lift in the *tail mean*, not the max — or (b) add a cheap accuracy-blind diagnostic to the preflight (linear-probe accuracy on block-3 features vs. block-5) that would establish whether middle-stage features are actually under-discriminative before spending the metric run.

### 6. [Idea 2] The auxiliary loss is confounded with an effective learning-rate increase on the stem and blocks 0–3
Adding `0.15 * L_aux` scales gradient magnitude on the first four blocks by roughly `1 + 0.15 * ||∂L_aux/∂θ|| / ||∂L_main/∂θ||`. A gain could be a front-loaded LR reweighting, not supervision. The proposal's audits report loss aggregates but no per-region gradient-norm ratio.
**Fix:** record (post-training, no per-step sync) the aggregate ratio of auxiliary-path to main-path gradient norm on the shared trunk at the sparse audit steps. This costs nothing and makes the null and the positive both interpretable.

### 7. [Idea 1] FP32 audit accumulation may trip your own `1e-5` integrity abort
The proposal permits "GPU-side FP64 **or** FP32" scalars, then hard-requires decomposition relative error ≤ 1e-5. Summing squares over 2,745,264 FP32 elements can accumulate relative error near or above that bound purely from rounding — an integrity abort caused by the audit, not the mechanism.
**Fix:** mandate FP64 accumulators (or `torch.linalg.vector_norm` in FP64) for the four audit scalars, and confine the 1e-5 bound to the FP64 path.

### 8. [Idea 1] The mechanistic argument needs the BN-nullspace objection answered, and the removed-energy diagnostic should be preregistered as the interpretation key
Real objection, stated only softly in the proposal's counter-hypothesis: in a PreAct WRN every conv output feeds a BN. Adding a constant `c` to all weights of a filter shifts its pre-activation by `c·Σx`, which the following BN largely re-centers — so the filter-mean gradient direction is close to (though not exactly in) the downstream BN's null space. That predicts GC's projection removes a component the network was already partly insensitive to, i.e. a small effect.
**Fix (cheap, high value):** you already compute `removed/raw` norm ratio. Preregister the reading: a ratio of ≲1% supports the redundancy counter-hypothesis and makes a null result *informative*; a ratio of ≳5% means GC is removing real signal and a null would instead implicate the "discarded useful coordinated evidence" branch. Right now the audit proves the code ran but is explicitly barred from informing interpretation — that wastes it.

### 9. [Idea 1] The 10-row classifier is the most likely place GC hurts, and it is bundled unavoidably
The proposal notes it ("the classifier's row mean may be useful") but centralizes all 17 tensors with no way to attribute a null. Given EXP-013's failed cosine-classifier result, the final layer is a known-sensitive spot on this lineage.
**Fix:** don't split the experiment (correctly, that would be a sweep), but record the classifier-row removed/raw ratio *separately* from the 16 conv tensors in the terminal audit. One extra scalar, and it makes a follow-up child well-posed instead of guesswork.

### 10. [All three] The branch-from-EXP-002 choice caps realizable value — say so explicitly in the write-up
`01-brainstorm.md` correctly flags that 95.33 is a local threshold and only ≥95.71 moves the goal. Ideas 1 and 2 are at least *composable* with the EXP-004→EXP-011 frontier (GC is an optimizer transform orthogonal to SAM's ascent step; the companion head is a training-time loss term orthogonal to EMA), so a local pass has a concrete next move. Idea 3 is not composable — EXP-011 already has EMA. Make composability an explicit success criterion in the chosen proposal's decision rules.

### 11. [Ideas 1 and 2] The declared "mechanism-strength" bars are below the known noise floor — both proposals admit this and neither acts on it
Idea 1 sets 95.33 (+0.10) against acknowledged 0.14–0.29-point selected-run variability; Idea 2 sets a 95.53 mechanism bar for exactly this reason. Idea 2's handling is better.
**Fix for Idea 1:** adopt Idea 2's two-tier framing — 95.33 as the formal tree verdict, ~95.53 as the mechanism-support bar — and report the final-16 tail mean/range/premium alongside the max, which Idea 1 already plans to compute. Do not add reruns.

### Sound as specified
Idea 1's parent-preservation contract (bitwise parity through `loss.backward()`, RNG neutrality, centralize-before-decay Nesterov ordering verified against an FP64 reference) is genuinely rigorous and is the correct way to make a single fixed-seed run interpretable. Idea 3's kernel arithmetic (normalized weights, mass identity, ESS, FP64 reconstruction) is the most mathematically careful spec of the three — its problem is target-node selection, not correctness.

---

## Scored verdict

**Full-Run Eligible-Weight Gradient Centralization**
- Evidence & reasoning: **7/10** — ECCV 2020 reports CIFAR/ImageNet gains on BN ResNets, the projected-optimization argument is coherent, and the parity/ordering contract is airtight; docked because the BN-nullspace redundancy objection (item 8) is raised but never answered, and no CIFAR-10-at-300s evidence exists.
- Potential impact: **7/10** — plausible +0.1–0.3 at zero parameter cost and near-zero exposure loss, and uniquely *composable* onto the EXP-004→EXP-011 frontier where the 95.71 goal threshold actually lives.

**Training-Only Fourth-Block Companion Classifier**
- Evidence & reasoning: **4/10** — DSN's evidence comes from non-residual, pre-BN networks whose gradient-delivery pathology this 6-block PreAct WRN does not have; the 0.15 coefficient and block-3 tap are uncalibrated single-shot guesses, and the effect is confounded with an early-layer LR increase.
- Potential impact: **5/10** — a real representation-regularization channel with a decent ceiling if the diagnosis is wrong about the limiter, but the most likely outcome is a wash or mild over-regularization of an already heavily-regularized first 75%.

**Readiness-Gated Clean-Tail Full-State EMA Without SAM**
- Evidence & reasoning: **6/10** — the estimator is validated in-tree (EXP-011) and the corrected zero-mass kernel is mathematically the best-specified artifact here; but EXP-002's 0.04-point best-to-final gap means there is almost no late-iterate variance left to average away.
- Potential impact: **3/10** — an ablation whose both answers are already actionable-free: EMA is on the frontier, ~95.4 is unreachable-from-relevant, and the readiness gate actively suppresses max-selection.

### Pick: **Full-Run Eligible-Weight Gradient Centralization**

It is the only candidate whose mechanism matches the diagnosed limiter under the diagnosed constraint — stable generalization improved through optimizer geometry, with no extra forward, no parameters, no data-exposure loss, and no tunable coefficient to overfit against the test metric. That matters given this branch's failure history: EXP-009 (SE gates, +20.7% latency), EXP-010 (depth realloc), EXP-013 (cosine classifier) all died from added structure or added cost; GC adds neither. It beats the companion head because its supporting literature comes from the *same* regime (BN ResNets on CIFAR) rather than a 2015 pre-BN regime whose pathology is absent here, and because it has no unvalidated coefficient. It beats the EMA isolation study decisively on impact: GC composes with SAM+EMA, so a 95.33+ result on EXP-002 licenses a direct attempt at the 95.71 goal threshold from EXP-011, whereas the EMA branch has no such move.

Two changes are required before launch, in order: **lift the `torch._foreach_*` ban** (item 1 — otherwise the most likely result is a preflight reject with no data at all, and bitwise parity to the loop reference is provable in preflight), and **preregister the removed/raw energy ratio as the interpretation key, with the classifier reported separately** (items 8–9 — this converts a probable null from "GC doesn't work here" into a diagnosis that directs the next child). Adopting Idea 2's two-tier 95.33/95.53 threshold framing (item 11) is a cheap third improvement.
