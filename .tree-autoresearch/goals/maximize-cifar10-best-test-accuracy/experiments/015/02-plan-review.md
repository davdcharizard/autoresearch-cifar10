VERDICT: PASS

Per-category verification (all clear):
- **Mathematical.** With ε=−0.25, L=CE(q,p)−0.25(1−q·p). Hard: d/dz[−0.25(1−p_y)] = −0.25p_y(p−e_y), so total = (1−0.25p_y)(p−e_y), factor∈[0.75,1] — correct, strictly positive, unique stationary point at p=e_y. Soft: ∂(q·p)/∂z_j = p_j(q_j−q·p), giving p−q−0.25·p⊙(q·p−q) — correct, and correctly *not* a scalar multiple of (p−q), so the absence of a scalar ratio bound is right, not an omission. The soft form reduces exactly to the hard form at q=e_y. Both gradients sum to zero over classes (softmax constraint intact). Loss is bounded below by 0 since CE ≥ 1−q·p ≥ 0.25(1−q·p): no negative-loss or unbounded-descent pathway.
- **Counter.** O+C+A=steps with O/C/A disjoint; A=D; Poly calls counted independently as O+C+D=steps; total=steps+SAM ⇒ CE calls = A exactly. This does pin the SAM base pass to CE (a Poly base pass would make Poly calls = steps+A ≠ steps). Closed and non-degenerate.
- **SAM.** Zero-grad between passes, RNG replay, BN off, restore before the single update, one optimizer step per SAM step (LR schedule and weight decay applied once), SAM/CutMix disjoint so no soft targets reach the hard-target path. Consistent.
- **Gating.** First-vector-decisive with numeric-fail ⇒ no metric and no rerun removes retry/cherry-pick pressure; exactly one timeout-600 CUDA0 run; timeout matches the 600 outer bound; drift tolerance 0.03 < the 0.10 parent→formal gap, so drift cannot manufacture an improvement.
- **Reward-hack.** Improvement is scored against the formal 95.71, not the softer parent 95.61; the 95.69–95.71 tail is explicitly non-improvement.
- **Classification.** Integrity violation and crash both map to invalid/NaN, so no failure path yields a scoreable metric.

BLOCKERS: none

NONBLOCKING CONCERNS:
1. **"155 EMA" unit is unpinned and one reading is unsatisfiable.** At 195 steps/epoch × 130 epochs = 25350 ≥ 25300, the step/epoch projections are self-consistent, but a ≥155 floor on *per-epoch* EMA updates cannot be met inside 130 epochs. Only the per-step (or every-N-steps) reading is satisfiable, where ≥155 is a weak floor (<1 epoch of tail). Pin the unit and cadence explicitly; if the implementation's EMA cadence is per-epoch, this gate line is unreachable and becomes blocking.
2. **Counters must increment at the call sites.** The identities have full detection power only if Poly and CE counters fire inside the loss functions, not inferred from step category. If incremented per step-type, a base pass silently using Poly (or a descent pass using CE) still satisfies every identity.
3. **"BN off" is ambiguous.** Freezing running-stat updates (standard SAM, batch stats retained) and eval-mode (running stats used for the perturbed gradient) give materially different descent gradients. Confirm which; only the former preserves standard SAM semantics with one stats contribution per step.
4. **Ascent/descent objective mismatch.** The perturbation is exact hard CE while the descent is Poly. Because the Poly hard gradient reweights per sample by (1−0.25p_y) before batch summation, the ρ-normalized direction differs from ascent on the actual training objective (they coincide only at batch size 1). Defensible as a design choice, but the measured effect conflates ε=−0.25 with this mismatch, weakening attribution to the loss alone.
5. **Gradient-scale/LR confound.** The [0.75,1] factor shrinks the effective step on confident examples by up to 25% as training converges, which is partly equivalent to an extra late LR decay under a fixed schedule. Worth noting when interpreting any gain.
6. **MAD requires all five rounds.** "First vector decisive" cannot literally apply to MAD/median ≤0.005, which is defined across rounds. Specify that the dispersion statistic uses all five rounds in fixed order and that "decisive" means no reordering, re-running, or round dropping.
7. **Eval-draw parity in the clean tail.** The objective is best test accuracy, so the reported number is a max over eval points; with "one eval/epoch" binding, the tail must evaluate exactly one of {raw, EMA}, not both. Evaluating both would double the draws (a test-set selection advantage over the parent) and is itself a constraint violation ⇒ invalid.
8. **Charged budget is not directly gated.** The vector bounds total <600 but the ≥25300-step/130-epoch lines are lower bounds on work. Make explicit that the projection is what fits within 300 charged including eval and EMA overhead, so throughput sufficiency cannot be satisfied by overrunning charged time.
9. **Device identity.** Confirm cuda:0 maps to physical GPU0 under the active CUDA_VISIBLE_DEVICES, since the constraint is stated physically.

## Resolution

- Counters now increment at actual loss call sites; no scheduled-category inference can hide a wrong objective.
- The second SAM pass explicitly remains in training mode with only `track_running_stats=False`.
- EMA floors are cadence-31 optimizer-step samples, and all five timing rounds are required before MAD with no drop/reorder/replacement.
- Charged-path projection is within the fixed 300-second training budget; synthetic evaluation timing is informational, while end-to-end total remains bounded separately.
- Evaluation remains exactly one inherited live-or-EMA source per epoch, and physical GPU-0 UUID must match visible `cuda:0`.
- The CE-ascent/Poly-descent and late effective-LR interpretations remain explicit causal limitations rather than implementation defects.
