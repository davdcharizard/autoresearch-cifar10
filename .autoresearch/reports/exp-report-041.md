# Report EXP-041: PolyLoss Poly-1 (objective gradient reshape)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-041.md
- **Plan**: plans/plan-041.md
- **Log**: logs/exp-log-041.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within the fixed 300s training-compute budget on
a single H20, editing only `train.py`. Baseline = **96.22%** (EXP-012, commit 6c417a4); bar = **96.32**.
This experiment probed the one untouched compute-free axis after 41 experiments — the training objective's
polynomial gradient shape.

## Idea & Hypothesis
Chosen from brainstorm-041: add the PolyLoss Poly-1 leading term to the loss — `L = CE_with_LS + ε·(1−p_t)`
with ε=1.0 and `p_t` = softmax prob of the true class (Leng et al., ICLR 2022). Reasoning: with ε>0 the
term amplifies the per-example gradient on hard/low-`p_t` examples → a mild convergence accelerator, which
fits the convergence-bound hypothesis; compute-neutral and convergence-neutral, changing the per-example
gradient (top-1-affecting, not loss-only polish) → dodges all three plateau walls. Hypothesis: better
hard-example convergence lifts best_test_acc above 96.32 at throughput-neutral ~91 ep. Honest expectation:
within-noise null (sibling objective tweaks LS-down EXP-023, cosine head EXP-039 were null; label smoothing
may partially cancel the poly term).

## Approach
Two edits to `train.py`: added `EPSILON_POLY = 1.0`; replaced the single `F.cross_entropy(...,
label_smoothing=0.1)` with `ce = CE+LS` then `pt = F.softmax(outputs,1).gather(1,targets[:,None]).squeeze(1)`
and `loss = ce + EPSILON_POLY*(1-pt).mean()`. Label smoothing 0.1 retained; model/params (4,299,866)/data/
optimizer/schedule/seed/eval all unchanged. No deviations from plan-041.

## Execution
Two runs:
- **Run 1 — DISCARDED (contention)**: launched on GPU 0 (idle at launch); a neighbor Protenix distributed
  job (`protenix_base_constraint`, torch.distributed nproc=2) saturated BOTH H20s mid-run → dt band 19ms×88
  + 9ms×164, only 70 epochs (vs ~91), under-trained. Invalid; discarded.
- **Run 2 — CLEAN (reported)**: an idle-GPU fair-run launcher (sandbox-disabled bash poll loop; the Monitor/
  sandbox kills `sleep`-based loops with exit 144, so it ran with the sandbox disabled) waited ~2h for the
  Protenix neighbor to release a GPU, then caught GPU 0. Clean: dt 642@8ms + 58@9ms, 90 epochs, accepted by
  the launcher's ≥85-ep gate. total_seconds 408.4, exit 0.

## Results

- **Primary metric**: best_test_acc 96.11% (baseline: 96.22, delta: **−0.11**, −0.11% — within noise)
- **Observations**:
  - **final_test_loss 0.1583 vs baseline 0.195 — a ~29% drop in eval CE loss**, likely the lowest test
    loss in the project's history (below even SWA's ~0.18, EXP-020). Yet top-1 did NOT rise.
  - Throughput-neutral and fair: 90 ep ≈ baseline ~91, steady dt 8ms (the extra softmax+gather is trivial).
  - Run 1 (contended, 70 ep) showed the SAME fingerprint (loss 0.1635, top-1 95.73 confounded by
    under-training) — the loss-crash effect is robust to epoch count.
- **Analysis**: This is a **textbook polish-vs-top1 result** (project wall #2). PolyLoss ε>0 minimizes
  `(1−p_t)` → drives `p_t→1` → the model becomes much more confident on the true class → dramatically lower
  cross-entropy / NLL (better calibration), but the *decision boundary* (which class has the max logit) is
  essentially unchanged → top-1 flat/within-noise. The hoped-for "hard-example convergence accelerator →
  top-1" did not materialize: on this BN-conditioned, well-converged 90-ep recipe, the extra gradient just
  sharpened confidence, not separation. The label-smoothing interaction (LS softens targets, ε>0 sharpens
  hard-example confidence) did not cancel — the confidence push dominated, hence the very low loss. Fits the
  recent null cluster (036–041) and confirms the objective-confidence axis behaves like the other polish
  levers (EMA/SWA/GC/LS-down): moves loss, not top-1.
- **Key Learning**: Reshaping the CE objective to emphasize hard-example confidence (PolyLoss Poly-1, ε=1)
  is a pure polish lever here — it produces the project's lowest eval loss (0.158) with NO top-1 gain
  (96.11, −0.11pp). On this saturated recipe, objective-confidence ≠ decision-boundary improvement.

## Verification
- **Conditions**: NECESSARY primary-metric condition FAILED (96.11 < bar 96.32, < baseline 96.22 within
  noise). Clean-completion and no-constraint-violation conditions passed (total_seconds 408.4 < 600, exit 0,
  diff = train.py only, seed 42, eval lines 90 == num_epochs 90 ≤1/epoch, num_params 4,299,866 unchanged).
- **Review Notes**: Run 2 trustworthy — clean uncontended run (642@8ms, launcher-verified, 90 ep ≈ baseline),
  metric cross-consistent with the loss/dt evidence and reproduced (qualitatively) by the contended Run 1.
  No integrity concern (a standard cited loss function; eval frozen, still reporting plain CE). The discarded
  Run 1 was correctly excluded under the fairness gate.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, fair throughput-neutral run; necessary primary-metric condition failed.

## Unexplored Avenues
- **ε sweep (e.g. +2)**: the paper's ImageNet optimum is higher; but since ε=1 already drove loss far down
  with no top-1 gain (pure polish), a larger ε would deepen the confidence/loss effect, almost certainly NOT
  top-1 — low value. A NEGATIVE ε (down-weight hard examples, more LS-like) is also unlikely to help (LS-down
  EXP-023 already null). The objective-confidence sub-lever is effectively closed.
- **Poly-loss with the LS removed** (replace LS with the poly term rather than stacking): a cleaner objective,
  but same mechanism (confidence shaping) → expected same polish outcome.
- The broader takeaway: compute-free objective/loss reshaping (LS-down, cosine geometry, PolyLoss) all move
  loss/calibration, not top-1 — the objective axis is now well-mapped and closed.

## Next Steps
- **Treat the objective/loss-shape axis as closed** (confidence: high): PolyLoss joins LS-down (EXP-023) and
  the cosine head (EXP-039) as compute-free objective tweaks that improve loss/calibration but not top-1.
- **The pivotal open question remains convergence-bound vs epoch-saturated** (confidence: medium): no
  experiment has cleanly ADDED real epochs (cudnn.benchmark EXP-040 was a dt no-op). The only way to test it
  is a working dt reducer. The least-risky remaining option is an OFFLINE measurement of
  `torch.compile(mode="max-autotune")` compile-time + steady dt (compile + time 50 steps, NOT a budgeted
  run) to decide if its compile-tax is repayable; if steady dt drops enough, a real run could add epochs and
  resolve the question.
- **If the throughput axis is also exhausted**, the plateau at 96.22 should be documented as the robust
  ceiling for this k=4 ResNet-20 at 300s, and remaining moves are radical-architecture gambles at iso-dt
  (high risk, capacity closed) — confidence: low that incremental levers remain.

## Exit Action Results
<!-- No exit actions defined in the goal file. -->
- None defined.
