# Report EXP-042: Deep supervision — auxiliary layer2 classifier with a decayed aux loss

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-042.md
- **Plan**: plans/plan-042.md
- **Log**: logs/exp-log-042.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within the fixed 300s training-compute budget on a
single H20, editing only `train.py`. Baseline = **96.22%** (EXP-012, commit 6c417a4); bar = **96.32**
(+0.1pp). This loop probed the one generalization-class lever the project's own High Importance insights
endorse but never tried: deep supervision via an auxiliary intermediate-layer loss.

## Idea & Hypothesis
Chosen from brainstorm-042 (after discarding two leads in planning: the cooldown axis is CLOSED, and TTA is
integrity-rejected). Add a lightweight auxiliary classifier on layer2 (mid-level, 128-ch) with
`L = CE_main(+LS) + λ(t)·CE_aux(+LS)`, `λ(t)=0.3·(1−frac)` decaying to 0; the aux head is train-only and
discarded at inference (eval scores the unchanged main head). Reasoning: the polish-vs-top1 wall is
axis-independent and explicitly covers OPTIMIZATION/objective changes, but project-insights L82 explicitly
prescribes injecting mid-level signal via "an AUXILIARY decayed loss on early layers" (written after the
EXP-032 multi-scale-head failure). Deep supervision changes WHAT the intermediate features encode
(generalization/feature-quality) while preserving the coarse-to-fine hierarchy. Hypothesis: auxiliary
mid-level supervision lifts best_test_acc above 96.32 at a throughput-neutral ~91 ep. Honest prior: deep
supervision's benefit is depth-scaling, so on a shallow 9-block net the gain may be modest/within-noise.

## Approach
Four edits to `train.py` (scope-clean): (1) `LAMBDA_AUX = 0.3`; (2) `self.aux_fc = nn.Linear(w2=128, 10)`
(Kaiming-init via the existing `apply`); (3) forward path; (4) training-loop two-term loss with the decayed
λ tied to the same elapsed-time fraction as the LR schedule. Model/data/optimizer/schedule/seed/eval all
otherwise unchanged. num_params 4,299,866 → 4,301,156 (+1,290).

## Execution
Two runs on idle GPU 0 (both H20s confirmed idle at each launch; GPU0 uncontended throughout both):
- **Run 1 — DISCARDED (throughput-confounded)**: the first implementation branched `forward` on
  `self.training`, returning a tuple in training and a tensor at eval. This **broke reduce-overhead
  CUDA-graph capture**: dt went bimodal/interleaved from ep1 (majority 14ms, 92×16ms, only 98×8ms) → just
  55 epochs (vs ~91) → severe under-train → best_test_acc 93.36 (−2.86pp). The interleaved-from-ep1 pattern
  (not a clean mid-run transition) and the idle GPU ruled out external contention; the cause was the
  graph-breaking conditional/variable-output forward.
- **Run 2 — CLEAN, FAIR (reported)**: per the EXP-030→EXP-031 precedent (re-test throughput confounds
  before concluding), removed the branch: `forward` is byte-identical to baseline (single-tensor main path →
  clean CUDA-graph → eval unchanged), and a separate `forward_train` ALWAYS returns `(main, aux)` (stable
  output structure) and is the compiled training target (`torch.compile(model.forward_train,
  reduce-overhead)`). dt returned to a **steady 8ms** (613×8ms + 81×9ms) → 90 epochs = baseline-equivalent
  → a fair throughput-neutral test. total_seconds 408.7, exit 0.

## Results
- **Primary metric**: best_test_acc **95.91%** (baseline 96.22, delta **−0.31pp**, −0.32% — a small
  regression, slightly beyond the ±0.15–0.25pp noise band).
- **Observations**:
  - **The aux head is throughput-free when compiled correctly**: Run 2's steady 8ms (identical to baseline)
    proves the aux global-pool + 128×10 linear + the second backward through layer1/layer2 fuse into the
    CUDA graph at no measurable dt cost. Run 1's 14ms was 100% a CUDA-graph-capture artifact of the
    data-dependent `self.training` branch, NOT intrinsic aux-backward cost.
  - **final_test_loss 0.2026 > baseline 0.195** — deep supervision did not even produce a polish/loss win
    (unlike GC/PolyLoss). It mildly hurt both top-1 and loss.
  - Fair test: 90 ep ≈ baseline ~91, dt 8ms, scope-clean, eval untouched (main head only).
- **Analysis**: The hypothesis is falsified. Auxiliary mid-level supervision did not sharpen the main head's
  generalization on this net — it mildly **regressed** it. This is a coherent, milder echo of EXP-032
  (multi-scale-head, −1.5pp): forcing layer2's features to be linearly class-discriminative (via the aux
  classifier's gradient) pulls them toward premature, head-friendly representations that partially fight the
  tuned coarse-to-fine hierarchy the main head depends on. The decayed λ→0 limited the damage (−0.31 vs
  EXP-032's −1.5, and the final objective was pure main), but the trajectory through the aux-supervised
  phase still landed slightly worse. Deep supervision's literature benefit is depth-driven (it eases signal
  propagation / vanishing gradients in very deep nets, e.g. DSN/GoogLeNet at 20–100+ layers); the shallow
  9-block ResNet-20 already trains cleanly with BN + warmup, so there is no propagation problem for the aux
  signal to fix — only a hierarchy to mildly perturb. Fits the same pattern as zero-init-γ (EXP-026,
  depth-driven null) and the "shallow-net trick non-transfer" project insight.
- **Key Learning**: Deep supervision (aux layer2 classifier, decayed λ 0.3→0) is throughput-free when
  compiled with a stable output structure, but mildly REGRESSES top-1 (−0.31pp) AND loss on this shallow
  9-block net — its depth-driven benefit doesn't transfer, and the aux gradient mildly perturbs the tuned
  coarse-to-fine hierarchy (a gentler EXP-032). Auxiliary-loss / deep-supervision axis CLOSED here.

## Verification
- **Conditions**: NECESSARY primary-metric condition FAILED (95.91 < bar 96.32, < baseline 96.22). Clean-
  completion and no-constraint-violation conditions passed (total 408.7s < 600, exit 0, diff = train.py only,
  eval lines 90 == num_epochs 90 ≤1/epoch, seed 42, no new deps, num_params 4,301,156).
- **Review Notes**: Run 2 is trustworthy — clean uncontended 8ms/90-ep run, metric cross-consistent with the
  loss/dt evidence. No integrity concern: deep supervision is a standard train-only technique; the frozen
  eval scores the unchanged single-tensor main head (the aux head is discarded at inference — explicitly NOT
  a TTA-style eval-protocol change). Run 1 was correctly discarded as throughput-confounded under the
  fairness gate.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, fair, throughput-neutral run; necessary primary-metric condition failed (mild
  regression).

## Unexplored Avenues
- **Aux head on layer1 or multiple aux heads**: a shallower aux point or several would supervise even
  earlier features — but EXP-032 + this result both show that pushing class-discrimination into early/mid
  layers hurts on this shallow net, so more/earlier aux heads would very likely regress more, not less. Low
  value.
- **Much smaller λ (e.g. 0.05) or a later aux onset**: could shrink the regression toward zero, but the
  best plausible outcome is "within-noise null," not a +0.1 gain — the mechanism doesn't help here. Low value.
- The broader takeaway: BOTH ways of using mid-level signal (input-concatenation into the head EXP-032, and
  auxiliary supervision EXP-042) hurt on this net. The intermediate-feature-routing family is now exhausted.

## Next Steps
- **Treat the auxiliary-loss / deep-supervision axis as CLOSED** (confidence: high): joins multi-scale-head
  (EXP-032) — both forms of injecting mid-level signal regress on this shallow, well-tuned, hierarchy-
  dependent net. Do not retry aux heads / deep supervision / λ sweeps.
- **The optimizer FAMILY (AdamW) remains genuinely untested** (confidence: low it gains, medium it's worth
  one map-completing loop): all 44 experiments used SGD; the polish wall predicts a regression/null, but a
  single clean AdamW probe would close the last major optimization axis. (Brainstorm-042 idea 2.)
- **The plateau at 96.22 is increasingly confirmed as the robust ceiling** for this k=4 ResNet-20 at 300s
  (confidence: high): capacity, augmentation (incl. cooldown), schedule, optimizer-dynamics, objective,
  weight-averaging, classifier-head, AND now intermediate-feature-routing are all closed. Remaining moves are
  radical iso-dt architecture gambles (high risk under the dt-gated budget) or the AdamW family probe.

## Exit Action Results
<!-- No exit actions defined in the goal file. -->
- None defined.
