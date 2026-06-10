# Report EXP-039: Cosine / normalized-softmax classifier head

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-039.md
- **Plan**: plans/plan-039.md
- **Log**: logs/exp-log-039.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within the fixed 300s training-compute budget on
a single H20, editing only `train.py`. Baseline = **96.22%** (EXP-012, commit 6c417a4); success bar =
**96.32** (baseline + 0.1). This experiment tested whether changing the classifier-head decision
*geometry* — from a plain linear+bias softmax to a scaled-cosine (normalized-softmax) head — lifts top-1.

## Idea & Hypothesis
Chosen from brainstorm-039: replace the plain `nn.Linear(w3,10)` head's scoring with a normalized-cosine
rule — L2-normalize the global-avg-pooled features and the classifier weight rows, score by
`scale·cosθ` with fixed `scale=16`. Selected because, after ~38 experiments closing capacity (both
directions), all augmentation, the entire LR schedule, regularizer-adds, optimizer dynamics/objective,
and weight-averaging, the classifier-head SCORING geometry was the one untouched lever that *appears* to
dodge all three established plateau walls: compute-neutral (two L2 norms on tiny tensors), convergence-
neutral (not a regularizer-add), and top-1-affecting rather than loss-only polish (it changes WHERE the
decision boundaries sit). Hypothesis: angular boundaries better-condition class separation → top-1 above
96.32; honest expectation within-noise (~96.0–96.3), with a clean null closing the classifier-scoring
sub-lever.

## Approach
Single change to `train.py`, head only: (a) `self.fc = nn.Linear(w3, num_classes, bias=False)` and added
`self.logit_scale = 16.0` (a plain float — not a Parameter, so excluded from weight decay and
`model.parameters()`); (b) in `forward`, after global-avg-pool + view, `feat = F.normalize(out, dim=1)`,
`w = F.normalize(self.fc.weight, dim=1)`, `return self.logit_scale * F.linear(feat, w)`. No other recipe
element changed (widths {64,128,256}, PEAK_LR 0.2, batch 128, WD 1e-4, LS 0.1, Cutout 16,
TrivialAugmentWide, cosine-to-0 LR, Nesterov m0.9, seed 42, torch.compile reduce-overhead). params
4,299,856 (−10 = dropped fc bias). No deviations from plan-039.

## Execution
One run, no retries. Launched on an idle H20 (GPU 0; GPU 1 stayed 0%/0MiB throughout) —
`CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Clean, uncontended: dt = 570 steps @ 8ms,
69 @ 9ms, only warmup compile outliers (44/73/80/84ms). Ran to the 300s budget cut: 83 epochs / 32336
steps, total_seconds 405.4, exit 0. No errors. Smoke-test note: the Milestone-1 `train.ResNet(3,10)`
check printed a k=1 param count (272464, default width_mult) — a harmless artifact; the real k=4 count
4,299,856 was confirmed separately and in run.log.

## Results

- **Primary metric**: best_test_acc 95.89% (baseline: 96.22, delta: **−0.33**, −0.34%)
- **Observations**: final_test_loss 0.2099 (vs baseline 0.195, mildly inflated). num_epochs 83 vs
  baseline ~91 — the head was NOT perfectly throughput-neutral: the two `F.normalize` + `F.linear`
  shifted more steps to 9ms, costing ~8 epochs under the fixed 300s budget. peak_vram 491 MB (≈baseline).
- **Analysis**: The hypothesis (angular geometry lifts top-1) is not supported — top-1 regressed.
  Two compounding causes: (1) the cosine head gave no top-1 benefit on this balanced, BN-conditioned
  ResNet-20 (BN already normalizes feature scale, so removing norm DOF buys little); (2) the assumed
  compute-neutrality was approximate — the extra normalize/linear trimmed 91→83 epochs → mild
  under-training, and the slightly inflated loss (0.21) is consistent with both that under-training and a
  modest scale-16 under-confidence. Net result 95.89 sits exactly at the EXP-036 (SAM) value and within
  the cluster of recent sub-baseline nulls (037: 96.04, 038: 95.47).
- **Key Learning**: The classifier-head scoring geometry (cosine/normalized-softmax) does not add top-1
  on this BN-conditioned balanced net, and even "tiny" per-step ops (two L2 norms) are not free under a
  wall-clock-dt-gated budget — they cost ~8 epochs.

## Verification
- **Conditions**: NECESSARY primary-metric condition FAILED (95.89 < bar 96.32, < baseline 96.22).
  Clean-completion and no-constraint-violation conditions passed (total_seconds 405.4 < 600, exit 0,
  diff = train.py only, seed 42, ≤1 eval/epoch, params 4,299,856).
- **Review Notes**: Results trustworthy — clean uncontended run (dt 8ms, GPU 1 idle), metric
  cross-consistent with loss/epoch evidence. The 83-epoch count reflects the head's real per-step cost,
  not contention, so the run is fair (and even a perfectly throughput-neutral version would need to clear
  +0.33pp from the under-train penalty plus the bar — unlikely given the null geometry effect).
- **Verdict**: no-improvement
- **Verdict Basis**: valid, fair run; necessary primary-metric condition failed (metric below baseline).

## Unexplored Avenues
- **Higher fixed scale (24/32) or learnable scale (init 16)**: the mildly inflated loss hints scale 16 is
  slightly under-confident; a larger scale could recover loss. But this would not address the larger
  problem (no geometry benefit + epoch loss), and a learnable scale adds a parameter/step cost — low EV.
- **Cosine head WITHOUT the per-step weight normalize** (pre-normalize weights once per step via a hook,
  or normalize only features): could recover the ~8 epochs, but the core null (no top-1 gain from angular
  geometry on BN-conditioned balanced data) makes a win improbable.
- The broader classifier-scoring sub-lever is now effectively closed by this clean null.

## Next Steps
- **Stop probing compute-/convergence-neutral "free" levers** — EXP-036/037/038/039 form a consistent
  cluster of sub-baseline nulls; the saturated net at 300s does not yield to polish, geometry, or
  border-statistics tweaks (confidence: high this sub-family is exhausted).
- **Reconsider the throughput frontier directly**: every recent idea that helped needed MORE epochs, and
  the budget is wall-clock-dt-gated. A genuinely dt-REDUCING change (kernel/memory-layout efficiency that
  buys back epochs at fixed accuracy-per-epoch) is the under-explored axis — e.g., revisit whether any
  recipe element adds avoidable per-step wall-clock that could be reclaimed into more epochs (confidence:
  medium). This reframes the goal as "buy epochs" rather than "add a trick."
- **A more radical architectural change** that improves accuracy-per-epoch without adding wall-clock (the
  capacity axis is closed both ways, so this means topology/connectivity at iso-FLOP-and-iso-dt, e.g.
  a cheaper-but-deeper micro-config) — higher risk, but the incremental-trick space is now well-mapped
  (confidence: low-medium).

## Exit Action Results
<!-- No exit actions defined in the goal file. -->
- None defined.
