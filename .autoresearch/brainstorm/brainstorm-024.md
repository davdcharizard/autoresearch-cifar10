# Brainstorm EXP-024
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Zhang 2019, "Making Convolutional Networks Shift-Invariant Again" (ICML)** — anti-aliased downsampling /
  **BlurPool**. Strided downsampling (strided conv, max-pool) aliases: it violates the sampling theorem, making CNNs
  surprisingly shift-VARIANT and hurting generalization. Fix: precede every subsampling op with a low-pass (blur)
  filter — replace "stride-s op" with "stride-1 op → blur → stride-s subsample". Reported consistent accuracy AND
  shift-consistency gains across ImageNet/CIFAR architectures, with NO added learnable parameters (the blur kernel is
  a fixed binomial filter) and only a small extra compute. This is a generalization mechanism that does NOT add a
  stochastic/convergence-slowing penalty — distinct from all the regularizers that failed here.
- Knowledge base has trivialaugment, cutmix, swa, wrn-dropout. No BlurPool entry yet (will add in planning).

## Experimental History Review

Current best = **96.22%** (EXP-012, commit 6c417a4). 24 experiments; ~16 axes closed. Binding constraint:
generalization/CONVERGENCE at fixed k=4 capacity in 300s (~84–92 epochs).

**Key state:** ALL scalar recipe hyperparameters are now bracketed interior optima — LR-peak 0.2 (EXP-016/017),
Cutout 16 (EXP-013/021), label-smoothing 0.1 (EXP-023), WD 1e-4 (EXP-005). And every ADD-a-regularizer move failed
(WD↑/Mixup/CutMix/dropout, EXP-005/011/018/022) — the recipe is convergence-bound, not overfit-bound. Per
project-insights, gains must come from a **convergence-NEUTRAL** change, not more regularization, and scalar-knob
tuning is exhausted → time for a structural/mechanism change (directive: "try more radical architectural changes").

**Closed axes (do NOT revisit):** capacity k>4 (EXP-004/009), LR-peak (EXP-016/017), block-order/pre-act (EXP-015),
activation (EXP-010), SE attention (EXP-008), weight-decay (EXP-005), more-epochs (EXP-007), auto-aug policy
(EXP-014), Cutout-size (EXP-013/021), label-mixing (EXP-011/018), weight-averaging (EXP-006/019/020), in-block
dropout (EXP-022), label-smoothing value (EXP-023).

**Critical methodological caution (EXP-015):** FLOPs-neutral architecture changes are NOT necessarily wall-clock-
neutral under torch.compile — a restructured graph can fit fewer epochs and confound the comparison. ANY structural
change MUST verify realized epoch count against the baseline 91 and attribute deltas only after confirming it.

## Candidate Ideas

### 1. BlurPool / anti-aliased downsampling (Zhang 2019)
**Summary**: Replace the two stride-2 downsampling sites (layer2 & layer3 first-block `conv1` and their 1×1 projection
shortcuts) with anti-aliased downsampling: do the conv at stride 1, then apply a fixed depthwise binomial blur filter
with stride-2 subsampling. Add a parameter-free `BlurPool2d` module (a registered-buffer 3×3 [1,2,1]⊗[1,2,1]/16
depthwise kernel, groups=channels). conv1 path: `relu(bn1(conv1_s1(x))) → blurpool → conv2`. Shortcut path:
`blurpool(x) → 1×1 conv_s1 → bn`. Params UNCHANGED (blur kernels are buffers, not parameters); k=4 recipe otherwise
identical (TA+Cutout(16), LR 0.2, LS 0.1, compile, seed 42).

**Reasoning**: The model is generalization-bound and scalar knobs are all bracketed. BlurPool is a genuine,
well-evidenced generalization mechanism (anti-aliasing → better shift-invariance) that, unlike every failed
regularizer here, adds NO stochastic/convergence-slowing penalty and NO learnable params — it fits the
"convergence-neutral structural change" prescription exactly. It is the one architectural lever flagged across recent
reports' Unexplored Avenues. If it improves shift-consistency it can lift top-1 on the generalization-bound margin.

**Sources**: Zhang 2019 (ICML, anti-aliased downsampling); project-insights Medium (convergence-neutral lever);
exp-report-021/022/023 Unexplored Avenues; train.py L80-92 (BasicBlock strided path), L104-106 (downsample stages).

**Estimated Effort**: medium — a ~10-line `BlurPool2d` module + rewiring the stride!=1 path in `BasicBlock`.

**Risk Assessment**: Two real risks. (a) **Compute/epoch confound (EXP-015):** the restructured strided path + 2
extra depthwise blur convs change the compiled graph and add compute → may fit fewer than 91 epochs and confound the
result. MUST check realized epoch count; if epochs drop materially the comparison is confounded (note in analysis).
The blur convs act on small (16²/8²) maps so added compute should be modest. (b) **Modest CIFAR ceiling:** only 2
downsample stages on 32×32 images → documented CIFAR gains are smaller than ImageNet. Fails gracefully
(no-improvement) if anti-aliasing doesn't help this shallow net. Params unchanged → otherwise a fair test.

### 2. Per-channel input std-normalization (std (1,1,1) → true CIFAR std)
**Summary**: Normalize inputs by true per-channel std (≈(0.247,0.243,0.261)) instead of (1,1,1), train.py L152-155.
Convergence-neutral; the last untried cheap single-knob.

**Reasoning**: Closes the input-normalization axis cleanly. Consistent with the convergence-neutral prescription.

**Sources**: train.py L152-155 (`std=(1,1,1)` comment); standard CIFAR practice.

**Estimated Effort**: low — one tuple.

**Risk Assessment**: First layer Conv→BN almost certainly absorbs a per-channel input rescale → expected NULL. Low
ceiling; an axis-closer, not a real lead.

### 3. Larger batch size (128 → 256)
**Summary**: Double the train batch to 256 (one constant). On the launch-bound k=4 net, fewer kernel launches per
image could raise throughput (more images seen in 300s).

**Reasoning**: A convergence/throughput lever never tested; if launch-bound, bigger batches mean more effective
training within budget.

**Sources**: train.py L22 (BATCH_SIZE=128); project-insights (launch-bound k=4).

**Estimated Effort**: low — one constant.

**Risk Assessment**: Multi-effect and confounded: doubling the batch halves SGD updates/epoch and reduces gradient
noise (less implicit regularization), and the LR is bracketed-tuned for batch 128 (linear-scaling would want higher
LR, but LR-peak is closed at 0.2) → batch 256 is effectively under-LR'd. Likely neutral-to-worse; hard to attribute.
Weaker than Idea 1.

## Idea Evaluation

**Evidence strength**: Idea 1 has strong external literature (ICML 2019, widely validated) for a mechanism that
matches the project's diagnosed need (convergence-neutral generalization gain, no added params/penalty). Idea 2 is an
expected null. Idea 3 is a confounded multi-effect knob with weak directional evidence.

**Mechanism clarity**: Idea 1 — clear (anti-aliasing restores shift-invariance → better generalization), with a known
competing negative (compute/epoch cost) that is measurable. Idea 2 — clear but BN-nulled. Idea 3 — muddy (several
opposing effects).

**Expected impact**: Idea 1 has the highest ceiling and is the only candidate with a real chance of clearing +0.1pp
via a new mechanism. Idea 2 ≈ 0. Idea 3 ≈ neutral/negative.

**Risk profile**: Idea 2 safest (but null). Idea 1's risk is the epoch-confound, mitigated by measuring epoch count
and the blur acting on small maps. Idea 3 riskiest-to-attribute.

**Feasibility**: Ideas 2/3 trivial; Idea 1 medium but self-contained in train.py (no new deps; fixed-buffer module).

Conclusion: **Idea 1 (BlurPool)** is the lead — with scalar knobs exhausted, it is the best-evidenced
convergence-neutral *structural* mechanism left, matching the project's diagnosed binding constraint, and the
directive calls for radical architectural changes at this stage. The EXP-015 epoch-confound risk is real but
manageable (measure epochs; blur on small maps). Idea 2 is a trivial axis-closer reserved for later; Idea 3 is
confounded and lower-value.

## Chosen Idea
**Selected**: BlurPool / anti-aliased downsampling (Zhang 2019)

**Why this idea**:
Scalar-knob tuning is exhausted (all bracketed) and every added regularizer failed — the project needs a
convergence-NEUTRAL structural change, and the directive calls for radical architectural moves at this stage.
BlurPool is the best-evidenced such lever: a parameter-free anti-aliasing mechanism that improves shift-invariance/
generalization (the binding constraint) without adding a stochastic convergence penalty. It is a fair test (params
unchanged) provided realized epoch count holds near the baseline 91.

**Hypothesis**:
Anti-aliasing the two downsampling stages with BlurPool will improve the model's shift-invariance and reduce the
residual generalization gap, lifting best_test_acc above the 96.32 bar — PROVIDED the restructured graph still fits
~91 epochs. If best_test_acc falls / epochs drop materially, either anti-aliasing doesn't help this shallow 2-downsample
net or the added compute confounds the budget (EXP-015 pattern), and the anti-aliased-downsampling axis is closed.
