# Proposal: Output-RMS-Matched Fixed-Temperature Cosine Classifier

## Decision and falsifiable hypothesis

Replace only the final affine function with a fixed-temperature cosine classifier while leaving the accepted global-average pooled representation unchanged:

```python
COSINE_SCALE = 22.786916732788086

features = F.normalize(out, p=2.0, dim=1, eps=1e-6)
weights = F.normalize(self.fc.weight, p=2.0, dim=1, eps=1e-6)
return COSINE_SCALE * F.linear(features, weights)
```

Keep constructing the existing `nn.Linear(128, 10)` in the same order, but omit its bias from the forward function. This preserves every initialized tensor, parameter count, module inventory, and post-construction RNG state. The unused bias must have `grad is None` and remain bitwise unchanged. Do not add a learnable temperature, angular margin, auxiliary loss, second head, phase switch, or special optimizer group.

For pooled feature `h_i` and class row `w_k`, the logits are

`z_ik = s * <h_i / max(||h_i||_2, eps), w_k / max(||w_k||_2, eps)>`,

with `s = 22.786916732788086` and `eps = 1e-6`. Hence `|z_ik| <= s` whenever both norms exceed epsilon, and the clamped definition remains finite below epsilon. The hypothesis is that removing feature- and class-weight-norm shortcuts will regularize confidence under RandAugment and area-soft CutMix without the sparse feature geometry that destroyed EXP014 and EXP031. It should retain feasible fixed-time exposure and raise seed-42 `best_test_acc` from 94.15% to at least 94.25%. A valid lower result falsifies this exact operating point; no scale rescue is permitted.

## Temperature selection: accepted-function calibration, not arbitrary scale tuning

Scale 8 from EXP037 is not defensible under the fixed global LR: on the first registered EXP022 strong batch it reduced initial logit RMS from 2.760 to 0.969 and the estimated first SGD update from 1.384 to 0.244. The loss-function survey likewise warns that cosine/logit-normalized objectives require explicit temperature and usually joint LR tuning. This goal cannot safely retune the global LR because optimizer-path changes have repeatedly caused collapse (EXP020/022/028).

The proposed scale is instead the unique output-RMS matching calibration on a pre-existing immutable batch, performed before candidate execution and without labels or accuracy:

- accepted seed-42 model at `7c1e7d8`, training mode;
- first tensor pair from EXP022's already registered strong corpus;
- accepted affine-logit RMS `r_affine = 2.7600300312042236`;
- unit-cosine-logit RMS `r_cos = 0.12112344801425934`;
- `s = r_affine / r_cos = 22.786916732788086`.

At this frozen scale, candidate RMS is `2.7600297927856445` on that CutMix batch and `2.7467775344848633` versus accepted `2.752229690551758` on the next hard batch. Corresponding initial CE losses are 6.0143 versus 5.9343 and 5.2436 versus 5.1748. Thus the proposal preserves the accepted initial logit/loss operating scale far more closely than scale 8 while changing only angular geometry. This is still one-batch calibration and not proof that temperature 22.79 is optimal; it merely removes a known arbitrary scale mismatch. Freeze the full-precision literal above. Do not recalibrate on another batch, round it to a nicer value, or inspect accuracy to choose it.

## Optimizer geometry at the unchanged global LR

Away from the epsilon clamp, with `u=h/||h||`, `v_k=w_k/||w_k||`, and upstream class gradient `g=p-y`,

`dL/dh = (s/||h||) * (I - uu^T) * sum_k(g_k v_k)`

and

`dL/dw_k = (s*g_k/||w_k||) * (I - v_k v_k^T) * u`.

The head therefore deletes radial loss gradients and retains tangent updates. Matching initial output RMS does not make these Jacobians identical to the affine head, so LR 0.1 and momentum 0.9 remain a real risk. Coupled `1e-4` weight decay is radial and almost functionally invisible instantaneously, but it can shrink raw row norms; the `1/||w_k||` Jacobian then amplifies later tangent gradients. Momentum can also carry components whose functional effect changes after renormalization. The safety controller must measure raw row norms, feature norms, tangent/radial gradient and update components, classifier/backbone update ratios, and momentum norms. Bounded logits cannot waive an update spike or class-collapse veto.

No per-head LR is proposed. Adding one would make the result inseparable from a new optimizer experiment and would violate the reason for output-RMS matching. The global LR transfers only provisionally and must earn authorization on the immutable trajectory.

## Hard and CutMix targets

The existing `F.cross_entropy(outputs, targets)` supports both integer hard targets and probability targets without modification. For CutMix target `y = lambda*onehot(a) + (1-lambda)*onehot(b)`, the logit derivative remains exactly `softmax(z)-y`, equivalently the same convex combination of the two hard-target derivatives. No target renormalization or special loss branch is needed.

The possible benefit is that a mixed example cannot raise confidence by globally inflating `||h||`; only its angular evidence matters. The counter-risk is equally important: feature magnitude may encode augmentation severity, sample quality, or mixture ambiguity, and normalization forces clean and mixed samples onto the same radius. Also, the nominal bound is wide (`max_k z_k - min_k z_k <= 45.5738`), so this is not a low-confidence head. Strong CutMix and weak hard phases must be reported separately, and lower preflight loss is not an accuracy oracle.

## Exact implementation and construction oracles

- Leave `self.fc = nn.Linear(c3, num_classes)`, `_weights_init`, model traversal, global average pooling, and flattening unchanged.
- Add only the frozen scale constant and replace `return self.fc(out)` by the normalized expression above.
- Preserve all residual blocks, data transforms, CutMix probability/alpha, batch 128, width 2, ordinary SGD, LR schedule, decay, timer, evaluator, seed, and logging.
- Prove bitwise identity of every parameter and buffer to accepted seed-42 construction, identical CPU/CUDA RNG state afterward, 19 Conv/19 BN/one Linear, and 1,073,962 parameters.
- Reproduce the four frozen calibration numbers and scale ratio above from accepted source and the hashed first corpus batch before candidate replay.
- Compare FP64 logits plus input/weight gradients against the exact formula on random nonzero, tiny, and zero vectors. Require finite clamp behavior and `max(abs(logit)) <= s*(1+1e-6)` for nonzero vectors.
- Prove `fc.bias.grad is None` and its value stays bitwise fixed over multiple updates. Every other parameter remains exactly once in the unchanged optimizer group.

## Prospective immutable-corpus gate

Reuse without regeneration, filtering, or reordering:

- EXP022 strong corpus: 200 hard/CutMix batches, SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`;
- EXP028 weak corpus: 64 hard batches, SHA-256 `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`.

Hash files and tensor contents and validate schemas, shapes, target sums, hard/soft counts, and finiteness before and after replay. Before exposing the candidate, run two accepted/accepted calibrations on byte-identical batches under production CUDA settings. A statistic gets veto authority only if both accepted controls satisfy its frozen denominator-safe definition; otherwise the protocol is invalid, following EXP035.

Replay copied accepted/candidate states through all 200 strong steps at LR 0.1 and all 64 weak steps at the registered tail LR points. Serialize and fsync the complete report before assertions. Authorize timing only if:

- state, logits, loss, gradients, momentum, updates, BN buffers, and all diagnostics remain finite; BN counters are exact and target semantics/hashes remain identical;
- every candidate logit obeys the analytic bound;
- there is no candidate-only episode of either two consecutive or at least three total steps with greater than 95% maximum predicted-class share while the matched accepted model is at or below 95%; shared initialization concentration is recorded, not counted against one arm;
- denominator-safe candidate/control whole-logit, pooled-feature, gradient, and update ratios stay at or below 5x, with no candidate update above 25% of parameter norm or 5x its preceding 16-step median;
- each classifier row norm remains at least 0.50, pooled-feature norm remains at least 1.0, row max/min norm ratio remains at most 3, and no row update exceeds 50% of its pre-update norm;
- candidate/control classifier and backbone gradient/update statistics remain within the same control-qualified 5x envelope, and no candidate-only site concentration survives the accepted-control comparison;
- terminal debiased candidate/control loss-EMA is at most 1.5 separately for strong and weak phases; lower loss never waives another failure.

EXP014's zero branch, EXP031's bounded-at-init max residual, and EXP034's near-identical BN-normalized function all failed because their optimizer geometry became extreme after the first step. Therefore initial RMS parity and bounded logits are necessary evidence here, never sufficient evidence.

## Timing and scored-run gates

The recurring work is two feature reductions/normalizations and one 10-by-128 row normalization in both forward and backward. Arithmetic is tiny relative to the measured convolutional backward bottleneck, but extra reduction kernels may reduce fixed-budget exposure. After safety passes, run seven alternating fresh-process accepted/candidate H20 timing pairs on identical production hard/soft batches, including transfer, forward, CE, backward, SGD, and synchronization after warmup. Require finite execution, trial CV below 3%, aggregate median candidate/control step time at most 1.05, every pair at most 1.08, peak allocation below 650 MiB, and conservative projected total wall time below 540 seconds. Record projected updates; do not impose an extra exposure floor because the scored 300-second run naturally prices ordinary overhead into accuracy. Catastrophic timing failure retires the candidate.

If all gates pass, run seed 42 exactly once with the ordinary 300-second/600-second protocol, no more than 19 evaluations, and only tracked `train.py` changed. Improvement requires a valid `best_test_acc >= moving_baseline + 0.10`; no rerun, temperature adjustment, epsilon change, bias restoration, head-specific LR, or phase-specific normalization is allowed.

## Expected impact and candid feasibility

- **Expected primary impact:** approximately `-0.15` to `+0.20` points, with a modest chance of clearing `+0.10`. The intervention has plausible regularization value but no local accuracy evidence.
- **Accuracy feasibility: low-medium.** The scale now matches accepted initial logits, yet the cited loss-function study says normalized heads are sensitive to jointly tuned temperature/LR/decay; those degrees of freedom are intentionally unavailable.
- **Optimization feasibility: medium.** Initial hard and CutMix losses closely match accepted, and logits are bounded, but inverse-norm tangent dynamics plus functionally weakened decay can drift under LR 0.1.
- **CutMix feasibility: medium-low.** Angular evidence may help mixed regions, but deleting feature magnitude may erase useful ambiguity information.
- **Runtime feasibility: high.** The head is tiny relative to 19-convolution backward, though kernel-launch overhead must still pass paired timing.
- **Distinctness: high.** It preserves accepted average pooling and adds no max feature, so it does not repeat EXP014/031 pooling geometry; it also preserves hidden initialization and activation sites, avoiding EXP034/035 mechanisms.

Overall this is a valid but high-variance experiment, not the safest lead. It is substantially more defensible than arbitrary scale 8 because its sole temperature is frozen by accepted-function calibration, but the inability to jointly tune global LR and decay materially lowers its expected success probability.
