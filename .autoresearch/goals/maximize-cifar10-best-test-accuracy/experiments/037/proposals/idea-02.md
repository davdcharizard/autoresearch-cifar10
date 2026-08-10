# Proposal: Fixed-Scale Cosine Classifier

## Decision and falsifiable hypothesis

Replace only the accepted final affine readout with a bias-free-in-function cosine classifier,

```python
COSINE_SCALE = 8.0

# after the unchanged global average pool and flatten
features = F.normalize(out, p=2.0, dim=1, eps=1e-6)
class_weights = F.normalize(self.fc.weight, p=2.0, dim=1, eps=1e-6)
return COSINE_SCALE * F.linear(features, class_weights)
```

Retain the existing `nn.Linear(c3, num_classes)` construction and initialization so every initialized tensor, parameter count, traversal, and post-construction RNG state remains identical to the accepted model; `fc.bias` is deliberately omitted from the forward and must remain bitwise unchanged with `grad is None`. There is no extra branch, learnable temperature, angular margin, auxiliary loss, class prototype, or phase-dependent behavior.

**Falsifiable hypothesis:** constraining every class logit to `[-8, 8]` and removing feature- and class-weight-norm shortcuts will make the balanced CIFAR-10 readout learn angular separation rather than confidence through radial growth. This should improve generalization under RandAugment and area-soft CutMix while preserving the accepted backbone, retain at least 99% of accepted fixed-time exposure, and raise seed-42 `best_test_acc` from 94.15% to at least **94.25%**. A valid lower score falsifies this exact scale-8 cosine point. A prospective safety or timing veto makes it invalid and does not authorize scale tuning.

## Evidence, scale choice, and limits

[NormFace](https://arxiv.org/abs/1704.06369) normalizes both features and classifier weights and shows why unit cosine logits leave softmax at a high loss; it introduces a scale after cosine. [L2-constrained Softmax](https://arxiv.org/abs/1703.09507) independently derives that fixed feature radius is preferable to an excessively free learned radius and gives the ten-class, `p=0.9` lower bound `log(0.9*(10-2)/0.1)=4.28`. Scale **8.0** is a single preregistered compromise: it is well above that ten-class fitting bound without importing the much larger 20-30 scales used for thousands of face identities. NormFace's ideal balanced-class loss bound at `s=8` is already approximately `log(1 + 9*exp(-10*8/9)) = 0.00124`, so scale 8 does not impose a material theoretical training-loss floor.

The source evidence is face-verification evidence, not direct short-horizon CIFAR-10 evidence. This proposal is therefore exploratory rather than a literature-backed expected win. Its local mechanism is nevertheless relevant: EXP014's unnormalized max branch reached a 3.96x classifier-weight ratio and chance accuracy, EXP031's initially small pooling residual reached a 4.34x per-example contribution, and EXP034 showed that nearly equal initial features can conceal 1.95x updates and later class collapse. Unlike those interventions, cosine logits have a true per-example absolute bound for the full trajectory; unlike them, however, this proposal changes the accepted classifier function immediately.

A read-only seed-42 check on the first registered EXP022 strong batch provides scale plausibility, not an execution result or permission to retune. Accepted pooled feature norms were 11.79-23.04, class-weight norms 1.20-1.48, and initial logit RMS 2.760. Scale-8 cosine logits had RMS 0.969, loss 2.848 rather than 5.934, and an estimated first SGD update norm 0.244 rather than 1.384; both arms still predicted one class for all examples at initialization. Thus scale 8 reduces the ordinary initial excursion but does not by itself prevent class concentration.

## Geometry and CutMix interaction

For pooled feature `h` and class row `w_k`, `z_k = 8 <h/||h||, w_k/||w_k||>`. Cauchy-Schwarz gives `|z_k| <= 8` (up to declared FP tolerance), independent of feature or weight magnitude. Data gradients through either normalization are tangent projections scaled by `8/max(||x||, 1e-6)`. This removes radial classifier updates from the forward function, but does **not** bound parameter gradients if a norm approaches zero. Coupled weight decay is mainly radial for `fc.weight` and therefore functionally attenuated, while tangent SGD can increase raw row norms. The preflight must consequently track pooled-feature and class-row norm floors, tangent/radial update components, and backbone updates; bounded logits cannot waive an update or class-concentration failure.

CutMix remains semantically exact: the existing probability targets and cross-entropy are unchanged, and their gradient is still the convex target mixture. The hypothesis is that angular logits prevent an easy or high-norm region from gaining confidence merely by inflating the whole pooled descriptor. The counter-risk is substantial: feature norm may encode sample quality or CutMix ambiguity, and discarding it forces equally scaled evidence for clean and mixed samples. A fixed scale can also sharpen ambiguous mixtures or underfit hard classes. Strong and weak corpora therefore need separate terminal-loss diagnostics, but neither is an accuracy proxy.

## Exact implementation and invariants

- Add the single literal constant `COSINE_SCALE = 8.0` beside the model hyperparameters.
- Leave `self.fc = nn.Linear(c3, num_classes)`, `_weights_init`, and all constructor order byte-for-byte unchanged. Do not normalize parameters in-place.
- Replace only `return self.fc(out)` with the two explicit `F.normalize(..., eps=1e-6)` calls and bias-free `F.linear` shown above.
- Preserve all postactivation blocks, Option-A shortcuts, global average pooling, width/depth, parameter tensors, optimizer membership, all-parameter `1e-4` decay, LR/momentum, seed, precision, N1/M7, p=0.5 alpha-1 CutMix, 80% weak-tail switch, timer, evaluator, and logging.
- Do not add production diagnostics. Disposable hooks/controllers live only under the ignored experiment directory.

Construction checks must prove bitwise identity of every parameter and buffer to accepted seed-42 construction, identical CPU/CUDA RNG afterward, 19 Conv/19 BN/one Linear, and 1,073,962 parameters. Prove `fc.bias` receives no gradient and remains bitwise fixed through several optimizer steps, while every active parameter occurs exactly once in the unchanged optimizer group. On random finite tensors with norms above epsilon, compare logits and gradients to an FP64 cosine oracle and require `max(abs(logit)) <= 8.00001`; explicitly test zero/tiny vectors to verify finite `eps=1e-6` semantics.

## Prospective, control-qualified safety protocol

Reuse without regeneration or filtering:

- EXP022's 200-batch strong hard/CutMix corpus, SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`;
- EXP028's 64-batch weak hard corpus, SHA-256 `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`.

Recompute file/tensor hashes, schema, hard/soft counts, shapes, finiteness, and target sums before and after replay. First run **two accepted/accepted calibrations** under production-default CUDA on byte-identical batches. Each must pass the following frozen denominator-safe whole-model gates before the candidate has authority; otherwise the protocol is invalid, as EXP035 demonstrated. Then instantiate accepted/candidate models from identical state and train independently over the same 200 strong batches at LR 0.1 plus 64 weak batches at preregistered cosine-tail progress points, with ordinary SGD, coupled decay, and no evaluator.

Record each step's loss/EMA, class histogram, logit RMS/max, pooled-feature norm quantiles/minimum, class-row norms, whole gradient/update/parameter/momentum norms, backbone and classifier gradient/update norms, cosine extrema, BN state, and immutable hashes. Production is authorized only if:

- all state and diagnostics are finite, BN counters are exact, variances are positive, target semantics are identical, and candidate logits satisfy the analytic bound;
- there is no candidate-only concentration episode of two consecutive or at least three total steps above 95% maximum class share while the matched control is at or below 95%; shared ordinary initialization concentration is recorded rather than hidden;
- candidate/control whole logit-RMS, gradient-norm, and update-norm ratios never exceed 5; candidate update is never above 25% of whole parameter norm or 5x its preceding 16-step median;
- no classifier-row update exceeds 50% of its pre-update row norm, minimum class-row norm stays above 0.50, minimum pooled-feature norm stays above 1.0, and no class-row max/min norm ratio exceeds 3; these absolute positive-denominator gates protect the normalization Jacobians;
- `fc.bias.grad is None` throughout and its value is bitwise unchanged; every other active path has finite nonzero aggregate gradients;
- terminal debiased candidate/control loss-EMA is at most 1.5 in each strong and weak phase.

These are catastrophic-geometry gates, not proof of accuracy. Lower loss cannot waive a failure, and bounded logits cannot waive a one-class argmax trajectory. Serialize and fsync the complete report before assertions. Do not rerun, change epsilon, or select another scale after any veto.

## Fixed-time feasibility and scored run

Two feature reductions and a 128x10 weight normalization are tiny relative to the measured 75.46% convolutional backward bottleneck, but they launch recurring forward/backward kernels and may still cost more than one percent of an 11 ms step. After safety passes, run five alternating fresh-process accepted/candidate H20 pairs, each with 100 warmups and 500 timed production hard/soft steps including transfer, forward, CE, backward, SGD, synchronization, and identical data. Require trial CV <=2%, candidate/control median <=1.01, candidate p95 <=1.05x control, projected exposure at least `floor(0.99*26,898)=26,629` updates, finite state, and conservative total runtime below 540 seconds. Separately time evaluator forwards and require at most 1.02x inference cost and no more than the accepted 19 evaluation opportunities. Any timing veto retires this exact implementation; fused, cached, in-place, approximate, or phase-only normalization is a new idea.

Immediately before production, query `exp-index.sh baseline` rather than trust the current 94.15 value, require the experiment branch to descend from the integration baseline, only tracked `train.py` changed, no stale owned log, static/Ruff/format/pre-commit/scope checks passing, and exactly one idle 97,871-MiB H20. Execute exactly once:

```bash
timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
```

Require exit zero, one finite ten-field summary, training `[300.0, 301.0)`, total below 600 seconds, 1,073,962 parameters, one near-80% switch with eight workers stopped, 45-55% CutMix among strong batches, hard weak targets, and 18-19 unique at-most-once-per-epoch evaluations (never above 19). Record the switch, first weak, best/final, NLL, exposure, memory, and timing, but classify only from integrity plus the queried primary gate: `best_test_acc >= moving baseline + 0.10` is improvement; a valid lower value is no-improvement; a preflight/timing veto or malformed run is invalid/crash. Never rerun seed 42.

## Risks and no-rescue boundary

- **Accuracy/transfer risk — high:** direct evidence is from face recognition, while balanced CIFAR-10 top-1 may benefit from magnitude information.
- **Scale/underfit risk — high:** 8 is theory-grounded for ten classes but unvalidated for the accepted 300-second CutMix recipe; scale is the dominant hyperparameter.
- **Optimization risk — medium-high:** normalized logits are bounded, but inverse-norm Jacobians and functionally weakened classifier decay change SGD geometry immediately.
- **CutMix risk — medium:** equal-radius features may regularize regional mixtures or erase useful ambiguity/quality signals.
- **Runtime risk — low-medium:** arithmetic is tiny, yet small normalization kernels can consume the one-percent exposure margin.
- **Protocol risk — medium:** accepted controls begin class-concentrated, so only prospectively control-qualified persistent/one-sided events are authoritative.
- **Single-seed risk — medium:** a narrow pass remains weak causal evidence, but rerolls are forbidden.

No rescue is permitted: do not try scales 4/6/10/16/20/30, learn temperature, normalize only features or only weights, restore/add/tanh the bias, change epsilon, add margins, detach norms, alter decay/LR/warmup/clipping, use a second classifier branch, make the method strong/weak-phase-specific, relax gates, or run production after a veto. Each changes the hypothesis and requires a new experiment ID.
