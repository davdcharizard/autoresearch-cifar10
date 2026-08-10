# Report EXP-035: Fixed SiLU Throughout ResNet-20
- **Created**: 2026-08-06

## Goal

Maximize seed-42 CIFAR-10 `best_test_acc` above the moving 94.15% baseline, with at least +0.10 percentage points required, by changing only `train.py` under the fixed 300-second training and 600-second wall protocols.

## Idea & Hypothesis

Replace all 19 dynamic ReLU operations in the accepted width-2 postactivation ResNet-20 with fixed beta-1 SiLU while preserving initialization, residual ordering, data, optimizer, schedule, timer, and evaluator. The hypothesis was that smooth signed activations would preserve weak localized evidence and gradient flow through the short strong phase, retain at least 98% exposure, maintain healthy switch fit, and raise `best_test_acc` to at least 94.25%.

## Approach

`train.py` received exactly three functional substitutions: BasicBlock's first conv-BN activation, its post-add activation, and the stem activation changed from `F.relu` to `F.silu`. Those call sites execute 19 times dynamically. An ignored controller loaded the accepted source directly from `7c1e7d8`, proved identical model state/RNG/topology, checked the SiLU value/derivative oracle, instrumented all activation sites on real hard/soft batches, and replayed four accepted controls plus the candidate over the registered 200-strong/64-weak corpora. Seven paired timing trials and one production run were conditional on a complete trajectory pass.

## Execution

Static, construction, oracle, initial-function, and corpus-integrity checks passed. The single complete preflight attempt then hit two formal trajectory gates. Eighteen steps exceeded the fixed 5x site-gradient ratio and four early steps exceeded the absolute per-tensor relative-update limit. The report was serialized before assertions, and no thresholds or statistics were changed after observing the candidate. Because the accepted control/control calibrations themselves exceeded the site-gradient bound and zero-initialized BN biases made the relative-update denominator undefined, the veto was obeyed but not interpreted as candidate-specific instability. Timing and production were skipped; no root `run.log` or primary metric exists.

## Results

- **Primary metric**: NaN (baseline: 94.15%, delta: N/A; production not authorized)
- **Observations**: SiLU passed initial real-batch gates with candidate/control loss, logit-RMS, pooled-RMS, and gradient-norm ratios of 0.6183/0.6434/0.5867/0.4430 on hard data and 0.6121/0.6441/0.5867/0.4420 on soft data. The accepted control began 100% one-class while SiLU began at 87.5%/91.41%, so no candidate-only initial concentration occurred. Across 264 updates, SiLU remained finite, had exact BN counters, produced no candidate-only >95% class-share step, and stayed within global gates: maximum logit/pooled/gradient/update ratios were 2.4685/1.2207/2.1666/1.6355, whole update/parameter was 0.01068, and preceding-median update was 1.3214. Strong/weak terminal loss EMA ratios were 0.9333/0.8751. The formal site-gradient maximum was 8.2441x, but control/control maxima were 9.5078x and 5.6247x. The absolute tensor-relative statistic reached `4.09e28` candidate and `9.03e28` control because zero-norm BN biases were divided by a `1e-30` clamp.
- **Analysis**: The available evidence does not discredit SiLU's accuracy or stability hypothesis. Its global optimizer geometry was less extreme than several prior vetoed candidates, its losses were lower, and it never showed candidate-only class collapse. However, lower short loss is not accuracy evidence, and the registered site-gradient gate formally failed. The two failing measurements were not candidate-specific: control/control variation crossed the same site bound, and the tensor-relative statistic is undefined at analytically zero parameter norm. This makes the result partial and unscorable rather than a no-improvement. It also exposes a protocol-ordering flaw: a safety statistic must be denominator-safe and demonstrate specificity on accepted controls before it is granted authority to inspect or veto a candidate.
- **Key Learning**: SiLU stayed concentration-free and globally bounded, but non-specific ratio gates blocked scoring; control qualification must precede candidate authority.

## Verification

- **Conditions**: Baseline/source, exact activation scope, static semantics, topology/RNG/oracle, initial real-batch behavior, and immutable-corpus integrity passed; the formal site-gradient and per-tensor trajectory conditions failed; timing, production, and metric verification were skipped.
- **Review Notes**: The evidence is provenance-complete: candidate source SHA `d80faf3628593a194765a0f33ea0b3bd1cb11e7972ced176ece3b080007ff94b`, controller SHA `ccf14047ae0db07b6cdfdb483219b11dbd03ac2b8f61e22d09bff41457ae2cb4`, and report SHA `ad129e253686aa27872a3c89aea5dea10c625952eef82d46dc5bc1d31e7eaf99`. The false-failure risk is demonstrated directly by both predeclared control/control calibrations, not inferred after the fact. Obeying the frozen veto preserves process integrity, but the veto cannot be cited as evidence that SiLU itself is unsafe.
- **Verdict**: invalid
- **Verdict Basis**: Production and evaluation were blocked by a preregistered preflight condition, leaving partial evidence and no trustworthy primary metric; NaN is required.

## Unexplored Avenues

- All-site fixed SiLU remains accuracy-unmeasured. It should not be rerun through a post-hoc amended EXP035 gate; any future reconsideration requires a generally applicable, prospectively control-qualified safety protocol established independently of this candidate.
- Stage/site subsets, beta changes, gain changes, approximate kernels, or ReLU/SiLU mixtures could alter throughput and signed-feature behavior, but they are unsupported adjacent rescues and were not tested.
- The planned paired timing remains unknown, so this experiment provides no evidence about whether 19 SiLU backward sites retain 98% fixed-budget exposure.

## Next Steps

- **Make all future ratio gates denominator-safe and control-qualified before candidate replay** (high confidence): zero-start tensors need a predefined absolute/relative composite, and accepted controls must pass frozen specificity criteria first.
- **Advance the deferred reflection-padding boundary-prior experiment** (medium confidence): it is orthogonal, branch-free, and avoids activation-site denominator pathology while preserving the accepted curriculum.
- **Keep EMA deferred** (medium confidence): the nearly monotonic tail and prior averaging regression still provide little measured headroom.

## Exit Action Results

- No exit actions were configured.
