# Report EXP-042: Exact-Neutral Centered Content-Attention Pooling
- **Created**: 2026-07-27

## Goal

Maximize fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.48% baseline under the frozen 300-second counted-training contract. EXP042 tested whether content-adaptive final spatial pooling could improve the accepted pooled-head learner by at least 0.10 points.

## Idea & Hypothesis

One zero-initialized 128-element content query scored the final 8x8 positions. The candidate preserved accepted GAP and added `sum((softmax(q^T x)-1/64)*x)` before the accepted pooled MLP, making startup exactly accepted while giving the query a covariance-driven first gradient. The hypothesis required best at least 94.58% and at least 127 passes; final at least 94.45% and loss at most 0.2456 were corroboration.

## Approach

The scorer was a bias-free `Conv2d(128,1,1)` constructed after all accepted parameters in a restoring CPU RNG fork and overwritten with zeros. It entered the ordinary matrix-decay group and added 128 parameters. The centered correction preserved accepted reduction ordering and exact zero-start output; no temperature, scale, bias, position state, alternate loss, schedule, data, RNG, classifier, or evaluator change was introduced.

## Execution

Semantic qualification passed directly. Zero-query CPU/CUDA pooled features, logits, common state, and common gradients were accepted exactly; scorer gradients were nonzero in mixup/hard fixtures, the covariance oracle agreed to `2.22e-16`, and full update errors stayed below `1.19e-7`. Fresh updates produced broad but nonuniform attention with 63.77-64.00 effective sites. Timing retained 0.980831 exposure, projected 127.806 passes, stayed below 0.66% CV, and peaked at 614.92 MiB.

The sole score completed without retry or runtime error. Mixup stopped at step 15,982/195.0 seconds and RandAugment after the epoch-82 iterator exhausted at step 15,990/195.1 seconds. The run produced 26 unique every-fifth plus final evaluations.

## Results

- **Primary metric**: 93.80% (baseline: 94.48%, delta: -0.68 points, -0.72%)
- **Observations**: Final equaled best at 93.80%, and final loss worsened from 0.2456 to 0.2787. The run delivered 24,987 steps, 127.93344 passes, 129 epochs, 1,096.4 MiB peak VRAM, and 1,003,610 parameters in 300.0 counted / 346.1 wall seconds.
- **Analysis**: The exact accepted endpoint and common first-step training signal were preserved while the scorer opened through a real data gradient, so startup disruption or exposure loss cannot explain the regression. The complete adaptive pooling trajectory harmed boundary quality: uniform averaging may supply a useful invariance, one shared query may discard distributed evidence, or learned scores may remain too weak or become misleading. Scored attention telemetry was deliberately absent, so the result cannot distinguish those mechanisms or claim where the query focused.
- **Key Learning**: Preserve uniform GAP on this pooled-head frontier; exact-neutral single-query spatial selection worsens both top-1 and CE at normal exposure.

## Verification

- **Conditions**: Completion/resource contract passed; primary metric improvement failed.
- **Review Notes**: Results are trustworthy. One H20, one fixed-seed score, `train.py`-only scope, exact startup identity, analytic gradient/update/RNG checks, 127.933 passes, correct transitions, once-per-epoch cadence, and complete summary all passed.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid 93.80% missed baseline by 0.68 points and threshold by 0.78, with loss worse by 0.0331.

## Unexplored Avenues

- Spatial standard deviation and centered 2x2 contrasts remain formally untested, but their arbitrary active starts and stronger confounds make them lower priority after this negative spatial-readout result.
- Other temperatures, queries, initializations, scales, and cutoffs are declined as immediate post-result search policy; this score does not experimentally falsify them.
- Independently motivated second-order pooling remains distinct but needs stronger evidence and a protected reduction/backward cost.

## Next Steps

- **High confidence**: Restore uniform GAP and preserve the accepted pooled residual MLP, sole CE, classifier, decay, and schedule.
- **Medium confidence**: Seek a mechanism outside final classifier/loss/pooling constraints; a new representation idea should preserve the 127-pass regime and avoid arbitrary active startup.
- **Low confidence**: Test one-time boundary momentum reset only as a clean transient closure if no stronger orthogonal mechanism survives review.
