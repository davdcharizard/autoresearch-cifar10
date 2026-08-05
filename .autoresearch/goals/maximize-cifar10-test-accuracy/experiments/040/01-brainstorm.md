# Brainstorm EXP-040
**Created**: 2026-07-27

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): preserve the accepted compute-effective spatial learner and place any new geometry work after pooling.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): confidence control can improve generalization, but the note cautions against uncalibrated stacking with existing soft-target regularization.

No network, remote source, or new literature retrieval was used. This offline pass uses the persistent knowledge base, accepted source, measured system understanding, and 39 indexed experiments.

## Experimental History Review

- EXP036 remains the 94.48% frontier with 94.45% final accuracy, 0.2456 loss, and 130.304 passes. Its scaled pooled residual MLP shows that low-dimensional post-pooling geometry can improve both accuracy and loss.
- EXP037/038 bracket classifier decay at accepted `5e-4`: zero and `1e-3` both lose at normal exposure. This closes norm-shrinkage strength as an immediate lever but does not test how class-vector norms and directions are represented in the forward map.
- EXP008/039 bracket hard-tail LR motion around the accepted global cosine. Mixup strength/duration, masking, averaging, SAM, batch scaling, precision/layout, SE, late freezing, and nearby capacity changes are also closed or infeasible in tested forms.
- The learner nearly interpolates training but retains 0.2456 test CE. Compute is binding and memory is not; a classifier-geometry intervention over only `10 x 128` weights can target boundary quality with negligible spatial cost and preserve the >=127-pass regime.
- Full cosine feature/classifier normalization remains underdetermined because it needs a logit scale. A Frobenius-preserving row-norm treatment can remove only class-specific radial variation while deriving its common scale from the existing weight tensor, avoiding a temperature sweep.

## Collected Ideas

- **Frobenius-preserving equal-row-norm classifier** - replace only the effective classifier weight in `forward` by row-normalized directions multiplied by the root-mean-square raw row norm. This makes all ten effective class vectors share a norm while exactly preserving total classifier Frobenius norm and retaining feature norms, bias, parameters, initialization stream, and accepted decay coefficient. It targets radial class bias without a free scale.
- **Weight-normalized classifier with learned global gain** - split classifier rows into normalized directions and a single learned common gain initialized from the accepted RMS row norm. It exposes magnitude cleanly but changes parameterization/optimizer state and adds a gain whose learning rate/decay allocation become new choices.
- **Class-vector orthogonality loss** - add a training-only penalty on off-diagonal normalized classifier Gram entries, prospectively tied to accepted `WEIGHT_DECAY`. It encourages directional diversity without inference changes, but the penalty scale is not diagnosed and normalized geometry makes ordinary decay semantics indirect.
- **Bias-free classifier simplification** - remove the ten classifier biases. CIFAR-10 is balanced and a common post-BN pooled representation may need little class-prior offset; construction RNG and all weight bytes can remain exact. The intervention is exceptionally clean but likely below the 0.10-point margin.
- **Pooled-feature RMS normalization** - normalize each pooled feature vector and restore a training-derived common scale before the classifier. This attacks sample-norm confidence and approximates angular decisions, but deriving and freezing a scale without evaluator leakage introduces batch/EMA state and a calibration phase.
- **Mean-norm equalization after each optimizer step** - project classifier rows to their RMS norm after every update. It enforces the same geometry directly but invalidates Nesterov momentum's relation to parameters and adds an order-sensitive projection outside the optimizer.
- **One-time hard-boundary momentum reset** - clear accepted Nesterov buffers at the first hard-label step while leaving the now-protected global cosine untouched. It is parameter-free and isolates inherited optimizer state, but stale velocity decays below 1% in about 44 steps, giving low expected impact.
- **Training-only class-prototype moonshot** - maintain early classwise means of pooled features and apply a separation/attraction objective before the hard tail. It could directly shape inter-class geometry, but mixup semantics, state synchronization, and an arbitrary loss weight make it too compound for current evidence.

## Combinations

- **Equal row norms + bias removal**: together they impose a balanced radial classifier with no class-specific norm or offset. This is more symmetric than either alone, but combining a meaningful geometry restriction with a ten-parameter simplification would obscure attribution and could overconstrain the boundary.
- **Equal row norms + orthogonality penalty**: norm equalization controls class radii while the penalty separates directions, approximating a structured spherical classifier without feature normalization. It is more complete geometrically, but penalty strength remains a second unsupported degree of freedom.
- **Normalized classifier + direct-path auxiliary CE**: enforce balanced class-vector radii while supervising both raw and refined pooled representations. This could align two feature geometries to one classifier, but prior review found no diagnosed direct-path collapse and the auxiliary coefficient remains weakly grounded.

## Candidate Ideas

### Frobenius-Preserving Equal-Row-Norm Classifier
**Summary**: Keep raw `fc.weight`, its accepted initialization bytes, optimizer membership, `5e-4` coupled decay, and bias, but use `W_eff[i] = W[i]/||W[i]|| * ||W||_F/sqrt(10)` in every forward. This equalizes all effective class-vector radii while exactly preserving total classifier Frobenius scale, feature norms, directions, parameters, and state-dict layout. No temperature, gain, epsilon, feature normalization, or projection is added.

**What it targets**: Class-specific radial bias in the final `10 x 128` boundary after the successful pooled residual representation, at negligible spatial cost and without reopening classifier decay strength.

**Reasoning**: Accepted initialization has a measured 6.96% row-norm CV and 1.2725 max/min ratio; the transformation preserves Frobenius norm but changes weight space by 6.95% and initial logit RMS by only 3.77%. Thus it removes a real geometric degree of freedom without a global scale reset. Under the differentiable map, raw radial learning becomes a common global gain while class directions remain free and accepted coupled decay still shrinks that gain. Full contract: `proposals/idea-01.md`.

**Sources**: `experiments/036/04-analysis.md`; `experiments/037/04-analysis.md`; `experiments/038/04-analysis.md`; `02-system-understanding.md`; `proposals/idea-01.md`.

**Estimated Effort**: medium

**Risk Assessment**: Unequal class radii may encode useful class difficulty despite balanced counts; the shared RMS couples rows and changes conditioning; tiny norm-reduction kernels may cost launch time. A miss rejects only this exact scale-preserving forward map, not all normalized classifiers.

### Decay-Calibrated Centered-Simplex Regularization
**Summary**: Center classifier rows to remove softmax gauge freedom, normalize directions, and penalize deviation of pairwise cosines from the ten-class simplex target `-1/9`. Derive one fixed coefficient before training so the initial angular-gradient Frobenius norm equals accepted classifier decay-gradient norm, with no coefficient sweep; inference remains unchanged.

**What it targets**: Irregular directional spacing among class boundaries rather than radial norms, while preserving the accepted affine classifier and decay coefficient.

**Reasoning**: Raw orthogonality is gauge-dependent and mathematically inappropriate for centered ten-class weights, while simplex geometry is gauge/scale invariant. Initial centered pairwise cosine mean is already `-0.110607` versus target `-0.111111`, but individual pairs span `[-0.277645, 0.062672]`; the penalty would reduce that anisotropy at negligible spatial cost. Full contract: `proposals/idea-02.md`.

**Sources**: `experiments/036/04-analysis.md`; `experiments/037/04-analysis.md`; `experiments/038/04-analysis.md`; `02-system-understanding.md`; `proposals/idea-02.md`.

**Estimated Effort**: high

**Risk Assessment**: There is no diagnosis that angular irregularity is harmful; semantic class relationships need not form a regular simplex; initial gradient matching is defensible but not optimal and adds regularization to a recipe where additive regularizers often fail.

### One-Time Nesterov Reset at the First Hard-Label Step
**Summary**: Preserve the accepted global cosine and all SGD settings, but zero all 52 live momentum buffers exactly once before the first hard-label forward/backward at the 65% target transition. Do not reset parameters, BN state, gradients, or buffers at the later RandAugment boundary.

**What it targets**: Inherited mixed-target optimizer velocity during the abrupt switch to hard labels, independently of the now-closed tail LR amplitude family.

**Reasoning**: The first reset update removes exactly `0.81*b_old` from PyTorch Nesterov direction, and in-place zeroing has no free scalar or persistent compute. However inherited memory falls below 1% in 44 updates, only 0.225 passes and about 0.5 seconds; benefit would require path amplification. Full contract: `proposals/idea-03.md`.

**Sources**: `experiments/039/04-analysis.md`; `experiments/039/proposals/idea-01.md`; `02-system-understanding.md`; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: The transient is very short and may erase useful mixed-target velocity across the entire backbone. A miss closes full/partial/selective boundary reset rescues but not unrelated geometry.

## Review

The offline critic selected Frobenius-preserving equal-row normalization at 3/5 evidence and 3/5 impact. I adopted its limits: equal Frobenius norm is an instantaneous invariant, not an accepted scale trajectory; balanced labels do not prove equal optimal margins; and success cannot distinguish radius removal from changed tangential conditioning or shared radial coupling. A miss closes only this exact differentiable RMS map and algebraic equivalents. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. Equal-row normalization has a measurable intervention and parameter-free scale rule. Centered-simplex regularization lacks a diagnosed angular failure and uses a heuristic coefficient; momentum reset is clean but its direct effect lasts only 0.225 passes.

## Chosen Idea
**Selected**: Frobenius-Preserving Equal-Row-Norm Classifier

**Why this idea**:
Replace only the effective final classifier rows by their raw directions times the differentiable RMS raw-row norm. This preserves total classifier Frobenius norm at each state, pooled-feature norms, raw directions, bias, parameter count, initialization bytes, optimizer membership, and accepted `5e-4` decay while removing class-specific radii from logits. It avoids an arbitrary temperature and operates over only 1,280 weights after pooling.

**Hypothesis**:
If direct class-specific classifier-radius freedom is harmful in the accepted pooled-head learner, the exact RMS equal-row-norm map will retain at least 127 projected and realized passes and raise fixed-seed `best_test_acc` from 94.48% to at least 94.58%. Final accuracy >=94.45% and loss <=0.2456 are corroboration only. Success supports the complete reparameterization without isolating geometry from conditioning; a normal-exposure miss closes only this exact differentiable map and algebraic equivalents.
