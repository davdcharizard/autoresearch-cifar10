# Report EXP-034: Conv2d-Only Kaiming Fan-Out Initialization
- **Created**: 2026-08-06

## Goal

Maximize seed-42 CIFAR-10 `best_test_acc` above the moving 94.15% baseline, with at least +0.10 percentage points required, by changing only `train.py` under the fixed 300-second training and 600-second wall protocols.

## Idea & Hypothesis

Initialize Conv2d tensors with torchvision ResNet's explicit Kaiming fan-out/ReLU convention while retaining the accepted default fan-in call for the Linear classifier. The change was chosen because it has no recurring cost, preserves the accepted graph and data curriculum, and might improve gradient transport through the stem and widening transitions. The hypothesis was that BatchNorm would preserve the initial function closely enough for the altered optimizer geometry to remain safe and raise accuracy to at least 94.25%.

## Approach

`train.py` split `_weights_init` by module type. Nineteen Conv2d modules received explicit fan-out/ReLU initialization, while the single Linear module retained its literal default Kaiming call. Because equal-fan convolutions are unchanged for the same RNG draws, only the 3-to-32 stem and two widening convolutions rescaled. Model graph, parameters, data, optimizer, schedule, timer, evaluator, and runtime operations remained unchanged. An ignored controller proved exact construction and replayed the registered 200-strong/64-weak immutable corpora before authorizing production.

## Execution

Static, construction, initial-function, and corpus checks passed. The first controller attempt hit a comparison-harness-only `staticmethod` restoration error; restoring the saved function with its descriptor fixed that mechanical issue without changing candidate semantics, data, seed, or thresholds. The completed replay then failed its preregistered candidate-only class-concentration gate, so the seed-42 production run and evaluator were not executed. Two post-veto control/control repeats under production-default CUDA settings were not bitwise identical, but neither showed one-sided concentration or any greater-than-95% concentration after the ordinary shared first-two-step transient.

## Results

- **Primary metric**: NaN (baseline: 94.15%, delta: N/A; production not authorized)
- **Observations**: Construction changed only the three registered tensors at scales 0.306186/0.707107/0.707107, with identical post-construction CPU/CUDA RNG state. On real hard/soft batches, relative logit L2 stayed below 0.044% and losses within 0.011% of control, confirming BatchNorm's near-cancellation of initial forward scale. During immutable replay, however, the candidate reached 99.22-100% one-class predictions at steps 9, 14, 15, 16, 24, and 25 while controls remained at 65.63-89.84%. Candidate/control maxima were 3.097x for logit RMS, 2.398x for gradient norm, and 1.952x for update norm; the stem's update reached 13.988% of its parameter norm. Lower terminal strong/weak loss EMA ratios of 0.9765/0.9759 did not waive the safety veto.
- **Analysis**: The intervention achieved its intended nearly invariant initial function but not a continuous optimization trajectory. Smaller stem and transition parameter norms left absolute forward features almost unchanged through BatchNorm while amplifying relative SGD motion, producing six late early-phase class collapses absent from the control/control noise check. CUDA backend nondeterminism limits fine-grained causal attribution, and the short replay cannot measure full-phase underfit or accuracy, but the discrete candidate-only concentration pattern is sufficient under the preregistered protocol to reject production. This discredits the exact all-Conv fan-out operating point; it does not establish a primary-metric result.
- **Key Learning**: Near-identical BatchNorm-normalized initialization can conceal amplified relative SGD geometry; all-Conv fan-out caused six candidate-only one-class transients.

## Verification

- **Conditions**: Baseline/source, static semantics, exact construction, initial-function similarity, and immutable-corpus integrity passed; trajectory safety failed; production and evaluator checks were skipped.
- **Review Notes**: Report hash `7699baab7ef9395bf1b68fe568392b8bb6103e9ee0ea035dfff6922d22e38815` was serialized before assertions. Production-default control/control repeats confirm the backend is not bitwise deterministic, but neither reproduced the candidate's later concentration. The single-seed protocol and lack of a baseline noise estimate also mean even a narrow scored pass would have required cautious causal interpretation; no such score exists here.
- **Verdict**: invalid
- **Verdict Basis**: A preregistered safety gate blocked production before evaluation, leaving only partial preflight evidence and no trustworthy primary metric; the index therefore records NaN.

## Unexplored Avenues

- Excluding the stem, selecting only transitions, or interpolating fan-in/fan-out scales could reduce relative-step amplification, but those are post-veto parameter tweaks without independent positive evidence and are explicitly retired for this line.
- Learning-rate, decay, clipping, or warmup compensation might restore reparameterization invariance, but each adds a second optimizer intervention and conflicts with the validated ordinary-momentum recipe.
- A future initialization proposal would need to preserve both initial function and parameter-to-update scale by construction, with an independent representation hypothesis rather than a rescue of this exact point.

## Next Steps

- **Seek an intrinsically bounded representation or capacity lever** (medium confidence): prefer changes whose forward contribution and relative update are structurally constrained rather than only small at initialization.
- **Consider classifier symmetry as a narrow zero-cost hypothesis** (low-medium confidence): explicit bias symmetry avoids altering hidden parameter scales, though the likely effect may be below the required margin.
- **Defer channels-last to a systems-motivated experiment** (low-medium confidence): it remains measurable and semantics-preserving, but its accuracy link through a few percent more exposure is weak.

## Exit Action Results

- No exit actions were configured.
