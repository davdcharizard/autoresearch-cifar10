# Report EXP-011: Cadence-31 charged-time EMA
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the frozen 300-second charged-training protocol by adding a genuine model/training improvement in `train.py`. EXP-011 grew from parent EXP-004 at 95.40%, which was also the prior global best; improvement required at least 95.50% with physical GPU 0, one validation per epoch, a complete bounded run, and intact parent semantics.

## Idea & Hypothesis

Add a horizon-derived, full-state EMA only during EXP-004's clean/SAM final quarter. An odd cadence of 31 alternates ordinary and SAM-trained samples, while a time-derived 18.75-second half-life keeps the averaging horizon meaningful under throughput variation. The hypothesis was that smoothing late iterates would reduce trajectory variance and improve `best_test_acc` by at least 0.10 points without consuming another model forward or materially reducing optimizer exposure.

## Approach

`train.py` now maintains non-gradient shadows for all parameters and persistent floating/integer buffers. Starting after 75% step-entry charged progress, every 31st post-optimizer, post-SAM-restore state is sampled; floating tensors use `2**(-dt/18.75)` retention and integer buffers copy the latest value. EMA updates and consecutive-distance work occur before the existing synchronization so they are charged. Evaluation remains exactly once per epoch: live before activation and EMA afterward, using an exception-safe full-state swap with fresh-state exact restoration, optimizer identity, RNG, coverage, mode, finite-distance, and BatchNorm audits. The online WRN, CutMix, SAM, schedules, seed, and evaluator are unchanged.

## Execution

The first CPU swap smoke exposed an in-place leaf-parameter copy outside no-grad; decorating the evaluation swap/restore method fixed it. The first GPU harness used selected even SAM step IDs that accidentally triggered cadence samples and created artificial parity skew; reserving sampling for an explicit alternating cadence sequence fixed the harness without changing experiment code. The corrected CPU smoke and full-WRN GPU preflight passed. Five paired rounds gave candidate/parent weighted latency ratio 0.999591, projected 25,570 steps and 458.2 seconds total, with 1,277.0 MiB peak allocation.

The sole metric run completed exit 0 on physical GPU 0. Claude Opus performed implementation and raw-result adversarial reviews and found no blocking concern. Transient logs and harnesses were deleted only after exact durable transcription and review.

## Results

- **Primary metric**: 95.61% (parent: 95.40%, delta vs parent: +0.21 points, +0.22%; global best: 95.61%)
- **Observations**: The run completed 25,798 steps, 133 epochs/evaluations, 300.0 charged seconds, and 447.9 total seconds at 1,222.4 MiB peak VRAM. CutMix was 10,345/20,857 and SAM 2,471/4,941. EMA made 160 updates split exactly 80/80 ordinary/SAM, 27 swaps/restores, and zero restoration, coverage, nonfinite, or RNG failures. All 159 consecutive distances were finite and nonzero. Epochs 118-133 ranged 95.44-95.61 with mean 95.493125; final accuracy/loss were 95.46%/0.1552.
- **Analysis**: The configuration passed the formal improvement threshold and demonstrated that sparse full-state EMA can be added to the validated CutMix/SAM package with negligible charged overhead and sound BatchNorm/state behavior. The hypothesis is supported at the package level, not causally isolated: the protocol evaluates only EMA in the tail, this is one fixed-seed run, and the candidate realized 0.93% more steps than the parent through ordinary run variance. The 95.61 best is a normal maximum within a stable approximately 95.49 plateau, not a singular spike, but descendants should not treat it as the stable checkpoint level.
- **Key Learning**: A full-state clean-tail EMA reached 95.61% with negligible charged overhead, but its 95.49% tail plateau limits causal certainty.

## Verification

- **Conditions**: All passed. Accuracy exceeded 95.50%; the run completed within timing constraints; 25,798 steps and 160 samples cleared dose gates; evaluation, cadence, state, RNG, BN, summary, and tracked-scope checks passed.
- **Review Notes**: Claude independently recomputed cadence divisibility, decay arithmetic, eligibility counts, evaluation routing, state inventory, line counts, timing, and preflight reconciliation. It classified the result as trustworthy while cautioning against attributing the full +0.21 to EMA alone.
- **Verdict**: improvement
- **Verdict Basis**: All hard constraints and necessary conditions passed, and 95.61% is +0.21 points over parent EXP-004, exceeding the required +0.10 margin.

## Unexplored Avenues

- **Shorter or longer time horizon**: the final EMA/live parameter distance was only 1.51% relative, so a preregistered half-life change could trade stale-state bias against smoothing without altering online training.
- **Uniform late SWA**: epoch- or cadence-level uniform averaging may exploit trajectory diversity differently, though it still needs full-buffer handling and cannot use uncharged BN recalibration.
- **Fixed EMA/live interpolation**: a preregistered blend could reduce over-smoothing and raise the stable tail level, but introduces another coefficient and parameter/BN compatibility risk.

## Next Steps

- **Add a low-overhead representation or calibration mechanism on EXP-011 (medium confidence)**: the new formal threshold is 95.71 while the stable EMA tail is about 95.49, so prioritize mechanisms with plausible effect near 0.3 points rather than micro-tuning.
- **Test an averaging-horizon variant only with a strong mechanistic rationale (low confidence)**: the current EMA is valid and efficient, but one run does not identify whether 18.75 seconds is near-optimal.
- **Explore memory-rich, fused changes that preserve 25k+ steps (medium confidence)**: abundant H20 headroom remains, whereas extra full forwards directly compete with validated optimizer exposure.
