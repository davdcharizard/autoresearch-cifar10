# Report EXP-006: Plateau-Only Fixed-Square Cutout
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the moving baseline of `92.30%` at `11f8469`. A valid improvement had to reach at least `92.40%`, modify only `train.py`, run once on one idle H20 under the fixed 300-second counted budget, and finish within 600 seconds.

## Idea & Hypothesis

Replace EXP-004's plateau-only `RandAugment(num_ops=1, magnitude=7)` with one mean-valued 16x16 Cutout patch on every crop/flip training view, then retain the validated weak final 20%. External Claude adversarial review selected this as the highest-ceiling orthogonal invariance test, while requiring explicit underfitting diagnostics and an honest head-to-head comparison against an already successful regularizer. The hypothesis predicted at least 92.40% while retaining 98.5% of EXP-004's 38,358 steps.

## Approach

Changed only `train.py`. Removed the PIL RandAugment entry and appended `RandomErasing(p=1.0, scale=(0.25, 0.25), ratio=(1.0, 1.0), value=0, inplace=True)` after normalization, yielding one contained 16x16 normalized-zero patch. Renamed the phase state and provenance to Cutout, preserved the exact 80% switch and weak loader, and left model, hard-label loss, optimizer, schedule, seed, evaluator, and worker lifecycle unchanged. Setup assertions proved the live plateau loader used the reviewed transform without consuming RNG.

## Execution

Claude's mandatory idea and plan reviews both completed through the external harness with exit code 0; no fallback reviewer was used. Static checks passed. A disposable preflight verified 32 exact masks, 269,722 parameters, 390 batches/epoch, 247.5-256.2 warmed Cutout batches/s, clean termination of all eight workers, and a 3.020-second weak-loader transition. The initial benchmark invocation lacked `PYTHONPATH=.` and an early timing attempt included CPU tensor-scan overhead; both diagnostic issues were corrected without changing candidate code or configuration. One fixed-seed H20 training run then exited 0 without a training retry.

## Results

- **Primary metric**: `91.63%` (baseline: `92.30%`, delta: `-0.67` percentage points, `-0.73%` relative)
- **Observations**: The final Cutout checkpoint was 83.15% at epoch 78 versus EXP-004's final strong checkpoint of 84.60%. The switch occurred once at exactly 80.0% and stopped all eight workers. The first weak epoch reached 90.47% versus EXP-004's 91.43%; the tail rose to a 91.63% peak/final but never closed the gap. EXP-006 completed 38,028 steps, 98 epochs, 25 unique evaluations, 300.0 counted seconds, 339.2 total seconds, 330.1 MB peak VRAM, and 269,722 parameters.
- **Analysis**: The throughput half of the hypothesis passed: 38,028 steps retained 99.14% of EXP-004 exposure despite replacing the slower PIL transform. The accuracy half failed decisively. Lower clean accuracy before the switch and a 0.96-point deficit immediately after one weak epoch indicate that the weak tail inherited a weaker representation rather than merely needing more BatchNorm resettling. Under this short schedule and small ResNet, masking 25% of every plateau image appears less useful than RandAugment's broader geometric and photometric invariances, and may be too aggressive as an every-view replacement. Because the seed and augmentation stream differ, the exact -0.67 effect is not causal, but its size and persistent tail gap are larger than a marginal ten-image miss.
- **Key Learning**: Every-view 16x16 Cutout preserved optimizer exposure but lost 0.67 points; N1/M7 RandAugment is the better plateau regularizer here.

## Verification

- **Conditions**: Completion, numeric summary, timing, throughput, parameter count, worker lifecycle, and evaluation uniqueness passed. Primary accuracy failed: `91.63% < 92.40%`.
- **Review Notes**: Results are trustworthy. Only the reviewed `train.py` intervention changed; the sole H20 was idle; seed, evaluator, budget, and evaluation cadence were preserved; one live-loader-verified Cutout phase switched cleanly to the weak tail. The result is a valid statistical no-improvement, not an infrastructure or implementation failure.
- **Verdict**: no-improvement
- **Verdict Basis**: The run satisfied all process and integrity requirements but finished 0.67 points below the moving baseline and 0.77 points below the acceptance threshold.

## Unexplored Avenues

- Use a smaller mask or `p<1.0` while preserving the 80% boundary. This may reduce the apparent every-view underfitting, but it needs a new predeclared operating point and review rather than post-hoc tuning.
- Add a milder Cutout operation to the accepted RandAugment recipe instead of replacing it. This could combine broad invariance and occlusion, but compounded distortion and attribution make it lower priority.
- A variable-area eraser could distribute occlusion strength rather than always removing exactly 25%; it remains untested and has weaker isolation than the rejected canonical configuration.

## Next Steps

- **High confidence**: restore the full EXP-004 recipe and test an orthogonal lever without removing its validated broad invariances.
- **Medium confidence**: reconsider same-width preactivation only with a hypothesis scoped to interaction with the strong-view plateau, acknowledging shallow-depth evidence predicts a small effect.
- **Medium-low confidence**: use a reviewed sweep for nearby RandAugment magnitudes or milder Cutout settings if the optimizer can amortize multiple trials without violating the fixed evaluation protocol.

## Exit Action Results

- None defined.
