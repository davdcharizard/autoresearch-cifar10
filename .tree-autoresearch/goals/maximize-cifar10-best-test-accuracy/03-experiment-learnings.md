# Experiment Learnings: Maximize CIFAR-10 Best Test Accuracy

<!-- Per-goal qualitative distillation, agent-appended during the analyze phase.
     Three categories below. Each bullet follows the
     3-line, budget-strict format:

       - **{statement — ≤150 chars}** ({source refs})
         Evidence: {1-2 lines, MUST cite a source path — report / log / JSON / URL}
         Implication: {1-2 lines}

     HARD CEILING: if a bullet's total length exceeds ~3-4 sentences after composition, REWRITE it
     shorter. Push any detail that does not fit into the experiment report and cite the report
     path from Evidence. Do NOT commit an over-budget bullet. The Failed Approaches section
     uses Mechanism/Insight in place of Evidence/Implication — same budget shape. -->

## Protocol Findings

<!-- Measurement, verification, or benchmarking discoveries specific to this goal's metric.
     Examples:
       - "Stage-1 measurements are contaminated by CUDA autotune warmup — measure warm."
       - "Stored baselines expire same-day; require a same-day pristine reference run."

     Good format (fits budget length, cites report):
       - **Baseline pLDDT drifts ~0.3/day; require same-day pristine reference** (EXP-006)
         Evidence: goals/{slug}/experiments/006/04-analysis.md § Verification Results — 8-case drift table
         Implication: capture pristine baseline same-day as experiment measurement. -->

### High Importance
### Medium Importance

- **Sparse component audits can identify numerically redundant optimizer-order variants** (EXP-020)
  Evidence: L2-only plus cross terms were just `4.71e-6` of GC's removed energy (`experiments/020/04-analysis.md`).
  Implication: measure the proposed update delta before spending a metric run on an optimizer-order ablation.

- **Allocation checks need fixed diagnostics and bounded post-state tolerance** (EXP-017, EXP-018, EXP-019)
  Evidence: EXP017 retained losses; EXP018 stayed byte-flat; EXP019 grew only 640 KiB with fixed reserved memory (`experiments/019/04-analysis.md`).
  Implication: create persistent state first, release transients, and use a small preregistered live-allocation tolerance rather than byte identity.

- **Five-round 0.5% ratio-dispersion gates can reject measured zero-overhead candidates** (EXP-016)
  Evidence: median ratio was 0.999462 and maximum 1.005321, but MAD/median 0.005307 blocked the metric run (`experiments/016/04-analysis.md`).
  Implication: calibrate dispersion ceilings from measured round noise or use more rounds before making them decisive.

- **Candidate-only smoke can materially understate full-run peak allocation** (EXP-015)
  Evidence: the complete model/optimizer/SAM/EMA smoke peaked at 610.3 MiB versus 1,222.4 MiB in the metric run (`experiments/015/04-analysis.md`).
  Implication: keep memory gates broad or calibrate against parent smoke/run ratios; do not treat isolated allocation as the full-run footprint.

- **Feasibility gates must pass the measured parent by construction** (EXP-008)
  Evidence: the 30,720 img/s absolute floor rejected parent worst throughput 29,485.7, though candidate remained clearly slower (`experiments/008/04-analysis.md`).
  Implication: use same-harness parent-relative gates with dispersion and verify the parent before candidate rejection.
- **Paired latency can still underpredict realized fixed-budget dose** (EXP-012)
  Evidence: a 1.0079x paired ratio projected 25,595 steps, but the metric run realized 25,376 (`experiments/012/04-analysis.md`).
  Implication: treat projections as feasibility estimates; preregister a realized-dose gate and never metric-retry a projection miss.
- **Sub-0.30-point gains sit below observed single-run selection and tail noise** (EXP-003, EXP-006)
  Evidence: EXP-003 candidates reversed by 0.14-0.29 points; EXP-006's final four evaluations spanned 0.15 points (`experiments/006/04-analysis.md`).
  Implication: prioritize candidates with a plausible effect near 0.3 points and treat smaller single-run deltas as unresolved.
- **Transient logs make durable metric transcription part of verification** (EXP-010)
  Evidence: Claude confirmed internal consistency but could not re-derive deleted `run.log` or preflight JSON (`experiments/010/04-result-review.md`).
  Implication: copy exact summaries, counts, and line-sourced diagnostics into `03-execute.md` before protocol-mandated deletion.
- **Max-selected accuracy needs its late-checkpoint plateau for context** (EXP-011)
  Evidence: the formal 95.61 best came from an epoch-118-133 EMA plateau with 95.493 mean and 95.46 final (`experiments/011/04-analysis.md`).
  Implication: carry tail range/mean and final accuracy beside the formal best when setting expectations for descendants.

### Low Importance

- **Wall-clock progress makes fixed-seed phase exposure throughput-dependent** (EXP-004, EXP-006)
  Evidence: the same budget yielded 25,560/25,644 steps and 2,449/2,488 SAM pulses (`experiments/006/04-analysis.md`).
  Implication: report exact phase dose and do not causally attribute sub-0.10-point deltas across throughput-changing runs.
- **Half-overlap batching preserves steps but halves new identities and epoch evaluations** (EXP-005)
  Evidence: `experiments/005/04-analysis.md` - 389-step epochs presented 99,584 views of 49,920 identities and yielded 66 evals in 25,492 steps.
  Implication: prefer memory-based history targets or an explicit sampler control when evaluating temporal distillation under a fixed-time budget.

## Failed Approaches

<!-- Approaches tried for this goal's metric that did not work, with a root-cause mechanism.
     Promotion rule: count = 1 → Low; count ≥ 2 → Medium; count ≥ 3 → High. Within each tier,
     sort by count descending (most-hit at top). On promotion to High, add a
     `> **Recurring failure:** ...` callout at the top of this section if not already present.

     Entry format:

       - **{Approach name / root mechanism}** (count: N, EXP-NNN, EXP-MMM)
         Mechanism: {1 line — why the approach fails}
         Insight: {1-2 lines — what alternative approaches might still work} -->

### High Importance
### Medium Importance

- **Dense stage-3 widening is a repeated H20 feasibility failure** (count: 2, EXP-014, EXP-022)
  Mechanism: widths 288 and 320 both cost about 1.161x despite 8.33% versus 17.56% extra MACs, reducing fixed-time dose below their gates.
  Insight: close nearby scalar widths; require a qualitatively different kernel geometry and direct device timing for further capacity work.

### Low Importance

- **Multi-launch FP32 channel gates exceed the compact WRN budget** (count: 1, EXP-009)
  Mechanism: four active late-stage SE paths added 20.7% median step latency despite only 0.924% more parameters.
  Insight: prefer fused/single-kernel representation changes or require parent-relative latency before accuracy.
- **Dual-view CPU phase gating exhausts loader headroom** (count: 1, EXP-008)
  Mechanism: clean-FP32 plus augmented-uint8 paired output cost about 2.1x parent loader time before GPU work.
  Insight: prefer single-view GPU augmentation or ablate transport versus operation cost before another CPU policy.

## Patterns

<!-- Goal-specific patterns, mechanisms, and reusable recipes discovered during experiments.
     Examples:
       - "Primary-bucket-only gating by aligning a new gate's threshold with an existing gate's
          threshold keeps long-path behavior bit-identical — composable."
       - "Image-time static override of configs_base.py avoids runtime-mutation side effects."

     Entry format: 3-line statement / evidence / implication (budget-strict). -->

### High Importance
### Medium Importance
### Low Importance

- **Adaptive and Euclidean SAM radii require measured-norm calibration** (EXP-007)
  Evidence: adaptive `rho=0.5` produced `||epsilon||=0.450053`, about 9x the parent's Euclidean `rho=0.05` (`experiments/007/04-analysis.md`).
  Implication: describe package comparisons honestly; match measured perturbation norm before claiming a geometry-only effect.
- **Substitution tests identify only the difference between removed and added mechanisms** (EXP-006)
  Evidence: replacing about 2,557 CutMix batches with 2,561 manifold batches left accuracy within noise and worsened loss (`experiments/006/04-analysis.md`).
  Implication: prefer additive tests that retain validated mechanisms at full dose when the metric gate is narrow.

- **Seventeen per-step gradient reductions can remain cheap on H20** (EXP-019)
  Evidence: GC preflight was 1.0077x median and the run retained 27,976 steps versus EXP002's 27,950 (`experiments/019/04-analysis.md`).
  Implication: measure the exact operation mix; small reductions need not reproduce the multi-path forward overhead seen with SE gates.

- **Energy decomposition distinguishes active harm from optimizer redundancy** (EXP-019)
  Evidence: GC removed 22.05% of eligible squared energy yet lowered the stable tail, ruling out a no-op explanation (`experiments/019/04-analysis.md`).
  Implication: carry intervention magnitude beside accuracy before diagnosing a projected optimizer as redundant or underexecuted.

- **Wall-clock schedules stay aligned when architecture and throughput change** (EXP-001)
  Evidence: `experiments/001/04-analysis.md` - LR reached its 5% warmup peak and 0.002 endpoint despite a new 28,790-step horizon.
  Implication: keep optimization phases tied to charged time when comparing models with different step throughput.
- **WRN-16-4 leaves substantial H20 headroom under the fixed budget** (EXP-001)
  Evidence: `experiments/001/04-analysis.md` - 2.75M parameters used 1,178.9 MiB and completed in 471.9 total seconds at 94.62%.
  Implication: prioritize generalization and recipe improvements before treating memory or runtime as saturated.
- **Dedicated augmentation generators isolate controlled parent RNG streams** (EXP-002)
  Evidence: `experiments/002/04-analysis.md` - separate seed-42 CPU/CUDA generators added CutMix without consuming shuffle or drop-path RNG streams.
  Implication: use dedicated generators when testing new stochastic training interventions against a fixed parent.
- **Two-pass methods can preserve one-pass stochastic and BatchNorm semantics** (EXP-004)
  Evidence: `experiments/004/04-analysis.md` - CUDA RNG replay, second-pass BN suppression, and exact restore enabled 2,449 valid SAM pulses.
  Implication: isolate the extra gradient pass while charging its work and applying only one optimizer and BatchNorm update.
- **Odd sparse cadences balance period-two training paths exactly** (EXP-011)
  Evidence: cadence 31 yielded 160 clean-tail EMA samples split exactly 80 ordinary/80 SAM (`experiments/011/04-analysis.md`).
  Implication: choose cadence coprime to periodic mechanisms when a shadow or audit must represent every path evenly.
- **Full-state EMA can use memory headroom without reducing optimizer exposure** (EXP-011)
  Evidence: candidate/parent preflight latency was 0.9996x; the run retained 25,798 steps with 1,222 MiB peak and 27 exact restores (`experiments/011/04-analysis.md`).
  Implication: prefer sparse fused state updates over extra forwards when memory is abundant but charged compute is constrained.
- **Equal-MAC stage placement can materially change H20 step latency** (EXP-010)
  Evidence: moving one block from 64x32x32 to 256x8x8 reduced paired median latency from 10.000 to 9.242 ms (`experiments/010/04-analysis.md`).
  Implication: benchmark tensor shapes and activation traffic; FLOPs and parameter counts alone do not predict fixed-budget exposure.
- **Preallocated center-indexed GPU masks make per-image erasure cheap** (EXP-012)
  Evidence: the exact helper averaged 71.1 us over 1,000 calls with zero allocation growth; paired package latency was 1.0079x (`experiments/012/04-analysis.md`).
  Implication: reuse GPU mask banks as an augmentation primitive, while separately gating realized training dose and accuracy.
- **Architecture shape changes perturb shared initialization under a fixed seed** (EXP-010)
  Evidence: only 10/14 shared stem/first-block state tensors matched because construction consumed shape-dependent RNG (`experiments/010/03-execute.md`).
  Implication: treat architecture results as package-level unless initialization uses an explicitly isolated design.
