# Brainstorm EXP-024
**Created**: 2026-07-26

## Web Search & Literature Review

- **mixup** (`knowledge/papers/mixup.md`): alpha controls interpolation strength; the accepted alpha-0.2/65% batch-shared treatment remains the validated base.
- **EXP-017 attention evidence** (`experiments/017/04-analysis.md`): two full stage-3 SE gates scored 94.16, lowered final loss, and incurred 4.6% measured overhead that explicitly included training-only diagnostic accumulation.
- **EXP-018/019 mechanism evidence**: final-only and static first-gate approximations scored 93.67/93.86, showing that both placements and input dependence matter despite the first gate's low aggregate example variance.
- **EXP-023 timing evidence**: diagnostic-free full gates remain semantically tractable, but composing them with 160-channel width retained only 88.94%; return to accepted width for any attention-efficiency test.

No network source was consulted; this offline loop uses persistent paper distillations and completed local artifacts.

## Experimental History Review

- The accepted WRN-16-2 with batch-shared alpha-0.2 mixup until 65% scores 94.07% in 141.9 passes. The acceptance threshold is 94.17%.
- EXP-017 is the closest result at 94.16% and 133.64 realized passes. Its full two-gate mechanism was feature-driven, but runtime included nine scalar diagnostic accumulators per gate on every training forward plus a terminal report.
- Removing either the first conditional gate or its input dependence destroyed the signal. Cheap follow-ups must keep both residual placements and per-example behavior; final-only/static designs are closed with count three high-importance evidence.
- Width-plus-full-SE failed before scoring at 126.21 projected passes. Added width is therefore removed from this loop; accepted 128-channel topology isolates attention cost and mechanism.
- Mixup duration is bracketed, alpha 0.4 over-regularizes, and alpha 0.1 remains the only untested strength-side control. SAM, EMA, schedule, decay, BF16, dropout, CutMix, and adjacent capacity changes are closed.
- The limiting gap is a noise-scale top-1 boundary. The only locally demonstrated +0.09 mechanism is full conditional stage-3 routing; the key unresolved question is whether its last 0.01 was held back by observation overhead or whether a cheaper conditional mapping can retain the signal.

## Collected Ideas

## Combinations

## Candidate Ideas

### Weaker Alpha-0.1 Mixup
**Summary**: Change only `MIXUP_ALPHA` from 0.2 to 0.1 while retaining the accepted batch-shared coefficient, permutation behavior, 65% cutoff, model, and schedule.

**What it targets**: The sole unmeasured side of mixup-strength calibration at the now-bracketed duration, reducing strongly interpolated batches without altering throughput.

**Reasoning**: Alpha 0.4 was too strong, so alpha 0.1 is the remaining controlled test of whether alpha 0.2 lies above the local optimum. The one-line change has no implementation or exposure ambiguity.

**Sources**: `knowledge/papers/mixup.md`; EXP-002/004/005/015/020.

**Estimated Effort**: low

**Risk Assessment**: Beta(0.1,0.1) is endpoint-heavy and likely under-regularizes, consistent with the shorter-window failure. Its expected ceiling is lower than the attention candidates.

### Diagnostic-Free Full Two-Gate SE
**Summary**: Recreate EXP-017's two exact-neutral ratio-16 stage-3 SE gates on accepted 128-channel residual branches, using the same preregistered seed 17017 and placement, but remove every training-time diagnostic accumulator and terminal gate report. Model/training semantics otherwise remain unchanged.

**What it targets**: The 94.16 near miss by recovering updates consumed by pure observation overhead while retaining the full cross-channel, two-placement conditional mechanism that later ablations showed necessary.

**Reasoning**: EXP-017's score missed acceptance by 0.01 and its 4.6% overhead included nonfunctional diagnostic work. EXP-018/019 make mechanism simplification unattractive; stripping observation is the only way to improve fixed-time efficiency without weakening the validated gate. This is an efficiency variant, not a seed reroll: seed, ratio, initialization, placement, data, and evaluator remain fixed.

**Sources**: EXP-017, EXP-018, and EXP-019 reports; high-importance attention learning.

**Estimated Effort**: medium

**Risk Assessment**: Diagnostic work may be negligible, making this a noise-scale replay of a result-selected seed. Require matched timing to project at least 137 passes before scoring; never change seed/ratio or rerun a valid score.

### Two Diagonal Conditional Stage-3 Gates
**Summary**: Gate both accepted stage-3 residual branches with `2*sigmoid(weight[c]*pooled_feature[c]+bias[c])`, using zero-initialized 128-vectors for exact unit output and direct first-step gradients. This retains per-example/per-channel behavior and both placements with only 512 scalar parameters and no global MLP.

**What it targets**: The conditional component missing from EXP-019 while reducing full SE's small-kernel/pooling overhead. It asks whether each channel's own global response is sufficient, rather than removing conditioning or a gate.

**Reasoning**: Static attenuation failed despite learning EXP-017's mean, so per-example response is the minimal unresolved mechanism. The diagonal form can learn sign and magnitude from the same channel's pooled residual at both blocks while preserving exact accepted logits/RNG at construction.

**Sources**: EXP-017/018/019 reports; `03-experiment-learnings.md` recurring attention failure.

**Estimated Effort**: medium

**Risk Assessment**: Cross-channel interaction may be essential, making this another destructive simplification. Zero initialization must be preflighted for nonzero finite gradients on both vectors, and scoring requires >=138 projected passes.

## Review

The blind reviewer selected diagonal two-gate conditionality at 5/10 evidence and 6/10 impact. I adopt its framing and gates: this tests whether each channel's own pooled residual is sufficient or whether cross-channel mixing is essential; zero-initialized weight/bias gradients and >=138 projected passes must pass before scoring. Full-SE without diagnostics is outcome-seed-biased and relies on an exposure premise contradicted locally; alpha 0.1 lacks a favorable mechanism. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Diagonal gates preserve both placements and per-example conditioning while removing only the unresolved cross-channel interaction. Their low cost gives more ceiling than a diagnostic-free replay, despite the strong simplification-failure prior.

## Chosen Idea
**Selected**: Two Diagonal Conditional Stage-3 Gates

**Why this idea**:
This is the narrowest mechanism change not already disproved by EXP-018/019. It keeps both residual gates and input dependence, initializes to exact accepted behavior without consuming RNG, and should recover nearly all accepted exposure. A failed score closes per-channel self-gating rather than inviting parameter rescue.

**Hypothesis**:
Exact-neutral diagonal self-gates on both 128-channel stage-3 residual branches will project at least 138 passes and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17% if per-channel input dependence, rather than global cross-channel mixing, is sufficient for EXP-017's positive attention signal.
