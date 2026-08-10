# Project Insights

<!-- Cross-goal STRATEGIC wisdom about this project. Read during ideation.

     Record an insight here only if BOTH hold:
       (1) STRATEGIC — guides DIRECTION of future goals/experiments (theory, mechanism, hardware
           envelope, hard-to-refactor blocker). Not just structural.
       (2) CROSS-GOAL — still useful for a future goal with a DIFFERENT metric.
     If either fails, it belongs in the experiment report, experiment learnings, infra-errors, or
     context — not here.

     Examples of valid entries:
       - "Memory-trading optimizations are fair game; memory-compression is not — VRAM is 13-24 GB of 98 GB."
       - "Runtime mutation of model attributes has non-local side effects — prefer image-time static overrides."
       - "torch.compile blocked by custom fast_layer_norm kernel + scipy Rotation in forward path."

     One section below: ## Experimental (loop exit), tiered High / Medium / Low.

     Entry format (3-line, budget-strict; HARD CEILING ~500 chars per bullet):

       - **{Insight — 1 line, ≤150 chars}** ({source refs})
         Evidence: {1-2 lines, MUST cite a source path — report / log / JSON / URL}
         Implication: {1-2 lines — what future work should do differently}

     Contradictions: note inline, e.g., (EXP-003 contradicts EXP-001). -->

## Experimental

<!-- Strategic insights added by here. Source refs use experiment IDs (EXP-NNN).
    If a matching bullet exists, extend its source-ref list and promote tier if warranted — do NOT duplicate.

     Example:
       - **Runtime mutation of model attributes has non-local side effects** (EXP-004, EXP-006)
         Evidence: goals/{slug}/experiments/004/04-analysis.md § Analysis — `model.cfg.foo=X` altered logit scaling
         Implication: prefer image-time static config overrides over runtime mutation -->

### High Importance

### Medium Importance

- **Safety ratios need denominator-safe, control-qualified thresholds** (EXP-035)
  Evidence: zero-start BN biases yielded `9e28` relative updates, and two accepted control pairs exceeded a nominal 5x site-gradient limit.
  Implication: define zero-denominator behavior and reject non-specific gates on controls before exposing candidate results.

- **Constant-direction optimizer matching does not bound stochastic update spikes** (EXP-020, EXP-022, EXP-028)
  Evidence: EXP028 matched first/coherent directions yet reached a 12.35x paired update and class collapse; two prior state-path changes also concentrated.
  Implication: require immutable production-batch trajectory gates for optimizer changes; algebraic scale checks alone cannot establish safety.

- **Batch-dependent epochs can bias max-over-checkpoints metrics via extra evaluations** (EXP-013)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/013/02-plan-review.md` found 22-23 candidate looks versus 19 baseline looks.
  Implication: equalize elapsed evaluation count when batch/epoch length changes; per-epoch limits alone do not prevent observation-count bias.

- **Persistent DataLoader workers avoid severe Python 3.14 forkserver startup overhead** (EXP-001)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/001/02-plan.md` — epoch 2 measured 18.975s without persistence and 1.025s with it.
  Implication: reuse workers in future data-heavy goals unless worker-state semantics require epoch restarts.

- **Phase changes need deterministic persistent-worker lifecycle management** (EXP-004)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/004/04-analysis.md` § Execution — explicit shutdown stopped all workers; rebuild took 2.612s.
  Implication: hold iterator references, shut workers explicitly, verify exit, then rebuild; garbage collection alone is insufficient.

### Low Importance

- **Bounded outputs do not imply bounded optimizer geometry in normalized heads** (EXP-038)
  Evidence: RMS-matched cosine logits stayed globally safe, while per-class raw weight norms reached `4.51x` dispersion within15 SGD steps.
  Implication: track latent raw norms and inverse-norm effective learning rates for normalized parameterizations, not only output/gradient aggregates.

- **Wider CIFAR convolutions can scale sublinearly on H20** (EXP-023)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/023/04-analysis.md` — 1.456x estimated MACs cost only 1.163x paired step time.
  Implication: use paired device timing rather than MAC ratios when judging hardware-aware architecture candidates.

- **Short-fit wins can invert over the full fixed-time phase** (EXP-015)
  Evidence: `goals/maximize-cifar10-best-test-accuracy/experiments/015/04-analysis.md` — 64-step loss was 0.769x control, but switch accuracy fell 3.25 points.
  Implication: use short fits only as collapse vetoes; require phase-scale diagnostics before claiming trajectory quality.

- **Exact initial function does not guarantee optimization-continuous recruitment** (EXP-014, EXP-025, EXP-034)
  Evidence: EXP-034 kept initial relative logit L2 below 0.044% yet reached 100% one-class predictions by step 9, extending EXP-014/025.
  Implication: preflight multi-step parameter/output trajectories, not only initial equality or one update, for newly recruited branches.

- **Nominally cheap per-step changes can materially reduce work under synchronized budgets** (EXP-003, EXP-029)
  Evidence: Label smoothing reduced steps 6.7%; 19 gradient means/subtractions added 1.97% full-step cost on an idle H20.
  Implication: benchmark full-step throughput before fixed-time runs even when code, parameter, and memory changes are tiny.
