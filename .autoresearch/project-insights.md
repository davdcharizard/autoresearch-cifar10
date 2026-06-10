# Project Insights

<!-- Cross-goal STRATEGIC wisdom about this project. Read during ideation.

     Record an insight here only if BOTH hold:
       (1) STRATEGIC — guides DIRECTION of future goals/experiments (theory, mechanism, hardware
           envelope, hard-to-refactor blocker). Not just structural.
       (2) CROSS-GOAL — still useful for a future goal with a DIFFERENT metric.
     If either fails, it belongs in the experiment report, goal-learnings, infra-errors, or
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
         Evidence: reports/exp-report-004.md § Analysis — `model.cfg.foo=X` altered logit scaling
         Implication: prefer image-time static config overrides over runtime mutation -->

### High Importance

### Medium Importance

### Low Importance
- **Plain FP32 compile/channels-last beats reduced-precision variants for this small CIFAR CNN** (EXP-001, EXP-002, EXP-007)
  Evidence: reports/exp-report-001.md, reports/exp-report-002.md, and reports/exp-report-007.md — BF16 hit 91.48%, TF32 hit 91.39%, FP32 hit 91.95%.
  Implication: prefer plain FP32 throughput work before AMP or TF32 when optimizing small benchmark CNNs.
