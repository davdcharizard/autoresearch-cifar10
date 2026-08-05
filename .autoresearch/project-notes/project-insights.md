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

### Low Importance

- **Free H20 memory does not imply useful CNN batch scaling** (EXP-034)
  Evidence: `goals/maximize-cifar10-test-accuracy/experiments/034/04-analysis.md` - doubling batch used <2 GiB but improved complete-body image rate only 6.10%.
  Implication: gate batch changes on measured full-body throughput and decision loss, not allocation headroom.

- **CNN memory format can change deterministic FP32 numerics materially** (EXP-031)
  Evidence: `goals/maximize-cifar10-test-accuracy/experiments/031/04-analysis.md` - NHWC logits differed from NCHW by up to 8.89e-4 on identical logical state/input.
  Implication: treat layout optimization as a numerical intervention and predeclare cross-kernel semantic bounds before performance tests.

- **Parameter count poorly predicts training cost for high-resolution CNN layers** (EXP-028)
  Evidence: `goals/maximize-cifar10-test-accuracy/experiments/028/04-analysis.md` - freezing 3.4% of parameters cut complete hard-step time 35.9%.
  Implication: attribute backward cost by resolution/stage before using parameter share to select optimization targets.

- **Equal CNN MACs can differ about 20% in H20 training speed by tensor shape** (EXP-016)
  Evidence: `goals/maximize-cifar10-test-accuracy/experiments/016/04-analysis.md` - `[1,2,3]` retained 1.205x measured throughput at identical MACs.
  Implication: benchmark resolution/channel placement directly; static MAC totals do not predict fixed-time exposure.

- **The H20 has ample memory headroom for larger CIFAR models** (EXP-001)
  Evidence: `goals/maximize-cifar10-test-accuracy/experiments/001/04-analysis.md` § Results — WRN-16-2 used 1.1 GiB of 97.9 GiB
  Implication: future work may trade memory for capacity or throughput when the time budget benefits.
