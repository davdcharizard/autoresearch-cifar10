# EXP-028: SiLU/Swish activation (ReLU → SiLU everywhere)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-028.md
- **Plan**: plans/plan-028.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-028
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Milestone 1 per plan-028: swapped all three `F.relu(...)` calls in train.py to `F.silu(...)` — BasicBlock L89 (pre-residual) and stem L127 (both identical text, done via replace_all), plus BasicBlock L92 (post-residual, `return F.silu(out)`). No config or signature changes. Smoke test passed: 0 `F.relu` / 3 `F.silu` remaining; AST parse clean; `num_params == 4,299,866` (activation is parameter-free); forward (2,3,32,32)→(2,10) OK; `git diff --name-only` = train.py only.

### Surprises & Discoveries
None. The swap is a pure pointwise-op substitution; two of the three call sites share identical text so a single replace_all covered both.

### Decisions
- Left `_weights_init` (kaiming_normal_, ReLU-gain) unchanged to keep the test clean and isolate the activation effect — every conv is BN-followed so the slight init-gain mismatch is re-normalized per layer (per plan risk note (b)).
- Used the functional `F.silu` to match the existing `F.relu` call style rather than introducing `nn.SiLU` modules — minimal diff, identical numerics.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID recorded at launch — background task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A (no WandB in this project)
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- Full 300s-compute-budget training of the k=4 WideResNet with SiLU replacing ReLU at all 3 activation sites, on a single H20. Hypothesis: a smooth non-monotonic activation improves generalization (smoother landscape, no dead-ReLU zero region) and lifts best_test_acc above the 96.32 bar. CRITICAL secondary check: throughput-neutrality — SiLU is a pointwise op that torch.compile should fuse into the conv/BN epilogue, so epochs should hold ~91 / dt ~8ms; if epochs drop below ~85 the result is compute-confounded (SiLU didn't fuse) and not a fair test of the activation.

Observations:
- Run exited 0, clean compile (no graph break on SiLU), no NaN/Traceback (Traceback count 0). Loss decreased normally from step 1 (source: run.log).
- **REGRESSION (within noise+epoch-penalty)**: best_test_acc 95.98% vs baseline 96.22 (−0.24pp), below the 96.32 bar (source: run.log summary block).
- **dt rose 8→9ms (SiLU did NOT fully fuse away)**: 681/685 sampled step lines at 9ms (vs baseline 8ms). So SiLU added ~1ms/step → num_steps 35500→34254 (−3.5%), num_epochs 91→88 (−3). This is a MILD throughput cost, above the ~85 compute-confound floor → roughly fair test, but ~3 epochs of the −0.24pp is attributable to the dt penalty, not the activation per se.
- **Loss FLAT**: final_test_loss 0.1960 ≈ baseline 0.195 — SiLU did not improve loss either (unlike the polish levers which lowered loss). A smooth activation neither helped top-1 nor loss on this already well-tuned net.

Key Metrics:
- best_test_acc: 95.98% (source: run.log summary block)
- final_test_loss: 0.1960 (≈ baseline 0.195)
- num_epochs: 88 | num_steps: 34254 | num_params: 4,299,866 | peak_vram_mb: 511.8 | total_seconds: 396.6 (source: run.log)
- mean dt ≈ 9ms (681/685 sampled lines; baseline ~8ms)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar**: FAIL. best_test_acc = 95.98% < 96.32 (baseline 96.22 + 0.1). Per plan, stop at first failure. (source: run.log summary block)
- **Cond 2 — clean completion within budget**: PASS (informational). Summary block printed, Traceback count 0, total_seconds 396.6 < 600. (source: run.log)
- **Cond 3 — no constraint violations**: PASS (informational). `git diff --name-only` = train.py only; num_params 4,299,866 unchanged; 88 evals for 88 epochs (≤1/epoch); no new deps (`F.silu` core torch); seed 42 unchanged. (source: git diff, run.log)

**MANDATORY attribution note (epoch-wall + FLOPs-neutral-≠-wall-clock-neutral, EXP-015/024):** num_epochs 88, mean dt ≈ 9ms (vs baseline 91 ep / 8ms). SiLU is a pointwise op but did NOT fully fuse into the conv/BN epilogue — it cost ~1ms/step (~12%) and ~3 epochs. Epochs held above the ~85 floor, so this is a ROUGHLY FAIR (mildly confounded) test: a smooth activation does not lift top-1 here, and it carries a small but real throughput cost. The −0.24pp is within the noise floor (~0.2pp) plus the ~3-epoch penalty. Verdict: **no-improvement**.

### Informational Metrics

- peak_vram_mb: 511.8 (slightly above baseline ~467 — SiLU's σ(x) intermediate adds a small activation buffer; well within budget)
- num_epochs / num_steps: 88 / 34254 (vs baseline ~91 / ~35500 — mild ~3.5% throughput cost from SiLU not fully fusing)
- final_test_loss: 0.1960 (≈ baseline 0.195 — SiLU did NOT improve loss)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
