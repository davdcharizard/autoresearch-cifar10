# EXP-031: Throughput-optimized Gradient Centralization (compiled + hoisted)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-031.md
- **Plan**: plans/plan-031.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-031
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Re-implements EXP-030's Gradient Centralization with the execution optimized to remove the ~1ms/step host overhead that cost EXP-030 ~3 epochs. Three edits to `train.py` (Milestone 1): (1) module-level `_gradient_centralize(grads)` returns each weight grad centralized OUT-OF-PLACE (`g - g.mean(dim=fan_in, keepdim=True)`) + `_gc_compiled = torch.compile(_gradient_centralize)` (default mode); (2) hoist `gc_params = [p for p in model.parameters() if p.ndim > 1]` once before the training loop; (3) at the call site between `loss.backward()` and `optimizer.step()`, `centralized = _gc_compiled([p.grad for p in gc_params])` then reassign `p.grad = cg`. Identical GC math to EXP-030 — only the execution changed. Smoke test passed.

### Surprises & Discoveries
- Smoke test confirmed correctness AND a throughput win at the micro level: over 30 iters the compiled GC ran 8.4ms vs eager 10.1ms (ratio 0.83, ~17% faster); compiled output matches eager to 2.98e-8. Max |per-output-unit mean| after compiled GC = 7.57e-9 ≈ 0. 23 GC targets, 45 excluded 1-D params (BN γ/β + biases), params 4,299,866 unchanged.
- The micro-benchmark per-call delta (~0.06ms/iter) is smaller than EXP-030's steady-state ~1ms/step regression — because the isolated micro-bench has no concurrent CUDA-graph model replay to serialize against. The definitive throughput test is the realized epoch count in the full run (Milestone 3).

### Decisions
- **Out-of-place + reassign `p.grad`** (not in-place `g.sub_()`): avoids the in-place-mutation-under-torch.compile clone-writeback ambiguity (where Inductor could write to a clone, silently no-op'ing GC). Reassigning `p.grad` to a fresh tensor each step is the standard supported pattern; SGD reads `p.grad` at step time and momentum buffers are keyed on the param, not the grad identity. The smoke test verified the reassigned grads are actually centralized.
- **DEFAULT compile mode (not reduce-overhead)**: `optimizer.zero_grad()` defaults to `set_to_none=True`, so backward reallocates grad tensors each step → grad addresses change → CUDA-graph (reduce-overhead) static-address capture would be invalid. Default mode still fuses mean+subtract and issues from compiled code without per-op Python dispatch.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — background task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- Full 300s-compute-budget training of the k=4 WideResNet with throughput-optimized (compiled+hoisted) Gradient Centralization, single H20. Hypothesis: the compile+hoist fix restores epochs to ~91 / dt~8ms (vs EXP-030's 88/9ms), giving GC its throughput-neutral fair test; since GC tied baseline at a 3-epoch disadvantage, a fair test could lift best_test_acc toward/past the 96.32 bar. KEY CHECK: realized num_epochs and dt.

Observations:
- (Note: an earlier launch was killed externally by a user `/exit` after printing only the header — no training occurred; relaunched cleanly.)
- **THROUGHPUT FIX CONFIRMED EARLY**: dt back to **8ms steady** (vs EXP-030's 9ms; sampled 8-9ms across early steps), step/epoch pacing matches baseline's ~91-epoch trajectory. The compile+hoist optimization eliminated the GC loop's ~1ms/step host overhead. (source: run.log step lines through step ~3900 / ep 10 / 12.8%)
- Clean compile (model + GC fn), no NaN/Traceback, loss descending normally; trajectory tracked baseline throughout. (source: run.log eval lines)
- **THROUGHPUT FULLY RESTORED — the fix worked perfectly**: num_epochs **91** (= baseline; vs EXP-030's 88), dt **8ms steady** (641/703 sampled lines = 91% at 8ms, 61 at 9ms). The compile+hoist optimization completely eliminated the GC loop's host overhead. (source: run.log summary + step lines)
- **DECISIVE FINDING — at a FAIR throughput-neutral test, GC does NOT move top-1, only loss**: best_test_acc **96.14%** (−0.08pp vs baseline 96.22, within ±0.2pp noise = effectively tied/slightly below) while final_test_loss **0.1894 IMPROVED** (< baseline 0.195 AND < EXP-030's 0.1934). (source: run.log summary)
- This RESOLVES the EXP-030 ambiguity: the 3 lost epochs were NOT masking a top-1 gain. At 88 ep GC gave 96.21; at full 91 ep it gives 96.14 — both within noise of each other and of baseline. GC consistently improves LOSS but not ACCURACY → it is another convergence-POLISH lever (the polish-vs-top1 pattern: EMA/SWA/Bag-of-Tricks/LS-down).

Key Metrics:
- best_test_acc: **96.14%** (−0.08pp vs baseline 96.22; below the 96.32 bar) (source: run.log summary)
- final_test_loss: **0.1894** (BETTER than baseline 0.195 and EXP-030's 0.1934)
- num_epochs: 91 | num_steps: 35161 | num_params: 4,299,866 | peak_vram_mb: 469.8 | total_seconds: 405.1
- dt ≈ 8ms steady (throughput-neutral confirmed vs EXP-030's 9ms)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar**: FAIL. best_test_acc = 96.14% < 96.32 (baseline 96.22 + 0.1). Also below baseline itself (−0.08pp). Per plan, stop at first failure. (source: run.log summary)
- **Cond 2 — clean completion within budget**: PASS (informational). Summary printed, Traceback 0, total_seconds 405.1 < 600. (source: run.log)
- **Cond 3 — no constraint violations**: PASS (informational). git diff = train.py only; num_params 4,299,866 unchanged; 91 evals for 91 epochs (≤1/epoch); no new deps (torch + torch.compile, core); seed 42 unchanged. (source: git diff, run.log)

**MANDATORY attribution note (throughput-neutrality — THE crux of this experiment):** num_epochs **91**, dt **8ms** — IDENTICAL to baseline, and the throughput fix vs EXP-030 (88 ep / 9ms) WORKED COMPLETELY. So this IS the clean, throughput-neutral fair test of GC that EXP-030 could not deliver. Conclusion: GC's intrinsic per-epoch effect on this net is **loss-improving but top-1-neutral** — best_test_acc 96.14 sits within the ±0.2pp noise floor of baseline (and of EXP-030's 96.21), while loss improved 0.195→0.1894. Verdict: **no-improvement** (below bar), and now a DEFINITIVE characterization (not a near-miss): GC joins the convergence-polish cluster. The 96.21 of EXP-030 was the noise-favorable tail of the same null.

### Informational Metrics

- peak_vram_mb: 469.8 (≈ baseline; the out-of-place GC temporaries add ~16 MB, negligible)
- num_epochs / num_steps: 91 / 35161 (vs EXP-030's 88 / 33,953 — throughput fully recovered to baseline level)
- final_test_loss: 0.1894 (project-competitive low loss; BETTER than baseline 0.195, consistent with GC's landscape-regularization improving loss not top-1)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
