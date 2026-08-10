# Review — EXP-017 Plan (Full-Run Eligible-Weight Gradient Centralization)

## 1. Centralize-before-weight-decay breaks the property the experiment claims to test, and the audit cannot detect it
**Section:** Code Changes / transformation ("Weight decay remains uncentralized inside SGD"); Code Changes / audits.

The plan centralizes the *raw loss gradient*, then hands it to unchanged PyTorch SGD, which internally computes `d_p = grad + wd*p` before momentum. Since the weight rows are not themselves zero-mean, `wd*p` reinjects a per-row mean component into the actual update and into the momentum buffer. The applied step is therefore **not** a projection onto the zero-row-mean hyperplane — which is the entire mechanism GC's ECCV claim rests on. The reference SGD_GC implementation centralizes `d_p` *inside* the step, after the decay term is added; this plan's ordering is a silent deviation from the cited source.

This compounds with the audit design: all FP64 energy accounting is on pre-subtraction *raw* gradients, so the audit will report clean centralization (residual ≤1e-6, orthogonal decomposition) while the quantity that actually moves the weights retains a nonzero row mean of magnitude `wd * mean(p)`. A null result would then be attributed to "BN redundancy" (the preregistered ≤1% reading) when the real cause could be that GC was partially undone every step. Either match the reference ordering, or preregister this as a deliberate variant *and* add a diagnostic on the post-decay `d_p` row mean (or on the momentum buffer) so the interpretation section is falsifiable. Note Milestone 1's smoke test only checks "centralize-before-decay Nesterov ordering" — it verifies the chosen ordering is implemented, not that the ordering preserves the mechanism.

## 2. AMP/GradScaler interaction is unspecified, and the nonfinite abort can kill the single metric run legitimately
**Section:** Code Changes / transformation and audits; Abort Criteria ("nonfinite loss/state/audit").

The plan never states the gradient dtype or whether the parent uses `torch.autocast` + `GradScaler`. Two concrete failure modes:
- If a `GradScaler` is in use, gradients at the point of centralization are **scaled** and may legitimately contain inf/NaN on steps the scaler is designed to skip. The plan's "zero nonfinite count" summary assertion and the "nonfinite audit → terminate" abort criterion would then fire on a healthy run, and the protocol explicitly forbids a metric retry. Centralization must run after `scaler.unscale_`, or the nonfinite policy must be rewritten to tolerate scaler-skipped steps.
- "compute each FP32 mean" directly contradicts "No ... casting ... is introduced" if grads are ever fp16/bf16. Resolve which is true and state it.

There is also no statement about `torch.compile`/CUDA-graph capture. If the parent compiles the step region, inserting `_foreach_sub_` between backward and `step()` may trigger recompilation or a graph break that shows up as startup or charged-time cost the preflight harness (which builds models differently) will not reproduce.

## 3. `num_steps >= 27000` / `num_epochs >= 138` as *failure* conditions can discard a genuine improvement
**Section:** Abort Criteria (final bullet) and Verification Procedure step 5.

The goal's necessary conditions are: ≥0.10pp over parent, no crash, respect the fixed time budget, complete summary. The plan adds step/epoch floors and rules that a completed run with 26,900 steps is "`no-improvement`, never permission for a metric retry" — so a run producing, say, 95.60 with slightly fewer steps would be thrown away despite clearly satisfying the goal. Under a *fixed wall-clock* budget, fewer steps is the expected, honest consequence of per-step overhead, not an integrity failure. These should be reported as context (overhead accounting), not as verdict-determining gates. Also note the internal inconsistency: a 1.03 latency ratio (the preflight pass bound) against the parent's 27,950 steps projects ~27,136 steps, leaving almost no margin against the 27,000 floor — the two gates are near-degenerate.

## 4. The single-`_foreach_sub_` premise is likely wrong, and the plan has no fallback
**Section:** Configuration Changes / "Subtraction implementation"; Code Changes / transformation.

`torch._foreach_*` fast-path kernels require matching shapes; with 17 heterogeneous grad/mean pairs and broadcasting, the op almost certainly dispatches to the slow path, which is a per-tensor `sub_` loop. Bitwise parity with the loop reference will therefore *pass trivially* while delivering none of the launch-count reduction the plan cites as its overhead mitigation ("reduced by exact foreach subtraction rather than by weakening eligibility"). Meanwhile the 17 unbatched `mean` reductions — the larger launch cost — are unbatched regardless. The plan should (a) verify in preflight whether the fast path is actually taken (e.g. compare against an explicit loop under the profiler, not just bitwise output equality), and (b) name a preregistered fallback (e.g. flattened single-buffer reduction) rather than leaving "abort the leaf on ratio >1.03" as the only response to overhead.

## 5. `best_test_acc` selection semantics are never asserted unchanged, while new eval-adjacent bookkeeping is added
**Section:** Code Changes / summary ("final-16 evaluation mean/range/final/best premium"); Milestone 1.

The goal forbids gains from "evaluation changes." The plan adds final-16 evaluation tracking and a "best premium" statistic but never states that the definition of `best_test_acc` (max over per-epoch evaluations), the evaluation cadence, the eval transform, and the eval batch construction are byte-identical to a36dc09. Given the plan's own diff-scope discipline everywhere else, add an explicit requirement: the only diff hunks touching the eval/reporting path are additive prints of already-computed per-epoch accuracies. As written, "final-16 ... best premium" is diagnostics sitting directly on top of the primary metric — the one place scope creep is most costly.

## 6. Diagnostic cost accounting relative to the charged budget is unstated
**Section:** Code Changes / audits; Verification Procedure step 5 (`training_seconds` in [299.5, 301.0]).

Two things need pinning down explicitly, because both are silent ways to make the candidate look better than the parent at equal budget:
- The ~55 audit steps (FP64 accumulations, residual max, split conv/classifier energy) must be **inside** the charged timer, not excluded as "diagnostics."
- The abort criteria include "nonfinite loss/state" during the run. If that is implemented as a per-step `.item()`/host check, it adds a synchronization to all ~27k steps that the paired preflight harness may not model. Restrict nonfinite checking to audit steps and after training, and say so.

## 7. Interpretation bands and the 95.53 "mechanism bar" are not falsifiable from one fixed-seed run
**Section:** Configuration Changes / Interpretation; Verification Procedure step 6.

- The preregistered energy readings cover ≤1% (BN redundancy) and ≥5% (substantial removed signal). The 1–5% interval — the most likely outcome — has **no** preregistered reading, so any value there can be narrated either way post hoc. Define the middle band or drop the dichotomy.
- 95.53 is 0.30pp over the parent from a single seed. CIFAR-10 run-to-run spread at this scale is typically comparable, and the protocol (correctly, per the goal) forbids repeats. Licensing a downstream GC-on-EXP-011 branch off a single sample crossing 95.53 should be stated as a weak, noise-limited signal in `03-execute.md`, not as evidence that "supports the mechanism strongly."
- Similarly, "Report 95.53 mechanism support, 95.61 global-best match, and 95.71 resolution-clearing global improvement separately" (Milestone 4) creates three post-hoc bars against one number; ensure the write-up leads with the single formal verdict (≥95.33 vs. not) and marks the rest as commentary.

## 8. The 240-second preflight budget is tight, and a timeout burns the one allowed repair
**Section:** Verification Procedure step 4.

Inside 240s the harness must: build the CIFAR pipeline, construct and warm up two models, run parity-through-backward checks on clean and CutMix batches, run 1,024 candidate production-order steps, and run 5 × 2 × 120 = 1,200 paired steps — roughly 2,200+ optimizer steps plus double startup. At the parent's ~93 steps/s that is ~24s of stepping but potentially far more in startup/cuDNN warmup/data load, and the plan grants only **one** harness repair for a timeout. Either raise the timeout, or split the structural smoke (step 3 scope) from the timing rounds so a timeout in the expensive half does not consume the repair allowance for the whole gate.

## 9. Minor: assert constants need a documented reconciliation
**Section:** Code Changes / inventory (2,745,264 elements, 2,266 rows, 17 tensors) vs. Verification step 5 (`num_params=2,748,890`).

The 3,626-parameter gap (BN affine + biases) is never written down. Record the decomposition in the summary print so that an assertion failure is immediately diagnosable as "model changed" versus "inventory selector changed," rather than an opaque hard stop before the single metric launch.
