# Report EXP-063: Stream-parallel two-member ensemble — probe-gated NO LAUNCH (concurrency cost-closure)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-063.md
- **Plan**: plans/plan-063.md
- **Log**: logs/exp-log-063.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged training budget by modernizing train.py only. Current baseline: 96.71 @ 1990397 (EXP-006 recipe); success bar = 96.81 (baseline + 0.1, operationalized as family mean 96.57 + 1.5σ with the EXP-052 replicate-pair-mean protocol near the bar).

## Idea & Hypothesis

After 56 consecutive closures across every conventional axis, brainstorm-063 identified the one resource the record showed as never-spent: the GPU runs this model at ~7% compute utilization (232 GFLOPs/step ÷ 22.3ms ≈ 10.4 TFLOP/s vs ~148 bf16 peak), with latency-bound, width-independent kernels (EXP-034, EXP-048). The ensemble dichotomy from prior loops: function-space diversity gain is real (+0.3–0.5, EXP-043) but every funding source tried — steps (dilution −0.9) and kernel width (2.8× dt, EXP-042) — cost more than it paid. **Hypothesis**: training two independent members concurrently on two CUDA streams funds the diversity from idle SM capacity at near-zero step-time cost (P2 ≈ P1), and the logit-mean ensemble clears the bar. The project's HIGH-importance ensemble entry requires the gain-vs-dilution inequality be shown BEFORE running — so an uncharged two-stream probe was pre-registered as the launch gate with criterion: LAUNCH iff [load < 30 AND P2 ≤ 23.5ms] OR [load ∈ [30,40) AND P2/P1 ≤ 1.08 AND P1 ≤ 24.5].

## Approach

train.py was fully implemented per plan-063 M1 (and passed 10/10 CPU sanity checks): two 4x ResNet-20 members constructed sequentially under the fixed seed (distinct inits), separately torch.compiled, each with the byte-identical baseline recipe (512/member batch, nesterov SGD, selective WD, shared time-keyed cosine lr_at); timed step = H2D 1024 → split 512/512 → event-ordered fwd+bwd on two CUDA streams → join → both optimizer steps → synchronize; MeanEnsemble (logit mean) evaluated via the untouched Eval.evaluate every 4th loop-epoch. Charged-time semantics preserved exactly. The launch decision belonged to /tmp/exp063_gpu_probe.py: gate GPU 0 (apps==0, load<40), warm both members exactly as train.py does, time 40 single-member steps (P1), 40 joint two-stream steps (P2), one ensemble eval pass (E).

## Execution

Single uncharged probe run at the cleanest possible conditions (GPU-0 apps=0, load1=10.1 — the strictest branch of the criterion). Results: P1 = 22.48ms (inside the family band 22.0–22.8 → probe environment verifiably clean), **P2 = 40.70ms, ratio 1.810**, E = 4.26s, VRAM 1,795MB. Criterion requires P2 ≤ 23.5: failed by 73%, not marginal → **NO LAUNCH**. One additional uncharged attribution diagnostic (eager-mode two-stream, /tmp/exp063_eager_diag.py): eager ratio 1.820 vs compiled 1.810 — identical. train.py was never launched; zero charged seconds were spent; no run.log was ever created.

## Results

- **Primary metric**: NaN (no charged run; baseline 96.71 unchanged)
- **Observations**: The two streams serialize almost perfectly (joint ≈ 2× single, both eager and compiled). The 0.19 shortfall from exactly 2× is consistent with H2D/epilogue overlap only. Warmup for two compiles was just 9.7s. Memory was never a constraint (1,795MB at the probe's 1024-batch peak).
- **Analysis**: The hypothesis rested on reading "7% utilization" as idle capacity. The probe falsifies that reading: for this model's small, latency-bound kernels, the binding resource is the **serial kernel-dispatch/latency chain** — one process, one enqueue path, and a GPU that retires each tiny kernel roughly as fast as it is issued. A second stream therefore queues behind the same chain instead of running beside it. Compute idleness is real but unreachable: idle SMs are not spare *dispatch* capacity. This retro-explains EXP-034 (per-block dt width-independent — the chain is latency, not arithmetic) and EXP-048 (99.3% "kernel time" — the chain is dense, leaving no gaps a second stream could fill from the dispatch side either). The third and last candidate funding source for ensemble diversity (concurrency, after steps EXP-043 and width EXP-042) is closed; with it, the multiplicity axis is closed at the funding level: the function-space gain is real but no currency on this stack can buy it for less than it pays.
- **Key Learning**: "Idle GPU capacity" must be qualified by which resource is binding — for latency-bound small-kernel models the serial dispatch chain, shared by all streams in a process, is the bottleneck, and stream-level concurrency buys nothing.

## Verification

- **Conditions**: not reached — the pre-registered M2 launch criterion (which gates everything) failed; Conditions 1–3 skipped per plan branch (ii). Integrity of the gate decision itself: verified (probe at apps=0/load 10.1 with P1 inside the family band rules out load-inflation per the EXP-062 lesson; criterion applied verbatim).
- **Review Notes**: Results trustworthy — the decisive numbers were measured twice through independent paths (compiled probe, eager diagnostic) and agree to 1%. The NO-LAUNCH outcome is the plan's designed primary falsification point, not an infrastructure failure.
- **Verdict**: invalid
- **Verdict Basis**: Pre-registered branch (ii): probe NO-LAUNCH → `invalid` with metric NaN — a cost-closure (no charged run, so neither improvement nor no-improvement applies), following the EXP-040/042/044/045 gate-kill precedent.

## Unexplored Avenues

- **Multi-process concurrency (CUDA MPS / two processes)**: would bypass the single-process dispatch chain, but violates the single-run timer semantics (two processes cannot share one charged t0→dt accounting honestly) and is out of scope — recorded as do-not-retry in the exp-log.
- **CUDA Graphs to compress the dispatch chain** (`torch.compile(mode="reduce-overhead")`): if dispatch latency, not kernel execution, dominates the 22.4ms step, graph capture could shrink P1 itself — a *step-time* lever, not an ensemble lever. Two caveats: EXP-048's 99.3%-kernel-time measurement suggests gaps are small, and dynamic LR + cross-stream patterns complicate capture. Worth a probe-first look only if a brainstorm shows the family dt band can plausibly drop ≥1ms (≈ +6 epochs).
- **Within-step parallelism via batched members** (e.g., a single fused module computing both members as grouped convolutions): EXP-042 already measured grouped-conv ensembles at 2.8× dt — closed.

## Next Steps

1. **CUDA-Graphs step-time probe** (medium confidence): uncharged probe of `mode="reduce-overhead"` single-member dt vs the 22.0–22.8 family band; if dt drops ≥1ms the freed epochs are general-purpose budget. Risks: cudagraph + per-step LR writes + BN state interplay; EXP-048 bounds the upside.
- 2. **Re-read the measured-ceiling hypothesis adversarially** (medium confidence): with every axis (heat, noise, loss, numerics, structure, order, lattice, averaging, reg-dose/type, allocation, schedule-family, multiplicity-funding) measured closed, the next brainstorm should either find a falsifier for the ceiling or argue a genuinely novel composition — per the standing autopilot directive to think harder rather than stop.
3. **Literature excavation round** (low confidence): 0-for-18 external transfer is strong absorption evidence, but each excavation has still produced clean closures; target ideas whose mechanism is *budget-creating* (like eval thinning was) rather than *pressure-adding*.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
