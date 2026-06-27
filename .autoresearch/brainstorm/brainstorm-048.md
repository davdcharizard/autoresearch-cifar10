# Brainstorm EXP-048
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources from model knowledge + project record, recorded for downstream re-reading:

- **PyTorch performance guidance (model knowledge of official docs/recipes)**: standard charged-step de-overheading levers that do NOT change arithmetic: (1) produce the final tensor layout in the DataLoader's collate (runs in uncharged worker processes, like the rest of the aug pipeline) so no layout-conversion kernel runs in the timed step; (2) overlap H2D copies with compute via a side-stream prefetcher (copy of batch N+1 issued during batch N's compute). Both deliver byte-identical tensors to byte-identical kernels in the same order — numerics-identical by construction, unlike EXP-021's max-autotune/fused-SGD arithmetic changes.
- **In-project anchor (the strongest available)**: throughput→epochs is the project's ONLY repeatedly-certified positive mechanism — EXP-000 (bf16/TF32/channels_last/batch) and EXP-006 (torch.compile, +25 epochs → +0.48; conversion ≈ 0.019/epoch at unchanged hyperparameters and distribution). exp-report-047 § Next Steps explicitly directs this brainstorm here: "the per-step overhead OUTSIDE conv kernels was never itemized."
- **Arithmetic context**: ~1.0 TFLOP/step (ResNet-20 4x, batch 512, fwd+bwd) at 22.4ms ≈ 45 TFLOPS achieved vs ~148 peak bf16 on H20 — consistent with conv-bound small images; plausibly 1.5–3ms of the 22.4 is copy/layout/launch overhead rather than kernel math. The in-step charged overhead visible in code: `inputs.to(device)` (6.3MB H2D, ~0.3ms, on the compute stream = serial) followed by `.to(channels_last)` (an extra full-tensor permutation kernel every step).

## Experimental History Review

State after 48 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16; 41 consecutive non-improvements/invalids. Post-EXP-047 frontier:

- **ALL structural classes are closed** (exp-report-047): absorbed-null (046 anti-aliasing), active-negative (047 routing −2.6σ, 030 head pooling −0.91), cost-priced (020/037/040–045). GAP(stage3)→fc head measured load-bearing. External transfer 0-for-14 including toll-free.
- **Recipe constants audit-complete**; schedule/heat/noise/batch/optimizer-arithmetic/init/aug all closed both directions. Numerics-equivalence law: EXP-021's faster-but-DIFFERENT arithmetic (max-autotune, cudagraphs, fused SGD bundle) converged lower — speed gains must not perturb update math.
- **The one open seam**: the charged step's non-kernel overhead. The baseline recipe itself was built by exactly this mechanism (EXP-000/006), and no experiment ever itemized or removed the remaining in-step copy/layout overhead. dt-gate protocol (D0 median, 26ms off-rung) doubles as the measurement instrument: D0 itself reports the saving.
- Protocol carry-overs: dual launch gates, D0 gate, ≥200-step windows, replicate-pair band 96.70–96.80, integrity pre-condition; conversion law 0.019/ep prices any dt saving (−1ms ≈ +6 ep ≈ +0.12).

## Candidate Ideas

### 1. Numerics-identical charged-step de-overheading: collate-side channels_last + side-stream H2D prefetch
**Summary**: (a) Custom `collate_fn` that calls the default collate then returns `x.contiguous(memory_format=torch.channels_last)` — the layout permutation moves into the uncharged DataLoader workers (where ToTensor/Normalize/TA already run), and the in-step `.to(memory_format=channels_last)` is dropped; (b) a small CUDA side-stream prefetcher that issues batch N+1's H2D copy during batch N's compute, so the copy leaves the critical path. The timed region's code/timer semantics are untouched: `torch.cuda.synchronize()` still fences ALL device work (including any in-flight prefetch) inside the charged window — no GPU work escapes charging; it is only overlapped, and CPU layout work joins the existing uncharged aug pipeline exactly like Normalize does today.

**Reasoning**: This is the only mechanism class with repeated in-project POSITIVE evidence (EXP-000, EXP-006: throughput → epochs → accuracy at unchanged hyperparameters), and the one seam never itemized. Unlike EXP-021 (count-1 failed approach: max-autotune + cudagraphs + FUSED SGD = different arithmetic), this is numerics-identical by construction — identical fp32 values in identical layout reach identical compiled kernels in identical order; RNG streams untouched; update math untouched. Honest dose arithmetic: expected saving 0.4–1.0ms/step (copy ~0.3ms + permutation kernel ~0.1–0.4ms + serialization slack) → +3–6 epochs → +0.06–0.12 expected true shift — BELOW the +0.3 detectability bar on its own. The experiment is therefore framed as a measurement: D0 itemizes the overhead at ~2 GPU-minutes; the full run prices the conversion. Value either way: if D0 barely moves, the de-overhead class closes with "the baseline step was already overhead-free" — the last open seam audited; if D0 drops ≥1ms the conversion law gets a clean in-regime test at small dose, and the changes are certified-mechanism components that could compound with nothing else left.

**Sources**: project-insights (EXP-006/EXP-021 entries; dt-pricing law); goal-learnings § Patterns (EXP-006 conversion); reports/exp-report-047.md § Next Steps; train.py L215–219 (in-step copy + layout conversion), L150–158 (DataLoader), L236 (synchronize fence).

**Estimated Effort**: low-medium (collate function + ~25-line prefetcher class + loop wiring; careful CPU sanity on layout/value identity).

**Risk Assessment**: (a) GATE pass with tiny saving → graceful mean-band null, class closed; (b) prefetcher bug (stale/skipped batch) → caught by CPU sanity (sequence-identity test) and the divergence guard; (c) channels_last pinned-memory interaction with persistent workers — stock PyTorch, low risk; (d) compile graph unchanged (input layout identical). No destabilizing mode; worst case −0 epochs and a null.

### 2. CUDA-graphs-only step replay (torch.compile mode="reduce-overhead") — runner-up
**Summary**: Re-run the compile with `mode="reduce-overhead"` (cudagraph replay of identical kernels), WITHOUT max-autotune and WITHOUT fused SGD — isolating the numerics-identical component of EXP-021's bundle to eliminate per-step kernel-launch overhead.

**Reasoning (and why not the lead)**: Kernel replay is numerics-identical in principle, and launch overhead at ~hundreds of launches/step could be ~0.5–1ms. But: EXP-021 is the adjacent count-1 failure and attribution within its bundle is uncertain; cudagraphs constrain the step (static input buffers — the dynamic per-step LR assignment is fine since the optimizer is eager, but input tensors get an extra static-buffer copy that eats part of the gain); eval shares weights with the graphed module (base_model eager — interaction risk with cudagraph weak refs); and a recompile/fallback failure mode burns a run. Idea 1 collects the cheaper, safer part of the same budget first and its D0 measurement tells us whether anything is left for graphs to claim.

**Sources**: goal-learnings EXP-021 entry; project-insights numerics-equivalence law; torch.compile docs (model knowledge).

**Estimated Effort**: low code, medium risk.

**Risk Assessment**: Moderate — graph capture failures or hidden numerics drift (memory-pool reuse) would cost a full run; gate covers only the slow direction.

### 3. drop_last=False (use all 50,000 images per epoch) — documented rejection
**Summary**: Stop discarding the final 336-sample partial batch each epoch.

**Reasoning (and why rejected)**: The partial batch changes the gradient-noise scale at every epoch boundary (336 vs 512), and the batch/noise axis is closed BOTH directions (EXP-012/022/023/024: the recipe sits at a measured gradient-noise optimum; any nonzero noise-sign change loses). Coverage gain is negligible (336/50,000 = 0.7% of one epoch; shuffling already rotates which images are dropped). Epoch-boundary law also warns the boundary is where signatures are most sensitive.

**Sources**: goal-learnings batch-scaling + momentum-trade entries; project-insights gradient-noise-optimum entry.

**Estimated Effort**: trivial.

**Risk Assessment**: Expected small negative via noise perturbation; no class-closing value. Rejected.

## Idea Evaluation

- **Evidence strength**: Idea 1 rests on the project's only repeatedly-POSITIVE mechanism (EXP-000/006 conversions) and is numerics-identical by construction, dodging the EXP-021 failure mode explicitly. Idea 2 isolates a component of a measured failure with uncertain attribution. Idea 3 contradicts a closed-both-directions axis.
- **Mechanism clarity**: Idea 1's is exact and priced: remove K ms of non-math work from the charged window → +K/0.0224 × 139/300… → ≈ +6 epochs per ms → +0.12 per ms by the conversion law. The D0 instrument reports K directly.
- **Expected impact**: honestly low-medium (likely +0.06–0.12 true, below one-draw detectability) — but every alternative is expected-zero-or-negative by measured law, and this one doubles as the audit that closes the last open seam.
- **Risk profile**: Idea 1 is the safest construction available (no arithmetic change, no schedule interaction, sanity-testable byte-identity, standard PyTorch patterns); failure modes all land as clean nulls or pre-launch sanity catches.
- **Feasibility**: moderate code with strong local verifiability (value/layout/sequence identity assertable on CPU and the gate measures the saving in 2 minutes).

Idea 1 dominates. Ideas 2/3 recorded to pre-empt re-derivation (Idea 2 remains available if Idea 1's D0 shows large residual overhead).

## Chosen Idea
**Selected**: Idea 1 — Numerics-identical charged-step de-overheading (collate-side channels_last + side-stream H2D prefetch)

**Why this idea**:
With every structural, regularization, schedule, and routing class measured-closed, the charged step's non-kernel overhead is the last unaudited seam — and throughput→epochs is the only mechanism with repeated positive in-project conversions (EXP-000/006). This construction collects that budget while staying byte-identical in arithmetic (the certified recipe's update math untouched), explicitly avoiding EXP-021's faster-but-different failure. The D0 gate doubles as the measurement instrument, so even the null outcome closes the de-overhead class with a number.

**Hypothesis**:
Moving the layout permutation to the uncharged worker pipeline and overlapping H2D copies with compute removes ~0.4–1.0ms from the charged step (D0 ≈ 21.4–22.0ms vs 22.4 family), converting to ~+3–6 epochs at unchanged arithmetic and, per the EXP-006 conversion law, a small true mean shift. Pre-registered branches: (i) best_test_acc ≥ 96.81 → improvement (replicate pair if 96.70–96.80); (ii) D0 saving < 0.3ms AND mean-band read (96.42–96.72) → the baseline step was already overhead-free; de-overhead class closed; (iii) D0 saving ≥ 0.5ms with mean-band read → epochs delivered but the dose is below detectability, consistent with the conversion law — class closed with the conversion datum; (iv) GATE_KILL (D0 > 26ms) or early loss trajectory off-family → implementation defect, verdict invalid/crash per evidence, fix-or-stop per retry rules.
