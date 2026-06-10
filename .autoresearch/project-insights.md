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
- **"Regularization saturated" is mechanism-specific — increasing augmentation DIVERSITY is the only lever that lifts top-1 here, confirmed 3× then MAPPED to its frontier** (EXP-012 +0.22, EXP-052 +0.12, EXP-054 +0.11; frontier mapped EXP-053/055; corrects EXP-005, EXP-011)
  Evidence: exp-report-012 — after WD (EXP-005) and mild Mixup (EXP-011) read as null ("saturated"), TrivialAugment lifted 96.00→96.22. exp-report-052 — AugMix(w2,d1) REPLACING TA lifted 96.22→96.34. exp-report-054 — `RandomApply([AugMix() w3], p=0.5)` (full 3-chain on ~50%) lifted 96.34→96.45. Each was the ONLY gain in a long no-improvement streak; weak/redundant regularizers saturate, more-DIVERSE augmentation need not. The lever is chain COUNT/distinctness — magnitude is null (EXP-053).
  Implication: do NOT declare the augmentation axis "closed" from weak-variant nulls — test the strongest, most diverse variant before concluding (3× the plateau-breaker). **UPDATE (EXP-055): the CPU-feasible frontier is now FULLY MAPPED at w3 / p=0.5 = 96.45. The three sub-axes are all closed: magnitude (EXP-053 null), width>3 (EXP-055 −0.44pp regression to 96.01), coverage<50% (EXP-055). Do NOT retry w4+/w5 or p<0.5.** The genuinely-strongest variant (full-coverage rich chains) is wall-INFEASIBLE on CPU at 8 workers (EXP-052: full AugMix w3 uniform → >600s). **EXP-056 VALIDATED the GPU-side throughput unlock (move aug into the train loop on the idle GPU): it is cheap (+1ms dt, 84 ep, dt-bound wall ~390s) and affords full coverage — BUT a naive full-coverage STACK of all ops (no clean-mix) is far too harsh → 95.39 (−1.06pp).** So the path is open but the GPU policy must BOUND the per-image shift like the proven CPU augs: AugMix-style clean-image convex mixing, TA-style single-op-per-image, and/or stochastic p<1 coverage. **UPDATE (EXP-057): the clean-mix policy fix WAS applied — faithful GPU AugMix (multi-chain Dirichlet + Beta clean-mix) at full coverage — and STILL regressed to 95.64 (−0.81pp), though the clean-mix softened it +0.25pp vs EXP-056's harsh stack. The binding constraint was COVERAGE, not policy: even shift-bounded AugMix over-regularizes at 100% coverage. With EXP-055 (coverage<50% hurts) this brackets ~50% as a TRUE interior optimum, not a CPU-wall artifact.** The augmentation axis is now FULLY EXHAUSTED — magnitude (EXP-053), width>3 (EXP-055), coverage<50% (EXP-055), coverage=100% (EXP-057), naive-harsh-GPU (EXP-056) all ✗; only GPU-AugMix-on-50%-subset remains. **UPDATE (EXP-059): tested GPU-AugMix W3@50% → 95.57, ≈ GPU 100% (EXP-057 95.64), both ~0.85pp under CPU 50% (96.45). DECISIVE: CPU augmentation is FREE w.r.t. the Σdt/epoch budget (runs in parallel dataloader workers, OFF the timed step) — that is why CPU AugMix gets w3/50% AND 91 epochs. Moving identical aug onto the GPU puts it INSIDE the timed loop → ~20 fewer epochs → underfit. The augmentation lever is now CLOSED from BOTH delivery paths: CPU (free-epochs, wall-limited to the w3/p=0.5=96.45 optimum) and GPU (unlimited-coverage, epoch-limited to ~95.6).** STRATEGIC RULE: on a Σdt-budgeted goal, keep augmentation on the CPU dataloader (free w.r.t. the budget); only move it to the GPU for aug that CPU workers genuinely cannot deliver within the wall — and even then the epoch cost usually sinks it. The next gain must come from a NEW lever (NOT augmentation either path, NOT EMA/SWA/schedule/capacity/normalization/optimizer/head — all closed); 96.45 appears at/near the k=4/300s ceiling. Generalizes to any fixed-compute-budget training-quality goal.
- **Massive VRAM headroom — memory-trading is essentially free, but capacity scaling is NOT free in wall-clock** (EXP-000, EXP-001; refined by EXP-004, EXP-009)
  Evidence: exp-report-000 peak 164 MB; exp-report-001 a 15.8× wider model peaked at only 490 MB of 98 GB. BUT
  exp-report-004 (k=6) and -009 (k=5) both regressed: wider nets turn compute-bound → too few epochs to converge.
  Implication: caching/precompute/larger batches are unconstrained by memory; capacity (width/depth) is bounded by
  a monotone wall-clock EPOCH WALL once the net is compute-bound — find the capacity sweet spot, don't max it.
- **At a fixed wall-clock budget, any COMPUTE- or SEQUENTIAL-LAYER-adding change hits an epoch wall that torch.compile cannot lift** (EXP-004, EXP-009, EXP-015, EXP-024, EXP-036, EXP-038, EXP-044, EXP-058)
  Evidence: exp-report-009 — compiled k=5 (18ms/step) fit only 41 epochs (vs k=4's 77) → under-fit 94.21. EXP-004
  eager k=6 → 35 ep → 95.26. EXP-015 pre-act graph → 78 ep. EXP-036 sparse SAM (2 fwd-bwd on every 5th step, even cudagraph-replayed) → mean dt 8→10.2ms → 76 ep → 95.89 (−0.33pp). EXP-024 BlurPool (anti-aliased downsampling, params
  UNCHANGED but conv1 moved to stride-1 → ~4× FLOPs at the 2 heaviest convs) → dt 8→9.5ms → 77 ep → under-fit 95.66.
  EXP-044 deeper-narrower ResNet-32 (k=3, 15 blocks vs 9) at ≈iso-FLOP (97.8%) STILL rose dt 8→12ms (+50%) → 60 ep → severe under-fit 92.58 (−3.64pp): MORE SEQUENTIAL LAYERS cost dt even at constant FLOPs (lower per-layer arithmetic intensity → memory-bound), the steepest wall yet.
  EXP-066 progressive resolution (24×24 early, 32×32 tail) — a change MEANT to REDUCE compute — STILL raised dt: a 2nd input shape under reduce-overhead spawns a 2nd CUDA-graph, inflating the 32×32-tail dt 8→10ms and cancelling the 24×24 saving → 89 ep < 91 (−0.63pp). Even a FLOP-REDUCING change is net-negative if it perturbs the single compiled graph.
  Implication: it is NOT just FLOPs — ANY change that adds non-trivial FLOPs (width) OR sequential layers (depth) OR
  restructures the graph (pre-act, anti-aliasing, OR a 2nd input shape/CUDA-graph) costs epochs and regresses via under-training at a fixed budget,
  regardless of the change's intrinsic merit (the merit gets masked/confounded). The CAPACITY axis is now closed from
  ALL FOUR directions — width (EXP-004/009), FLOP-neutral width-realloc (EXP-038), deeper-narrower (EXP-044), and shallower-WIDER (EXP-058: ResNet-14 k=5 iso-param STILL rose dt 8→12ms — identical to EXP-044 — proving **the memory-bandwidth wall is set by channel WIDTH, not block-count**, so reducing depth buys NO dt headroom for width). k=4 {64,128,256} ResNet-20 is definitively the compute-optimal frontier. Do NOT retry ANY width/depth/block-count/realloc variant. Only pursue
  compute-NEUTRAL, iso-LAYER-count changes while launch-bound; always verify realized epoch count before attributing a delta.

### Medium Importance
- **Compute-neutral convergence-POLISH levers improve test LOSS/calibration but NOT top-1 once the recipe is at its capacity ceiling** (EXP-006, EXP-019, EXP-020, EXP-023, EXP-026, EXP-030, EXP-031, EXP-039, EXP-041, EXP-064; generalizes the SWA entry)
  Evidence: every compute-neutral "polish" move lowered loss or flatness without lifting top-1 — EMA/SWA (loss 0.18 << 0.195 but top-1 ≤ baseline, EXP-006/019/020), LS-down (CE loss 0.156 < 0.195, top-1 −0.19pp, EXP-023), Bag-of-Tricks zero-γ + no-bias-decay (loss 0.195→0.190, top-1 −0.04pp, EXP-026), Gradient Centralization at a FAIR throughput-neutral test (loss 0.1894 < 0.195, top-1 96.14 within noise, EXP-031 — the EXP-030 96.21 "near-miss" was the noise-favorable tail of this null), and most strikingly **PolyLoss Poly-1 (EXP-041): eval CE loss CRASHED to 0.158 (project-lowest, ~29% below baseline 0.195) yet top-1 96.11 (−0.11pp within noise)** — the objective-confidence lever maxes calibration, not accuracy. **EXP-064 (gradient-norm clip max_norm=2.0): final_test_loss 0.1939 < EXP-054 0.1968, top-1 96.34 (−0.11pp within noise) — identical signature, closing the last gradient-side knob.** reports/exp-report-064.md, -041.md, -031.md, -026.md, -020.md, -023.md.
  Implication: for a fixed-capacity TOP-1 goal whose recipe is already converged+well-tuned, do NOT spend loops on convergence-polish (weight-averaging, init tricks, no-bias-decay, LS retuning, gradient-centralization/standardization, OBJECTIVE/LOSS-SHAPE reshaping incl. PolyLoss/cosine-geometry). This now spans WEIGHT-AVERAGING, INIT/WD, OPTIMIZER/GRADIENT-DYNAMICS, OBJECTIVE/LOSS-SHAPE, and — as of EXP-043 — the OPTIMIZER FAMILY itself (AdamW lr2e-3/wd0.05 regressed −0.35pp at a fair stable 8ms/91ep run; the adaptive generalization gap, NOT under-training): no optimizer change (family or grad/objective mod) moves top-1 here. The polish pattern is axis-independent, and a lower test LOSS is a near-certain false-positive signal for top-1 here. Top-1 gains require capacity or fundamentally different generalization, not optimization/objective polish — and capacity (k both ways + realloc), augmentation, schedule, classifier-head, and intermediate-feature-routing are ALL closed too, so the 96.22 plateau is the robust k=4/300s ceiling (33 straight no-improvements).
- **The "launch-bound" regime is BATCH-DEPENDENT: bigger batches go compute-bound, and under a compute-time-gated budget that COLLAPSES optimizer updates → regression** (EXP-025; refines the launch-bound claims in the torch.compile + epoch-wall entries)
  Evidence: exp-report-025 — at batch 128 k=4 is launch-bound (~8ms/step); doubling to batch 256 raised dt to ~15ms steady / ~24-28ms warmup (mean ~21.5ms ≈ 2.7×) = compute-bound. The 300s budget gates on Σ(per-step compute dt) (train.py L242, timer starts after the dataloader yields), so #steps ≈ 300/mean(dt): batch 256 collapsed updates 61% (35.5k→14k), epochs 91→72, acc 93.84 (−2.38pp, loss↑ 0.195→0.258).
  Implication: for any compute-`dt`-gated fixed-budget goal, do NOT raise batch size to "buy epochs" unless you first confirm dt stays flat — the launch-bound headroom is consumed by batch ~128 here. The budget rewards MORE updates at smaller effective batch; larger batches reduce both images and updates. Verify the dt-vs-batch curve before assuming launch-bound at a new batch size.
- **ImageNet/deep-net-proven tricks do NOT reliably transfer to the shallow / small-image (32×32), already-well-tuned CIFAR regime — validate the transfer, don't assume it** (EXP-024, EXP-027, EXP-028; also EXP-026 zero-γ)
  Evidence: anti-aliased/information-preserving downsampling (+0.3-0.5% on ImageNet R50) REGRESSED from both sides — BlurPool (EXP-024, compute-confounded) AND compute-neutral ResNet-D (EXP-027, fair 89-ep: 95.75 −0.47pp, loss WORSE). Smooth activation SiLU/Swish (EXP-028) NULLED — flat loss, −0.24pp, AND cost ~1ms/step (didn't fuse). Zero-init residual γ (EXP-026) nulled — benefit is depth-driven. Common thread: each trick's mechanism scales with DEPTH / IMAGE-SIZE / hard-to-optimize-ness (multi-stage downsampling, dead-ReLU avoidance, deep signal-propagation init), none of which bind on a shallow 9-block 32×32 net that already trains cleanly with BN+warmup+a tuned recipe. reports/exp-report-028.md, -027.md, -024.md.
  Implication: before importing a trick proven on deep/large-image nets, ask whether its mechanism still applies at THIS scale; budget it as a genuine experiment, not a free win. Corollary (EXP-028): the launch-bound "pointwise ops fuse to ~free" assumption is NOT guaranteed — SiLU's σ(x) did not fuse and cost real dt; verify dt empirically. The downsampling/anti-aliasing AND activation axes are now CLOSED here.
- **At a short fixed budget, ADDING regularizers fails — the recipe is convergence-bound, not overfit-bound; gains come from convergence-neutral changes** (EXP-005, EXP-011, EXP-018, EXP-022, EXP-047; refines the EXP-012 High entry)
  Evidence: once the recipe was well-regularized (TA+Cutout+LS+WD), every ADD-a-regularizer move regressed or nulled —
  WD↑ (EXP-005 null), Mixup (EXP-011 null), CutMix (EXP-018 −1.08pp), in-block dropout (EXP-022 −1.37pp, loss
  0.195→0.224 under-fit), and Ghost BatchNorm normalization-noise (EXP-047 −1.06pp, loss 0.195→0.220, slower
  convergence — a noise regularizer is just another penalty here, and GhostBN's gain is large-batch-specific so it
  doesn't bind at batch 128). The lone gain (TrivialAugment, EXP-012) SUBSTITUTED/diversified input aug without adding a
  convergence-slowing penalty. reports/exp-report-022.md, -018.md, -047.md.
  Implication: for any fixed-short-budget training-quality goal, don't keep stacking regularizers once converged-and-
  regularized — added penalties (incl. normalization noise) cost epochs/convergence the budget can't spare. Prefer
  convergence-NEUTRAL levers (aug diversity/substitution, input normalization, schedule shape) or REDUCING a
  regularizer; reserve more regularization for when the epoch budget grows. (On this CIFAR goal every accuracy axis —
  incl. normalization, EXP-047 — is now closed; the 96.22 ceiling is fully mapped.)
- **A per-PARAMETER Python for-loop running each step (eager, outside the compiled region) costs measurable dt via kernel-launch overhead — `torch.compile` of that loop fully removes it** (EXP-030, EXP-031 confirms the fix)
  Evidence: exp-report-030 — Gradient Centralization as a Python loop over 23 conv/fc grad tensors (≈46 tiny kernel launches/step) raised dt 8→9ms → epochs 91→88. EXP-031 fixed it: hoist the weight-param list once + wrap the centralization in `torch.compile` (DEFAULT mode; out-of-place + reassign `p.grad` to dodge in-place-mutation-under-compile clone ambiguity) → **dt back to 8ms / 91 ep = baseline** (fix confirmed). NOTE: reduce-overhead/CUDA-graph is INVALID here because `zero_grad(set_to_none=True)` reallocates grads each step (addresses change). reports/exp-report-031.md, -030.md.
  Implication: for any per-step gradient/param op under a compute-`dt`-gated budget, do NOT iterate `model.parameters()` in eager Python — hoist the param list out of the loop and `torch.compile` the op (default mode, out-of-place writeback). A throughput-confounded near-miss is a strong signal to re-test throughput-neutral before concluding — though in this case the fair test (EXP-031) revealed the masked effect was loss-only, not a top-1 gain. Mirrors the EXP-028 "pointwise ops don't fuse free" corollary; here the cost is host-side launch count, removed by compiling.
- **Forcing EARLY/mid-level features to satisfy a classifier — by EITHER input-concatenation into the head (EXP-032) OR auxiliary deep-supervision loss (EXP-042) — disrupts the tuned coarse-to-fine hierarchy and regresses; the WHOLE intermediate-feature-routing family is closed** (EXP-032, EXP-042)
  Evidence: exp-report-032 — a multi-scale head (concat global-avg-pooled layer2[128]+layer3[256] → fc) gave the classifier a direct path to mid-level 16×16 features + a direct gradient path to layer2. Convergence collapsed early (ep1 19.3% vs 55.4%), never recovered → 94.72 (−1.50pp) AND loss WORSE 0.195→0.231. exp-report-042 — the auxiliary-decayed-loss form once flagged as the "safer" variant (aux layer2 classifier, λ 0.3→0, main-head-only eval, throughput-NEUTRAL 8ms/90ep) ALSO regressed −0.31pp (95.91) AND loss (0.2026); milder only because λ→0 and the main forward path was untouched.
  Implication: on this well-tuned SHALLOW 9-block net the learned coarse-to-fine hierarchy is load-bearing, and ANY gradient pressure pulling mid-level features toward premature class-discriminability hurts — input-concat hard (−1.5pp), aux-loss mild (−0.31pp). Deep supervision's literature benefit is depth-driven (eases signal propagation in 20–100+ layer nets); it does not transfer to a net that already trains cleanly with BN+warmup (cf. zero-init-γ EXP-026). The intermediate-feature-routing axis is now CLOSED from both sub-levers; combined with closed capacity/optimizer-dynamics/aug/schedule/objective/weight-averaging/classifier-head axes, the k=4/300s net is firmly generalization-bound at fixed capacity. The optimizer FAMILY (AdamW vs SGD) is the last major untested axis.
- **bf16 autocast is safe and beneficial on H20 (no GradScaler needed)** (EXP-000)
  Evidence: reports/exp-report-000.md § Execution — bf16 ran stably, ~6-7ms/step vs fp32 ~8.6ms (+21% throughput), no NaN.
  Implication: default to bf16 autocast + channels_last for any compute-bound training on this hardware.
- **`torch.compile(mode="reduce-overhead")` gives ~30% throughput on small LAUNCH-BOUND nets; benefit shrinks to near-zero once compute-bound** (EXP-007, EXP-009)
  Evidence: exp-report-007 — launch-bound k=4 dt 10–11ms→8ms, 77→89 epochs (default mode net-negative, 1.03× for
  ~13.6s cost). exp-report-009 — compute-bound k=5 only reached 18ms (2.25× compiled-k4 for 1.56× FLOPs), i.e. the
  CUDA-graph launch-overhead win barely helped. Keep eval on the eager handle to avoid recompiles.
  Implication: reduce-overhead compile is a real lever ONLY while the net is launch-bound; it does NOT re-open
  capacity trade-offs for compute-bound model sizes (EXP-009 disproved that EXP-007 hope). Use it to add cheap
  per-step ops on k=4, not to afford bigger models.
  EXP-040 corollary: once reduce-overhead is on, the CONV DT FLOOR is already reached — `torch.backends.cudnn.benchmark
  = True` is a throughput NO-OP (dt stayed 8ms; it only swapped in equal-speed HIGHER-memory conv algos, peak_vram
  491→971 MB). So cheap throughput flags can't buy net-new epochs on top of compile; only an aggressive kernel
  re-autotune (max-autotune) or moving work off the timed GPU path could cut dt.
  EXP-045/046 RESOLUTION (now fully de-confounded): EXP-045 — max-autotune + an off-budget compile-warmup added real
  epochs for the first time (91→96) yet best_test_acc FELL to 95.71 (−0.51pp), converged flat tail. EXP-046 — the clean
  reduce-overhead control (SAME warmup, baseline kernels, dt steady 8ms, ep1 45.7% normal) landed 96.20 ≈ baseline. So
  the "throughput→buy epochs" axis is CLOSED and the TA recipe is epoch-SATURATED at ~91 kernel-INDEPENDENTLY: baseline-
  kernel +epochs = baseline acc, so EXP-045's −0.51pp was purely the max-autotune KERNEL-NUMERICS penalty, not the epochs.
  CONFIRMED sub-finding: throughput-optimal conv kernels (max-autotune Triton EXP-045, cudnn.benchmark EXP-040) BOTH land
  ~0.3-0.5pp below the reduce-overhead baseline at ≥baseline epochs → reduce-overhead kernels are near-accuracy-optimal;
  faster kernels trade a hair of top-1. Keep mode="reduce-overhead". FURTHER: reduce-overhead's own compile is only ~4.4s
  (EXP-046 startup 2.1→6.5s), so an off-budget warmup buys at most ~+1 epoch at baseline kernels — the warmup reclaim is
  only large (~+5 ep) for heavier compiles (default/max-autotune), and those heavier kernels cost more top-1 than the epochs gain.

- **SWA/weight-averaging needs a terminal-LR floor to engage; it improves loss/flatness more than top-1 and only approaches a tuned cosine-to-0 from below** (EXP-006, EXP-019, EXP-020)
  Evidence: exp-report-006 — EMA on cosine-to-0 = no-op (settled tail). SWA floor sweep: 0.05→95.97 (exp-019),
  0.02→96.13 (exp-020), both project-lowest loss ~0.18 < 0.195, both < the 96.22 cosine-to-0 baseline. As SWA_LR→0
  the constant tail degenerates into cosine-to-0, so SWA's supremum over the floor IS the baseline (approached from below).
  Implication: for any fixed-budget TOP-1 goal, weight averaging (SWA/EMA/Lookahead) will not beat a well-tuned
  cosine-to-0 schedule — it maxes out on calibration/loss, not accuracy. Worth it only if the metric is loss/ECE,
  or if cosine-to-0 is NOT already tuned. Don't sink more runs into floor/start-frac/cyclic variants once the
  monotone-from-below trend is established.
- **FLOPs-neutral architecture changes are NOT necessarily wall-clock-neutral under torch.compile — down to the sub-ms-OP scale** (EXP-015, EXP-038, EXP-039)
  Evidence: exp-report-015 — pre-activation block reorder (same FLOPs as post-act baseline) fit only 78 epochs vs
  EXP-012's 91 at the same 300s budget; the restructured block + extra final BN→ReLU yielded a less-efficient
  compiled graph, silently costing ~14% of the step budget and confounding the accuracy comparison. EXP-038 is a
  STRONGER case: a fat-head width reallocation {64,128,256}→{48,128,304} (≈−0.9% FLOPs by w²·area) raised CLEAN
  uncontended dt 8→10.5ms (+31%) — the wider 304-ch stage3 is MEMORY-BANDWIDTH-bound, not FLOP-bound — → 73 ep →
  under-trained 95.47 (−0.75pp). (Also: channel counts must be multiples of 8/16 or cuDNN drops off the tensor-core
  path — {44,...} ran ~38ms.) So "FLOP-neutral capacity reallocation" CANNOT dodge the epoch wall.
  Implication: for any fixed-wall-clock goal, never assume "same/fewer FLOPs ⇒ fair same-budget test" — wall-clock
  tracks MEMORY-BOUND execution time, not FLOPs. Always verify realized epoch/step COUNT against baseline AND that
  the GPU was uncontended (see infra-errors: shared-node contention also inflates dt). This closes the CAPACITY axis
  from both directions: uniform widening (EXP-004/009, FLOP add) and FLOP-neutral reallocation (EXP-038, wall-clock
  premium) — k=4 {64,128,256} is the compute-optimal frontier; stop probing capacity.
  EXP-039 extends this to the SMALLEST scale: a cosine/normalized-softmax head added only two `F.normalize`
  + one `F.linear` on tiny pooled tensors (params −10, "obviously compute-neutral") yet shifted enough steps
  8→9ms to cut 91→83 ep (mild under-train, loss 0.195→0.210) → 95.89 (−0.33pp). So even sub-millisecond
  per-step ops are NOT wall-clock-free here — assume nothing is free; verify realized epoch count every time.

### Low Importance
