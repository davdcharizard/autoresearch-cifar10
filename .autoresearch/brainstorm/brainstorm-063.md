# Brainstorm EXP-063
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Status declaration (required by exp-report-062 Next Steps)

This brainstorm executed **option (a), deeper excavation via /lit-search** (venue-constrained: papers.nips.cc, openreview.net, proceedings.mlr.press, openaccess.thecvf.com) — it surfaced NO screen-passing external candidate (details below) — and then found its candidate via **option (b), a genuinely novel non-interpolation composition**: stream-parallel two-member ensemble training, which spends a measured resource (idle SM capacity at ~7% GPU utilization) that none of the 62 prior experiments ever spent. It is not a re-measurement: EXP-043 measured the sequential time-split ensemble endpoint and EXP-042 the in-one-kernel endpoint; the stream-parallel point buys diversity with a THIRD currency neither touched, and the standing High-importance ensemble entry explicitly licenses the attempt if the gain-vs-dilution inequality is shown BEFORE running — which a 90-second two-stream GPU probe does deterministically.

## Web Search & Literature Review

- **/lit-search venue sweep (NeurIPS/ICLR/ICML/CVF, 2024–2026 focus)**: no paper matching "≥0.3pp gains in heavy-aug, budget-matched regimes" surfaced. Notable hits, all non-candidates: "A unified LR schedule for budgeted-iteration training" (openreview.net — anneal-family member, axis closed at family level EXP-062); YOCO/You-Only-Cut-Once (arXiv 2201.12078 — aug-diversity add, dose+absorption-closed); "Data-Efficient Augmentation" (arXiv 2210.08363 — per-sample selection, EXP-050/051-closed).
- **Combined Image Data Augmentations diminish the benefits of Adaptive Label Smoothing** (arXiv 2507.16427, 2025)
  Independent external CORROBORATION of this project's absorption law (combined augmentations absorb regularizer roles — our EXP-035/036/037/046/060 series). Strengthens the measured-ceiling reading; contributes no candidate.
- **On the Variance of Neural Network Training** (arXiv 2304.01910, Jordan)
  Corroborates the run-level σ framing (EXP-027): CIFAR run variance is largely irreducible given the training distribution; ensembling is the known variance/level lever.
- **Throughput-geometry facts already in the record (the basis for the chosen idea)**: per-block dt ≈ 2.5ms is WIDTH-INDEPENDENT (EXP-034) and the charged step is 99.3% kernel time (EXP-048) — i.e., kernels are LATENCY-bound, not FLOPs-bound: ~232 GFLOPs/step ÷ 22.3ms ≈ 10.4 TFLOP/s achieved vs ~148 bf16 peak ⇒ **~7% utilization, >90% of SM capacity idle**. Two independent models' kernels on separate CUDA streams can co-schedule on the idle SMs; launch-path serialization from one Python thread is bounded by the 0.15ms/step non-kernel overhead (EXP-048). Deep-ensembles function-space gain for 2 members: +0.3–0.5, measured IN-PROJECT (EXP-043) — the gain is real; only its price was fatal.

## Experimental History Review

- Current best 96.71 @ 1990397; family mean 96.57, σ 0.16; bar 96.81 = mean+1.5σ. **56 consecutive closures**; every catalogued axis measured-closed (goal-learnings), schedule axis now closed at the FAMILY level (EXP-062), external transfer 0-for-18.
- **The ensemble dichotomy (the record this idea targets)**: weight-space averaging = zero gain (EXP-011/032); function-space 2-member gain REAL at +0.3–0.5 but bought with member steps at −0.9 (EXP-043, alternating full-batch steps → each member got half the steps); in-one-kernel grouped implementation hardware-closed at 2.8× dense (EXP-042). The standing implication (project-insights HIGH): "multi-model proposals must show their diversity gain exceeds the per-member compute dilution BEFORE running." No experiment ever tested CONCURRENT (stream-parallel) member training, because nothing in the record before EXP-034/040/044/048 established that the GPU is latency-bound with ~93% of its capacity idle at this model size.
- **What the conversion law says the inequality needs**: dilution ≈ 0.014/epoch × Δepochs (starvation ladder). Members step TOGETHER in one timed step of duration dt_both; member epochs = 300000/dt_both/97.65. At dt_both = 23.5ms → 131 ep → dilution ≈ −0.11; at 26ms → 118 ep → −0.29; at 30ms → 102 ep → −0.52. With gain ∈ [0.3, 0.5]: net clears the +0.3 screen only if **dt_both ≲ 23.5ms** (near-full overlap); it is sub-bar arithmetic at dt_both ≥ ~26ms. The probe decides which world we are in for ~90 GPU-seconds, before any charged run.
- Constraints this respects: VRAM is the goal's explicit soft slack (2× model ≈ 3.2GB of 98GB); MeanEnsemble eval contract validated EXP-043 (the evaluated module IS the trained system — not a TTA wrapper); eval thinning + 16 loader workers are validated wall levers (EXP-031/043); validation stays once per epoch.

## Candidate Ideas

### 1. Stream-parallel two-member ensemble (concurrency-funded diversity), gated by a two-stream dt probe
**Summary**: Train TWO independent 4x ResNet-20 members simultaneously inside each timed step, each member's forward+backward issued on its own CUDA stream so their latency-bound kernels co-schedule on the ~93%-idle SMs; one fetch of 1024 images split 512/512 (disjoint per-member data each step → decorrelation, 16 loader workers per EXP-031), independent optimizers, byte-identical recipe per member; eval once per epoch via the EXP-043-validated MeanEnsemble module (logit mean). GATE: a pre-launch GPU probe times the two-stream step; **launch only if P2 ≤ 23.5ms** (pre-registered from the conversion-law arithmetic: dilution ≤ −0.11 vs measured gain +0.3–0.5 → expected net ≥ +0.2–0.4). P2 ∈ (23.5, 26] → no launch, cost-closure with the interior point priced; P2 > 26 → no launch, the concurrency hypothesis is refuted at zero charged cost (EXP-040/042/044/045 precedent).

**Reasoning**: (1) The gain side of the inequality is MEASURED IN-PROJECT (+0.3–0.5, EXP-043) — no external-transfer risk, the only failed component was the price. (2) The price side rests on a measured fact no prior ensemble attempt knew: kernels run at ~7% of peak and per-block cost is width-independent (latency-bound), so a second member's kernels are candidates for near-free co-scheduling. (3) This is the only construction in the record whose screen-pass is DECIDABLE by probe rather than assumed: the High-importance entry's "show the inequality before running" is satisfied literally. (4) Spends the goal's explicitly-granted slack (VRAM/SM idle capacity) — the one resource 62 experiments never touched. (5) Composition of validated machinery only: MeanEnsemble eval contract (EXP-043), eval thinning + 16 workers (EXP-031), dual-model compile warmup (EXP-059 dynamic=False per-shape lesson does not apply — both members see one static 512 shape each; two separate compiled graphs).

**Sources**: EXP-043 (gain + eval contract), EXP-042 (in-kernel closure), EXP-034/040/044/048 (latency-bound throughput geometry), EXP-031 (loader/eval wall levers), knowledge/papers/deep-ensembles-function-space.md, project-insights § ensembling HIGH entry.

**Estimated Effort**: medium-high — two-model training loop + streams + dual compile warmup + MeanEnsemble eval + 1024-fetch split; the probe is ~40 lines and decides launch/no-launch before any charged run.

**Risk Assessment**: (a) Python-side launch serialization (GIL) could make P2 ≈ 2×P1 despite idle SMs — bounded by EXP-048's 0.15ms/step non-kernel measurement, but torch.compile's per-step host work for TWO graphs is unmeasured; the probe resolves this. (b) Streams + autograd + torch.compile interaction (backward kernels follow forward's stream; two graphs from one thread) — CPU sanity + probe verify mechanism engagement via the dt signature itself (P2 < 2×P1 IS the engagement proof). (c) Loader: 1024 img/step at ~23ms needs ~45k img/s — 16 workers validated at 38k+ (EXP-031); stalls land outside dt but inside the 600s cap; eval thinning available. (d) Worst research case: probe clears, run lands family-band → the EXP-043 dichotomy's last open interior point is measured and the multiplicity axis closes completely; graceful no-improvement.

### 2. Schedule-Free SGD lr bracket (lr 1.2)
**Summary**: One-run dose probe of EXP-062's open variant (paper guidance: schedule-free optimum lr often ≥ scheduled peak).

**Reasoning**: Would convert the family-level schedule closure into a 2-point bracket.

**Sources**: exp-report-062.md; knowledge/papers/schedule-free-road-less-scheduled.md.

**Estimated Effort**: low (constant change on the EXP-062 diff).

**Risk Assessment**: VIOLATES the record's own do-not-retry (goal-learnings EXP-062 entry: lr retunes priced — historical retune recovery ≤0.5 vs −1.84 deficit). Discarded.

### 3. Stage-1 kernel size 3×3 → 5×5 (receptive-field increase behind the dt gate)
**Summary**: Larger stem-stage kernels at lattice widths, dt-gated.

**Reasoning**: Kernel SIZE is the one kernel-family dimension never probed (only groups).

**Sources**: EXP-040/042/044/045 (gate protocol + off-lattice tier law).

**Estimated Effort**: low.

**Risk Assessment**: Fails arithmetic BEFORE the probe: 5×5 = 2.78× stage-1 conv FLOPs; even on a fast tier the step-time increase prices a starvation deficit larger than any plausible receptive-field gain (absorption gives the gain side a negative prior too). Discarded.

## Idea Evaluation

**Evidence strength**: Idea 1 is unique in the record: BOTH sides of its value equation are in-project measurements (gain +0.3–0.5 from EXP-043; price decidable by the same probe class that charted the kernel envelope in EXP-040–045). Ideas 2–3 are priced negative by existing entries before any run.

**Mechanism clarity**: Idea 1's mechanism is explicit and two-step: (i) latency-bound kernels co-schedule on idle SMs → member steps near-free; (ii) function-space logit-mean of decorrelated members raises accuracy by a measured +0.3–0.5. Failure of (i) is detected by the probe; failure of (ii) would contradict an in-project measurement.

**Expected impact**: At P2 ≤ 23.5ms: net +0.2–0.4 → plausibly ≥ 96.81. This is the only candidate generated since EXP-060 whose upside arithmetic reaches the bar without assuming an unmeasured transfer.

**Risk profile**: Safest possible: the launch decision is made uncharged; a cleared-probe run that lands family-band still closes the multiplicity axis completely (real information); no NaN/instability mechanism beyond standard infra.

**Feasibility**: The heaviest engineering of the three, but every component (MeanEnsemble eval, dual warmup, eval thinning, 16 workers, composite watchdog) is validated machinery from EXP-031/043/059.

**Honesty check (not a re-measurement)**: EXP-043 measured time-multiplexed members (each got HALF the steps — diversity bought with steps); EXP-042 measured space-multiplexed members in one kernel (2.8× dense — diversity bought with kernel efficiency). Idea 1 buys diversity with SM occupancy — a currency whose availability (~93% idle) was only established by the throughput-geometry measurements of EXP-034/040/044/048, all AFTER the ensemble closures. The standing HIGH entry does not forbid ensembles; it demands the inequality be shown before running. The probe shows it.

## Chosen Idea
**Selected**: Stream-parallel two-member ensemble (concurrency-funded diversity), gated by a two-stream dt probe

**Why this idea**:
It is the only construction available whose bar-clearing arithmetic uses ONLY in-project measurements (EXP-043's gain, the conversion law, the latency-bound throughput geometry), it spends the one explicitly-granted resource never touched (idle GPU capacity / VRAM slack), it complies literally with the High-importance ensemble entry's precondition (inequality shown BEFORE running, via a ~90s uncharged probe), and its worst case is cheap and informative (probe-refusal closes the concurrency hypothesis at zero charged cost; a family-band run completes the multiplicity axis).

**Hypothesis**:
If the two-stream probe prices the joint two-member step at P2 ≤ 23.5ms (near-full kernel overlap on the idle SMs), then training two independent members concurrently and evaluating their logit mean will land best_test_acc at members' level (≈ mean − 0.11 dilution) + function-space gain (+0.3–0.5) ≥ 96.81 (branch i). Pre-registered alternates: (ii) P2 > 23.5 at gate → no launch, concurrency refuted/cost-closed at zero charged cost (verdict invalid/NaN per gate-kill precedent); (iii) probe clears but the run reads family band [96.41, 96.73] → the decorrelation gain shrinks at converged member level — multiplicity axis closed completely; (iv) < 96.41 → stream interference degrades member trajectories (numerics/contention mechanism, diagnosed by member-level eval lines); (v) infra → relaunch ≤ 2.
