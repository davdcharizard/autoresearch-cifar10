# Goal Learnings: Maximize CIFAR-10 Test Accuracy

**Goal**: goals/maximize-cifar10-test-accuracy.md
**Experiment Index**: experiment-indices/maximize-cifar10-test-accuracy.tsv

<!-- Per-goal qualitative distillation, agent-appended during the analyze phase.
     Three categories below. Each bullet follows the
     3-line, budget-strict format:

       - **{statement — ≤150 chars}** ({source refs})
         Evidence: {1-2 lines, MUST cite a source path — report / log / JSON / URL}
         Implication: {1-2 lines}

     HARD CEILING: if a bullet's total length exceeds ~3-4 sentences after composition, REWRITE it
     shorter. Push any detail that does not fit into the experiment report and cite the report
     path from Evidence. Do NOT commit an over-budget bullet. The Failed Approaches section
     uses Mechanism/Insight in place of Evidence/Implication — same budget shape. -->

## Protocol Findings

<!-- Measurement, verification, or benchmarking discoveries specific to this goal's metric.
     Examples:
       - "Stage-1 measurements are contaminated by CUDA autotune warmup — measure warm."
       - "Stored baselines expire same-day; require a same-day pristine reference run."

     Good format (fits budget length, cites report):
       - "Baseline pLDDT drifts ~0.3/day; require same-day pristine reference" (EXP-006)
         Evidence: reports/exp-report-006.md § Verification Results — 8-case drift table
         Implication: capture pristine baseline same-day as experiment measurement. -->

### High Importance

- **RUN-LEVEL σ̂ = 0.123 at n=5 (updated EXP-067): baseline draws {96.71, 96.59, 96.40, 96.53, 96.44}, pooled mean 96.534 — the recorded 96.71 is the distribution TOP (+1.4σ̂), so the +0.1 bar (96.81) sits at mean + 2.2σ̂; NO DRIFT detected months in** (EXP-027, EXP-067)
  Evidence: reports/exp-report-027.md § Results (first pair, byte-perfect signatures); exp-report-067.md § Results — second pristine pair (96.53/96.44) in-band with exact family ledgers (13,455/13,461 steps), pooled n=5 σ̂ 0.123 vs 0.16 at n=3; level ledger re-anchored 2026-06-11
  Implication: candidates need TRUE effects ≥ +0.28 ≈ 2.2σ̂ to clear the bar on a single draw (effect-size screen ≥ +0.3 stands); results within ±0.15 of the recorded baseline are noise-band ("no detectable effect", NOT measured loss); structural laws rest on deficits ≥0.2–0.3 and replicated designs; standing decision arithmetic: mean 96.534, σ̂ 0.123, band [96.29, 96.78].

- **Per-epoch eval overhead rivals the 300s training budget — 345 evals ≈ 295s; run finished 596.7s of the 600s cap** (EXP-000, EXP-031)
  Evidence: reports/exp-report-000.md § Results — total_seconds 596.7 with 345 epochs at ~0.85s/eval; exp-report-031.md — Run 1 WALL_CAP_KILL at 601s/81% charged (~1.3s uncharged eval/epoch + ~120s loader stalls: 8 workers cannot feed 13.5ms steps); Run 2 fixed via phase-1 eval thinning (every 3rd epoch — once/epoch is a CEILING, not a floor) + 16 workers → 457s at 185 epochs
  Implication: budget any high-epoch idea as 300 + startup + epochs×~1.3s + stalls ≤ 600 (epochs ≲ 200 hard max); eval thinning in non-plateau phases and 2× loader workers are validated wall levers (bought 144s in EXP-031).

### Medium Importance

- **A pre-registered replicate-pair (n=2, MEAN decision) correctly DECLINED a bar-clearing single draw: 96.84 then byte-identical 96.56, mean 96.70 sub-bar** (EXP-052)
  Evidence: reports/exp-report-052.md § Results — both runs pristine (D0 22.5/22.0ms, 138 ep each); spread 0.28 ≈ 1.2σ of a pair difference; mean protocol 1.6% false-positive under H0 vs single-run 6.7%
  Implication: a single ≥-bar draw near the bar is ~weekly-expected under H0 at this σ — for any near-bar result on a re-tested or low-prior candidate, require a replicate mean before accepting a new baseline; the max of a pair is never a decision input. (pct prints 1 decimal = 0.3s; 50 steps → ms ∈ {18.0, 24.0, 30.0, …}) — slow-window thresholds set between rungs misfire on clean runs** (EXP-037)
  Evidence: logs/exp-log-037.md § Experimental Adjustments — two clean-load runs at true dt 24.2ms showed 5 then 9 windows reading exactly 30.0ms (>29 threshold) while ~400-step watchdog windows never exceeded 25.5ms; one rerun spent refuting a phantom contamination
  Implication: set fine-profile slow thresholds OFF the rungs (e.g., 31ms not 27–30) or profile on ≥200-step windows; "all exceedances are exactly one rung value" is the quantization signature — check it before classifying contamination.

- **Early dt gate (kill at step ~100 if projected epochs < ~65) works: 38ms measured vs 36–40ms predicted, 83 epochs vs 80–85 — zero loop waste** (EXP-008, EXP-026, EXP-040, EXP-042, EXP-044, EXP-045)
  Evidence: reports/exp-report-008.md § Execution; exp-report-026.md — inline watchdog variant (kill at 3 consecutive windows >23.5ms within first 7 ticks) screened GELU and SiLU out in ~90s each, then passed hardswish for a full run; exp-report-040.md — D0-median variant (first 3 windows, GATE_KILL >36ms, contention D0×1.25 after) screened 5x and 4.5x width in ~90s each and discovered the 256-channel kernel cliff; exp-report-042.md — same D0-median variant (threshold 28ms) screened grouped convs at 63.0ms in ~90s, discovering the 2.8× grouped-kernel penalty
  Implication: standard for any run with uncertain epoch count AND now the standard screen for architecture-class changes — set the kill threshold at the dt where projected epochs make the bar arithmetically implausible; plateau-at-end = converged (architecture problem), climbing-at-end = starved (throughput problem); pre-register the kill branch's verdict so a screened-out idea costs minutes, not a loop. Kernel-class changes (grouped/depthwise/strided) AND any non-64-multiple width MUST be gated — dense pricing laws hold only on the kernel lattice (EXP-044: 160ch cost 4× the law's prediction).

- **The step-count ledger is a finer throughput instrument than windowed dt — but single-pair step deltas < 1% are themselves within scatter** (EXP-048, EXP-053)
  Evidence: logs/exp-log-048.md — num_steps 13,515 vs EXP-046's 13,428 read a ~0.15ms/step saving that D0/windows could not resolve; exp-report-053.md — the same code change then delivered 13,428/13,434 on two pristine replicates: the +87-step saving did NOT reproduce. Single-epoch eval reads (ep1) are also not a reliable tripwire — judge numerics by trajectory + plateau + family test_loss.
  Implication: for throughput claims below ~0.5ms, compare num_steps between pristine 300.0s runs — but require REPLICATION before banking a <1% step delta as a component; never set single-read integrity bands without characterizing scatter first.

- **External contention produces silently-plausible runs — exit 0, sane metric, just 36% fewer epochs; printed dt misses it (samples 2% of steps)** (EXP-011)
  Evidence: logs/exp-log-011.md § Run 1–2 — 89/105 epochs vs 139 clean; window-avg dt from pct_done deltas showed 24/48ms alternation; foreign GPU-0 PID confirmed live (Run 1)
  Implication: before trusting any metric, sanity-check num_epochs against the clean projection (~139 @ 22ms); diagnose via pct_done deltas between step prints; rerun contaminated runs — never analyze them as research results.

- **Mechanism engagement must be verified by its physical signature (dt drop), never by flip/marker prints — and a ~90s GPU probe before launch prices any graph change for free; time the BASELINE net in the same probe session as an internal control (P_norm = 22.4×P/B) to self-validate against load inflation** (EXP-055, EXP-056, EXP-063, EXP-064)
  Evidence: logs/exp-log-055.md § Experimental Adjustments — Run 1's requires_grad freeze printed its FREEZE marker yet saved nothing (compile no-op); the probe then measured engaged vs disengaged dt (22.04 → 15.15ms) before Run 2; exp-log-056.md — the probe caught the pre-act reorder's surprise +1.0ms fusion toll pre-launch, letting bands be revised and the read toll-adjusted; exp-log-063.md — a pre-registered probe LAUNCH CRITERION (not just a price reading) terminated the experiment at zero charged cost when the two-stream overlap hypothesis read 1.81× instead of ~1.0×
  Implication: GPU-probe (compile + warm + time 40 steps) EVERY graph change before committing a full run — it both validates mechanism engagement and prices unexpected dt shifts; treat in-run prints as necessary, never sufficient. Probe-to-run D0 offset ≈ +0.5–1.0ms at low probe load — but at probe load ~33 the offset INVERTED (EXP-062: probe 23.28 vs run 22.27): probe inflation grows smoothly with host load well below the <40 gate, so when probe load > ~30 treat the FAMILY ledger as the binding integrity band and fast-side band deviations as benign (contamination only ever yields FEWER steps).

### Low Importance

- **Steady-state idle-loop policy (measured-ceiling regime; explicit-bound backlog EMPTY as of EXP-070)** (EXP-067, EXP-068, EXP-069, EXP-070)
  Evidence: reports/exp-report-067.md § Next Steps (protocol adoption); exp-report-068/069/070 (three zero-GPU closure loops; enumeration completed EXP-069; backlog emptied EXP-070 — infra angle + data addition were the last nominations)
  Implication: sweep on schedule (periodic double-screened lit sweep, next due ~2026-06-17, then weekly-scale — now INCLUDING the infra modality, whose closure is availability-dependent unlike the arithmetic ones), close on nomination (pre-run inequality / by-inspection bound at zero charged cost), replicate only when a near-bar decision actually pends; never burn charged GPU on candidates failing the screens, and never manufacture runs to look busy. Reusable: any throughput proposal must first beat the EXP-070 inequality (bar-clearing needs ~11% numerics-IDENTICAL speedup ≈ −2.4ms of 22.4ms).

## Failed Approaches

<!-- Approaches tried for this goal's metric that did not work, with a root-cause mechanism.
     Promotion rule: count = 1 → Low; count ≥ 2 → Medium; count ≥ 3 → High. Within each tier,
     sort by count descending (most-hit at top). On promotion to High, add a
     `> **Recurring failure:** ...` callout at the top of this section if not already present.

     Entry format:

       - **{Approach name / root mechanism}** (count: N, EXP-NNN, EXP-MMM)
         Mechanism: {1 line — why the approach fails}
         Insight: {1-2 lines — what alternative approaches might still work} -->

### High Importance

> **Recurring failure:** uniform width increases under the fixed time budget — failed 4x (8x eager, 5x unaligned, 6x aligned+compiled, and EXP-040's gate-killed 4.5x/5x). Closed at the HARDWARE level: dt cliff above 256 channels (22.4→54ms). Do NOT retry any width with >256 channels anywhere in the net.

- **Capacity increase under the fixed budget — closed in EVERY currency incl. instrument availability: fast kernels exist only at power-of-2 widths {64,128,256}** (count: 6, EXP-002, EXP-005, EXP-007, EXP-040, EXP-044, EXP-045)
  Mechanism: width trades against epochs (8x→40ep −0.82, 5x→52 −1.11, 6x→55 −0.71); >256 = flat 54ms cliff (EXP-040); off-power-of-2 widths = flat ~33ms tier regardless of FLOPs (EXP-044: 160→32.4ms at +18%; EXP-045: 192→33.4ms at +40%, refuting the "64-multiple" hypothesis since 192 = 3×64); on-lattice increases (64/256/256 ≈34ms, 128/128/256 ≈36ms) are starvation-priced before gating.
  Insight: 64/128/256 is the UNIQUE hardware-fast increasing triple — optimal by exhaustion. The capacity-at-converged-epochs question is permanently unmeasurable on this hardware (no fast instrument exists). Do NOT re-derive width candidates; any future shape change must use only lattice widths and still be dt-gated.

### Medium Importance

- **The LOSS AXIS is fully closed — per-sample treatments lose in BOTH directions around plain mean CE+LS, and the logit-scale optimum holds via the OPTIMIZER path too (fc-WD axis bracketed both sides)** (count: 4, EXP-050, EXP-051, EXP-057, EXP-058; joins target-distribution closures EXP-009/036 and EXP-011/032)
  Mechanism: uniform logit-margin up-pressure (m=0.75, EXP-050) → test_loss IMPROVED 0.185→0.1505 yet accuracy fell to 96.19 = mean−2.4σ (converged-depressed plateau). Confidence-keyed down-weighting (detached w=p_true^0.7, GCE geometry, EXP-051) → test_loss DEGRADED to 0.239 and accuracy collapsed to 95.32 = mean−7.8σ with a still-climbing tail (undertrained: the weight built a permanent anti-curriculum). Opposite CE signatures, both accuracy-negative — plain mean CE+LS sits between two measured cliffs.
  Insight: (1) test-CE and test-accuracy DECOUPLE HARD — CE-improving interventions are at best neutral (011/032) and at worst negative (050); treat test_loss as a non-proxy and give loss-motivated candidates (distillation, calibration, margins, robust losses) a measured negative prior. (2) EXP-050's "destroyed-view amplification" reading is REFUTED by 051: low-p views under heavy aug are hard-but-GENUINE load-bearing signal (suppressing them is 3× worse than over-pushing them); 050's damage was generic gap-inflation over-pressure. (3) Do not retry: band-pass weights (interpolate both cliffs), focal/anti-GCE (mirror of 050), GCE doses (anti-curriculum at any q>0 by mechanism). The loss is at a measured local optimum in target-distribution, logit-geometry, AND per-sample-weighting dimensions. (4) EXP-057/058 complete the optimizer-side bracket: fc.weight WD 0 → 96.36 (−1.3σ), 5e-4 → family mean, 2e-3 → 96.24 (−2.1σ) — a measured interior maximum AT the default, both deviations showing the same accuracy-down/CE-flat decoupling from opposite mechanisms (over-confidence vs norm starvation). Logit-scale interventions are closed via loss-up, loss-down, optimizer-relief, AND optimizer-tightening pathways; per-layer WD coverage is COMPLETE and the fc axis is bracketed — do not retry interior doses (bounded sub-resolution by the BN-momentum bracketing logic) or fc-LR multipliers (both directions lose by proxy plus head-lag entanglement).

- **BN running-stat momentum deviations from the 0.1 default in EITHER direction (0.02 smoother EXP-038; 0.25 fresher EXP-039)** (count: 2, EXP-038, EXP-039)
  Mechanism: the estimator's two error terms cross AT the default — m=0.02 is lag-dominated (−0.30 mean, 4× plateau scatter, hot-phase evals collapse to 35 vs family ~64 because the cosine tail keeps drifting weights); m=0.25 is variance-dominated (mean noise-band 96.449/best 96.64, 3× scatter, hot phase family-equal — a 10-batch EMA already tracks even hot-phase drift, so there was no residual lag to harvest).
  Insight: BN-constants law fully measured: constants must come from the augmented training stream (EXP-029) AND use the default freshness — m=0.1 is the two-sided optimum of the lag/variance trade. Interior doses closed by bracketing (both endpoints ≤0); phase-varying momentum refuted in both phases. The eval-constants axis has no remaining corner, and this was the program's last in-regime measured slope.

- **Weight averaging for eval — EMA across the anneal AND canonical SWA basin-tail with correct BN re-estimation** (count: 2, EXP-011, EXP-032)
  Mechanism: both share the improved-loss/no-acc-gain signature. EMA (decay 0.995, live BN buffers) lost −0.25; SWA (cosine frozen at 85% → lr 0.030, ~21 equal-weight snapshots, augmented-loader update_bn before every tail eval) landed AT the baseline mean (96.60, test_loss 0.1756 vs family ~0.185). Fixing the EXP-011 BN flaw (re-est at n=1 measured +0.71) removes damage but adds no level: the time-keyed cosine anneal already performs the equivalent iterate averaging implicitly, and under fixed wall clock the SWA phase is CARVED OUT of the anneal, not added after it.
  Insight: the accuracy ceiling is decision-boundary-limited, not confidence/calibration-limited — solution-smoothing moves logit geometry, not argmaxes. Axis closed from both ends; paper SWA gains presuppose a fixed-epoch budget with an appended SWA phase (external transfer 0-for-12).

- **Batch scaling 512→1024 under any LR rule OR any SCHEDULE — level bracketed (EXP-012/022) and the late-switch schedule (Smith et al. tail noise-halving) reads family-band too** (count: 3, EXP-012, EXP-022, EXP-059)
  Mechanism: throughput delivers identically every time (level: +12 ep at dt ~41ms; schedule: ~6% per-image tail dividend, +2 equiv epochs, 142 ep/11,933 steps) but the metric never moves — level loses at both canonical LR points (linear 0.8 → 96.66, √ 0.566 → 96.57); the late 512→1024 step at p≥0.75 with LR unchanged read 96.51 = mean−0.4σ: tail noise reduction is REDUNDANT with the cosine's own anneal.
  Insight: noise axis closed in BOTH level and schedule — the explicit LR anneal already saturates late-phase sharpening, so no batch trajectory (constant, ramped, stepped up OR down) has a separate mechanism; brainstorm-059 pre-registered that the multi-step ramp and the 512→256 noise-up tail inherit this null. Batch 512 constant is THE optimum. Durable instrument: any multi-shape compiled run MUST use `torch.compile(dynamic=False)` + per-shape warmup (see infra-errors EXP-059).

- **Raising integrated LR heat on the time-keyed schedule (peak 0.4→0.6; warmup 0.15→0.08)** (count: 2, EXP-010, EXP-014)
  Mechanism: with progress = elapsed/budget, EVERY shape change that starts the anneal earlier or peaks higher raises lr(p) at every instant; the extra heat defers mid-schedule progress the fixed-length tail cannot repay — dose-response: peak +50% → −0.57pp; warmup-halving → −0.22pp.
  Insight: "shorter warmup = less wasted ramp" is a fixed-iteration intuition that INVERTS under time-keyed schedules (the anneal completes regardless, so nothing is freed). Hot side closed twice; cold side now measured too (EXP-049: peak 0.3 → mean-band flat) — the heat axis is closed in BOTH directions.

### Low Importance

- **Training-set composition (label-noise curation, subset selection/importance sampling, near-duplicate handling) — the last residual-space member, closed at zero charged cost** (count: 1, EXP-069; completes the out-of-recipe enumeration)
  Mechanism: cleaning gains are added-noise artifacts (the +0.9pp confident-learning figure lives at 20–40% synthetic noise; natural rate ~0.54% ⇒ ≤0.1pp ceiling under LS+TA+RE, arXiv 1911.00068/2103.14749); selection has a budget-matched published NULL (arXiv 2110.14283) and weighting is the measured anti-curriculum (EXP-051/065); test-blind duplicate removal strictly deletes information about the pinned eval distribution.
  Insight: regime-check published gains on NOISE RATE as well as augmentation/budget. With this, every member of the residual space ("data composition/order, objective shaping, architecture") carries a closure citation — candidate inflow now requires new double-screened external evidence or an anchor revision. Do not re-propose cleaning/coreset/duplicate variants; data ADDITION inherits cost-landing + pretrained-knowledge closures (bound it explicitly if ever nominated).

- **Sequential self-distillation, resolution-up training, and adaptive/sign optimizer swaps — the last "radical" classes, closed by measured-anchor inequalities at zero charged cost** (count: 1, EXP-068; EXP-063/064 NO-LAUNCH lineage)
  Mechanism: KD splits price both phases on the starvation ladder (best split ≈ 96.6 even granting the published-max +1.0 fixed-epoch credit, EXP-043/002/007); upsampled training adds zero information while Eval pins 32px (EXP-029 mismatch sign) atop a ×1.56 FLOPs floor (EXP-066 square-kernel pricing); the optimizer family is subsumed by its measured best member (Muon at mean, EXP-028; schedule-free −1.84, EXP-062) plus pointwise tolls.
  Insight: the class-level closure ledger is now complete — candidate inflow requires new heavy-aug budget-matched external evidence or a revision of a standing anchor; route any such claim through a pre-run inequality (reports/exp-report-068.md) before spending charged seconds. Do not re-propose KD variants, mixed/raised resolution, or AdamW/Lion/cautious retunes.

- **5x5 stem conv — the kernel-size corner, probe-priced free (+0.26ms) yet a REAL structural negative** (count: 1, EXP-066; closes the ceiling audit's last unpriced corner)
  Mechanism: 96.14 = mean − 2.7σ at byte-clean signatures (13,266 steps, 137 ep, test_loss 0.1929) with the plateau still creeping at cutoff (EXP-030's still-organizing signature) — 75-weight stem filters averaging 25 of 32² pixels organize slower through peak heat and degrade all 19 downstream convs. Probe datum: square 5x5 is FAST-PATH (stage-3 P_norm 31.02 ≈ dense-law 30.3 at 1.59× FLOPs) — kernel-shape slowness is DIMENSIONALITY (1D, EXP-044/045), not size.
  Insight: the stem is triangulated load-bearing in content (EXP-019), gain (EXP-018), and now geometry; kernel-size axis closed (bigger/dilated stems dominated, stage-3 starvation-priced, mixed branches EXP-064-closed). The 2026-06-11 adversarial ceiling audit also closed Normalize std=(1,1,1) BY INSPECTION (eval pins the input distribution; RE-fill amplitude covered by EXP-060) — the measured-ceiling hypothesis survived its designated falsification attempt.

- **Structural reparameterization — ACNet/DBB-family multi-branch convs (3x3 ∥ 1x3 ∥ 3x1 per-branch BN; eval-exact fold proven to 2e-6)** (count: 1, EXP-064; family-level closure)
  Mechanism: probe-gated NO LAUNCH with internal baseline control (B = 22.36 mid-family-band, same session): full ACB toll ratio 1.930 (P_norm 43.2 vs ≤ 26.0 — required gain 1.19 > published max 1.11); minimum possible variant (3x3 ∥ 1x1, +1 conv +1 BN per site) still 28.61ms → required gain ≈ 0.69 > single-branch ablation gains. 1D convs land on slow odd-shape kernels (EXP-044/045 law); toll superlinear in branch count.
  Insight: "free at inference" inverts under a train-time budget — the technique's entire price lands on the metered resource. Family closed: any branch subset, any site subset (gain and toll co-scale with sites). Do not retry: ACB/DBB/RepVGG-style branches, partial-site variants, per-kernel-region optimizer surgery (the no-BN equivalent — closed by EXP-057/058 bracketing logic). Durable instrument: the two-net probe with INTERNAL BASELINE CONTROL self-validates against load inflation — prefer it over load-conditional criteria. Zero charged seconds spent.

- **Stream-parallel two-member ensemble — concurrency as the diversity funding source (two CUDA streams, event-ordered fwd+bwd, MeanEnsemble eval)** (count: 1, EXP-063; completes the funding closure of EXP-043/042)
  Mechanism: probe-gated NO LAUNCH at the cleanest conditions (apps=0, load 10.1, P1 = 22.48 inside the family band): joint two-stream step P2 = 40.70ms (ratio 1.810; eager diagnostic 1.820 — identical, so not a compile artifact). For latency-bound small-kernel models the binding resource is the SERIAL kernel-dispatch chain shared by all streams in one process; the ~93% idle compute (EXP-034/048 geometry) is not spare dispatch capacity, so a second stream queues behind the chain instead of running beside it.
  Insight: multiplicity is now closed at the FUNDING level — steps (−0.9 dilution, EXP-043), kernel width (2.8×, EXP-042), and concurrency (1.81× serialization, EXP-063) all cost more than the +0.3–0.5 function-space gain. Do not retry: any same-process multi-stream training on this family (any member count, eager or compiled); multi-process/MPS violates single-run timer semantics. Zero charged seconds spent — the pre-run-inequality requirement did its job. Open residue: if dispatch latency is a component of the 22.4ms step, CUDA Graphs (`reduce-overhead`) is a STEP-TIME lever worth a ~90s probe (EXP-048's 99.3%-kernel-time bound says upside is small).

- **Schedule-Free SGD substitution (constant lr 0.4 post-warmup, eval at lr²-weighted average x, BN refreshed at x; Defazio NeurIPS 2024)** (count: 1, EXP-062)
  Mechanism: the x-curve delivered the paper's SHAPE (strictly monotone, zero scatter) yet was still climbing at ep139 — 94.87 = mean−10.6σ at exact family signatures (13,471 steps, dt-FREE foreach step, VRAM 1,613); test_loss 0.229; 3 evals within 0.15 of best (no plateau, only a slow climb's tail). Averaging hot iterates cancels noise but cannot do the anneal's late-phase basin refinement — at a 13.5k-step horizon the averaged trajectory sits far before its catch-up point vs the annealed one.
  Insight: the schedule axis is closed at the FAMILY level — anneal-internal probes (EXP-010/014/016/049) plus the averaging-based alternative; EXP-032's converse is now measured (averaging does NOT perform implicit annealing). Any-horizon external claims must be checked against the horizon's position on the catch-up curve. Do not retry: lr retunes (historical recovery ≤0.5 vs −1.84), schedule-free AdamW (same horizon objection), anneal-tail hybrids (reintroduce the measured optimum). Reusable: reference-exact ScheduleFreeSGD + BN-refresh-at-x machinery (knowledge/papers/schedule-free-road-less-scheduled.md); x-at-init explains ep1 ≈ 22 (a legitimate tripwire-exception class, judge by ledger).

- **Occlusion-TYPE substitution at the dose optimum — CutMix (signal-fill + area-weighted mixed labels) swapped for RandomErasing at p=0.5** (count: 1, EXP-060; closes the seam EXP-009 left open)
  Mechanism: precise family null — 96.69 = mean+0.75σ at byte-clean signatures (D0 22.7, 13,285 steps, 137 ep, probe-priced toll 0.11ms); test_loss up (0.224 vs 0.185) is mixed-target CE arithmetic with accuracy untouched. Under TA the network is indifferent to what fills the occluded box AND to occlusion-correlated label mixing: occlusion acts purely as information deletion, so dose (already optimal) is the only live variable. The largest published prior (+0.5–1.0) of any untested construction absorbed to zero — external transfer 0-for-17.
  Insight: regularization axis now closed in TYPE as well as dose; substitution at constant count is SAFE where stacking was over-pressure (no EXP-009-class deficit), but carries no gain. Do not retry: mixup-substituted (dominated member, inherits null + off-manifold risk), CutMix dose/α variants (dose bracketed), hybrid fills (interpolations of null vs optimum). Reusable: sync-free eager per-batch input mixing outside the compiled graph costs ~0.1ms and needs no compile precautions.

- **Full pre-activation block reorder (ResNet v2 / WRN-native B(3,3): clean identity, bare stem, final BN-ReLU; params exactly preserved)** (count: 1, EXP-056)
  Mechanism: toll-adjusted null — raw 96.49 = mean−0.5σ with a probe-measured +1.0ms/step fusion toll (~−7 epochs ≈ −0.08 priced), so the ORDER effect itself ≈ 0 at family signatures; He's v2 gain is a depth phenomenon (vanishes at depth 20) and the residue absorbs under TA+RE (external transfer 0-for-16).
  Insight: block-order class closed; the standard-modernization audit is COMPLETE — every published-toolkit entry is now measured on this recipe. New throughput datum: op ORDER changes dt at an identical op multiset (inductor fuses post-act epilogues better) — GPU-probe any reorder before launch.

- **FreezeOut-style tail freezing of stem+layer1 (compressed anneal completing at p=0.70, graph-visible detach freeze, dual-warmup cached)** (count: 1, EXP-055)
  Mechanism: the conversion DELIVERED in full — dt 22.5→15.8ms (31%), +1,550 tail steps, +16 plateau evals, zero recompile toll — yet read 96.32 = mean−1.6σ converged-flat: freezing ⅓ of conv FLOPs (only 5.2% of params) costs ~0.3 of plateau LEVEL, far more than the surplus repays.
  Insight: the tail-pressure law is now PARAMETER-side too (data-side: EXP-025/033) — nothing, distribution or weights, may stop moving before budget end. Do not retry freeze variants (amplitude-scaled/later/uncompressed are interpolations against closed heat/anneal laws); the detach-flag+dual-warmup pattern and the measured "stem+layer1 tail refinement is worth ≥ ~0.3" price are the reusable outputs.

- **Cross-axis compound of certified-free components (anti-aliased shortcut + de-overhead prefetch), n=2 mean replicate** (count: 1, EXP-053)
  Mechanism: pair read 96.61/96.28, mean 96.445 = mean−0.8σ — BELOW the shortcut-alone pool (+0.11, n=3); the prefetch's once-measured +87-step saving regressed to zero on both replicates (13,428/13,434 = 046-family ledger), so the second component delivered nothing and the pair still underran additivity.
  Insight: cross-axis "free + free" does not compose into a free positive — the compound-of-frees region (the last catalogued positive-direction space) is closed. Do not re-compound the current component pool; a future compound needs a NEW component with a replicated ≥ +0.1 estimate first.

- **Lowering integrated LR heat (PEAK_LR 0.4→0.3, 0.75× heat at byte-identical signatures)** (count: 1, EXP-049; completes the bracket opened by EXP-010/014)
  Mechanism: mean-band null — 96.52 = mean − 0.3σ at family test_loss (0.1882), family step count (13,456), converged-FLAT plateau (not heat-starved); the run lost ~2pp mid-schedule and the anneal fully repaid it, confirming the anneal (not the peak) sets final basin quality in this regime (consistent with EXP-016 tail-shape sensitivity).
  Insight: the LR optimum is FLAT over [0.3, 0.4] and breaks between 0.4 and 0.6 — the last open recipe-constant bracket is closed; the recipe-constant audit is COMPLETE. Do not probe 0.5 (no mechanism to beat a flat optimum; cartography only) or warmup 0.25 (the heat variable itself is now measured flat-below).

- **Charged-step de-overheading (collate-side channels_last + side-stream H2D prefetch, numerics-identical)** (count: 1, EXP-048; bounds brainstorm-048 Idea 2 / cudagraphs)
  Mechanism: total recoverable non-kernel overhead measured at ~0.15ms/step (0.7%): pinned non_blocking H2D already overlaps queued work, and the channels_last permutation is bandwidth-trivial; saving delivered exactly +1 epoch (+87 steps) → +0.02 by the conversion law, invisible; read 96.57 = mean exactly at family signatures.
  Insight: the charged step is 99.3% irreducible kernel math and the kernel lattice is at its floor (EXP-040–045) — the EXP-000/006 throughput→epochs mechanism is EXHAUSTED. This bound also closes cudagraphs/reduce-overhead (recoverable launch overhead < 0.15ms) without a run. Do not retry step-time engineering of any flavor.

- **Multi-scale decision head — fc over concat[GAP(stage2), GAP(stage3)] (384-d classifier input)** (count: 1, EXP-047; triangulates with EXP-030, EXP-037)
  Mechanism: a REAL structural negative, not an absorbed null — 96.15 = mean − 2.6σ at byte-clean signatures (22.5ms, 138 ep, params 4,287,306, test_loss 0.19); the plateau is uniformly depressed. Mid-level stage-2 GAP features carry weak class-linear information; concatenated into one linear head they dilute the decision margin all training long.
  Insight: decision-layer routing class closed FROM BELOW — GAP(stage3)→fc is measured load-bearing, a local optimum triangulated three ways (pooling op EXP-030 −0.91, channel gating EXP-037 null−deficit, input routing EXP-047 −0.56). Do not retry concat heads, gated h2 paths, or projected stage-2 inputs — dose-128 at −2.6σ makes smaller doses implausible. With this, ALL structural classes are closed (absorbed-null, active-negative, or cost-priced).

- **Anti-aliased shortcut downsampling (F.avg_pool2d replacing the pad shortcut's [::2,::2] strided slice at both stage transitions)** (count: 2, EXP-046, EXP-052)
  Mechanism: zero params/dt/noise change; n=1 read 96.65 (+0.5σ); pre-registered n=2 mean replicate (EXP-052) drew 96.84/96.56 → mean 96.70 < bar; pooled 3 draws +0.11 vs recipe mean = +1.2σ_mean — unresolvable from zero (resolving +0.1 needs n≈20); the Zhang-2019 gain (+0.5–1.0, weak-aug fixed-epoch) absorbs to ≤ +0.1 under TA+RE.
  Insight: closed PERMANENTLY with adequate statistics — any true effect is sub-bar by direct measurement, not classification. Further sampling = variance mining. Blurpool (grouped, 2.5–3× dense) is the same mechanism at positive cost — do not retry. avg_pool2d confirmed free on the kernel lattice.

- **Ensemble multiplicity under the fixed budget — full-alternation 2 × 4x members, logit-mean inference (function-space averaging)** (count: 1, EXP-043; brackets with EXP-011/032/042)
  Mechanism: the 2-member decorrelation gain is REAL (+0.3–0.5 over members priced at ~95.6 by the starvation ladder) but halving each member's steps costs ~−0.9 vs the single-model mean → 96.07 at byte-clean dense signatures (22.34ms, 139 ep, 470.3s); test_loss 0.196 not below family; plateau scatter 3× (starved members still moving at anneal end).
  Insight: averaging dichotomy fully measured — weight-space zero gain (EXP-011/032), function-space real-but-unaffordable (this); both diversity-starvation endpoints known, interior (mid-fork) interpolates ≲96.6 sub-bar; in-one-kernel variants hardware-closed (EXP-042). The max-statistic plateau wants ONE fully-trained model. MeanEnsemble eval-contract + eval-thinning predicate are validated reusable engineering (run finished 470.3s despite 2× eval cost).

- **Derandomized alternating horizontal flip ((epoch+index)%2 via shared-memory epoch tensor into persistent workers)** (count: 1, EXP-041)
  Mechanism: the coverage benefit is real but regime-bound — early evals ran at/above family (ep5 66.6 vs ~64) where per-image flip samples are few, then vanished by the plateau (iid has ~70 samples/orientation by ep139); meanwhile alternation makes each image's last-seen orientation DETERMINISTIC by parity at every epoch end → plateau scatter 0.75 (5× family) and mean mildly down (96.27); best 96.49 at byte-identical signatures.
  Insight: third independent case (with BN-momentum EXP-038/039) of the epoch-boundary law: best-over-evals punishes any mechanism that makes epoch ends structurally special. Data-order/coverage class closed; the shared-memory epoch tensor is validated reusable engineering for live per-epoch state in persistent workers.

- **Progressive resizing 24→32 (first 50% of budget at 24px via in-step charged GPU downsample)** (count: 1, EXP-031)
  Mechanism: throughput fully delivered (dt 13.5ms = 0.60×, +46 epochs, stall-free switch, clean phase-2 numerics) yet plateau unchanged at 96.69 (noise-band) — low-res epochs are NOT baseline-epoch-equivalents; the formative high-LR phase learns 24px statistics whose advantage is transit-speed only, decaying to zero at the converged plateau (max-statistic law). CIFAR 24px discards signal where ImageNet 160px discards redundancy.
  Insight: the EXP-006 conversion law (+25 ep = +0.48) is conditional on epochs carrying the SAME training distribution. Even regime-MATCHED (wall-clock-budgeted) external evidence failed to transfer — record now 0-for-11. Interior points (28px, other fracs) unbracketed but expected ≤ +0.1: the toll is distribution-inheritance, not adaptation shock (switch dip healed in 1 epoch).
  Mechanism: dt-free (+0.3ms) and only +2,560 params, yet behind family at EVERY waypoint and still climbing at cutoff (best at ep136), −0.91 — max-pool's argmax-routed gradients are a discontinuous, high-variance credit path that drags head+trunk organization through the whole high-LR phase.
  Insight: a SIXTH deferral variant — free in dt/params/numerics/noise/normalization but loses on per-step optimization QUALITY at heat. Speedrun architecture evidence is now 0-for-10 transferring. LogSumExp pooling would fix the discontinuity but pays the EXP-026 pointwise tax — gate-price if ever tried.

- **Eval-time BN stat substitution (clean-data recalibration before each eval, forward-only)** (count: 1, EXP-029)
  Mechanism: weights/affines are calibrated to the AUGMENTED batch-stat normalization constants used in every training forward; swapping in clean-data constants mis-scales all 20 BN layers and compounds — −10.93 at convergence (test_loss 0.49 vs 0.185) with training provably untouched.
  Insight: eval must REPRODUCE training-time constants, not match the eval distribution; interpolations are bracketed (96.71 pure-augmented ↔ 85.78 pure-clean). EXP-025's +0.35 was feature adaptation, not stat alignment. Alignment axis closed with inverted sign.

- **Muon optimizer for conv weights (NS-5 orthogonalized nesterov momentum, airbench anchor lr 0.24/m 0.6)** (count: 1, EXP-028)
  Mechanism: the sample-efficiency gain is REAL (ep10 85.7 vs ~78) but decays to zero by plateau — it buys arrival time, not plateau height; the plateau lands AT the baseline mean (96.42/96.53 two clean draws) in a slightly worse basin (test_loss 0.193 vs 0.185) while NS-5's +2.9ms/step costs 16 epochs.
  Insight: optimizer-GEOMETRY axis closed alongside the SGD-internal axes; speedrun optimizer evidence is calibrated to ~10-epoch budgets where arrival time IS the metric. Remaining escapes (NS-3, retune, SGD tail) all fail the +0.3 effect-size screen on the basin-deficit arithmetic.

- **Smooth activations (GELU(erf), SiLU, hardswish) in place of ReLU** (count: 1, EXP-026)
  Mechanism: activation math is a throughput lever at this dt — GELU +4.5ms (+20%), SiLU +2.8, hardswish +1.1 per 22.4ms step (19 sites, fwd+bwd); hardswish's full run netted exactly its 7-epoch deficit (−0.10) plus an early-heat init toll (ep1 20.2 vs family ~38) and a truncated plateau (best AT final epoch).
  Insight: ReLU is uniquely free in dt + calibrated to the certified init/heat + mask-cheap in backward VRAM; activation axis closed on cost grounds. Speedrun-lineage activation evidence (airbench GELU) does not transfer across dt budgets.


- **SE channel attention (r=16, all 9 blocks, near-identity init: fc2 zero-weight + bias 2.0 → gate 0.881 at step 0)** (count: 1, EXP-037)
  Mechanism: engineering flawless — deferral dodged (ep1 ~35 vs family ~38, not EXP-018's ~20; init verified post-kaiming-pass), dt only +1.7ms (~0.19ms/module; micro-attachments are an order cheaper than the 2.5ms/block whole-block cost) — yet two clean draws (96.34/96.37, 129/128 ep) land exactly at mean − deficit with family-identical test_loss (0.188 vs 0.185). Published CIFAR SE gain (+0.5–1.2 fixed-epoch) appeared at ZERO strength.
  Insight: the heavy-augmentation absorption law extends from regularizers (SAM EXP-035, LS EXP-036) to capacity-conditioning modules — SENet's CIFAR baselines are crop+flip; under TA+RE + completed anneal the gates buy nothing. Attention axis closed (stage-3-only/ECA doses of a zero-effect mechanism inherit the null). NEW SCREEN: candidates need evidence under heavy-aug budget-matched regimes specifically.

- **Label-smoothing dose 0.1→0.2 (in-domain anchor value; the last unmeasured recipe constant)** (count: 1, EXP-036)
  Mechanism: a precise null — 96.58 = baseline-recipe mean (96.57) at byte-identical signatures (22.3ms, 139 ep, clean profile); neither the predicted margin-pressure gain nor the over-smoothing harm appeared; test_loss 0.286 vs 0.185 is pure hard-label CE arithmetic under compressed logits, accuracy untouched.
  Insight: LS is FLAT on [0.1,0.2] under TA+RE — heavy augmentation absorbs the regularization role LS plays in the lighter-aug anchor recipes (cifar10-fast/airbench use derandomized flip+cutout). Anchor transfer requires AUGMENTATION-regime match, not just dataset/schedule match. Loss-target axis closed; LS=0 down-bracket is low-expected-value (flat top end + hard-target risk under heavy TA).

- **Distribution lightening of ANY phase — clean-data tail (EXP-025), light-aug tail (EXP-033), AND light-aug HEAD/warmup (EXP-065): the pressure-profile law is four-quadrant complete** (count: 3, EXP-025, EXP-033, EXP-065)
  Mechanism: tail variants bank a one-time alignment transient (+0.35 clean / +0.48 light) then lose the anneal's endgame (−0.87 / −0.46). The head variant (TA+RE off for the 21 warmup epochs, crop+flip floor, full recipe for the entire anneal) trained visibly faster during the light phase (loss 0.76 vs family ~1.1) yet the advantage washed out and mildly inverted: 96.38 = mean −1.2σ at byte-clean signatures. Banked easy-distribution alignment does not transfer forward — the EXP-018 "switches on during peak heat loses" signature, data side.
  Insight: augmentation pressure, like LR heat and parameter trainability, must be constant-on from step 0 to budget end. The curriculum intuition (easy→hard) is directly measured-negative in this regime; the reverse ordering (hard→easy) is EXP-033. No time-profile of the recipe's pressures remains unmeasured: do not retry ramps, gates, tapers, or phase-restricted aug in EITHER direction or ANY phase. Alignment transients are distribution-proximity artifacts, not banked capability (EXP-025/029/033/065 — four collection attempts, all net-negative).

- **Explicit sharpness minimization — periodic SAM (k=5, ρ=0.05, global-norm, BN stats frozen on perturbed pass, eager second pass)** (count: 1, EXP-035)
  Mechanism: implementation flawless (cost model exact: 48.9ms/SAM-step, 27.7ms mixed, 112 epochs; no instability at peak LR) yet the result decomposes to deficit-only: −0.32 vs mean ≈ −27 ep × 0.014, retained gain ≈ 0, and test_loss 0.1945 vs family 0.185 — no flatness signature in EITHER currency.
  Insight: TA+RE+LS at batch 512 already saturates implicit flatness — explicit worst-case sharpness pressure is redundant with heavy-aug gradient noise. SAM's published CIFAR gains are weak-augmentation-calibrated. Closes the flatness family at the explicit end (EXP-023/024 closed the implicit end); ASAM/ESAM/ρ-k retunes closed by the zero-gain reading + deficit arithmetic.

- **Depth-for-width capacity reshaping (ResNet-26 at 56/112/224 ≈ matched FLOPs/params; fallback 48/96/192; mirror: EXP-008 ResNet-14 wider-shallower)** (count: 2, EXP-008, EXP-034)
  Mechanism: doubly closed. Hardware: per-block cost ~2.5ms is width-independent at these widths, so +3 blocks = +8.3ms whatever the width — the matched-dt depth design point does not exist (gate-kills at 31.0/27.4ms). Statistics: the full run at its own pace (30.5ms, 102 epochs, anneal complete) converged FLAT at 96.01 with test_loss 0.196 vs baseline 0.185 — a worse basin LEVEL, not a transit deficit.
  Insight: depth 20 at 4× width is a measured local optimum of the depth-width plane (shallower lost EXP-008, deeper loses both on dt and on level); width saturates depth's marginal value — WRN's claim confirmed under fixed wall clock. Gate-screens ResNet-32+ permanently via the per-block law.

- **Heat-constant momentum trades in BOTH directions (β 0.95 + peak 0.2; β 0.8 + peak 0.8; lr/(1−β)=4 held)** (count: 2, EXP-023, EXP-024)
  Mechanism: both premises delivered perfectly — signatures byte-identical to baseline both runs — yet both converged below it: horizon ×2 (noise down) −0.30; horizon ×½ (noise up) −0.22, stable, no bounce. β=0.9 is at the top of the noise parabola.
  Insight: momentum axis closed BIDIRECTIONALLY; recipe certification complete (every constant bracketed alone or in compensated trades); interpolations (β 0.85/0.925) closed by bracketing logic. See GRADIENT-NOISE LAW in Patterns.

- **Step-time engineering via max-autotune + cudagraphs + fused SGD (faster-but-different arithmetic)** (count: 2, EXP-021, EXP-070 family-level closure; precision counterpart EXP-054)
  Addendum (EXP-070): EXP-048's 99.3%-kernel bound closes OVERHEAD only; the kernel-MATH side closes by EXP-021's numerics law + the required-gain inequality (bar needs ~11% numerics-identical speedup; first infra-scoped search found no such option in torch 2.9/cuDNN — pytorch.org/blog/pytorch-2-9; cuDNN algo knobs are numerics-different by construction). Throughput escape shut by composition; re-check availability in future sweeps only.
  Mechanism: speedup fully delivered (dt 22.4→20.9ms, 149 epochs, +10) at byte-identical hyperparameters, yet converged −0.20pp (replicated: 96.41/96.51) — Triton kernel reduction orders/TF32 choices + graph execution degrade the trajectory ~−0.4pp mid-run, more than the epochs repay.
  Insight: the EXP-006 conversion law needs numerics equivalence, not just hyperparameter constancy; throughput axis closed at its last tier unless a numerics-PRESERVING speedup exists (none identified — default-compile dt is near the floor).

- **fp16 autocast + GradScaler in place of bf16 (8× finer mantissa at equal tensor-core speed) — the finer-precision direction of the numerics axis** (count: 1, EXP-054)
  Mechanism: precise mean-band null — 96.54 (mean −0.2σ) at family signatures (windows 22.0–23.3ms, family trajectory/plateau/test_loss 0.1877); the scaler's unscale+inf-check pass cost 2–3 epochs (136 vs 138–140; ≈ −0.05 priced), so the precision effect itself ≈ 0; fp16 fully STABLE at peak LR 0.4 (zero skips/NaN).
  Insight: the recipe sits at a measured PLATEAU in precision-space — bf16 rounding was never a cost (EXP-021 closed the coarser side at −0.20). Do not revisit precision, compile modes, or throughput flavors; fp16 plumbing is stability-certified if a future construction needs it.

- **Projection shortcuts at stage transitions (ResNet option B, WRN-faithful 1×1+BN)** (count: 1, EXP-020)
  Mechanism: two stacked costs — Kaiming-initialized projections must be LEARNED during peak heat (ep1 34.6 vs 38.2–39.0 family) AND +0.6ms/step cost 4 epochs (≈ −0.08pp by EXP-006 arithmetic); converged −0.13pp. The pad shortcut's zero channels are filled by the residual branch anyway at 135 epochs.
  Insight: fixed-epoch architecture evidence does not transfer even from the best-matched regime (WRN: CIFAR-10, our widths/depths) — an architecture change must be free in BOTH early heat and epochs to escape; shortcut-topology axis closed.

- **Whitening init for the stem conv (airbench patch-eigenvector filters ± negations, learnable)** (count: 1, EXP-019)
  Mechanism: washed out — early trail within eval noise of baseline (ep1 38.95 vs 38.20), plateau converged −0.26pp; bn1 immediately renormalizes the stem (airbench feeds whitening into an activation with NO intervening BN), and 139 epochs learn an equivalent basis anyway (vs the ~10-epoch speedruns where it pays).
  Insight: init axis now closed BOTH directions (expressivity-removing −0.99, information-adding −0.26); "information at init" only pays when the budget is too short to learn it — at ~139 epochs nothing init-time moves the converged plateau.

- **Zero-γ init of residual-final BNs (Bag of Tricks identity-at-init)** (count: 1, EXP-018)
  Mechanism: blocks spend the warmup/peak phase "turning on" from γ=0 — effectively a stem-only net during the hottest heat (ep5 35.3% vs 63.8% comparator), converging flat at −0.99pp; the trick's documented benefit (stability headroom for bigger peak LR) is one we cannot spend, since peak 0.4 is certified optimal via quality not stability (EXP-010).
  Insight: init interventions priced in EARLY HEAT are deferral and will lose; only init changes that ADD early learning (e.g. whitening first conv — information at init) escape the objection. Init axis's cheap end closed with a strong negative sign.

- **Per-stage depth reallocation at equal FLOPs — BOTH directions measured below uniform: [2,3,4] −0.28 (EXP-017) and the mirror [4,3,2] −0.32/floor−0.02 (EXP-061)** (count: 2, EXP-017, EXP-061)
  Mechanism: both reallocations converge FLAT below uniform at family test_loss. [2,3,4]: removing a stage-1 block costs more early representation than +1.11M stage-3 params return. [4,3,2]: a fourth stage-1 block adds nothing while two 256-wide blocks under-process the final features — stage-3's block COUNT is load-bearing even though its added params were not (the EXP-017/061 distinction). Probe datum: the per-block dt law holds for stage-heavy shapes ((4,3,2) = 22.51ms despite 4× activations in stage 1 — block count, not activation volume, prices the step).
  Insight: the allocation curve PEAKS AT uniform [3,3,3] — the axis is closed bidirectionally, and with it the depth-allocation plane is fully measured (total depth EXP-008/034; per-stage split EXP-017/061). Slopes measured by REMOVAL do not extrapolate through the optimum into ADDITION. Do not retry: interior splits ([4,2,3]/[3,4,2] — interpolations of a peaked curve), reallocation + width compensation (off-lattice/starvation-closed). This was the last in-record favorable directional signal; the record now contains NO untested construction with measured evidence in its favor.

- **Linear-to-zero anneal in place of cosine (identical peak/warmup/total heat)** (count: 1, EXP-016)
  Mechanism: linear's hot late-middle (lr ≈0.028 vs cosine ≈0.005 at p≈0.94) keeps the net refining to the very end — still climbing at cutoff, best at the FINAL epoch, −0.50pp; cosine's flat cold tail manufactures ~10 converged evals for the max-statistic to harvest, linear got one.
  Insight: final-value schedule theory (Defazio linear-beats-cosine) INVERTS under best-over-checkpoints + fixed wall clock — the tail's job here is a long converged plateau, not an optimal final iterate. Schedule axis now closed in every probed dimension (heat ±, warmup, family); only heat-compensated hybrids remain, low prior.

- **Halving weight decay (5e-4 → 2.5e-4) — the pressure-DOWN probe** (count: 1, EXP-015)
  Mechanism: no over-fit tail — the run simply trained marginally worse at every stage and converged flat at −0.30pp; consistent with the WD-with-BN effective-LR account (less WD = larger weight norms = cooler effective LR + less gradient noise).
  Insight: the recipe was AT the regularization optimum, not past it; WD axis closed both ways (finer steps are within ±0.1pp noise). The cooling reading also adds a cold-side point to the heat curve — heat is at optimum too.

- **Reflection padding for RandomCrop — "data quality" reframing of augmentation still lands on the saturated axis** (count: 1, EXP-013)
  Mechanism: reflected borders = more diverse, harder crops ⇒ effectively stronger augmentation; trajectory ran 6–8pp below baseline mid-schedule, converged −0.14pp (96.57); its PIL cost also grew loader stalls ~50→197s, busting the 600s cap (626.9s).
  Insight: on this recipe ANY increase in train-crop diversity is regularization regardless of framing — the dose-response curve now has four points (RE +0.83, TA +0.17, reflect −0.14, mixup −0.46); only pressure-REDUCING moves (e.g. WD down) remain untried on this axis.



- **Stacking a fourth regularizer (mixup α=0.2) on the LS+TA+RE recipe** (count: 1, EXP-009)
  Mechanism: over-regularization, not undertraining — run converged (flat final plateau) 0.46pp below baseline; four soft/occlusion/policy regularizers over-constrain 4.29M params even at 137 epochs.
  Insight: the regularization axis is saturated (dose-response RE +0.83 → TA +0.17 → mixup −0.46); substitution (swap RE for mixup) or LS-removal variants exist but expected value is low.

- **Depth-for-width reshaping at constant time (ResNet-14 @ 6x)** (count: 1, EXP-008)
  Mechanism: fails by CONVERGENCE not starvation — 83 epochs (above floor, dt model exact) plateaued flat at 95.76 (-0.95pp); depth 20→14 loses more representation than 1.43x params at 6x width restores.
  Insight: WRN's depth-width equivalence does not transfer down to 14 layers on this variant/budget; ResNet-20 4x @ 139ep is the topology optimum — rotate to per-epoch signal (Mixup/CutMix), optimization quality (EMA), or throughput.

## Patterns

<!-- Goal-specific patterns, mechanisms, and reusable recipes discovered during experiments.
     Entry format: 3-line statement / evidence / implication (budget-strict). -->

### High Importance

- **Time-keyed one-cycle LR (progress = elapsed/TIME_BUDGET_S) guarantees anneal completion under any throughput — validated, composable** (EXP-000, EXP-001)
  Evidence: reports/exp-report-000.md § Results; exp-report-001.md — schedule self-adapted to 3x lower throughput with zero retuning
  Implication: keep this schedule under all future architecture/pipeline changes; never key schedules to predicted step counts.
- **Accuracy scales steeply with width under the fixed time budget: 4x width (4.29M params) = +2.07pp; heavier models also cut eval count and wall-clock risk** (EXP-001)
  Evidence: reports/exp-report-001.md § Results — 95.23% at 114 epochs, total 395.8s vs 596.7s at EXP-000
  Implication: next capacity step (6-8x width) is high-value; watch undertraining below ~50 epochs and consider peak-LR 0.3.
- **Occlusion regularization (RandomErasing p=0.5, scale 0.02-0.4, random fill) is free accuracy once capacity exists: +0.83pp at zero throughput/VRAM cost** (EXP-003)
  Evidence: reports/exp-report-003.md § Results — 96.06% with epochs/VRAM identical to EXP-001; final_test_loss 0.2447 → 0.2084
  Implication: keep in all future recipes; augmentation pressure raises signal per epoch, so the width-epoch optimum likely moved right — retry 5-6x width on this recipe.
- **Regularization dose-response at 4x width is mapped on BOTH sides and PEAKED at the current recipe: WD-half −0.30 (below) | RE +0.83, TA +0.17, reflect −0.14, mixup −0.46 (above)** (EXP-003, EXP-004, EXP-009, EXP-013, EXP-015)
  Evidence: reports/exp-report-015.md § Results — pressure-DOWN run also CONVERGED FLAT (no over-fit tail), like both over-pressure runs; deficits on both sides are dynamics, not schedule length
  Implication: keep WD 5e-4 + LS+TA+RE exactly as-is — the axis is closed in BOTH directions; ANY diversity-increasing change counts as pressure regardless of framing (EXP-013); fine WD steps (4e-4/7.5e-4) are within-noise, skip. TIME-VARYING doses also lose and are now bracketed monotone in tail pressure: zero-aug tail −0.87 via overfit (EXP-025); crop+flip tail −0.46 via learning freeze WITHOUT overfit (EXP-033) — pressure must be constant AND at the certified level through the LAST step.

- **The EXP-006 recipe is a measured LOCAL OPTIMUM of single-constant changes AND of structural perturbations: ten constant probes (EXP-007…015) plus four structural edits (EXP-017 free params, EXP-018 easier optimization, EXP-019 information-at-init, EXP-020 reference-faithful shortcut modernization) all converged below baseline** (EXP-007…EXP-020)
  Evidence: experiment-indices TSV rows 007–020; exp-report-015.md § Analysis (constant bracketing); exp-report-020.md § Results (structural arc incl. the best-evidenced architecture change available)
  Implication: RECIPE-SPACE IS CLOSED — and as of EXP-036 the closure is AUDIT-COMPLETE: LABEL_SMOOTHING, the one constant that had entered bundled (EXP-000) and was never dosed, read exactly flat at 0.2 (96.58 = mean). Every constant in train.py has now been individually dosed at least once and the incumbent won or tied every time. Brainstorms must screen candidates against ALL FOUR laws — deferral, numerics equivalence, max-statistic, gradient-noise optimum (now bidirectional) — and operate out-of-recipe only (data composition/order, objective shaping, architecture free in all four currencies).

- **GRADIENT-NOISE LAW (BRACKETED, now BOTH implicit and explicit): the baseline noise scale (batch 512, β=0.9) is the measured MAXIMUM of the noise curve — both directions lose at byte-identical signatures; EXPLICIT flatness (periodic SAM, EXP-035) adds zero on top, confirming the implicit budget is saturated** (EXP-011, EXP-022, EXP-023, EXP-024, EXP-035)
  Evidence: exp-report-024.md § Results — noise down: EMA −0.25, 2× batch −0.14, 2× horizon −0.30; noise up: ½ horizon (β 0.8 + peak 0.8) −0.22, stable, converged tight plateau; both compensated trades had signatures identical to baseline
  Implication: reject candidates that move effective gradient noise in EITHER direction; with this bracket, recipe-space is closed in the strongest sense — remaining candidates must be out-of-recipe (data composition/order, objective shaping, architecture) AND free in heat+epochs+numerics+noise.

- **torch.compile is pure profit on this setup: 1.22x img/s for 22.8s startup; warmup-before-timed-loop + eager-eval-reference make it budget-clean** (EXP-006)
  Evidence: reports/exp-report-006.md § Results — 139 vs 114 epochs, +0.48pp, training_seconds exactly 300.0, startup 22.8s
  Implication: keep compiled (DEFAULT mode) in all future recipes; a "converged" (final=best) one-cycle run can still be epoch-starved — throughput gains convert ONLY when hyperparameters stay fixed (EXP-012: batch-driven throughput forced LR up, metric-neutral) AND the arithmetic stays numerically equivalent (EXP-021: max-autotune/cudagraphs/fused gave +10 epochs yet lost 0.20pp — different kernel numerics degraded the trajectory more than the epochs repaid).

### Medium Importance

- **Compiled throughput projections must use measured compiled dt, not eager scaling exponents — inductor gains shrink as width grows** (EXP-007)
  Evidence: reports/exp-report-007.md § Results — 6x cost 2.59x time for 2.25x FLOPs vs eager exponent ~0.76 predicting 75-80 epochs; got 55
  Implication: before any capacity step, compute projected epochs from a measured dt at the target width (a few warmup steps suffice); require ≥70.
- **bf16 autocast + TF32 + channels_last + batch 512 gives 3.75x img/s on H20 for small CNNs; host DataLoader is the remaining bound** (EXP-000)
  Evidence: reports/exp-report-000.md § Results — 60k vs 16k img/s, dt ~8ms at batch 512
  Implication: further throughput needs GPU-resident data or bigger batches; the precision/layout lever is now exhausted.

### Low Importance
