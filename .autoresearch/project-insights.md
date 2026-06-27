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

- **H20 step time for small models is governed by alignment AND per-layer launch overhead, not FLOPs: 5x widths (80/160/320) paid 2.19x time for 1.56x FLOPs; +3 blocks cost +8.3ms regardless of width compensation (~2.5ms/block, width-independent at ≤64-wide stage 1)** (EXP-005, EXP-034)
  Evidence: reports/exp-report-005.md § Results — 8.6k img/s vs 18.7k at 4x; exp-report-034.md § Results — two-point measurement: ResNet-26 at FLOPs 1.02x ran 31ms, at FLOPs 0.75x ran 27.4ms (∂dt/∂FLOPs ≈ 13.3ms/unit; the rest is per-block) vs baseline 22.4ms at 9 blocks
  Implication: any architecture change on this hardware must keep channels multiples of 32 (ideally 64) AND treat layer COUNT as a first-class dt cost — "matched-FLOPs" depth trades do not exist at this scale; price every shape in measured dt within the first 2 minutes before reasoning about quality.

- **Under a fixed WALL-CLOCK budget, every lever that defers progress fails identically — the schedule cannot extend to repay it** (EXP-002, EXP-005, EXP-007, EXP-009, EXP-010, EXP-014, EXP-018, EXP-020, EXP-030, EXP-066)
  Evidence: reports/exp-report-010.md § Results — capacity (starved), hot LR, heavy regularization all lost via the same mechanism; exp-report-014.md — warmup-shortening is a heat increase under time-keyed schedules (−0.22pp); exp-report-018.md — even an INIT trick (zero-γ identity-at-init) is deferral: blocks turned on during peak heat, ep5 35% vs 64%, −0.99pp; exp-report-020.md — a reference-faithful shortcut modernization (WRN option B) paid early heat + 4 epochs, −0.13pp; exp-report-030.md — a dt-free, +2,560-param max-pool head lost −0.91 purely on optimization QUALITY (argmax-routed gradients dragging the whole high-LR phase, still climbing at cutoff)
  Implication: evaluate any proposed change by WHEN its payoff arrives AND whether it degrades per-step optimization quality at heat; external results benchmarked at fixed iterations/epochs do not transfer to fixed-time training — even from regime-matched references (WRN EXP-020; cifar10-fast head EXP-030; and EXP-031's progressive resizing, the first WALL-CLOCK-regime-matched evidence, also failed: dataset-scale mismatch — ImageNet downsizing discards redundancy, CIFAR 24px discards signal; external transfer now 0-for-11). A change must be free in early heat, epochs, AND gradient quality. 10th confirmation, 7th mechanism class (capacity, schedule, regularization, init, topology, gradient path, and receptive-field geometry — EXP-066's dt-free 5x5 stem read mean − 2.7σ still-creeping-at-cutoff: coarse first-layer filters organize slower through peak heat).

### Medium Importance

- **A standing baseline recorded once on a max-statistic is selection-biased HIGH in its own run distribution — measure run-level σ with zero-diff replicates before interpreting near-baseline results** (EXP-027)
  Evidence: reports/exp-report-027.md — baseline replicates {96.71 standing, 96.59, 96.40}: recorded value sits +0.9σ above the config's true mean; five prior "misses" at −0.05…−0.15 were within 1σ (no detectable effect, not loss)
  Implication: on any goal whose baseline was adopted from a winning run, the de-facto bar is (bar − mean) ≈ bar + selection offset; buy the σ measurement early (2 replicate runs) and demand candidate effect sizes ≥ ~2σ; never cite noise-band deficits as mechanism evidence.

- **Best-over-checkpoints metrics REWARD a long converged plateau: smoothing (EMA/SWA) cannot raise it, final-value schedules starve it, TRANSIT-SPEED gains decay to zero before it — and schedule-REPLACEMENT by averaging never reaches it at short horizons** (EXP-011, EXP-016, EXP-028, EXP-031, EXP-032, EXP-062)
  Evidence: reports/exp-report-011.md — EMA lost 0.25pp while improving test loss; exp-report-016.md — linear anneal at IDENTICAL total heat lost 0.50pp, best at final epoch; exp-report-028.md — Muon ran +7pp ahead at ep10 yet plateaued at the baseline mean in a worse basin; exp-report-031.md — progressive resizing's +46 cheap low-res epochs converted at ZERO; exp-report-032.md — canonical SWA with correct augmented-loader BN re-estimation tied the baseline mean (96.60) with strictly better test loss: an annealed time-keyed schedule already performs the equivalent iterate averaging implicitly, and under fixed wall clock the SWA phase is carved OUT of the anneal, not added after it; exp-report-062.md — the converse measured: Schedule-Free SGD (NeurIPS 2024, claims at-or-above EVERY cosine horizon) ran a perfectly smooth monotone x-curve still climbing at ep139, 94.87 (−1.84) — averaging hot iterates cancels noise but does not do the anneal's late-phase basin refinement
  Implication: for any max/min-over-checkpoints goal, judge interventions by the converged-plateau LENGTH and LEVEL they produce, not by final-iterate theory or early-trajectory speed; literature calibrated on final-epoch or time-to-threshold comparisons does not transfer to max-statistics; "any-horizon" schedule claims must be checked against the horizon's position on the averaging catch-up curve; step-count conversion laws are conditional on unchanged data distribution; and solution-smoothing improves calibration (loss), not argmax decisions — accuracy ceilings are decision-boundary-limited.

- **Throughput converts to metric gains only under NUMERICS EQUIVALENCE: faster-but-different arithmetic (autotuned kernels, cudagraphs, fused optimizers) can cost more trajectory quality than the gained steps repay** (EXP-021)
  Evidence: reports/exp-report-021.md § Results — max-autotune+fused delivered dt −6.7% / +10 epochs at byte-identical hyperparameters yet converged −0.20pp, replicated twice; mid-run deficit ~0.4pp predates convergence
  Implication: a recipe tuned under one execution regime is tuned TO that regime's arithmetic; when chasing speed on any fixed-budget goal, verify the faster path is numerically equivalent (same kernel family/reduction orders) or re-validate quality — never assume "same math, faster" from API equivalence.

- **A tuned short-budget recipe sits at a gradient-noise OPTIMUM (bracketed both directions): noise-reducing changes lose (−0.25/−0.14/−0.30) AND noise-increasing loses (−0.22), all free in heat/epochs/numerics** (EXP-011, EXP-022, EXP-023, EXP-024)
  Evidence: reports/exp-report-024.md § Results — compensated momentum trades in both directions had byte-identical signatures to baseline, isolating the noise variable; deficit roughly symmetric around β=0.9
  Implication: treat gradient-noise scale as a first-class tuned quantity like LR; once a recipe is tuned, moving its noise in EITHER direction by any mechanism (averaging, batch, momentum horizon) is a measured loss — screen proposals for their noise sign and reject nonzero ones.

- **Techniques published under LIGHTER augmentation do not transfer into heavy-augmentation recipes — the augmentation absorbs their role (inert at best, deficit-costly at worst); holds for regularizers, capacity-conditioning modules, zero-cost architectural fixes, AND augmentation-type swaps** (EXP-035, EXP-036, EXP-037, EXP-046, EXP-060)
  Evidence: reports/exp-report-035.md — periodic SAM (canonical CIFAR ρ) returned exactly its epoch deficit, zero retained flatness gain under TA+RE+LS; exp-report-036.md — doubling label smoothing to the cifar10-fast/airbench anchor value read a precise null (96.58 = baseline mean) at byte-identical signatures; exp-report-037.md — SE channel attention (published +0.5–1.2 on crop+flip CIFAR ResNets), deferral-proof init verified and dt only +1.7ms, returned exactly mean − deficit in two clean draws; exp-report-046.md — anti-aliased shortcut (ResNet-D/Zhang-2019 mechanism, +0.5–1.0 published), the first candidate free in EVERY currency (zero params, zero dt at 22.4ms/139 ep, zero noise), still nulled at mean+0.5σ with family test_loss; all anchors/baselines use far lighter augmentation than TA+RE
  Implication: absorption is NOT cost-accounting — even toll-free techniques null, so freedom from deferral/dt/noise costs is necessary but NOT sufficient. When importing ANY technique from an anchor recipe or fixed-epoch paper, require the AUGMENTATION regime to match, not just dataset/architecture/schedule; candidates must additionally argue a mechanism the heavy-aug ensemble cannot itself supply. External transfer record now 0-for-17 (EXP-060: CutMix, the largest published prior tested — +0.5–1.0 — read a precise family null when SUBSTITUTED at matched dose, extending absorption to augmentation type itself).

- **Batch-driven throughput under fixed wall-clock cannot be rescued by LR scaling rules: at 2× batch BOTH linear and √-scaled peaks converged below baseline** (EXP-012, EXP-022)
  Evidence: reports/exp-report-022.md § Results — 1024-LR axis bracketed (√ 0.566 → 96.57, linear 0.8 → 96.66, baseline 96.71) with +12 epochs delivered both times; √ ran smoother yet worse
  Implication: extra steps convert only at UNCHANGED hyperparameters; a speedup that forces any hyperparameter move pays a trade that measured net-negative here — prefer speedups invisible to the optimizer (and numerically equivalent, see EXP-021 entry).

- **Throughput scaling exponents do not transfer across execution regimes: eager sublinear (~0.76) became ~linear-or-worse under torch.compile** (EXP-007)
  Evidence: reports/exp-report-007.md § Results — compiled 6x cost 2.59x time for 2.25x FLOPs; eager EXP-002 measured 2.85x for 4x FLOPs
  Implication: never project runtime for a new config from measurements taken under a different compiler/precision/layout regime — spot-measure dt in the target regime first (a few steps suffice).

- **A tuned config can be CERTIFIED a single-change local optimum by bracketing: probe each constant in both directions until every neighbor measures worse with a consistent mechanism** (EXP-007…EXP-015)
  Evidence: reports/exp-report-015.md § Analysis — ten clean converged single-constant probes (pressure ±, heat ±, capacity ±, batch, smoothing) all below baseline on maximize-cifar10-test-accuracy
  Implication: once bracketed, stop knob-tuning — further gains require multi-constant trades (hold one quantity, trade others) or structural changes (different schedule family/optimizer/architecture shape); single-knob retries just re-measure the optimum.

- **Pointwise ops are NOT free at small-CNN step times — and neither is op ORDER: special-function activations cost 5–20% of a 22.4ms step even fused (erf +4.5ms, sigmoid +2.8, piecewise +1.1), and reordering an IDENTICAL op multiset (post-act → pre-act) cost +1.0ms (4.5%)** (EXP-026, EXP-056)
  Evidence: reports/exp-report-026.md § Results — three-activation cost ladder via the early-dt gate; exp-report-056.md — pre-activation reorder at identical ops/FLOPs/params measured 23.08 vs 22.04ms in a controlled GPU probe (inductor fuses conv→BN→ReLU epilogues better than BN→ReLU→conv prologues)
  Implication: on any fixed-wall-clock goal, price EVERY graph change — op substitutions AND reorderings — in measured dt before reasoning about quality; "same ops, same FLOPs" does not imply same step time; a ~90s compile+time probe is sufficient instrumentation.

- **BN running statistics are PART of the learned function — they must come from the TRAINING distribution (EXP-029) and the default m=0.1 freshness is the measured TWO-SIDED optimum of the lag/variance trade (EXP-038/039)** (EXP-029, EXP-038, EXP-039)
  Evidence: reports/exp-report-029.md § Results — recalibrating stats on clean (test-transform) data cost −10.93 with training provably untouched; exp-report-038.md — smoothing the EMA horizon 10→50 batches cost −0.3 with 4× plateau scatter (even the "converged" plateau drifts, so stale constants misalign exactly where a max-statistic harvests); exp-report-039.md — freshening 10→4 batches read flat-mean with 3× scatter and ZERO hot-phase gain: a 10-batch EMA already tracks even hot-phase weight drift
  Implication: on any BN network, treat normalization constants as calibrated parameters with two requirements — train-distribution sourced and default-recent; the error terms (lag vs estimator variance) cross at m=0.1, eval must reproduce training-time constants, and estimator intuitions from static-weights settings fail in BOTH directions while weights drift.

- **Conv kernels on this box (H20 + torch.compile default + channels_last + bf16) are fast ONLY at power-of-2 widths {64,128,256}: off-lattice widths land on flat slow tiers (~33ms below 256-max, 54ms above), grouped convs 2.8× dense — but square kernel SIZE is fast-path and FLOPs-priced** (EXP-040, EXP-042, EXP-044, EXP-045, EXP-066)
  Evidence: reports/exp-report-040.md — 288 AND 320 both 54.0ms vs 256's 22.4ms (flat tier); exp-report-042.md — groups=2 at 63.0ms for 1.125× FLOPs; exp-report-044.md — 160-wide stage 32.4ms vs 24.8 dense-law; exp-report-045.md — 192 (= 3×64, refuting the "64-multiple" hypothesis) at 33.4ms despite +40% vs +18% FLOPs — off-lattice dt is FLAT, FLOPs are second-order; retro-consistent with EXP-005 (80/160/320) and EXP-034 (48/96/192, 27.4ms); exp-report-066.md — square 5x5 at lattice widths prices within 2.4% of the dense law (31.02 P_norm vs 30.3 predicted at 1.59× FLOPs): kernel-SHAPE slowness is dimensionality (1D/odd), not size
  Implication: the dense pricing laws (13.3ms/FLOPs-unit, 2.5ms/block) hold ONLY at power-of-2 channel counts ≤256 with dense SQUARE kernels (any size); off-lattice widths and 1D shapes fall to one slow implementation per regime. Architecture search on fixed hardware is search over the lattice {64,128,256}, and any shape/kernel-family change must be dt-gated (~90s) before reasoning about statistical value — the gate probes charted the full envelope at ~2 GPU-min each.

- **Ensembling under a FIXED training budget cannot pay — now closed in ALL THREE funding currencies: steps (dilution −0.9 vs gain +0.3–0.5), kernel width (2.8× dt), and concurrency (two CUDA streams serialize, ratio 1.81)** (EXP-043, with EXP-011/032/042; EXP-063)
  Evidence: reports/exp-report-043.md § Results — full-diversity endpoint 96.07 (gain visible over starvation-priced members yet far sub-mean); zero-diversity endpoint = SWA/EMA at mean with zero gain (EXP-011/032); in-one-pass implementations 2.8× kernel-closed (EXP-042); interior mid-forks interpolate below the single-model mean + bar offset; reports/exp-report-063.md — stream-parallel two-member probe at clean conditions read P2 = 40.7ms vs P1 = 22.5 (ratio 1.810 compiled, 1.820 eager — identical, so not a compile artifact): the ~93% idle compute is unreachable because the binding resource for latency-bound small-kernel models is the SERIAL kernel-dispatch chain, shared by all streams in a process
  Implication: on any fixed-compute goal, multi-model/multi-member proposals must show their diversity gain exceeds the per-member compute dilution BEFORE running — for small ensembles on tuned recipes it measured 2–3× short; prefer one fully-trained model per budget. "Idle GPU capacity" arguments must name the binding resource: compute-idleness ≠ spare dispatch capacity, and stream-level concurrency buys nothing for dispatch-chain-bound models.

- **Screen any technique by WHERE its cost lands relative to the metered resource: "free at inference" structural tricks (reparameterization, branch folding) concentrate their entire price on TRAIN time — exactly the budget a train-time-metered goal cannot pay** (EXP-064, EXP-068)
  Evidence: reports/exp-report-064.md — ACNet branches tolled 1.93× step time (control-validated probe, required gain 1.19 > published max 1.11); the minimum possible variant (one 1x1+BN branch) still cost +28% step time vs a sub-0.5 published ablation gain; both priced at kernel launches with 1D convs landing on slow odd-shape implementations; reports/exp-report-068.md — with a dense enough anchor ledger the inequality needs NO new probe: three family closures (sequential KD, resolution-up, optimizer swaps) for zero GPU seconds, all bounds sub-bar under candidate-favoring arithmetic
  Implication: papers advertise costs at THEIR metered resource (usually inference FLOPs); re-derive the cost at YOURS before planning. Probe-gated pre-registered inequalities convert these mispriced candidates into zero-charged-cost closures (EXP-063/064: two family closures for ~4 GPU-minutes), and mature ledgers make them probe-free (EXP-068). Use an internal baseline control inside any probe so the reading self-validates against load.

- **Fine-tune-class techniques (FixRes, end-on-test-distribution, augmentation taper, FreezeOut) presuppose a CONVERGED model; under a time budget the schedule tail IS the convergence phase — repurposing it loses even when its known failure mode is fixed, and the law is TWO-SIDED in pressures AND now BOTH-ENDED in time: neither the data distribution nor any parameter subset may deviate from full pressure at ANY phase (head-side curriculum lightening also measured negative — EXP-065: light warmup trained faster yet finished mean −1.2σ; easy→hard ordering refuted alongside hard→easy)** (EXP-025, EXP-033, EXP-055, EXP-065)
  Evidence: reports/exp-report-025.md § Results — clean-data tail gave a one-time +0.35 alignment boost then overfit-flatlined, −0.87; reports/exp-report-033.md § Results — crop+flip tail fixed the overfit (test_loss flat) yet learning froze, banking +0.48 alignment but forfeiting the anneal's ~+0.4 endgame, −0.46; tail-pressure response is monotone (full/light/zero = 96.6/96.2/collapse); reports/exp-report-055.md § Results — the PARAMETER-side mirror: freezing stem+layer1 at p=0.70 after a completed compressed anneal delivered a fully-instrumented 31% step-time saving (+1,550 tail steps, +16 plateau evals) and still lost −1.6σ — tail refinement of layers holding just 5.2% of params outweighs a third of the network's compute converted to extra steps
  Implication: screen literature techniques by what training state they assume — "after training, then X" results need a converged checkpoint that fixed-budget runs never produce; the tail's training pressure is load-bearing to the LAST step in BOTH currencies (data and weights); fixed-epoch "freeze is free" results invert under fixed time even when the freed compute is fully recycled into steps.

### Low Importance

- **Equal-FLOPs capacity reallocation is not free: a faster, bigger [2,3,4] depth variant lost to uniform [3,3,3] — high-resolution early-stage blocks carry irreplaceable representation at shallow depth** (EXP-017)
  Evidence: reports/exp-report-017.md § Results — +26% params, +5 epochs, −0.9ms/step, −185MB VRAM, yet −0.28pp with a fully converged plateau
  Implication: params/FLOPs are not fungible capacity — allocation POSITION is a first-class variable; population-level allocation results (RegNet) measured at other total depths do not transfer downward.
