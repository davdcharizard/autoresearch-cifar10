# Brainstorm EXP-030
**Created**: 2026-08-06

## Web Search & Literature Review

- **SGDR** (`knowledge/papers/sgdr.md`; ICLR 2017)
  Cosine schedules improve CIFAR anytime performance, but their useful starting amplitude and horizon are empirical. The accepted elapsed-time schedule has not tuned its abrupt weak-tail start LR.
- **CutMix** (`knowledge/papers/cutmix.md`; ICCV 2019)
  Regional donor pixels and area-weighted targets improve CIFAR generalization. Local experiments validate the method but show probability 0.75 over-regularizes, leaving a plausible lower-probability operating point.
- **Population Based Augmentation** (PMLR 97, 2019: https://proceedings.mlr.press/v97/ho19b.html)
  Phase-varying augmentation policies can outperform static ones, supporting schedule-aware data strength; direct population search is infeasible here, so any local phase rule must be a single preregistered point.
- **PyTorch channels-last** (`knowledge/references/pytorch-channels-last.md`)
  Layout remains an untested backward-speed lever, but EXP029 shows that adding even 1.97% per-step work is enough to violate the exposure gate and EXP013 leaves exposure-to-accuracy causality unresolved.

## Experimental History Review

- EXP010's 94.15% recipe remains the frontier. Width 2, standard momentum, all-parameter decay `1e-4`, N1/M7 plus p=0.5 alpha-1 CutMix through a simultaneous 80% LR/data transition, and a weak hard tail are protected as the default composition.
- EXP005 moved only augmentation to weak at 75% while keeping LR 0.1 and lost 0.18; EXP027 removed only CutMix at 70% and collapsed fit. Neither tested moving the shared LR/data boundary earlier, so a 75% simultaneous transition remains distinct.
- EXP011's p=0.75 CutMix lowered switch fit by 2.91 and lost 0.15. The response shows excess mixed-target frequency is harmful but does not establish whether p=0.4 can retain most regional regularization with slightly better fit.
- EXP012/026 peaked at 94.22, only 0.03 below the gate, through interventions that deepened strong underfit but recovered in the tail. The frontier may respond to a small reallocation of time or regularization strength rather than another global representation change.
- Global optimizer paths now have three class-collapse failures, while literal all-Conv GC was safe but lost 1.97% exposure. Favor zero/near-zero-overhead constants and worker-side data rules; retain exact-corpus safety for any changed target policy.

## Collected Ideas

- **75% simultaneous LR/data transition** — Set the single existing `LR_HOLD_FRACTION` to 0.75 so N1/M7+CutMix and LR 0.1 end together, followed by 25% weak hard-label cosine refinement. It targets the tail recovery gap without reproducing EXP005's harmful weak-at-high-LR interval.
- **Weak-tail start LR 0.02** — Preserve the complete accepted 80% strong phase and raise only `ANNEAL_START_LR` from 0.01 to 0.02 before cosine decay to `1e-4`. It gives the weak tail more early optimization amplitude at zero overhead while keeping the final convergence target.
- **CutMix probability 0.40** — Preserve alpha 1 and the complete timing/schedule, but reduce mixed strong batches from half to 40%. It interpolates away from EXP011's over-regularized p=0.75 side and may improve switch fit without abandoning the regional regularizer that added 0.60 points.
- **CutMix alpha 0.5** — Keep p=0.5 but make the beta distribution favor smaller/larger pasted regions rather than half-image composites. This changes geometry rather than frequency and is worker-side, but the direction of regularization is less directly supported by local results.
- **77.5% simultaneous boundary** — A midpoint between accepted 80% and the distinct 75% hypothesis might better balance exploration/refinement. It is scientifically weaker than testing a round preregistered 75% point and smells like interpolation before either endpoint is measured.
- **Weak-tail start LR 0.005** — A smaller abrupt drop could emphasize stable convergence/calibration. The tail already recovers rapidly and ends near its best, so reduced amplitude is more likely to underuse the short horizon.
- **TrivialAugmentWide instead of N1/M7** — Swap the worker-side policy for one random operation with broad magnitudes, importing a low-tuning augmentation idea at no GPU cost. It discards a locally validated policy and risks host throughput/strong underfit without direct local evidence.
- **Fixed 19-look channels-last** — Test whether NHWC gives a genuine FP32 speedup while holding evaluation opportunities constant. It attacks the measured backward bottleneck, but even a systems pass still lacks a direct generalization mechanism.
- **Moonshot scheduled CutMix probability decay within the strong phase** — Keep p=0.5 early, then linearly decay to p=0.25 before the 80% transition. It may reconcile exploration and fit, but adds a new policy family and persistent-worker control problem before either static weaker point is known.

## Combinations

- **75% simultaneous boundary + LR 0.02 tail start**: More weak-tail time plus higher initial refinement amplitude could accelerate recovery, but both alter tail optimization and a miss would be uninterpretable. Test each scalar alone first.
- **p=0.40 CutMix + 75% boundary**: Slightly weaker mixed-target frequency and longer hard refinement may preserve fit twice over, but likely gives up too much regularization and confounds frequency with horizon.
- **Channels-last + scalar schedule change**: Layout could add exposure while the scalar supplies an accuracy mechanism, but changed kernel numerics and two simultaneous effects make attribution worse than either isolated test.

## Candidate Ideas

### Move the Coupled Strong/LR Boundary to 75%
**Summary**: Change only `LR_HOLD_FRACTION` from 0.80 to 0.75. The existing single constant moves the LR drop, strong-loader exit, weak/hard loader rebuild, and dense-tail evaluation together, reallocating 15 counted seconds from strong high-LR learning to low-LR weak refinement. Full specification: `proposals/idea-01.md`.

**What it targets**: The accepted run finishes at its best after rapid weak-tail recovery. This candidate tests whether the CutMix representation is mature by 75% and whether roughly 1.3k additional hard weak updates improve conversion and NLL.

**Reasoning**: It is distinct from EXP005's weak-at-LR-0.1 interval and EXP027's hard-N1/M7 bridge because all coupled sources of difficulty still change simultaneously. It has zero per-step overhead and preserves the tested tail endpoints, but challenges the locally protected long-exploration horizon.

**Sources**: EXP002, EXP005, EXP010, EXP027; `02-system-understanding.md`; SGDR/PBA notes above.

**Estimated Effort**: low.

**Risk Assessment**: Medium-high scientific risk. Removing about 15 seconds of accepted strong exploration may leave invariance learning incomplete, and earlier dense evaluations add wall time. Exact boundary semantics and target provenance remain essential.

### Reduce CutMix Probability to 0.40
**Summary**: Change only `CUTMIX_PROBABILITY` from 0.50 to 0.40 while preserving alpha 1, N1/M7, and the complete 80% synchronized schedule. Full specification: `proposals/idea-03.md`.

**What it targets**: Strong-phase fit versus regional regularization. It keeps 80% of accepted mixed-batch exposure while making 10% more strong batches hard-label examples.

**Reasoning**: EXP010 proves p=0.50 is valuable and EXP011 shows p=0.75 over-regularizes. A slightly lower point may restore fit without creating EXP027's contiguous hard-label bridge, but there is no evidence the response is monotone below 0.50 and the accepted point may already be optimal.

**Sources**: CutMix paper; EXP010, EXP011, EXP012, EXP026, EXP027; `02-system-understanding.md`.

**Estimated Effort**: medium-high because exact-corpus policy validation and fresh paired timing remain mandatory.

**Risk Assessment**: Medium-high. The candidate directly weakens the intervention responsible for the frontier's 0.60-point gain, and its likely effect is close to single-seed resolution.

### Raise Only the Weak-Tail Start LR to 0.02
**Summary**: Change only `ANNEAL_START_LR` from 0.01 to 0.02, retaining the full 80% accepted strong phase and the same `1e-4` cosine endpoint. Full specification: `proposals/idea-02.md`.

**What it targets**: Tail adaptation amplitude. It nearly doubles integrated LR over the 60-second weak hard-label tail so more useful parameter motion can occur before the terminal LR becomes tiny.

**Reasoning**: EXP010's best occurs at the end and EXP012/026 recover strongly in the weak tail, while no local run tuned the abrupt tail start independently. The candidate preserves every protected data/target boundary and adds no work, though its first weak update and effective coupled-decay displacement are about twice the accepted values.

**Sources**: EXP002, EXP010, EXP012, EXP026; `02-system-understanding.md`; SGDR notes above.

**Estimated Effort**: low.

**Risk Assessment**: Medium. The inherited strong-phase momentum may overshoot at the data/target transition, and the larger mean LR leaves less effective time in the low-LR basin. The one-line intervention nevertheless has unusually clean attribution.

## Review

Claude's independent adversarial review (`01-idea-review.md`) selected **Raise Only the Weak-Tail Start LR to 0.02**, scoring its evidence/reasoning 4/5 and potential impact 3/5. The coupled 75% boundary scored 2.5/5 and 3/5 because it removes strong exploration against the strongest promoted local pattern and also advances dense evaluation, mildly biasing a maximum-over-checkpoints metric. CutMix p=0.40 scored 2/5 on both axes because evidence only constrains the harmful upper side of p=0.50, the candidate weakens the method responsible for the frontier's largest gain, and its validation cost is high for a likely sub-resolution effect.

I adopt the reviewer's core argument. The 0.02 tail start uniquely leaves every historically protected structure intact: the complete 80% strong phase, synchronized data/target transition, augmentation policy, CutMix geometry/frequency, ordinary momentum, and evaluation cadence. It tests an unmeasured scalar with a clean one-line intervention and no exposure cost.

Two reviewer caveats become explicit interpretation constraints. First, EXP010 ending at its best is directionally ambiguous: it may indicate insufficient tail motion or an already well-tuned schedule. The preflight must therefore verify the exact twofold first-weak displacement, and production must compare the first weak checkpoint with EXP010's 93.16% to diagnose overshoot. Second, larger LR also increases the effective coupled-decay displacement; any gain is the net schedule effect, not proof that loss-gradient amplitude alone helped, and final NLL versus 0.1934 must be reported.

## Idea Evaluation

- **Raise only the weak-tail start LR to 0.02** — Advance. It preserves the complete accepted strong curriculum and changes only the amplitude of an untested refinement phase. Its null result remains cleanly informative, while boundary, cadence, and throughput stay fixed.
- **Move the coupled strong/LR boundary to 75%** — Reject for EXP030. Synchronization makes it distinct from EXP005/027, but it removes about 1.3k strong updates against the goal's strongest positive schedule evidence and adds evaluation opportunities to the primary maximum metric.
- **Reduce CutMix probability to 0.40** — Defer. EXP011 proves that more than p=0.50 is harmful but supplies no evidence that less is beneficial; this point weakens the proven +0.60 intervention and requires disproportionate corpus/timing validation.

## Chosen Idea
**Selected**: Raise Only the Weak-Tail Start LR to 0.02

**Why this idea**:
It is the only finalist that leaves the full accepted 80% data/target curriculum and evaluator cadence unchanged while probing a locally untuned, zero-overhead optimization amplitude. EXP010's terminal-best trajectory and EXP012/026's rapid weak-tail recovery provide a concrete—though directionally ambiguous—reason to test more refinement motion. The exact one-line scope gives better attribution than shortening strong exploration or weakening CutMix.

**Hypothesis**:
Changing only `ANNEAL_START_LR` from 0.01 to 0.02 will preserve EXP010-like strong-phase fit and exposure, accelerate useful adaptation after the 80% switch, and raise seed-42 `best_test_acc` from 94.15% to at least 94.25%. Point prediction is 94.30%. A valid lower result rejects this exact scalar; first-weak accuracy and final NLL will distinguish under-refinement from overshoot but cannot override the primary verdict.
