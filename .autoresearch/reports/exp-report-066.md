# Report EXP-066: Kernel-size corner — 5x5 stem behind the internal-control GPU probe
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-066.md
- **Plan**: plans/plan-066.md
- **Log**: logs/exp-log-066.md

## Goal
Maximize best_test_acc (%) of the ResNet-20-family CIFAR-10 model within the fixed 300s charged
training budget by modernizing train.py only. Baseline 96.71 @ 1990397 (run-distribution top;
family mean ≈ 96.57, σ ≈ 0.16); bar = 96.81 (+0.1 absolute).

## Idea & Hypothesis
This loop's brainstorm ran the pre-registered adversarial audit of the measured-ceiling
hypothesis. The audit (a) closed the one apparent recipe-constant gap by inspection — Normalize
std=(1,1,1) is pinned by prepare.py's eval transform, and the only train-side-expressible
residue (RandomErasing fill amplitude) is covered by EXP-060's fill-content indifference — and
(b) identified kernel SIZE as the last unpriced structural dimension: the lattice law
(EXP-040/042/044/045) prices channel widths and 1D/odd kernel shapes, but square 5x5 dense
kernels had neither a dt price nor a level measurement. Chosen experiment: probe-gate two graded
variants (stem-only 3→64 5x5, +3,072 params, ~zero FLOPs; stage-3 5x5, +59% net FLOPs,
probe-only by pre-run arithmetic) with the EXP-064 internal-control probe; launch the stem
variant only if its normalized toll ≤ +0.5ms. Hypothesis: 5x5 lands off the fast path (full
NO-LAUNCH closure); if the stem is free, its run reads a family-band absorption null, with a
replicated ≥-bar read as the ceiling-falsification branch.

## Approach
Milestone 1 sanity-validated both variants (exact param counts 4,289,098 / 10,053,194; spatial
preservation; loss decrease). Milestone 2 probe (B/S/T compiled charged steps, one session,
P_norm = 22.4×P/B): session valid at B = 22.18ms. The launch branch fired for the stem
(P_norm_S = 22.66 ≤ 22.9). Implementation was the planned single line in `ResNet.__init__`:
`nn.Conv2d(3, w1, 5, stride=1, padding=2, bias=False)`. No other changes; no deviations.

## Execution
Two-stage execution, both clean. Probe: gates at load 7.5–10, zero GPU-0 apps; B=22.18,
S=22.44, T=30.72ms. Full run via the exp046-standard composite (gates cleared poll 1, load 8):
D0 = 22.7ms exactly matching the probe, all 30 watchdog windows 22.0–23.3ms, zero slow streaks,
rc=0, 480.5s total, no retries, no errors.

## Results
- **Primary metric**: 96.14 (baseline: 96.71, delta: −0.57, −0.59%)
- **Observations**:
  - **Pricing half (probe)**: square 5x5 kernels are FAST-PATH — T's 31.02ms P_norm sits within
    2.4% of the dense-law prediction (22.4 + 0.59×13.3 ≈ 30.3ms for 1.59× FLOPs). Kernel SIZE
    prices at FLOPs, unlike kernel SHAPE (1D/odd → flat slow tiers, EXP-044/045) and unlike
    off-lattice widths. The stem variant was effectively free (+0.26ms ≈ −1 epoch).
  - **Level half (run)**: with the cost variable removed, the stem-5x5 read 96.14 = mean − 2.7σ
    at byte-clean signatures (13,266 steps, 137 ep, VRAM 1,613, test_loss 0.1929 vs family
    ~0.185) — a REAL structural negative in the EXP-030/047 class, far below the absorbed-null
    band the absorption law predicted.
  - The plateau was still creeping at cutoff (last 8 evals 96.05→96.14, best AT the final
    epoch) — the still-organizing signature (EXP-030 rhyme), indicating a per-step optimization
    QUALITY drag that lasted the whole run, not a capacity or schedule effect.
- **Analysis**: The hypothesis was half right and half wrong in an informative way. Wrong on
  cost: 5x5 is not slow-tier — the H20/inductor fast path covers square kernels, so the
  kernel-size corner closes on LEVEL, not cost. Wrong on level direction: not an absorbed null
  but an active negative. Mechanism reading: at 32×32 input, a 75-weight stem filter spans
  ~2.4% of the image and averages over 25 pixels — coarser, harder-to-organize first-layer
  features whose poor early organization propagates through all 19 downstream convs during peak
  heat (deferral-law family, 10th confirmation, now including receptive-field geometry). The
  stem's 3x3 kernel is load-bearing, completing the stem triangulation: init content (EXP-019
  washout), zero-γ (EXP-018 deferral), and now kernel geometry (−0.43 vs mean). The
  measured-ceiling hypothesis survives its designated falsification attempt: the audit's last
  unpriced corner measures NEGATIVE, strengthening the claim that the baseline is the optimum
  of its reachable space.
- **Key Learning**: Square 5x5 convs are FLOPs-priced fast-path on this stack (the kernel-shape
  law is about dimensionality, not size), but the 5x5 stem loses −0.43 at zero cost — early
  spatial resolution is load-bearing representation, and the ceiling audit's final corner
  closes negative.

## Verification
- **Conditions**: Integrity pre-condition PASS (family step ledger 13,266; exact params;
  pristine windows; D0 = probe reading). Condition 1 FAIL: 96.14 < 96.81 — pre-registered
  branch (iii) (below family floor 96.41). Conditions 2–3 informationally pass (480.5s; one
  eval/epoch).
- **Review Notes**: results confirmed trustworthy — probe-to-run dt agreement (22.66 vs 22.7),
  byte-clean telemetry, and the deficit (−2.7σ) is far outside the noise band; no false-failure
  mechanism plausible.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid measured result below the bar; hard constraints
  all respected — single train.py line, GPU 0, ≤600s, once-per-epoch eval).

## Unexplored Avenues
- 7x7 or larger stems, dilated stems: dominated — same coarsening mechanism at stronger dose;
  the measured −0.43 at 5x5 makes them strictly worse bets. Do not retry.
- 5x5 elsewhere (stage 1/2 interiors): stage-3 is starvation-priced (31.02ms); stage-1/2 5x5
  would carry both partial FLOPs tolls AND the same coarsening risk with no independent gain
  mechanism. Closed by combination of this run's level read and the dense-law pricing.
- Mixed stems (parallel 3x3+5x5 branches): multi-branch convs are family-closed on cost
  (EXP-064 reparam closure — branch toll superlinear in branch count).
- The probe datum opens nothing new: fast-path square kernels only matter if a kernel-size
  change had a positive level mechanism somewhere — this run measured the best-positioned one
  (free, stem) at −2.7σ.

## Next Steps
1. The measured-ceiling hypothesis now has audit-complete support (recipe constants, structural
   classes, funding currencies, pressure profiles, and the kernel-size corner all measured or
   closed by inspection). The next brainstorm's honest options are down to instrument
   investment: the σ-tightening replicate pair (pre-registered no-improvement, pools σ to n≈5
   and re-anchors the family signature ledger) — confidence medium that it is the
   highest-information remaining spend.
2. Periodic fresh-literature sweep restricted to the double screen (heavy-aug budget-matched
   evidence AND cost landing off the charged step) — nothing survived this loop's sweep;
   re-check occasionally as 2026 work appears. Confidence low.
3. If the loop must keep attempting metric movement: the only nonzero-prior territory left is
   compositions never priced (e.g., a NEW component with a replicated ≥+0.1 estimate is a
   prerequisite per EXP-053 — none exists). Treat any future candidate to the pre-run
   inequality + probe gate before charging budget. Confidence low.

## Exit Action Results
