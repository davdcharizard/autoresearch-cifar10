# Brainstorm EXP-066
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

Primary task this loop (per reports/exp-report-065.md § Next Steps): adversarial audit of the
measured-ceiling hypothesis, then a literature sweep restricted to mechanisms surviving BOTH
standing screens (absorption: heavy-aug budget-matched evidence required; cost-landing: the
technique's price must not land on charged train time — project-insights EXP-064 entry).

- **cifar10-airbench current state** (https://github.com/KellerJordan/cifar10-airbench; arXiv 2404.00498)
  airbench96 element-by-element vs our closure record: Cutout dose (occlusion dose/type closed,
  EXP-003/009/060), derandomized flip (EXP-041), Muon (EXP-028), dirac/identity init (init axis
  closed both directions, EXP-018/019), Lookahead/EMA (averaging closed, EXP-011/032/062),
  GPU-resident data + step de-overheading (EXP-048: charged step 99.3% kernel math), wider/deeper
  blocks (lattice + depth closures), **multi-crop TTA (their 96% figures include TTA — banned here:
  Eval.evaluate() is the fixed ground truth)**, data filtering (see Candidate 2). Nothing new
  survives.
- **Augmentation multiplicity** (arXiv 2105.13343 — "Drawing Multiple Augmentation Samples Per
  Image... Decreases Test Error")
  Multiple augs of the same image per batch at fixed batch size = correlated samples = gradient-noise
  DOWN; at grown batch = batch axis. Both directions closed (EXP-011/022/023/024 noise bracket;
  EXP-012/022/059 batch level+schedule). Light-aug ImageNet evidence; fails both screens. Rejected
  in ideation.
- **"How Important is Importance Sampling for Deep Budgeted Training?"** (arXiv 2110.14283)
  Finding: under tight budgets uniform sampling is hard to beat; augmentation dominates selection.
  Independent external support for the measured-ceiling reading; also pre-refutes dynamic
  example-selection candidates (consistent with EXP-051's −7.8σ confidence-weighting result).

## Experimental History Review

67 experiments; baseline 96.71 @ 1990397 (distribution top; true mean ≈ 96.57, σ ≈ 0.16; bar
96.81 = mean + 1.5σ; effect-size screen ≥ +0.3). Last improvement EXP-006. External transfer
0-for-20. Closure record (goal-learnings + project-insights, all measured): heat both ways,
schedule family-level, noise level+schedule, momentum both ways, reg dose+type+stacking, loss
axis both ways (margin/weighting/fc-WD bracket), init both directions, activations (cost), BN
momentum + stat sourcing, eval-side averaging (EMA/SWA), ensembles in all three funding
currencies, reparam family (cost), step-time engineering (99.3% kernel floor), precision both
ways, pressure-profile four-quadrant, data order, resolution, shortcut topology ×2, head/routing
from below, pre-act order, attention, Muon, SAM, FreezeOut, depth/width/allocation
both ways on the unique fast lattice {64,128,256}.

**Adversarial ceiling-audit results (this loop's primary task — line-by-line re-read of train.py
and prepare.py for unpriced degrees of freedom):**

- **Normalize std=(1,1,1) — the one constant never dosed — is closed BY INSPECTION**: prepare.py
  L13 pins eval inputs to `Normalize(mean, (1,1,1))`; any train-side std change creates a
  train/eval input-distribution mismatch (EXP-029 mechanism, −10.9). The only
  train-side-expressible residue — RandomErasing's N(0,1) fill being ~4× brighter than the
  std-1 image statistics — is covered by EXP-060: signal-fill (perfectly in-distribution content)
  vs noise-fill read identical family nulls; the network is indifferent to fill statistics.
  Zero-cost closure; the recipe-constant audit is now complete WITH this corner.
- Stochastic depth: mask-form = pure added pressure (dose law, peaked); skip-form = dynamic
  graph patterns under torch.compile (2^9 recompiles) — cost lands on charged time. Law-closed.
- Gradient clipping: buys stability headroom the recipe cannot spend (EXP-018 lesson; peak 0.4
  certified via quality, run is NaN-free). Law-closed.
- GhostBN / GroupNorm: BN-stat granularity noise = pressure (peaked axis); BN constants are
  load-bearing at default freshness (EXP-029/038/039); GN slower kernels. Law-closed.
- Crop-padding dose, TA-Wide→TA dose-down: peaked pressure axis (EXP-015/033). Law-closed.
- Per-stage/layer LR profiles: bracketed by heat law (up) + freeze law EXP-055 (down) + fc-LR
  rejection (EXP-057/058 logic). Law-closed.
- Weight-tied recurrent depth, bottleneck/ResNeXt/ConvNeXt blocks: +launches/+FLOPs on a
  launch-bound box (2.5ms/block law; grouped 2.8×). Cost-closed.
- Shorter "epochs" to multiply eval draws: games the once-per-epoch ceiling for max-statistic
  harvesting — reward hacking, rejected.
- **Residual found: kernel SIZE is the last unpriced structural dimension.** The lattice law
  (EXP-040/042/044/045) prices channel widths and 1D/odd shapes; square 5x5 dense kernels at
  lattice widths were never probed — neither their dt (fast path or slow tier?) nor their level
  effect. Two graded variants exist: stem-only (27→75-weight conv, +3,072 params, ~zero FLOPs
  delta — plausibly free) and stage-3 (+59% net FLOPs — almost certainly starvation-priced).

## Candidate Ideas

### 1. Kernel-size corner: 5x5 stem (and stage-3 ladder) behind the internal-control GPU probe
**Summary**: Price the last unpriced structural dimension. GPU-probe (EXP-064 internal-control
pattern: baseline net timed in the same session, P_norm = 22.4×P/B) two variants: (a) stem conv
3→64 as 5x5/padding=2 (+3,072 params, FLOPs delta ~0.1%), (b) stage-3 convs 3x3→5x5 (+59% net
FLOPs; expected slow tier or starvation price). Pre-registered branches: any variant with
P_norm ≤ 22.9ms (toll ≤ +0.5ms ≈ 3 epochs) carries a launchable full run; stage-3 expected
NO-LAUNCH on cost; stem launch (if priced free) tests whether enlarged input receptive field
moves the plateau. Expected full-run outcome under the absorption law: family-band null
(96.41–96.73); a replicated read ≥ bar would falsify the measured-ceiling hypothesis at the
structural corner.

**Reasoning**: At measured ceiling, loops should maximize information per GPU-second. EXP-063/064
established that probe-gated pre-registered inequalities buy family-level closures at zero charged
cost (~2 GPU-min each). Kernel size is the ONLY structural dimension with neither a dt price nor
a level measurement; closing it completes the structural map begun by EXP-040–045. The launch
branch is honest: free-structural changes have nulled 5/5 times, so the gain prior is ~0, but
this is the one corner where that null is inferred (absorption law) rather than measured.

**Sources**: goal-learnings § Failed Approaches (lattice law EXP-044/045; probe pattern EXP-064);
project-insights (launch-overhead law EXP-005/034; pointwise/op-order pricing EXP-026/056).

**Estimated Effort**: low (probe script from exp064 template; stem change is ~3 lines).

**Risk Assessment**: Worst case both variants toll out → invalid/NaN closure row, zero charged
seconds (precedented, EXP-040/042/044/045/063/064). If stem launches and reads family-band null
(likely), it's a clean no-improvement at one run's cost. No destabilization path.

### 2. Train-set composition: label-noise filtering (airbench "data filtering" lineage)
**Summary**: Drop the ~150–500 known-mislabeled/duplicate CIFAR-10 train images (hard-coded index
list in train.py from published label-error audits) so every epoch trains on cleaner gradients;
steps unchanged (time-budgeted), epochs +~1%.

**Reasoning**: Data composition is the one out-of-recipe class never measured, and the lineage is
budget-matched (airbench). Targets plateau LEVEL via decision-boundary quality — the named
limiting factor (EXP-032 insight).

**Sources**: airbench repo (data filtering); arXiv 2110.14283 (counter-evidence: selection null
under budgets); EXP-051 (adjacent mechanism, −7.8σ).

**Estimated Effort**: medium (external index-list provenance is the hard part — no new packages).

**Risk Assessment**: Three stacked objections: (i) no heavy-aug evidence — LS+TA+RE plausibly
already neutralizes ~0.5% bad labels (absorption); (ii) the fixed test set carries the SAME ~0.5%
label noise — training on matching noise can help predict the original test labels (EXP-029
rhyme: eval must reproduce training conditions), so cleaning may even invert; (iii) index list is
externally derived and unverifiable in-scope. Expected ≈ 0 ± small both ways → fails the +0.3
effect-size screen.

### 3. σ-tightening replicate pair (instrument investment)
**Summary**: Two more zero-diff baseline runs, pooled with EXP-027's pair → σ from n≈5;
pre-registered no-improvement (metric = pair mean).

**Reasoning**: Tighter σ improves every future near-bar decision (EXP-052 protocol).

**Sources**: EXP-027, EXP-052; goal-learnings § Protocol Findings.

**Estimated Effort**: low (zero code change; 2 × ~8 min wall).

**Risk Assessment**: No improvement possible by design. Marginal: the family band already pools
~15 mean-band reads across byte-identical-signature nulls (96.41–96.73), so σ is effectively
characterized; n+2 adds little. Deferred again (was "low, defer" in exp-report-065 too).

## Idea Evaluation

**Evidence strength**: Candidate 1 rests on the validated probe-closure economics (EXP-063/064:
two family closures for ~4 GPU-min) and addresses a verified gap (no kernel-size price exists in
the record). Candidate 2's supporting evidence is light-aug and small, with a published
budget-regime null AGAINST selection (2110.14283) and an adjacent measured −7.8σ (EXP-051);
its test-noise-matching counter-mechanism makes even the SIGN uncertain. Candidate 3 has solid
protocol grounding but the marginal information is small given ~15 pooled family-band reads.

**Mechanism clarity**: Candidate 1's cost mechanism is crisp (kernel-shape → implementation tier,
extrapolating the EXP-044/045 law) and its level mechanism is the honest open question (input-RF
enlargement at the stem is the one structural change with no direct null). Candidate 2's level
mechanism is muddled (cleaning vs noise-matching pull opposite ways). Candidate 3 has no level
mechanism at all.

**Expected impact**: None of the three clears the bar in expectation — that is what
measured-ceiling means; the audit found no candidate above the +0.3 screen anywhere in scope.
Candidate 1 maximizes map-completion per GPU-second AND retains the only nonzero (if small)
falsification path for the ceiling hypothesis through its launch branch. Candidate 2 burns a
full run on a sign-uncertain sub-screen effect with provenance risk. Candidate 3 produces no
metric-relevant information.

**Risk profile**: Candidate 1 fails safest (NO-LAUNCH → NaN at zero charged cost, or one clean
family-band run). Candidate 2 risks an uninterpretable read (composition + absorption + noise
matching entangled). Candidate 3 is risk-free but value-poor.

**Feasibility**: 1 and 3 trivial; 2 blocked on index provenance.

## Chosen Idea
**Selected**: Kernel-size corner: 5x5 stem (and stage-3 ladder) behind the internal-control GPU probe

**Why this idea**:
It is the only genuinely unmeasured structural dimension left after the adversarial audit (which
otherwise CONFIRMED the measured-ceiling hypothesis, closing the Normalize-std corner by
inspection). It follows the validated zero-charged-cost closure economics, completes the
structural pricing map, and its stem-launch branch is the cheapest remaining honest test that
could falsify the ceiling. The graded ladder (stem ~free-FLOPs vs stage-3 +59% FLOPs) prices the
kernel-shape law at two well-separated points in one probe session, mirroring the EXP-026/040
ladder pattern.

**Hypothesis**:
Square 5x5 kernels land OFF the fast path: the stage-3 variant prices at P_norm ≥ ~28ms
(starvation, NO-LAUNCH), and the stem variant — if 5x5@64ch falls on the slow tier — also tolls
> +0.5ms (full NO-LAUNCH closure of the kernel-size corner at zero charged cost). If instead the
stem variant probes ≤ 22.9ms P_norm, its full run reads inside the family band 96.41–96.73
(absorption-law null), further confirming the ceiling; a replicated read ≥ 96.81 would falsify
the ceiling at the structural corner. Primary expected verdict: invalid/NaN (cost closure) or
family-band no-improvement.
