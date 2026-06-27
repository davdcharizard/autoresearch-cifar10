# Brainstorm EXP-034
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Wide Residual Networks** (knowledge/README.md References; arXiv 1605.07146): width beats depth on CIFAR at FIXED width-or-depth sweeps — but those sweeps hold the OTHER dimension constant and use fixed epochs. Our regime question is different: at FIXED compute (dt) and ≈fixed params, does moving capacity from width into depth change the plateau level? WRN never measures that trade directly at our shape (depth 20, 4× width).
- **He et al. 2015 (ResNet paper, cited in train.py)**: CIFAR family is depth-parameterized 6n+2 (20/32/44/56...) at width 16/32/64; accuracy rises monotonically 20→110 at fixed width. Depth direction has headroom in the original design space; our runs sit at its shallowest point with 4× the width.
- **RegNet design spaces** (knowledge/papers/regnet-design-spaces.md): optimized spaces are third-stage-heavy — but EXP-017 measured that allocation transfer FAILS at depth 20 ([2,3,4] −0.28). Informs candidate filtering: allocation reshapes are low-prior; total-depth changes are the open variable.
- No new external sources needed — this loop's candidates are arithmetic consequences of in-project measurements.

## Experimental History Review

- 35 experiments, 6 improvements, last improvement EXP-006; baseline 96.71 @ 1990397 (run-level mean ≈96.57, σ ≈0.16 per EXP-027); bar = 96.81 requires a TRUE effect ≥ +0.3.
- **Capacity-MAGNITUDE axis closed on the width side**: 5× (EXP-005), 6× (EXP-007), 8× (EXP-002) all lost — more width costs dt/epochs and the plateau gain doesn't cover it. 4× is the measured width optimum.
- **Depth-DOWN measured BAD**: EXP-008 (ResNet-14 @ 6×, wider-but-shallower at roughly matched compute) lost — removing depth hurt even when width compensated. This is the only direct depth-direction datum, and its sign favors MORE depth per unit compute, not less.
- **Depth-UP never probed.** EXP-017 moved blocks BETWEEN stages at constant total depth (lost −0.28, allocation); no experiment changed NUM_BLOCKS upward. ResNet-26/32 at compensated width is unmeasured territory.
- **Recipe/data/eval/optimizer axes all measured-closed** (EXP-007…033): constants bracketed both directions, schedule family/shape closed, optimizer geometry closed (EXP-028), weight averaging closed (EXP-011/032), resolution closed (EXP-031), tail distribution closed monotone (EXP-025/033), eval-side BN closed with inverted sign (EXP-029). EXP-032's diagnosis: accuracy is decision-boundary-limited, not calibration-limited.
- **Laws that constrain design**: max-statistic (only converged-plateau LEVEL pays — so a candidate must change the level, and capacity SHAPE is the canonical level-changer); epoch-deficit arithmetic (every +1ms dt ≈ −6 epochs; deficits >~10 epochs have repeatedly cost ~0.1–0.2); early-dt gate kills cheaply (~90s, EXP-026 precedent); normalization-constant preservation; heat preservation (any change that alters early trainability interacts with the time-keyed schedule — EXP-018/020).
- **Wall/contamination protocol**: dual launch gates (GPU-0 empty AND 1-min load <60), composite watchdog, 600s cap, eval thinning + 16 workers available as wall levers (EXP-031/032).

## Candidate Ideas

### 1. Depth-for-width trade at matched compute — ResNet-26 at stage widths 56/112/224
**Summary**: NUM_BLOCKS 3→4 (ResNet-26: 12 blocks) with WIDTH scaled by √(3/4) ≈ 0.875 → stage widths 56/112/224 (all 8-aligned, two 16-aligned). Conv FLOPs scale as blocks × width²: (4/3) × (56/64)² = 1.02× baseline — dt and params (~4.38M vs 4.29M) both ≈ matched, so epochs stay ~139 and the max-statistic confound is held out. Everything else byte-identical: recipe constants, schedule, compile, eval. Strict early-dt gate: kill if windowed dt > 24.5ms in the first ~2 minutes (epoch deficit then >12 — pollutes the comparison).

**Reasoning**: This is the last unprobed capacity direction, and the only direct datum on it points UP: EXP-008 (depth-down at width-up, ≈matched compute) lost. Mechanism: depth multiplies the composition count of nonlinearities — at fixed compute it buys boundary expressivity where width buys parallel features; EXP-032 diagnosed the ceiling as decision-boundary-limited, which is exactly what composition depth targets. The original ResNet CIFAR results show monotone depth gains 20→110 at fixed width; we sit at the family's minimum depth with the width dial already optimized (4×). Risk is bounded by the gate: if the 12-block sequential structure (more BN/ReLU memory traffic, +33% kernel launches, partially fused by compile) blows dt, the run dies at ~90s for pure information.

**Sources**: EXP-008/002/005/007 rows (experiment-indices TSV); exp-report-032 (boundary-limited diagnosis); exp-report-033 § Next Steps; knowledge README WRN + He-2015 entries; goal-learnings GRADIENT-NOISE/heat-preservation entries (unchanged here — recipe untouched).

**Estimated Effort**: low — two-constant change (NUM_BLOCKS=4; per-stage widths 56/112/224 requires replacing the single WIDTH_MULT with explicit stage widths, ~5 lines in ResNet.__init__) + standard composite run.

**Risk Assessment**: (a) dt could exceed the gate → 90s gate-kill, axis closed at this design point (acceptable outcome); (b) non-64-aligned channels (56) could cost cudnn/tensor-core efficiency — the gate catches it; (c) depth could need its own LR/warmup tuning (deeper = harder early optimization) — but ResNet-26 is far below the depth where plain CIFAR ResNets degrade (110+), and heat is unchanged; (d) worst case: clean run at baseline level → depth direction closed both ways, capacity axis fully bracketed.

### 2. Raw depth add — ResNet-26 at unchanged 4× widths (64/128/256)
**Summary**: NUM_BLOCKS 3→4 keeping 64/128/256. FLOPs ×1.33 → dt ~29-30ms → ~105 epochs; params 5.71M.

**Reasoning**: The WRN/He evidence in its purest form (depth added, nothing removed). But it confounds the probe: capacity magnitude rises (5.71M sits between the failed 4×→5× step) AND epochs fall by ~34 — both measured-negative directions. EXP-005 (6.69M, fewer epochs) already lost on exactly this arithmetic.

**Sources**: EXP-002/005/007 (capacity-magnitude bracketing); EXP-001 (the one capacity win, at a far larger level jump than this).

**Estimated Effort**: low (one-constant change).

**Risk Assessment**: High prior of repeating the EXP-005 failure mode; a loss is uninterpretable (depth? params? epochs?) — weak information value.

### 3. Deeper-extreme dose point — ResNet-32 at 48/96/192
**Summary**: NUM_BLOCKS 3→5 (15 blocks) with width ×0.75 → FLOPs ×(5/3)×0.5625 = 0.94× baseline; params ~4.0M.

**Reasoning**: Stronger dose on the same depth-for-width axis. If Candidate 1 wins, this is the natural follow-up to map the dose-response; running it FIRST risks overshooting (more launch overhead at 15 blocks, slight params drop adds a small capacity-down confound).

**Sources**: same axis evidence as Candidate 1.

**Estimated Effort**: low.

**Risk Assessment**: Higher dt-overhead risk (15 sequential blocks) and a −7% params confound; informative mainly as a second point after Candidate 1.

## Idea Evaluation

All three live on the one remaining open axis (capacity shape, depth direction). Candidate 2 is dominated: it re-enters the measured-failed capacity-magnitude + epoch-deficit regime, and any outcome is confounded three ways — discard as a first probe. Candidate 3 is the right SECOND point on the axis but the wrong first one: its 15-block launch overhead risk is higher, its params drop adds a confound, and its result is hard to interpret without the intermediate point. Candidate 1 is the clean instrument: dt ≈ matched (epochs comparable, max-statistic law satisfied), params ≈ matched (pure shape, not magnitude), recipe byte-identical (heat/noise/normalization laws all preserved), gate-screened (worst case ~90s), and it is the direct mirror of EXP-008 whose sign supports it. Evidence strength: medium (one in-project datum with the right sign + the original ResNet depth ladder); mechanism clarity: high (composition depth targets the diagnosed decision-boundary limit); expected impact: the only axis left where a LEVEL change is structurally possible; risk: gate-bounded.

## Chosen Idea
**Selected**: Depth-for-width trade at matched compute — ResNet-26 at stage widths 56/112/224 (Candidate 1)

**Why this idea**:
It probes the single remaining unmeasured capacity direction with every confound held out: compute matched (1.02×), params matched (+2%), recipe byte-identical, epochs ≈ unchanged. The only in-project datum on the depth direction (EXP-008: shallower-but-wider lost at matched compute) points exactly this way, and the mechanism (nonlinearity composition → boundary expressivity) targets EXP-032's decision-boundary diagnosis. The failure mode is cheap and informative: a dt-gate kill (~90s) or a clean baseline-level run both close the axis.

**Hypothesis**:
At matched dt (windowed dt ≤ 24.5ms, epochs within ~8 of 139) and matched params (~4.38M), ResNet-26 @ 56/112/224 converges to a HIGHER plateau than the 96.57-mean baseline family because composition depth adds decision-boundary expressivity per unit compute — predicting best_test_acc ≥ 96.81. Falsified by: dt gate kill (mechanism unavailable at matched compute), or a clean full run with plateau within or below the baseline noise band (96.4–96.7), which closes the depth direction and with it the capacity-shape axis.
