# Report EXP-047: Multi-scale decision head — fc over concat[GAP(stage2), GAP(stage3)]
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-047.md
- **Plan**: plans/plan-047.md
- **Log**: logs/exp-log-047.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged budget by modifying train.py only. Baseline: **96.71** @ 1990397; bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16 (EXP-027).

## Idea & Hypothesis

After EXP-046 established the strongest-form absorption law (even toll-free external techniques null — candidates must supply something augmentation cannot), the last never-dosed structural class was decision-layer information ROUTING: every prior experiment kept the classifier's input fixed at GAP(stage3). Chosen idea: classify from concat[GAP(stage2) 128-d, GAP(stage3) 256-d] — give the linear decision layer direct access to mid-level features (applied-practice anchors: fast.ai concat-pool head, hypercolumn principle). Free in every priced currency: +1,280 linear params, ~zero dt, zero noise/schedule interaction, smooth gradients (deliberately avoiding EXP-030's max-pool credit-assignment failure). Hypothesis: best ≥ 96.81 at baseline signatures; branches (ii) mean-band inert null, (iii) ≤ 96.42 dilution, (iv) GATE_KILL.

## Approach

Two-line change in `train.py` `ResNet`: `fc = nn.Linear(w2 + w3, num_classes)`; forward pools both stage outputs, flattens, concats, classifies. CPU sanity: params exactly 4,287,306; manual-recomputation equivalence (routing wired right); stage-2 fc columns receive gradient; train smoke decreasing at lr 0.01/6 steps (the plan's lr-0.05/2-step toy smoke overshot on random labels — flaky criterion, not a bug; recorded in exp-log Decisions). Composite launcher `/tmp/exp046_composite.sh` reused verbatim (gate 26ms).

## Execution

Single pristine run: gates cleared at poll 1; GATE_DECISION D0=22.7ms (projected 136 epochs) — branch (iv) eliminated; all 32 windows 21.7–22.8ms, no slow streaks, rc=0, 138 epochs, 300.0s charged, 520.0s total. No retries, no errors.

## Results

- **Primary metric**: best_test_acc 96.15 (baseline: 96.71, delta: −0.56, −0.58%)
- **Observations**: Signatures baseline-identical (dt ≈22.5ms, 138 ep, VRAM 1639MB, params 4,287,306 as projected) — the deficit is NOT cost-mediated. Plateau LEVEL uniformly depressed: last 8 evals 96.09–96.15 (family 96.5–96.7); test_loss 0.1905–0.1939 vs family ~0.185. Plateau scatter normal (~0.06) — converged, stable, just lower.
- **Analysis**: Pre-registered branch (iii) — dilution, and decisively so: −0.42 vs mean ≈ −2.6σ, one of the largest clean structural deficits measured (cf. max-pool head −0.91, stage-reallocation −0.28). Root cause reading: GAP(stage2) features at depth-7 are mid-level (TA+RE-perturbed edge/texture statistics) and carry far less class-linear information than stage-3 features; concatenating them into a SINGLE linear layer forces the softmax to contend with 128 noisy dimensions whose weights receive the same WD/LR — the head's decision margin is built on a 384-d basis where a third of the directions are weak, and the whole network's gradient signal through fc is split across them all training long (the stage-2 shortcut path also bypasses stage-3's representational pressure slightly). This discredits the specific approach (raw concat into one linear head) AND, combined with EXP-030 (pooling operator) and EXP-037 (channel gating), triangulates the head region as a measured local optimum: GAP(stage3)→fc is the right decision interface for this network. The routing class is closed from below — the FIRST structural class closed by an active negative rather than an absorbed null, which is itself informative: the baseline head is not merely "good enough", it is load-bearing.
- **Key Learning**: Decision-layer information routing is a real, negative axis here — mid-level features actively dilute a linear head (−2.6σ at byte-clean signatures); GAP(stage3)→fc is measured load-bearing, closing the last never-dosed structural class.

## Verification

- **Conditions**: Integrity pre-condition PASSED (pristine profile, 138 epochs, params exact, 300.0s, evals ≤ epochs). Condition 1 (best ≥ 96.81) FAILED: 96.15. Conditions 2–3 skipped per first-failure-stop (informationally both pass: 520.0s ≤ 600; 138 ≤ 138).
- **Review Notes**: Results confirmed trustworthy — clean profile, no contention, metric grepped from run.log, eval contract untouched (architecture change flows through base_model; Eval.evaluate() unmodified). The deficit is consistent across the whole plateau (not a single bad read).
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure — valid result far below the bar.

## Unexplored Avenues

- **BN or per-path scaling on h2 before concat** (e.g., a learnable scalar gate initialized near zero): would let the network down-weight stage-2 features instead of being forced to use them — but a near-zero-init gate converging to zero just reproduces the baseline at extra heat cost (EXP-037's SE arc showed gates stay near init under this recipe), and converging away from zero re-introduces the measured dilution. Expected outcome: baseline-or-worse; not worth a run.
- **Stage-2 features through a small projection (128→32) before concat**: reduces the dilution dose but adds a Kaiming-init module that must be learned at heat (EXP-020/037 deferral signature). The measured −2.6σ at dose-128 makes a positive at dose-32 implausible.

## Next Steps

1. **Treat the structural frontier as exhausted and the baseline as at the recipe's measured ceiling** (high confidence): all structural classes are now closed — absorbed-null (046), active-negative (047, 030), cost-priced (020, 037, 040–045). The honest statement: no remaining constructible single change passes the screen stack with positive expected value.
2. **Next brainstorm must therefore go meta**: either (a) revisit the only axis with measured POSITIVE in-regime conversions — throughput→epochs at numerics-IDENTICAL arithmetic (anything that shaves charged ms/step without touching update math; the lattice is charted but the per-step overhead OUTSIDE conv kernels — e.g., the per-step `loss.item()` host sync pattern, LR loop overhead — was never itemized), or (b) micro-dose combinations of certified components only (no external imports). Medium confidence either yields a screen-passing candidate.
3. **Carry the protocol stack forward unchanged** (high confidence): D0 gate, dual launch gates, replicate-pair band, integrity pre-condition — all five of the last five runs resolved exactly per pre-registration.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
