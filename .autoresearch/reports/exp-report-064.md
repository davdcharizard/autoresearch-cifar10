# Report EXP-064: ACNet asymmetric convolution blocks — probe-gated NO LAUNCH (reparameterization family cost-closure)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-064.md
- **Plan**: plans/plan-064.md
- **Log**: logs/exp-log-064.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged budget by modernizing train.py only. Baseline: 96.71 @ 1990397 (bar 96.81; family mean 96.57, σ 0.16; bar-over-mean +0.24).

## Idea & Hypothesis

After EXP-063 closed multiplicity at the funding level, brainstorm-064 identified train-time structural reparameterization (ACNet, ICCV 2019: every 3x3 conv becomes 3x3+BN ∥ 1x3+BN ∥ 3x1+BN summed, folding exactly into one conv at eval) as the last unmeasured plateau-raising mechanism class — it evades the capacity closure (eval params unchanged) and the reg-dose closure (not a regularizer; it reparameterizes optimization geometry). Published CIFAR-10 fixed-epoch gains +0.35–1.11 incl. WRN-16-8. **Hypothesis**: probe reads dt ≤ ~26ms (epochs ≥ 118, dilution ≤ 0.29) and the realized gain ≥ +0.53 clears the bar. Pre-registered launch criterion with an internal baseline control: LAUNCH iff B ∈ [21.5, 25.0] AND P_norm = 22.4×P/B ≤ 26.0.

## Approach

train.py: ~25-line ACB module swapped into all 19 conv+BN sites (+32/−11 diff), all hyperparameters byte-identical; Eval untouched — the branched module in eval mode computes exactly the folded plain-conv function (CPU sanity proved fold-equivalence to 2.15e-06). Sanity 6/6; NUM_PARAMS_ACB = 7,149,002 pinned. Launch decision delegated to /tmp/exp064_gpu_probe.py, which timed 40 full steps of the plain baseline net (internal control) and 40 of the ACB net in the same session — the control normalizes out load inflation by construction (EXP-062 lesson, improved over EXP-063's load-conditional criterion).

## Execution

Single uncharged probe at clean conditions (apps=0, load 14.9): control **B = 22.36ms** (mid-family-band → probe self-validated), ACB **P = 43.15ms**, **toll ratio 1.930**, P_norm = 43.23 vs ≤ 26.0 → **NO LAUNCH** (fails by 66%). Uncharged family diagnostic: DBB-lite (3x3 ∥ 1x1 — the fewest extra launches any reparameterization can have, +1 conv +1 BN per site) timed **28.61ms** → ~107 epochs → dilution −0.45 → required realized gain ≈ 0.69, above what single-branch ablations deliver. train.py was never launched; zero charged seconds; no run.log ever existed.

## Results

- **Primary metric**: NaN (no charged run; baseline 96.71 unchanged)
- **Observations**: The full ACB toll (+20.8ms for +76 small-kernel launches) is far superlinear vs the DBB-lite toll (+6.25ms for +38) — the 1D convs (1x3/3x1) land on slow odd-shape kernel implementations, consistent with the EXP-044/045 law (off-standard shapes get one flat slow implementation; FLOPs second-order). ACB compile warmup was 36.4s (3× baseline) — 57 convs to compile.
- **Analysis**: The reparameterization family's value proposition ("free at inference") is priced entirely at TRAIN time, and this goal's budget is train-time — the technique's cost lands exactly where this regime cannot pay it. The pricing arithmetic closes the whole family: gain and toll both scale with site count (partial-site variants inherit the same inequality), and the minimum-launch variant (1x1 branch) already requires ≈ 0.69 realized gain against a sub-ACB published ablation band. With reparameterization closed, every plateau-raising mechanism class in the record — capacity, regularization, loss geometry, averaging, schedule, ensembling/multiplicity, attention, activation, normalization constants, and now train-time over-parameterization — is measured-closed. Two consecutive probe-gated NO-LAUNCH closures (EXP-063/064) at zero charged cost also validate the pre-run-inequality discipline: four GPU-minutes bought two family-level closures.
- **Key Learning**: "Free at inference" techniques invert under a train-time budget — their entire price concentrates on the only resource this goal meters; on a launch-bound box the minimum possible reparameterization branch already costs ~28% step time.

## Verification

- **Conditions**: not reached — the pre-registered M2 launch criterion failed; Conditions 1–3 skipped per plan branch (ii). Gate integrity: the internal control read 22.36ms (mid-family-band) in the same session at the same load, ruling out load-inflation by construction; criterion applied verbatim; arithmetic re-checked (43.23ms → 71 ep → dilution −0.95 → required gain 1.19 > published max 1.11).
- **Review Notes**: Results trustworthy — control-normalized, decisive margin (66%), and the family diagnostic measured through an independent build.
- **Verdict**: invalid
- **Verdict Basis**: Pre-registered branch (ii): probe NO-LAUNCH → `invalid`, NaN — cost-closure with no charged run (EXP-040/042/044/045/063 precedent).

## Unexplored Avenues

- **Online folding (train ON the folded conv, branch only for gradient shaping)**: would dodge the dt toll but requires re-deriving the branch gradients analytically — that is exactly equivalent to per-kernel-region learning rates, which is optimizer-side machinery the optimizer closures (Muon EXP-028, momentum trades EXP-023/024) price negatively; and the equivalence only holds without per-branch BN, which the papers identify as the actual gain source.
- **Fused triple-conv kernel (write a custom kernel computing all 3 branches in one launch)**: out of scope (no new packages; Triton authoring is in-scope in principle via torch, but the inductor already had the chance to fuse and didn't) — and EXP-021's max-autotune null bounds custom-kernel upside.
- **Per-branch-BN effect via plain BN parameter groups**: the papers attribute the gain to per-branch adaptive scaling; mimicking it without branches (e.g., separate WD/LR for kernel center vs corners) is per-region optimizer surgery — closed by the fc-WD bracketing logic (EXP-057/058) and momentum-trade closures.

## Next Steps

1. **Adversarial audit of the measured-ceiling hypothesis** (medium confidence): with all ten mechanism classes now measured-closed, the next brainstorm's first job is to attack the record itself — re-derive from the in-scope files (train.py, prepare.py read-only) whether any charged-side resource or eval-legitimate structure remains unpriced, before reaching for external techniques (transfer record 0-for-19 counting reparam pricing).
2. **Targeted lit excavation for budget-creating mechanisms** (low-medium confidence): the only external candidates worth screening are ones that REDUCE charged dt at identical numerics (the one lever EXP-048 left with a hard null but EXP-021 measured only via changed arithmetic) or that raise the plateau through data information content — both heavily bounded, but excavations have produced clean closures at worst.
3. **Replicate-pair tightening of the family band** (low confidence): if brainstorming yields only sub-screen candidates, a byte-identical baseline replicate pair would sharpen σ and the bar's position — protocol value rather than metric value; defer unless the idea drought is real.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
