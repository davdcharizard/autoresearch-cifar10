# Report EXP-001: Widen the ResNet (WideResNet-style, k=4) + projection shortcuts
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-001.md
- **Plan**: plans/plan-001.md
- **Log**: logs/exp-log-001.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) for a ResNet trained under a fixed 300s wall-clock budget on a single
GPU, editing only `train.py`. Higher is better. Baseline at experiment time: **92.06%** (EXP-000). Success
bar: ≥ 92.16 (baseline + 0.1 pp).

## Idea & Hypothesis
Chosen idea: widen the network (WideResNet-style) on top of the EXP-000 recipe. Motivated by EXP-000's key
finding that the model was capacity-bound (a +21% step increase moved accuracy only +0.33 pp) while VRAM is
essentially free. The Wide ResNet result (Zagoruyko & Komodakis 2016) holds that width is the most
compute-efficient capacity knob under a wall-clock budget — wide layers parallelize well and converge in
fewer epochs than deep-thin nets. Hypothesis: k=4 widening ({64,128,256}) + 1×1 projection shortcuts would
lift best_test_acc meaningfully above 92.06% (target ~93%+), staying in budget.

## Approach
Architecture-only change in `train.py`, EXP-000 recipe held fixed (bf16, channels_last, time-fraction cosine
LR, Nesterov, label smoothing, batch 128, WD 1e-4, PEAK_LR 0.2, seed 42). (1) `BasicBlock` downsample/channel-
change blocks use a 1×1-conv + BN projection shortcut (`self.shortcut`) instead of channel-padding identity;
other blocks use `nn.Identity()`. (2) `ResNet(width_mult=k)`: stem stays 16 channels, stages widen to
{16k,32k,64k}, fc input 64k; k=1 reproduces the original net. (3) `WIDTH_MULT=4` → {64,128,256}. Param count
4,299,866 (15.8× the k=1 net). Single run, no retries.

## Execution
One clean run on GPU 0. Started with 4.3M params; per-step time ~9-18ms vs EXP-000 ~6-7ms — only ~1.5-2.5×
slower despite 15.8× params, confirming the previous net was overhead-bound and the H20 readily absorbs the
extra FLOPs. Fit **79 epochs / 30,498 steps** (vs EXP-000 109 / 42,156 — ~28% fewer epochs). Loss decreased
smoothly, no NaN/divergence; LR warmed to 0.2 then cosine-annealed. Completed in budget: 300.0s training,
385.7s total. peak VRAM 490.8 MB. No errors.

## Results
- **Primary metric**: **94.90%** (baseline: 92.06%, delta: **+2.84 pp**, +3.08%)
- **Observations**: best 94.90 @ epoch 75; final 94.83%, final loss 0.249 (lower than EXP-000's 0.31 — the
  bigger model both fits better and generalizes better here). Trajectory: 91.73 (BASE) → 92.06 (EXP-000,
  recipe) → 94.90 (EXP-001, width). The single largest jump so far, from a single architectural lever.
- **Analysis**: Hypothesis strongly confirmed, and the magnitude exceeded the ~93% expectation. This validates
  the EXP-000 diagnosis that capacity — not training budget — was the binding ceiling: a 15.8× capacity
  increase bought +2.84 pp while costing only ~28% of epochs, because wide convs are GPU-efficient (the tiny
  baseline barely used the H20). VRAM (490 MB / 98 GB) remains a non-constraint, so further capacity is still
  on the table. The new question is where the *next* ceiling sits: more width (k=6/8?), regularization
  (the model now has the capacity to overfit — Cutout/mixup likely help), or recipe tuning for the bigger net.
- **Key Learning**: Widening is the dominant lever for this task — 15.8× capacity (k=4 WRN-style) lifted acc
  +2.84 pp at only ~28% fewer epochs; the H20 makes width nearly free, so capacity scaling is the main axis.

## Verification
- **Conditions**: all passed (clean completion in budget; 94.90 ≥ 92.16; only train.py changed, eval
  once/epoch, no new deps, seed unchanged).
- **Review Notes**: Results confirmed trustworthy. Gain came purely through an in-scope architecture change;
  frozen eval harness untouched; single fixed-seed run, no seed hacking. Adversarial check: a +2.84 pp jump
  from added model capacity is a genuine generalization gain that would survive benchmark recomposition —
  not metric gaming. Magnitude is far beyond run-to-run noise.
- **Verdict**: improvement
- **Verdict Basis**: all necessary conditions passed + primary metric improved by +2.84 pp (well above +0.1 bar).

## Unexplored Avenues
- **Scale width further (k=6, k=8)**: VRAM and throughput headroom remain large; if still capacity-bound,
  more width may keep paying — but watch the epoch budget (k grows FLOPs ~k²) and the onset of overfitting.
- **Regularization for the larger model**: with 4.3M params the net can overfit pad+crop+flip; Cutout and/or
  mixup, and/or higher weight decay, likely add another increment — now more relevant than at k=1.
- **Depth × width tradeoff**: a moderately deeper-and-wider config (e.g. NUM_BLOCKS 4-5 with k=2-3) may beat
  pure width at equal wall-clock; untested.
- **Recipe re-tuning for the wide net**: PEAK_LR / WD were tuned implicitly for the small model; the wide net
  may prefer higher WD (WRN uses 5e-4) or a different peak LR — cheap to probe.

## Next Steps
1. **Add Cutout (+ possibly raise weight decay to ~5e-4)** on top of k=4 — *high confidence* the larger model
   is now regularization-limited; WRN recipes pair width with Cutout/WD for the last couple points.
2. **Push width further (k=6/8) or a deeper-wider mix** — *medium confidence*; test whether capacity is still
   the ceiling or if returns have started to diminish; mind the epoch budget.
3. **Re-tune PEAK_LR/WD for the wide net** — *medium confidence*, cheap, isolates recipe gains from the
   architecture change held fixed this loop.

## Exit Action Results
- None defined for this goal — skipped.
