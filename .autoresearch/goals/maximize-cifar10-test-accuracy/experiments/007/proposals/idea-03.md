# Proposal idea-03: Second ReZero-gated residual block at the proven layer2/8×8 stage

## Summary
One-token edit to `train.py:150`. Current:
`self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), GatedResidual(256))`
Becomes:
`self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), GatedResidual(256), GatedResidual(256))`

The `GatedResidual` class (`train.py:119-137`) is reused verbatim — no new class, no other line. The appended block adds two `conv_bn(256,256)` convs = 2·(256·256·3·3) ≈ 1.18M params (plus negligible BN/α), all at 8×8 spatial resolution. ReZero `alpha=nn.Parameter(torch.zeros(1))` makes the block exact identity at init, so the deeper net starts numerically equivalent to the EXP-004 baseline (modulo extra kaiming RNG draws before `fc`) and `PEAK_LR=0.4` and the whole schedule (`train.py:285-291`) stay unchanged — no LR retune. A 2-step trainability smoke is required per the project-insights ReZero entry: at α=0 the branch convs receive zero gradient on step 1 (gradient flows only to `alpha` via `∂L/∂α=⟨grad_out, c2(c1(x))⟩`); verify `alpha.grad≠0` on step 1 AND that branch-conv grad becomes nonzero on step 2 once α has moved off zero. α.grad≠0 alone is necessary-but-not-sufficient (it was the dead-block failure mode the EXP-004 plan-review caught).

## What it targets
Residual-capacity headroom at the **one stage where added depth demonstrably paid off**. EXP-004 added the *first* `GatedResidual(256)` at layer2/8×8 and gained +0.13pp (95.87→96.00, the current baseline). This probes whether that same stage has *more* headroom under a second identical block.

## Reasoning
Two prior results bound the design. EXP-004 (`004/04-analysis.md`) proves 8×8/256 ReZero capacity is real: the deeper net matched EXP-003 early (identity start) then led mid-training (ep25 92.63 vs 88.84) and held a higher tail floor despite 32 fewer epochs. EXP-005 (`005/04-analysis.md`) proves *where not* to add it: a 2nd block at layer3/4×4 gave −0.10pp because (a) 4×4 coarse capacity was unused and (b) a 4×4/512 conv runs ~10% slower than a FLOP-equal 8×8/256 conv via cuDNN kernel selection. This proposal places the 2nd block at 8×8/256 — full-rate kernels (EXP-004 ran ~26k img/s), directly avoiding EXP-005's throughput failure mode, at the only stage with a positive capacity signal. It is the explicit "Unexplored Avenue" #1 in `005/04-analysis.md`.

## Estimated effort
Low — one-token diff, reused class, no schedule change, plus the standard 2-step smoke.

## Risk assessment
1. **Diminishing returns (primary risk).** The EXP-006 idea-reviewer flagged that the first block's +0.13pp does NOT imply monotonic gain from a second; layer2 may be near its sweet spot, leaving the second block net-negative after fewer low-LR-tail updates. This is the central assumption that must hold. Validate on the *trajectory* (epoch count + tail curve vs EXP-004), not just final best.
2. **Throughput.** Two extra 8×8/256 convs per step at ~26k img/s; expect epochs to drop from EXP-004's 142 toward ~125-135. Capacity gain must outrun the lost annealing budget (it did for one block; margin is thinner for two).
3. **Noise floor.** Per the project-insights Medium entry, the time-budgeted loop fits a host-throughput-dependent step count, giving a ~0.1pp run-to-run noise floor (142/131/150 epochs across byte-identical runs). The +0.1pp bar sits AT this floor, so the gain must clearly exceed it to register from a single run.

**Vs. widening layer2 (256→384, alternative placement):** widening would change channel count and break the layer3 `conv_bn(256,512)` input contract, forcing edits to `train.py:150-151` and reshuffling more kaiming RNG — a larger, multi-line change that cannot be ReZero-identity-initialized cleanly (new channels are not a residual branch), so it would likely need an LR retune. This depth-stacking variant is the cleaner single-variable probe and lower-risk to execute, though its upside ceiling is plausibly lower than a well-tuned width bump. If this fails on diminishing returns, width-at-8×8 (with ReZero-gated new channels) is the natural follow-up.
