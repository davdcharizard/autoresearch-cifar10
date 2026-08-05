# Idea-01: Compile-funded DEPTH at the proven 8×8 stage (a second ReZero block at layer2)

## Summary
Add representational DEPTH to the backbone — the one architectural axis with a *positive* prior here (EXP-004) and direct external corroboration (airbench's 95→96 step is literally "add a third convolution to each block") — and fund the per-step cost with `torch.compile`'s validated +12% throughput (EXP-014, math-equivalent, off-budget warmup) so the deeper net still fully anneals in 150-ish epochs instead of under-annealing (the #1 failure mode that sank EXP-005/007).

Concretely, the **recommended minimal operating point**: insert a SECOND `GatedResidual(256)` (ReZero, α init 0 → identity at init) into `layer2` (the 8×8/256 stage), immediately after the existing one. This is the exact location and block type that EXP-004 validated as the only-ever architectural win (the FIRST gated block at layer2, +0.13pp), so the experiment asks the sharp question "if one gated 8×8 block helped, does a second?" — now with compile funding the cost that EXP-005/007 could not afford. Identity-init guarantees the deeper net starts bit-equivalent to the current proven net and ramps capacity in gradually.

torch.compile is the load-bearing enabler: EXP-014 already implemented and validated it on this exact harness (off-budget `t_start_training` warmup, BN-buffer restore, a separate uncompiled EMA eval path, no recompile leak) and measured a clean +12% img/s with bit-equivalent math. EXP-014 spent that headroom on WIDTH (256→320) and it saturated; it was NEVER spent on DEPTH, which has a positive prior. This experiment redirects the banked lever to the axis the evidence actually favors.

## Reasoning
- **Multi-source evidence for depth specifically**: (a) airbench (arXiv:2404.00498) reaches 96% from 95% by adding a third conv per block — depth, not width; (b) EXP-004 here: adding the first GatedResidual at 8×8 was the single architectural improvement in the whole project; (c) EXP-005 (depth at 4×4/layer3) and EXP-007 (width at layer2) both LOST *to under-anneal, not to saturation* — the capacity was useful but unannealed. The common thread: depth at the proven full-speed 8×8 stage helps *if it can anneal*. Compile removes the anneal constraint.
- **Directly addresses the ceiling diagnosis's loophole**: EXP-014 concluded "generalization ceiling, not epoch/throughput-bound" — but that test added epochs and WIDTH, never depth. Depth changes the function class (more nonlinear composition), which width at a saturated stage does not. The ceiling claim has an untested crack exactly here.
- **Identity-init de-risks**: ReZero α=0 means the deeper net = current net at init; any divergence is monotonic capacity ramp, not a cold-start regression (validated mechanism, EXP-004).

## Sources
- airbench / Keller Jordan, arXiv:2404.00498 — "add a third convolution to each block" for 96% (knowledge/references/fast-cifar10-recipes.md).
- EXP-004 (experiments/004) — first GatedResidual@layer2, the only architectural win.
- EXP-014 (experiments/014, knowledge/references/torch-compile-throughput.md) — torch.compile +12% validated, math-equivalent, off-budget warmup; banked as headroom.
- knowledge/references/rezero-identity-init.md — ReZero α=0 identity-init mechanism.

## Estimated Effort
Medium. Two coupled changes in train.py: (1) port the EXP-014 torch.compile wrapper (off-budget warmup, EMA eval path) — reference implementation exists; (2) add one GatedResidual(256) to layer2. Smoke: num_epochs must stay ≥135 (under-anneal gate) WITH compile on; verify best==per-epoch-max and identity-init (ep1 ≈ current net). Same-session control + confirmation pair mandatory (the low-control-draw artifact has recurred 4×).

## Risk Assessment
- **Primary risk — under-anneal**: even +12% may not fully fund a second 256-ch block's two 3×3 convs at 8×8; if num_epochs drops below ~135 the gain won't anneal (EXP-005/007 pattern). Mitigation: layer2-only (cheapest positive location); pre-measure epochs in a smoke; fall back to a single extra conv (not a full 2-conv block) if epochs dip.
- **Secondary risk — the ceiling is real**: EXP-014's diagnosis may hold even for depth → ties. But this is the best-evidenced remaining structural bet, and a tie still closes the depth axis definitively.
- **Compile fragility**: recompile leaks / BN-buffer mishandling silently corrupt math — mitigated by reusing the EXP-014-validated wrapper and asserting math-equivalence on the `SCHEDULE=tri` control (compile-off vs compile-on bit-check at step 2).
