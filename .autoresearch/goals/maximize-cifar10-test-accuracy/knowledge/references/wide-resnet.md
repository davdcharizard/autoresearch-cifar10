# Wide Residual Networks (WRN) — pre-activation, for CIFAR-10

Chosen backbone for EXP-022 — the first wholesale-different backbone after 16 within-DavidNet nulls. The canonical "higher-ceiling" CIFAR-10 conv net.

## Source
- **Wide Residual Networks**, Zagoruyko & Komodakis, BMVC 2016 (arXiv:1605.07146).
- **Identity Mappings in Deep ResNets** (pre-activation block), He et al. ECCV 2016 (arXiv:1603.05027).
- **Cutout** (DeVries & Taylor 2017, arXiv:1708.04552): WRN-28-10 + cutout = **97.1%** CIFAR-10 — the headline ceiling, under the exact aug class we run (Cutout12+RandomErasing).

## Architecture (CIFAR variant)
- Naming: **WRN-d-k** = depth `d`, widen factor `k`. `d = 6N + 4` where `N` = blocks per stage (3 stages). E.g. WRN-16-k → N=2; WRN-22-k → N=3; WRN-28-k → N=4.
- **Stem**: single `Conv3×3, 3→16` (no BN/ReLU before first block in pre-act form; first block's BN handles it).
- **3 stages** at spatial 32×32 / 16×16 / 8×8, widths `16k / 32k / 64k`. Stage 1 stride 1; stages 2,3 stride 2 at the first block (downsample).
- **Pre-activation basic block**: `BN→ReLU→Conv3×3→BN→ReLU→Conv3×3`, plus identity (or 1×1 conv projection when channels/stride change) shortcut added to the output. Optional dropout between the two convs (WRN paper uses 0.3 for some; we likely omit — we already regularize heavily).
- **Head**: final `BN→ReLU→GlobalAvgPool→Linear(64k→10)`. NOTE: this replaces DavidNet's global **MaxPool + scale_out·Linear** — there is no `scale_out` in standard WRN.

## Param/size reference (CIFAR, approximate)
- WRN-16-4 ≈ 2.7M; WRN-22-4 ≈ 4.3M; WRN-16-8 ≈ 11.0M; WRN-28-4 ≈ 5.8M; WRN-28-10 ≈ 36.5M.
- VRAM is a non-constraint here (H20 98GB; DavidNet uses ~1.6GB) — size is gated by **throughput → num_epochs**, not memory.

## Why it plausibly breaks the DavidNet ceiling
- Different topology: pre-activation ordering (clean identity gradient), **N>1 blocks per stage** (more depth at each resolution), a full **16×16 stage** (DavidNet pools 32→16→8→4 with only one block at 8×8), **GAP** readout. WRN's documented 97.1%-with-cutout is matched-epoch evidence its ceiling > DavidNet's ~96.4.

## RESULT — EXP-022 NO-IMPROVEMENT (do NOT re-run budget WRN here)
Tested in EXP-022 and it does NOT transfer at 300s. Same-session compiled triple: **cA WRN-22-4 (4.30M, 133 ep/12880 steps ≥ the 12610 anneal gate) = 96.31**, **cB WRN-16-4 (2.75M, 196 ep) = 96.34**, both TIE the compiled DavidNet control **c0 96.32** (−0.01 / +0.02pp, within noise); neither cleared 96.48. Both fully annealed (not under-anneal), peak_lr 0.4 (= linear-scaled 0.1×512/128, smoke-confirmed stable), identical recipe + data stream. **Why the 97.1% does not transfer**: that is WRN-28-10 (36.5M) at 200 MATCHED epochs; the budget-feasible WRNs that anneal in 300s are small (≤4.3M) and do NOT inherit the big-net ceiling, while the higher-capacity **WRN-16-8 (10.96M) under-anneals at 68 ep** (disqualified). Net lesson: the ~96.4 ceiling is **backbone-family-independent** (DavidNet AND WRN both top out there at full anneal) → the limiter is the recipe/data/compute regime, not topology. Do NOT re-test budget WRN sizes/LRs; a non-conv backbone is unlikely to break a recipe/data-bound ceiling. See experiments/022/04-analysis.md.

## Constraint notes for THIS goal (300s budget)
- **#1 risk = under-anneal** (EXP-007/021): a too-large WRN fits <110 ep and loses on epochs, not ceiling. **Gate on measured `num_epochs ≥ 130`** (healthy band 130-173 with compile); pre-smoke img/s for WRN-16-4 / 22-4 / 16-8 and pick the largest that holds the band. Reject sub-110-ep cells as under-anneal artifacts, not backbone verdicts.
- **Fund per-step cost with the banked torch.compile +12%** (twice-validated EXP-014/021; recipe in `torch-compile-throughput.md`).
- **Recipe transfer / confounds**: keep backbone-agnostic wins (EMA 0.998, tail flip-TTA, one-cycle time-based, LS 0.2, Cutout12+RandomErasing, bf16/channels_last, batch 512). Drop `scale_out` (GAP head). Start WITHOUT the whitening stem (it changes the WRN stem input shape — add as a follow-up rider). Peak-LR 0.4 (mean-loss) was tuned for DavidNet; sanity-check the WRN early trajectory and re-tune only if divergent/flat.
- **Protocol**: same-session DavidNet compiled control + confirmation pair (the low-control-draw artifact has recurred 4×).
