# Designing Network Design Spaces (RegNet)

- **Source**: arXiv 2003.13678 (Radosavovic, Kosaraju, Girshick, He, Dollár — CVPR 2020)
- **Consulted**: EXP-017 brainstorm (2026-06-10)

## Key Claims

- Population-level analysis (thousands of trained nets): the best design spaces allocate MOST blocks to the third stage and few to the first — "higher flop models have a large number of blocks in the third stage", mirroring standard large ResNets.
- Good design spaces have simple regularities (quantized-linear width/depth functions); uniform per-stage depth is not what emerges when allocation is optimized at scale.
- Increasing the depth of DEEPER stages is more beneficial than shallower stages (corroborated by NAS literature in the same search).

## In-Project Relevance

- Motivated EXP-017: [3,3,3] → [2,3,4] at depth 20 / 4x width — per-block MACs are equal across ResNet stages, so the move was FLOPs-neutral (+1.11M params).
- **EXP-017 result: did NOT transfer — −0.28pp despite running FASTER (144 vs 139 epochs) with +26% params.** The run converged with a full plateau, so the deficit is representational: at total depth 20, removing one of three 32×32 stage-1 blocks costs more than the added 8×8 capacity returns. RegNet's third-stage-heavy populations live at much larger total depths and optimize final accuracy at fixed iterations — neither condition holds here (shallow extreme; max-over-evals at fixed wall clock). Use this paper for allocation intuitions only at depth ≳ 30–50; at the shallow extreme, uniform allocation measured at/near optimal (reports/exp-report-017.md).
