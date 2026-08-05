# Idea Review EXP-039

Offline fallback adversarial review of randomized finalists:

| Candidate | Evidence | Potential impact | Verdict |
|---|---:|---:|---|
| Regime-Aligned Hard-Tail Cosine Rephase | 7/10 | 7/10 | Lead |
| Gamma-1 Focal Loss Only in the Hard Tail | 3/10 | 6/10 | Reject |
| Training-Only Direct-Path Auxiliary CE | 3/10 | 4/10 | Reject |

**Pick**: Regime-Aligned Hard-Tail Cosine Rephase.

The rephase targets an observed objective discontinuity, acts over roughly 9,000 tail steps, costs essentially no exposure, and derives every anchor from accepted constants. EXP008 supports preserving late motion but does not prove that more is beneficial; near interpolation and best-near-final behavior are material contrary evidence.

Focal loss would shrink most easy-example tail gradients by roughly two orders of magnitude near the observed training CE, conflicting with EXP008's under-update signal. Direct-path auxiliary CE lacks a diagnosed raw-feature failure, weakens supervision to the only recently validated head, and reuses a feature scale as a loss coefficient without evidence.

Corrections adopted: the rephase increases both data-gradient motion and coupled-decay integration; only best accuracy determines success; endpoint/loss/exposure are interpretation evidence; a miss falsifies the exact 39.46%-larger tail-area curve and deprioritizes nearby tuning without formally closing every schedule; isolated momentum reset remains open; success supports only the complete fixed-seed package, not unique causality or general under-updating.
