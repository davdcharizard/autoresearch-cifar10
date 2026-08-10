# Claude Adversarial Implementation Review: EXP-013

- **Reviewer**: Claude Code 2.1.220
- **Evidence**: complete current `train.py` diff versus `d68f73a`
- **Initial verdict**: BLOCK
- **Final verdict**: PASS

## Blocking Findings And Resolution

- A per-epoch geometry error could originally propagate before the terminal summary. The implementation now catches and records it, requests loop exit, prints all terminal audits and summary values, then raises nonzero.
- EMA-source epochs originally printed only restored online geometry. They now print distinct online and direct EMA-shadow geometry without changing swap or selection state.

## Confirmed Invariants

- Stored/trainable counts are exactly 2,748,890/2,748,880.
- Bias is zero-initialized, frozen before ownership construction, retained in optimizer/EMA state, excluded from SAM, and remains exactly zero.
- Scale-40 normalized logits execute in FP32 under disabled autocast with no RNG or data-path effect.
- Geometry masking/distances and deferred failure ordering are sound; all parent CutMix/SAM/EMA semantics remain intact.

## Claude Preflight Authorization Review

- **Reviewer**: Claude Opus via Claude Code 2.1.220, no fallback model, read-only tools
- **Verdict**: PASS
- Claude independently recomputed the weighted timing arithmetic and confirmed all gates: parent drift `0.00993782 <= 0.03`, ratio dispersion `0.00256197 <= 0.005`, median candidate/parent ratio `1.00957979 <= 1.03`, projected steps `25553.21 >= 25000`, and projected total `452.19s < 600s`.
- It confirmed that the sole harness correction reduced a two-element finite predicate with `.all().item()`, occurred before any completed gate measurement, and did not invalidate the first complete preflight.
- It confirmed formula, RNG neutrality, bias/optimizer/SAM/EMA ownership, CutMix identity, production-faithful SAM replay, balanced cadence-31 EMA sampling, exact restoration, and the accuracy-blind paired early trace.
- **Authorization**: no blocking issues; launch exactly one fixed-seed metric run and do not retry its measured outcome.
