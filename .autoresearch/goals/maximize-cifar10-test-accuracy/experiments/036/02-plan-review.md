# Plan Critique - EXP-036

Cross-model review was intentionally unavailable in this strictly offline/local
session. The fallback local plan critic returned **REVISE**.

1. **[Construction RNG] `torch.manual_seed` would contaminate CUDA state.** A CPU-only fork restores only CPU, but `torch.manual_seed(36036)` seeds all devices. Use `torch.random.default_generator.manual_seed(36036)` inside the CPU fork and prove CPU restoration plus byte-identical CUDA state. A local API probe confirmed this method leaves CUDA state unchanged.
2. **[Exposure classification] Plan and proposal disagree below 130 realized passes.** Exposure is an interpretive mechanism floor, not a task hard constraint. The reviewed plan explicitly supersedes the proposal: a complete valid score below 130 remains the sole goal result and cannot be rerun, while the mechanism is inconclusive.
3. **[Gradient semantics] Per-tensor nonzero common gradients are brittle.** Require every present gradient finite, nonzero aggregate backbone/classifier norms, and finite nonzero gradients/updates for both new matrices. Keep reported group magnitudes diagnostic.
4. **[Timing semantics] Alternating arms need exact reset state.** Start each paired window from fresh cloned common parameters, BN buffers, optimizer state, CPU/CUDA RNG, and deterministic inputs/targets; reset peak memory immediately before candidate timing.

## Verdict

**Revise, then proceed.** The fixed intervention, source scope, H20 exposure
formula, sole-score rule, and primary threshold are otherwise sound.
