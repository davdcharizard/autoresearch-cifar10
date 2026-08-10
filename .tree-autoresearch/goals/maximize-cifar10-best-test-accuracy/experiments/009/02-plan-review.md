# Claude Adversarial Plan Review: EXP-009

Claude Opus reviewed the plan against the goal, parent source, EXP-002 evidence, learnings, and brainstorm. The material concerns adopted into the plan are:

1. `F.linear` remains BF16 under outer autocast unless the SE block explicitly disables autocast; production dtype assertions must cover descriptor and both affine outputs.
2. A global maximum gate deviation is outlier-sensitive and has no saturation ceiling. Require per-module mean activity, fraction beyond two BF16 ULPs, and saturation bounds.
3. The fixed 5x LR gives peak gate LR 1.0; live-gate smoke must reject both inactivity and gates pinned near zero/two.
4. Evaluation is FP32 under inference mode. Run an eval-mode smoke and ensure training activity counters neither mutate nor include evaluation batches.
5. Avoid per-forward host synchronizations. Accumulate device-side finite/activity statistics and inspect after synchronized steps or at exit.
6. The 26,000-step floor implies a ratio about 1.075, making the stated 1.10 ratio irrelevant. Project only as `27,950 / measured_ratio` from a matched parent harness.
7. Total runtime tends to fall with fewer epochs under a fixed charged budget; projected evaluation count, not an easy 550-second ceiling, is the relevant max-over-epoch exposure diagnostic.
8. Any step/evaluation loss is part of the package and must accompany causal claims even when the formal metric verdict remains valid.
9. Repair permissions must explicitly exclude all locked SE design values.
10. Define standardization as biased channel variance with `rsqrt(var + 1e-5)`.
11. Require bitwise initial forward parity, post-`model.apply` zero-final-layer identity, and CutMix-generator isolation, not only shared tensor equality.
12. Materialize actual parent source and all harnesses outside the repository.
13. Verify visible CUDA UUID/PCI identity against physical `nvidia-smi -i 0`, and enforce stable co-tenancy through benchmark and launch.
14. Keep `se:` audits outside the frozen summary and require exactly one primary/memory summary key.
15. A formal 95.33-95.52 improvement must be explicitly labeled below the preregistered 0.30-point evidence bar.
