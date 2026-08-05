1. [Milestone 3 / Verification NC3] The plan allows env-overridden winning cells, then “bake” defaults after the run without requiring a rerun. That can report a metric produced by a different command/config than the final `train.py`, especially after deleting logs.

2. [Code Changes: env-overridable `LABEL_SMOOTHING`/`CUTMIX_P`] Adding runtime env knobs widens the search surface beyond the concrete chosen method and creates a measurement/bookkeeping hole: final logs do not print `LABEL_SMOOTHING`, `CUTMIX_P`, or `CUTMIX_ALPHA`, so results are not self-describing.

3. [Training step `use_cutmix` coin / helper CPU draws] The “CPU draw = throughput-free” claim is incomplete. `float(torch.rand(1).item())`, `Beta(...).sample()`, and two `torch.randint(...).item()` calls allocate/draw CPU tensors inside every timed step; this avoids CUDA sync but still adds on-budget Python/CPU overhead and possible CPU RNG bottleneck.

4. [CutMix helper `cut_h, cut_w = int(h * r), int(w * r)`] The box size can truncate to zero for high λ, producing many no-op CutMix batches with `lam=1.0`. That changes the effective `CUTMIX_P` downward and weakens the hypothesis without the verification measuring applied/non-empty CutMix rate.

5. [CutMix helper / determinism claim] The plan says CPU draws are deterministic under existing seeds, but DataLoader workers and CPU augmentation already consume RNG state. Moving λ/box/coin draws to the main CPU RNG changes the global CPU RNG stream relative to baseline and couples CutMix draws to any future CPU-side code path changes.

6. [CutMix loss composition] The linear LS composition is mathematically valid for two hard targets, but the plan does not guard `perm == arange` cases. Same-sample pairs create nominal CutMix with unchanged labels but pasted identical images, again reducing effective regularization while still counted as CutMix.

7. [Milestone 1 smoke test] The proposed assertion “mixed differs from inputs only inside the pasted box” is not directly checkable from the helper because it does not return box coordinates. With random `perm`, same-class or same-index regions can also be equal, making the smoke either under-specified or brittle.

8. [Verification NC2 / multi-cell verdict] Running cell-2 only after cell-1 fails converts the experiment into a conditional mini-sweep, but NC2 still treats a single best cell crossing 96.48 as decisive. That inflates false-positive risk near the stated ~0.1pp noise floor.

9. [Milestone 2 throughput guard vs Abort Criteria] The plan has inconsistent thresholds: Milestone 2 says `<135` means fix before trusting, Abort Criteria says only `<110` aborts, Verification says `[142,155]` gates trust but is “not an NC.” This leaves room to accept a below-band result despite the throughput premise failing.

10. [Verification NC3 `grep -c "evaluator.evaluate" train.py == 1`] This check is too syntactic to prove “at most one validation run per epoch”; comments, aliases, wrappers, or conditional calls can evade or falsely fail it. It does not actually validate runtime evaluator call count.

11. [Milestone 3 optional `CUTMIX_P=0.25`] The optional third cell is triggered by trajectory interpretation and can produce a winning result, but it is not part of the primary hypothesis and further expands the hyperparameter search without adjusted success criteria.

12. [Log cleanup / verification evidence] “Remove `run*.log` after recording” conflicts with auditability. Since the plan relies on trajectory details, env-run commands, throughput bands, and summary equality checks, deleting logs before durable capture makes later verification weak.
