1. **Verification Protocol §5 / Milestone 4:** The pass gate only requires best GBN cell `>=96.48` and treats same-session c0 as “context” ([02-plan.md:100-104]). That permits a false win like c0=96.47, cA=96.48, especially after taking `max(cA,cB)`. The chosen idea requires “AND clearly above same-session c0” ([idea-01.md:31-34]); the plan does not enforce that.

2. **Milestone 1 equivalence smoke:** The `GHOST_SIZE=512` smoke only exercises the bypass branch `return super().forward(x)` ([02-plan.md:52-55]). It does not test the custom ghost path at all: reshape, per-ghost normalization, manual full-batch running-stat update, affine/cast, or channels_last preservation. A broken `GHOST_SIZE=128/64` implementation could pass M1.

3. **Code Changes / running stats + EMA:** The plan claims full-batch running stats make `AveragedModel(use_buffers=True)` sound ([02-plan.md:63-67], [train.py:255-257]) but has no verification that EMA buffers are actually correct after `ema_model.update_parameters(model)`. Gradient smoke does not touch EMA. Add a multi-step check comparing raw BN buffers and EMA BN buffers against expected full-batch-stat EMA behavior.

4. **Code Changes / “clean eval stats” overclaim:** Updating each BN layer from full-batch moments removes only that layer’s per-ghost stat noise ([02-plan.md:63-67]). Downstream BN running stats are still collected from activations produced by upstream ghost-normalized train-mode layers, while eval uses non-ghost running-stat paths via `model.eval()` ([prepare.py:32-35]). The plan treats eval stats as fully clean but does not test raw-vs-EMA eval stability or BN train/eval mismatch.

5. **Milestone 2 throughput model:** The plan frames the cost as mostly “reshape + grouped-stat” ([02-plan.md:20-22], [02-plan.md:73]), but the custom path replaces fused cuDNN BatchNorm with several fp32 reductions, broadcasts, affine ops, casts, and autograd through `var`. That can be much more than reshape overhead across 7 BN sites. If the precheck shows >15% slowdown, “still run but caveat” leaves the main hypothesis under-annealed rather than fixing/aborting.

6. **Milestone 1 / ghost correctness checks:** No smoke verifies per-ghost math: per-group output mean≈0/var≈1, full-batch running_mean/var matching `nn.BatchNorm2d`’s update convention, or output memory format after the custom branch. The g=512 smoke cannot catch a wrong grouping dimension or wrong variance axis in `[G,g,C,H,W]` ([02-plan.md:56-62]).

7. **Proposal/plan mismatch:** The proposal says `GHOST_SIZE` supports `0/512 = standard BN` ([idea-01.md:5-7]), but the planned code uses `N % self.ghost_size` with no zero guard ([02-plan.md:54]). `GHOST_SIZE=0` would crash. If 0 is not supported, remove it from the accepted interface.

8. **Integrity check scope:** `git diff --name-only = train.py only` ([02-plan.md:18]) misses untracked files and does not prove `prepare.py` is byte-unchanged. The plan also writes logs under `experiments/016/` ([02-plan.md:83], [02-plan.md:106]). Use `git status --short` plus a checksum or `git diff -- prepare.py` if the hard constraint is enforced strictly.

9. **GPU contention detection:** The plan logs `nvidia-smi` only before each cell ([02-plan.md:23], [02-plan.md:29]), but Abort Criteria mention foreign jobs appearing mid-cell ([02-plan.md:90]). There is no mid-run sampling, so contention can silently bias `num_epochs`, wall time, and the same-session comparison.

10. **Multiple-comparison handling:** The plan acknowledges `max(cA,cB)` at a 0.1pp floor is unproven ([02-plan.md:34]) but still makes best-of-two the primary decision. This is a measurement-fishing path unless the gate requires a margin over c0 or a confirmation run before recording an improvement.

---

## Resolutions (folded into 02-plan.md)

1/10. **Gate must require clear margin over c0 + confirmation for hairline** → M4 + Verification 5b now require best ≥96.48 AND >c0 by >0.1pp; a <~0.15pp-over-c0 win triggers a confirmation re-run (winning cell + fresh c0) before recording `improvement`.
2/6. **Equivalence smoke only tested the g=512 bypass** → split into Smoke A (g=512/g=0 bypass ≡ nn.BatchNorm2d) and Smoke B (GHOST PATH math: per-group normalized mean≈0/var≈1; manual running stats == nn.BatchNorm2d's full-batch+unbiased update; grouping on batch axis).
3. **No EMA buffer verification** → Smoke C: 5 steps + ema update, assert ema BN buffers = EMA-avg of raw full-batch running stats, shape [C].
4. **"Clean eval stats" overclaim** → softened in Code Changes: full-batch update removes per-ghost NOISE but the standard BN train/eval gap remains (mildly amplified); not a bug, checked empirically via the per-epoch EMA eval.
5. **Throughput: custom path replaces fused cuDNN BN → could be ≫reshape cost; "caveat" leaves hypothesis under-annealed** → M2 now has a concrete mitigation ladder: if >15% slowdown, switch to FUSED `F.batch_norm` on the ghost-folded view (keeping clean C-sized buffers + manual full-batch running update), re-smoke/re-probe; if still slow, apply GBN to later BN sites only; under-anneal confound flagged only as last resort.
7. **GHOST_SIZE=0 crash (N%0)** → forward bypass guard now `ghost_size <= 0 or ghost_size >= N`; 0/512 both = standard BN.
8. **Integrity check missed untracked files / prepare.py bytes** → M1 + Verification use `git status --short` + `git diff --quiet -- prepare.py`.
9. **GPU contention only sampled pre-cell** → added a background mid-cell `nvidia-smi` sample per cell + num_epochs cross-check across cells as the contention detector.
