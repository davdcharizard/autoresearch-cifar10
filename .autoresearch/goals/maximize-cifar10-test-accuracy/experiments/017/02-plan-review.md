**Soundness Verdict:** **not sound as a decisive EXP-016 reproduction.** It is runnable as an exploratory activation-noise test, but several plan claims would make the result over-interpreted.

1. **[Plan §Hypothesis, §Code Changes lines 7, 60-73] NoisyBN is not algebraically equivalent to GhostBN.**  
   True GhostBN perturbs pre-affine normalized activations with group-shared, data-dependent mean/std errors. The plan applies independent per-sample Gaussian noise after affine BN: `y * (1 + σeps) + σγeps`. This also scales BN `beta`, which GhostBN does not. A tie would not close the BN-stat-noise axis, and a win would not prove the EXP-016 mechanism was reproduced.

2. **[Plan §Smoke C / §Code Changes lines 17, 72-74] “EMA/eval buffers stay clean” is overclaimed at full-model level.**  
   A NoisyBN module updates its own buffers before adding its own noise, but downstream BN layers receive upstream noised activations and update buffers from that altered distribution. The proposed smoke can pass for an isolated BN while missing this network-level train/eval-stat shift.

3. **[Plan §Calibration line 19, §Cells lines 29-31] σ calibration can mis-set the actual GhostBN-equivalent noise.**  
   It calibrates on an init model and one batch, then collapses separate additive and multiplicative ghost-stat errors into one scalar. It also measures `σ_ghost/σ_full - 1`, while the true multiplicative factor is closer to `σ_full/σ_ghost - 1`. This makes cA/cB only loosely tied to EXP-016’s `g=128` noise.

4. **[Plan §Throughput lines 22-24, §Abort line 91] “Throughput-free” is not guaranteed.**  
   The implementation adds two RNG tensors plus elementwise ops at three layer3 sites, including a large `[512,512,8,8]` activation. The probe is necessary, but if layer3 noise is >8% slower or `num_epochs <142`, the plan has no real mitigation left while still framing the cell as epoch-neutral.

5. **[Plan §Verdict lines 37-38, §Verification lines 103-107] Single-run win logic is still vulnerable to low-control/high-cell noise.**  
   EXP-016 showed c0 can draw low by ~0.24pp. The plan only requires confirmation when the noise cell is <~0.15pp above c0, not when the absolute gain over stored 96.38 is barely above the 96.48 bar. A 96.49 result with a low c0 could be recorded from one noisy high draw.

6. **[Plan §Hypothesis line 7, §Verification lines 104-106] The null conclusion is too strong.**  
   If NoisyBN ties at healthy epochs, the plan says BN/activation-stat noise is redundant and EXP-016 was likely a weak-c0 artifact. That conclusion does not follow because the tested perturbation removes GhostBN’s group-shared, data-dependent structure and changes downstream BN behavior.

7. **[Plan §Gradient smoke line 18] The smoke wording may falsely fail on the frozen whitening conv.**  
   Current `train.py` has a frozen whitening conv with `requires_grad=False`; “all conv grads finite” is too broad unless it explicitly excludes non-trainable parameters.

---

## Resolutions (folded into 02-plan.md)

1. **NoisyBN ≠ GhostBN / scales β** → FIXED the code: division-free `y' = y + (y−β)·σε_mul + γ·σε_add` perturbs ONLY the normalized x̂, β untouched (faithful to GhostBN's structure). Remaining surrogate gaps (independent-per-sample vs group-shared; Gaussian vs data-dependent) are now explicitly ACKNOWLEDGED in the design note, not claimed away.
2. **EMA/eval cleanliness overclaimed at network level** → Smoke C reworded to assert PER-LAYER buffer math only; the inherent network-level train/eval stat shift (downstream BN on noised activations) is flagged as intrinsic to activation noise (as in GhostBN/dropout), with the per-epoch EMA eval curve (ep25, best≈final) as the empirical check.
3. **σ calibration form/method** → multiplicative corrected to `σ_full/σ_ghost − 1`; average over ≥4 real batches; measure σ_add* and σ_mul* separately, set σ_cal ≈ max(·); NoisyBN gained `mul`/`add` toggles for per-component noise if they diverge.
4. **Throughput not guaranteed** → M2 mitigation ladder added: layer3-only (3 sites) → mul-only (`add=False`) → confound flag; eps tensors are [512,C,1,1] (small) so cost is expected near-zero; re-probe after any mitigation.
5. **Single-run win vulnerable to low-c0/high-cell draw** → confirmation re-run made MANDATORY for ANY apparent win (not just hairline); `improvement` requires replication of (≥96.48 AND >c0+0.1pp) across both the original and a fresh same-session c0 re-run.
6. **Null conclusion too strong** → softened: a tie demotes but does not close the GhostBN axis (surrogate drops group-shared/data-dependent structure); the faithful closer (compile-funded layer3 GhostBN) is named; a confirmed win is attributed to "throughput-free BN/activation noise," not "EXP-016 mechanism reproduced."
7. **Gradient smoke too broad (frozen whitening conv)** → restricted to `requires_grad=True` params only.
