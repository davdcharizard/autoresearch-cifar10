# EXP-017 Candidate Review

## Prioritized Feedback

1. **Idea 1 is strongest, but the surrogate is not “exact GhostBN.”**  
   [idea-01.md](/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/017/proposals/idea-01.md:4) replaces ghost-stat noise with independent per-sample Gaussian affine jitter. Real GhostBN perturbations are data-dependent, group-shared, and layer-scale-dependent. That hidden assumption is the main risk. Improve it by gating first to the proven EXP-016 site, `layer3`, and calibrating `σ` from measured full-vs-ghost stat ratios, rather than starting all BN sites at fixed `0.10/0.20`.

2. **Idea 1 correctly attacks the diagnosed limiter.**  
   EXP-016 showed layer3 GhostBN beat same-session control by `+0.24pp` despite fewer epochs, but was capped by fused-BN throughput loss. A fused-BN-preserving surrogate directly targets that cost mechanism. Under-anneal risk is low, but not zero: random tensor generation at all 10 BN sites can still cost steps. Keep the `num_epochs >=142` rejection gate.

3. **Idea 2 has a concrete execution trap: spatial shape/padding.**  
   [idea-02.md](/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/017/proposals/idea-02.md:4) says `MaxPool2d(2,stride=1)` then blur stride-2. Without correct padding, the 32→16→8→4 chain in [train.py](/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/train.py:149) can become 32→15→7→3 and break or distort the final `MaxPool2d(4)`. Use a standard padded BlurPool implementation and smoke exact tensor shapes before the run.

4. **Idea 2’s evidence is externally plausible but locally weak.**  
   BlurPool is a real inductive-bias pivot, but this goal already tested several shift/transform-related levers: translate TTA was sub-noise over mirror TTA, and RandAugment tied baseline. The likely failure mode is “improves robustness/shift stability but not this already-cropped CIFAR-10 metric enough to clear `96.48`.” If run, make later-pools-only or compile-funded BlurPool the primary cell, not merely fallback, to avoid an under-anneal-confounded all-pools loss.

5. **Idea 3’s identity init is under-specified and can be wrong.**  
   [idea-03.md](/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/017/proposals/idea-03.md:4) says initialize sigmoid SE gate near 1. Plain sigmoid near 1 requires a large positive bias, which saturates the gate and weakens gradients; bias 0 gives gate 0.5 and changes the baseline function. Use `gate = 2 * sigmoid(z)` with zero-initialized second FC for exact gate 1 and live gradients.

6. **Idea 3 is capacity-adjacent despite the wording.**  
   SE is not width, but it is still added representational machinery. EXP-014 showed properly annealed extra capacity did not help this architecture, so the proposal needs a stronger reason why channel attention specifically escapes that verdict. It is cheap enough to test, but its local evidence is weaker than Idea 1’s BN-noise signal.

7. **All three must be judged against both gates, not just same-session delta.**  
   EXP-016’s control drew low (`96.14`), while the absolute bar remains `>=96.48`. A weak control cannot make a result valid unless the winning cell also clears the stored baseline by `+0.10pp`.

## Scored Verdict

| Idea | Evidence / Reasoning | Potential Impact | Verdict |
|---|---:|---:|---|
| **1. Throughput-free BN-affine noise** | **8/10**: best local evidence by far; directly follows EXP-016’s positive GhostBN signal and removes the measured throughput tax, but depends on a surrogate-noise assumption. | **7/10**: a true `+0.2pp` BN-noise effect at full epochs would clear `96.48`; fixed all-site Gaussian noise may tie or over-regularize. | **Best pick. Run with layer3-first/calibrated refinement.** |
| **2. BlurPool** | **5/10**: solid paper mechanism, but local shift/transform evidence is already mostly saturated and implementation has shape/throughput traps. | **6/10**: could clear the bar if anti-aliasing transfers, but expected gain in this cropped/TTA’d CIFAR setting is likely small. | Plausible pivot, second choice. |
| **3. SE attention** | **4/10**: valid literature mechanism, but local reasoning is mostly generic and capacity-saturation history cuts against it; init detail is risky. | **6/10**: SE can be high value-per-param in general, but this small saturated ResNet-9 has no local positive channel-attention signal. | Third choice unless identity init is fixed and other ideas fail. |

## Pick

**Run Idea 1: Throughput-free BN-affine noise.**

It is the only candidate with direct positive evidence in this exact harness: EXP-016 found BN-stat noise was beneficial relative to a same-session control, and the only clear blocker was the fused-kernel throughput tax. BlurPool and SE are reasonable exploratory pivots, but both lean mainly on external matched-epoch literature and have weaker connection to the observed limiter.

Recommended EXP-017 form: keep Idea 1, but refine it to **layer3-first BN-affine noise**, with `BN_NOISE_MIN_CH=512` or equivalent, plus one mild all-site arm only if throughput is confirmed. The cleanest possible variant would be calibrated noise or compile-funded faithful layer3 GhostBN, but among the three listed ideas, Idea 1 has the strongest merit.

Sources: EXP files in the repo; Hoffer GhostBN paper https://arxiv.org/abs/1705.08741, Zhang BlurPool paper https://arxiv.org/abs/1904.11486, Hu SE paper https://arxiv.org/abs/1709.01507.
