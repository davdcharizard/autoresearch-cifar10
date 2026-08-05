**Prioritized Feedback**

1. **Do not accept sub-noise wins.** The goal definition requires ≥96.10%, and `03-experiment-learnings.md` says ~0.1pp is already the run-to-run noise floor from variable epochs. Any 96.03-96.09 result is a null, not a win.

2. **Idea 3 is the riskiest retry of a known failure mode.** EXP-005 and EXP-007 both failed by cutting epochs; EXP-007’s 256→384 widen got only 94 epochs and was still climbing. The 256→320 version is materially different, but still depends on a fragile throughput projection and host load. Add a short step-time smoke/param-count check if run; judge by `num_epochs` plus tail shape, not accuracy alone.

3. **Idea 2’s key assumption is unverified: “epoch surplus” is not the same as proven overfitting.** The brainstorm infers regularization-bound behavior from airbench reaching ~96% in fewer epochs, but current logs do not show train accuracy/test gap. Failure mode: Cutout12 + RandomErasing + LS=0.2 makes the net underfit under the fixed time-based anneal. Mitigate by watching mid-trajectory vs EXP-004; if ep25/ep50 collapse, back off erasing.

4. **Idea 2’s “throughput-free” claim should be treated as a hypothesis.** `RandomErasing` runs in DataLoader workers, but it is still per-sample Python/tensor work on CIFAR batches. If `num_epochs` falls materially below the EXP-004/006 142-150 band, the result is confounded.

5. **Idea 1 is technically sound but its alpha argument is overstated.** `train.py:243-249` really does decay BN affine params and `GatedResidual.alpha`, but SGD weight decay on `alpha` is proportional to `alpha`, not a constant force; unless alpha grows large, the term is probably tiny versus the measured gradient. Record final alpha if possible. The BN no-decay part is the real evidence-backed mechanism.

6. **Idea 1 likely has the cleanest attribution but the weakest headroom.** Bag-of-Tricks/fastai/timm support no-decay on BN/bias, and the code change is safe, but this recipe is already heavily regularized with LS=0.2, Cutout, EMA, and wd=5e-4. Most likely outcome is a few hundredths pp, below the bar.

7. **No hard-constraint violations found.** All three can be done by editing only `train.py`; RandomErasing uses existing torchvision; optimizer grouping and width edits preserve the frozen eval harness and one-eval-per-epoch rule.

**Scored Verdict**

| Idea | Evidence / Reasoning | Potential Impact |
|---|---:|---:|
| 1. Decoupled weight decay | **7/10**: Standard, code-local, and throughput-free, but the ReZero-alpha benefit is weakly quantified. | **4/10**: Most likely sub-noise on this already-regularized 96% recipe. |
| 2. Cutout12 + light RandomErasing | **6.5/10**: Mechanism matches the regularization-bound diagnosis, but overfitting is inferred rather than measured. | **6.5/10**: The combined augmentation has the best chance among throughput-free ideas to clear +0.1pp. |
| 3. Layer2 256→320 widen | **6/10**: Strong in-repo capacity evidence from EXP-004, but it revisits the EXP-007 under-anneal family. | **7/10**: Highest ceiling if epochs remain sufficient, but also the largest regression risk. |

**Pick: Idea 2, stronger augmentation.**

It wins because it attacks the stated limiter most directly: saturated/regularization-bound behavior with surplus epochs, without intentionally spending GPU throughput. Idea 1 is cleaner but probably too small to beat the noise floor. Idea 3 is scientifically informative and has a higher ceiling, but after two capacity under-anneal failures it is a worse bet for actually clearing ≥96.10% in the next single run.
