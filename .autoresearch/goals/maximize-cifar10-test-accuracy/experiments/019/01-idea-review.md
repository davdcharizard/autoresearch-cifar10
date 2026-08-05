PRIORITIZED FEEDBACK

1. The threshold is not “beat 96.38 once”; it is a real ≥96.48 signal against same-session noise. All three ideas are plausible but small. Given `03-experiment-learnings.md` and `project-insights.md`, the prior is that any within-DavidNet tweak is likely sub-noise unless it changes the learned function in a way the prior 13 nulls did not.

2. idea-02’s `tail` variant is the biggest concrete flaw. Holding LR at `0.05 * PEAK_LR` through the end contradicts the project’s strongest lesson: complete the low-LR anneal. With `PEAK_LR=0.4`, the run ends at LR 0.02, not near zero. That is an under-anneal risk disguised as a tail. If schedule is chosen, prefer `cos`; revise `tail` to finish at zero.

3. idea-01 is the only finalist with a genuinely new modeling mechanism: content-adaptive channel recalibration. It is train.py-only, no deps, likely under the epoch-risk gate, and orthogonal to width/depth/optimizer/regularization/downsampling. Main refinement: use identity-preserving `2 * sigmoid` SE gates, not a default 0.5 gate in ungated residual blocks, because current `Residual` blocks are not ReZero-protected.

4. idea-03 is clean but probably too shallow for the diagnosed limiter. Adaptive avg+max pooling is cheap and correct, but it only changes the final linear readout over an already saturated representation. It is very likely a ±0.05-0.10pp shuffle unless the max-only head is surprisingly discarding class signal.

5. Same-session control and confirmation rerun are mandatory. The learning files establish ~0.1pp noise and repeated weak-control artifacts, especially EXP-016/017. Any apparent win below same-session c0 +0.1pp should be treated as no signal.

SCORED VERDICT

idea-01: Squeeze-Excitation channel attention  
Evidence/reasoning: 6.5/10. The mechanism is real and distinct from saturated axes, but ImageNet SE gains may not transfer to this small, heavily augmented CIFAR net.  
Potential impact: 5.5/10. Best upside of the three because it changes feature computation, but expected gain is still likely near the noise floor under this ceiling.

idea-02: One-cycle schedule shape  
Evidence/reasoning: 6/10. Throughput-free and previously flagged, but current triangular one-cycle is already a core validated recipe; `tail` as written risks violating the anneal lesson.  
Potential impact: 4.5/10. Cosine could produce a small basin-selection gain, but after EXP-014 showed extra epochs do not help, pure schedule shape is more likely to tie than break 96.48.

idea-03: AdaptiveConcatPool head  
Evidence/reasoning: 4.5/10. Mechanistically coherent and safe, but weakly connected to the robust generalization ceiling.  
Potential impact: 3/10. Likely sub-noise as a standalone EXP-019; better as a later rider on a stronger representation change.

PICK

Run idea-01, Squeeze-Excitation channel attention, as EXP-019.

It wins because it is the only finalist that adds a materially new functional form while staying close to throughput-neutral. The expected value is still modest, but it has the best chance of producing a real ≥96.48 result rather than another schedule/readout-level tie. I would run layer2+layer3 SE with identity-preserving `2*sigmoid` gates, enforce `num_epochs >= 135`, and require same-session c0 plus confirmation before calling it a win.
