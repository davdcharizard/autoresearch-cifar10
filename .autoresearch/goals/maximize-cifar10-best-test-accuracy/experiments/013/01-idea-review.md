# Adversarial Review — EXP-013 Candidate Ideas

## Prioritized Feedback (most important first)

**1. Idea-02 (Late Weak-Tail Averaging) — the local evidence actively argues *against* the mechanism. [near-fatal]**
The proposal's own foundation undermines it. EXP-010 finished with `final == best` and EXP-002's dense-tail probe found a best-vs-final gap of just **0.01 points** (`03-experiment-learnings.md` Low-Importance / `experiments/002` §Results). That means the terminal trajectory is smooth and monotonically improving, not noisy/oscillating. Uniform averaging pays off precisely when there is terminal *variance to cancel*; on a monotone-improving trajectory a uniform mean of ~7 endpoints is pulled *backward* by the earlier, lower-accuracy, less-annealed iterates — it is dominated by simply taking the final online point. Compounding this: from 90% onward LR is already ~0.005 and decaying, so the ~7 endpoints (separated by ~390 updates) are highly correlated — the opposite of the diverse high-constant-LR samples SWA relies on. The proposal names both risks ("uniform mean can lag an improving online iterate," "checkpoints too similar") but does not resolve them.
*Path to improve:* the honest fix is a mechanism change the scope forbids (an EMA/last-weighted scheme, or a constant-LR SWA excursion) — i.e., this needs to become a different experiment. As pre-registered (uniform mean, replace-online-eval), the evidence predicts lag, not gain.

**2. Idea-02 — replacing the online evaluation can *hide* a new online best. [design flaw]**
Because EXP-010's best is its final online iterate, evaluating only the averaged model at eligible endpoints forfeits exactly the online peaks that produced the current frontier, and evaluating both is forbidden. So even in the null case where averaging neither helps nor hurts, you risk scoring *below* 94.15% by never observing the online best. *Path:* not resolvable within the "one eval, averaged only" constraint — another reason this is a mechanism mismatch.

**3. Idea-01 (Batch-256) — central unverified assumption: that added image exposure moves the *diagnosed* limiter. [most important gap, but addressable]**
The accuracy bottleneck is stated as "generalization under a short strong phase" (`02-system-understanding.md` §Current Bottleneck), and the same doc warns "larger batches reduce beneficial gradient/BN noise." Batch-256 improves throughput/exposure (a *systems* lever) while attenuating the gradient/BN stochasticity that aids CIFAR generalization at fixed epochs — it may push the wrong lever. **However**, the proposal under-uses its strongest counter-evidence: EXP-010 finishes at `final==best` (still improving at budget end → the model is *exposure/underfit-limited*, not overfit), and EXP-007's width-2 win came *despite 29.2% fewer updates* (`04-results.tsv` 007), showing updates are not the binding constraint and more effective epochs plausibly help. *Path:* lead the diagnosis with these two facts — they convert "exposure is untested" into "the frontier shows underfit-at-budget, and capacity/exposure already beat update count once." That is the single best available justification and it is currently buried.

**4. Idea-01 — LR 0.2 with no warmup on a *short* 240s plateau is the largest execution risk. [addressable]**
Linear scaling is first-order only; the doc concedes curvature/BN/momentum can make the doubled step overshoot or select a sharper basin, and momentum-0.9's horizon now spans 2× the images. The predeclared no-warmup stance is defensible (2× is not the extreme-batch regime), but a single overshoot in the short plateau is unrecoverable within scope. *Path:* keep no-warmup (correct call for scope purity), but explicitly gate on the 87.08% switch checkpoint as a *diagnostic* (already present) and pre-register that a sub-87.08% switch attributes the loss to LR-0.2 sharpness rather than exposure — so a null result is interpretable. This is mostly sound as written.

**5. Idea-05 (Avg-Max Pooling) — uncontrolled feature-scale shift is a real confound the "parameter-free/identical-init" framing hides. [addressable]**
Post-ReLU max ≥ mean, so `lerp(avg,max,0.5)` raises classifier input magnitude with an *unchanged* classifier init (proposal §Failure Mechanisms, "Feature-scale shift"). Early logit scale and effective optimization geometry therefore change — the intervention is not the clean "only the readout statistic changed" test it's presented as. *Path:* this cannot be neutralized without touching classifier scale (out of the declared scope), so at minimum the hypothesis should stop claiming the confound is absent and pre-register the logit-scale change as part of the measured net method.

**6. Idea-05 — the "dilution" premise is weak at an 8×8 map, and evidence is thin. [moderate]**
Global average is over only 64 cells; a compact CutMix donor/part is not diluted across a large map the way the motivation implies. And CutMix targets are *area-proportional* while max is *area-insensitive* (proposal acknowledges), so max pooling is arguably *mis*-aligned with the very CutMix signal it claims to serve. There is no cited result that avg-max helps this recipe — it is a plausibility argument only. *Path:* strengthen by pointing at a concrete diagnostic prediction (e.g., first-weak-tail accuracy should *rise* if localized readout helps), so the run yields mechanism evidence even on a null.

**7. Idea-04 (Zero-Gamma) — self-identified fatal Option-A dead-channel deadlock. [fatal, correctly retired]**
The proposal's own Gate 5/6 analysis is correct: with parameter-free Option-A zero-padding, newly padded transition channels have zero shortcut + zero residual (γ=0) → exact-zero pre-ReLU → zero ReLU derivative → permanently dead γ/β; ~96 of 128 final channels never activate. The cited Goyal result used *projection* shortcuts, which don't impose this boundary — so the literature support does not transfer to this architecture. This is a genuine hard-constraint-compatible retirement, not a rescuable timing miss. Correctly disqualified as stated; do not launch a knowingly capacity-disabled run.

**8. Idea-03 (TorchInductor) — correctly retired at feasibility.** PyTorch 2.9.1 rejects `torch.compile` on Python 3.14 before capture; dependencies are frozen (`01-definition.md` hard constraints). No legal implementation exists. Sound, auditable no-go. Not a pickable candidate.

**Cross-cutting:** four of five proposals carry heavy, mostly-sound preflight/timing-gate machinery. That is appropriate rigor, but note two ideas (03, 04) spend most of their length auditing why they *cannot* run — treat them as documented dead ends, not finalists.

---

## Scored Verdict

Scored on (a) evidence & reasoning, (b) potential impact. Ideas 03 and 04 are self-retired (feasibility / structural deadlock) and are not scorable finalists.

| Idea | Evidence & Reasoning | Potential Impact |
|---|---|---|
| **01 Batch-256 + 2× LR** | **7/10** — quantified memory/throughput headroom (0.61% VRAM, backward-bound) + measured 256-knee; `final==best` and the width-2 "gained with 29% fewer updates" result are affirmative exposure-limited evidence, though under-emphasized. Risk (noise reduction vs the diagnosed generalization limiter) is real and honestly stated. | **6/10** — credible +0.10–0.30; ceiling is moderate (predicted 94.25–94.45) but backed by a concrete resource and a clean scaling rule. |
| **05 Avg-Max Pooling** | **5/10** — plausible mechanistic story tied to the validated CutMix win, and it preserves the healthy strong-phase fit; but no direct evidence, an 8×8 map weakens the dilution premise, and the feature-scale-shift confound is unaddressed. | **6/10** — cheap, non-destructive representation bet; if the readout hypothesis holds it could clear the gate, but the CutMix-area/max mismatch caps the upside. |
| **02 Weak-Tail Averaging** | **3/10** — mechanism contradicted by the local evidence (monotone-improving `final==best` trajectory + low decaying LR + correlated endpoints); replace-online-eval can hide the online best. Careful implementation cannot rescue a poorly matched mechanism. | **4/10** — best case is a small centering gain; realistic case is lag below the online frontier. |
| 03 TorchInductor | — retired (infeasible on Py3.14) | — |
| 04 Zero-Gamma | — retired (Option-A dead-channel deadlock) | — |

### Pick: **Idea-01 — Batch-256 Fixed-Time Training with Linear LR Scaling**

It is the only live candidate whose justification rests on *measured, quantified* headroom (backward-dominated step, <1% VRAM used, an empirically located batch-256 throughput knee) rather than a plausibility argument, and its scaling rule is theoretically grounded and fully pre-registered. Critically, the two facts it currently under-sells — EXP-010 finishing at `final==best` (underfit at budget) and EXP-007 winning *with 29% fewer updates* — are direct affirmative evidence that this frontier is exposure/capacity-limited, not update- or overfit-limited, which is exactly the regime where trading updates for ~20 more effective passes can pay off. Its main risk (reduced gradient noise vs the diagnosed generalization limiter, plus no-warmup LR 0.2) is genuine but honestly bounded and diagnosable via the 87.08% switch checkpoint.

It wins over **Idea-05** on strength of evidence: 05's dilution premise is weak at an 8×8 map, its max/CutMix-area alignment is questionable, and it carries an unacknowledged feature-scale confound — all plausibility, no measurement. It wins decisively over **Idea-02**, whose mechanism is contradicted by the very trajectory data it cites. **Ideas 03 and 04 are correctly self-retired and cannot be selected.**

**Required refinement before running Idea-01:** promote the `final==best` and "width-2 gained despite fewer updates" evidence to the front of the diagnosis to close the "does exposure move the limiter?" gap; and pre-register the sub-87.08% switch checkpoint as the discriminator that attributes any null to LR-0.2 sharpness rather than to the exposure hypothesis, so a single-seed no-improvement is still interpretable.
