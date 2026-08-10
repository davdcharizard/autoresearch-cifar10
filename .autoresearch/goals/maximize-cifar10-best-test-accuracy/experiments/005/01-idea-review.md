I have everything I need. Emitting the review.

---

# EXP-005 Adversarial Idea Review

## Prioritized Feedback (most important first)

### 1. CROSS-CUTTING, near-fatal for the whole round: the +0.1 pp bar sits *below* the noise floor, and all three ideas reshuffle the augmentation RNG stream
- The acceptance bar is `92.40%` — **+0.10 pp = 10 test images out of 10,000**. EXP-004's own analysis (`experiments/004/04-analysis.md`) and its plan review (`02-plan-review.md:concern 1`) already established two facts the EXP-005 brainstorm ignores: (a) single-seed ResNet-20/CIFAR-10 variance is well above 0.1 pp, and (b) *any* change to the RandAugment policy "necessarily changes the fixed-seed augmentation stream, so exact causal size cannot be separated from draw changes in one run." **All three candidates change the augmentation policy** (magnitude, num_ops, or switch timing) → all reshuffle the worker RNG stream → a single-seed pass at 92.40% is indistinguishable from a favorable draw. EXP-004 survived this because it cleared the bar by **+0.37 pp** (a real signal); a fine-tuning delta of +0.0–0.2 pp will not.
- This is functionally the outcome the no-seed-hacking constraint exists to prevent, entering through the augmentation RNG instead of the seed knob. The brainstorm's "Combinations/Collected Ideas" sections are stubbed and contain **no variance plan**.
- **Concrete fix (applies to whichever idea is chosen):** before running, commit to a variance-aware protocol — e.g. report the delta against a 2–3 seed estimate of EXP-004's spread (using extra seeds to *estimate variance*, not to cherry-pick the max, is legitimate due diligence, not seed hacking), and do not claim `improvement` on a within-noise single run. Absent this, the entire EXP-005 round risks producing an un-attributable "pass."

### 2. Magnitude 9 (idea A): the "torchvision default" justification is not CIFAR evidence, and the downside is real
- **Evidence gap:** "magnitude 9 is torchvision's default" is not evidence it beats 7 on a 270k-param net at 32×32. The RandAugment note itself warns of "excessive distortion for a small ResNet20" (`knowledge/papers/randaugment.md`), and Cubuk et al.'s CIFAR-optimal magnitudes for their (much larger) Wide-ResNet were modest, not the generic default. "Adjacent to the successful point" argues the direction is *plausible*, not that it's *up*.
- **Concrete failure mechanism:** EXP-004's final strong checkpoint was only **84.60%** (`04-analysis.md`) — the augmented distribution is already hard for this net in ~100 epochs. Pushing magnitude 7→9 raises task difficulty further, risking the short-horizon underfit the EXP-004 review named as the dominant risk, or destroying 32×32 semantics outright. The weak tail may not fully recover an over-perturbed representation.
- **Fix:** if pursued, test as a small ladder (7/8/9) or pair with a slightly longer weak tail to absorb the added difficulty — but this compounds concern #1.

### 3. Two ops @ mag 5 (idea B): un-gated throughput risk + changes two knobs at once
- **Throughput (the EXP-003 trap):** EXP-004 measured the strong loader at 165–176 b/s vs the ~7.8 ms GPU step (~128 b/s-equivalent) — the loader was *just barely* faster than the GPU, so augmentation stayed free. Doubling PIL work (N=2) could drop the loader **below** the GPU step time, making it the bottleneck and cutting optimizer steps — the exact update-loss mechanism that sank EXP-003 (`04-results.tsv:003`). The brainstorm mentions "worker preflight can determine" this but sets no gate. **Fix:** mandate a loader-throughput preflight identical to EXP-004's, and only proceed if the augmented rate stays above the GPU step rate with margin.
- **Confounded isolation:** N=1→2 *and* M=7→5 change simultaneously, so a null/positive result cannot be attributed to either. **Fix:** hold M and vary only N, or vice versa.
- **Weak ceiling argument:** the paper's N≥2 gains are on 36M-param Wide-ResNets; for a 270k net in a short horizon, more composed distortion is more likely to underfit than to "cover richer neighborhoods."

### 4. Switch off at 75% (idea C): the "late peak ⇒ refinement-limited" inference is over-read
- **The one real weakness:** EXP-004 peaking at epoch 98/99 is cited as evidence the weak phase is "refinement-limited." But a cosine anneal *always* peaks near its endpoint — peak-at-end is expected from the schedule, not proof that more weak-phase time helps. The genuinely strong signal is the **+6.83 pt jump at the switch**, which shows clean-distribution/BN adaptation matters — that supports giving BN some high-LR clean adaptation *before* annealing, which is exactly what 5% at lr=0.1 does. So the mechanism is sound; the *specific* supporting observation is partly misattributed. **Fix:** justify idea C on the switch-jump + BN-resettle argument, and drop the "late peak" claim.
- **Uncertain sign, but smallest downside:** it trades 5% strong-augmentation exposure (the EXP-004 regularization engine) for 5% clean high-LR adaptation. Net sign is uncertain, but overhead is ~zero, there's **no throughput or semantic-destruction risk**, and it's the cleanest single-knob isolation (only the switch boundary moves; augmentation strength is untouched). It is the only candidate grounded in *this goal's own* trajectory rather than generic appeals.
- **Minor:** requires a distinct `AUG_SWITCH_FRACTION=0.75` decoupled from `LR_HOLD_FRACTION=0.8` (`train.py:25,207,281`) — low complexity, but verify the transform swap fires at 0.75 while the LR still keys off 0.8, and that only clean-crop/flip batches run in 75–80% at lr=0.1.

### 5. Meta-observation: the candidate space is narrowed prematurely to augmentation micro-tuning
- The brainstorm fixes "worker lifecycle, evaluation cadence, model, LR timing" and confines all three ideas to one augmentation knob each. Given the round is fighting a noise-floor bar, these are inherently low-ceiling. Higher-ceiling levers deferred from EXP-004 remain untouched — preactivation ResNet-20, or exploiting the massive unused headroom (peak VRAM **330 MB of 98 GB**, `04-analysis.md`) via a wider/deeper model. Not a flaw in the three ideas per se, but worth flagging that the expected value of *any* winner here is capped well below EXP-004's +0.47.

---

## Scored Verdict

| Idea | Evidence & Reasoning | Potential Impact |
|---|---|---|
| **Switch off at 75% (C)** | **6/10** — cleanest single-knob isolation, zero throughput/underfit risk, mechanism (high-LR clean BN adaptation before anneal, backed by the +6.83 pt switch jump) is the most goal-specific of the three; docked for over-reading "late peak" as refinement-limited. | **5/10** — modest, uncertain sign, near noise, but the smallest downside and the most interpretable result. |
| **Two ops @ mag 5 (B)** | **5/10** — RandAugment does parameterize N, but the specific N=2/M=5 point isn't shown superior and two knobs move at once, muddying attribution. | **4/10** — plausible breadth gain, but un-gated throughput risk can re-trigger EXP-003-style step loss; ceiling weak for a 270k net. |
| **Magnitude 9 (A)** | **4/10** — "torchvision default" is not CIFAR evidence; adjacency argues plausibility of direction, not sign; the note warns of over-distortion at this scale. | **4/10** — real underfit/semantic-destruction downside given the 84.6% strong checkpoint; modest, coin-flip upside. |

## Winner: **Switch Augmentation Off at 75% (idea C)**

It wins on merit. It is the only candidate whose mechanism is grounded in *this goal's own strongest empirical signal* — the +6.83-point jump when EXP-004 swapped to clean crop/flip, which demonstrates that clean-distribution/BN adaptation is where accuracy is realized. It gives BatchNorm and the clean objective a slice of *high-LR* adaptation before annealing begins, a lever the current recipe never exercises (the transform and LR both switch at 80%, so clean adaptation only ever happens under a falling LR). Crucially, it isolates a single boundary knob with **no throughput risk (idea B) and no semantic-destruction/underfit risk (idea A)**, so its result is the most attributable — the property that matters most when the acceptance bar sits at the noise floor.

**Required refinements before adopting idea C:** (1) commit to the variance-aware protocol from feedback #1 — a single within-noise pass cannot be called `improvement`; (2) drop the "late peak ⇒ refinement-limited" justification and rest the case on the switch-jump/BN argument; (3) confirm the decoupled `AUG_SWITCH_FRACTION` fires the transform swap at 0.75 while LR holds to 0.8, so 75–80% runs clean crop/flip at lr=0.1 as intended.

Both alternatives are weaker: idea A bets on an unsupported direction with a real downside, and idea B changes two knobs behind an un-gated throughput risk that could re-run EXP-003's failure. Neither offers a higher, better-argued ceiling than C.
