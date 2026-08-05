# Idea Review EXP-037

Offline fallback critic verdict:

1. Classifier-only zero decay is the only clean one-run treatment. It preserves EXP036 and gives exact pre-step attribution, but EXP007 and the tiny 1,280-element scope create a weak prior.
2. Fixed-scale cosine classification bundles two normalizations, bias removal, and arbitrary scale 10; reject without a prospective training-derived scale.
3. Classical momentum materially changes update amplitude without LR compensation, while EXP036 shows no endpoint instability; reject.

| Candidate | Evidence | Potential impact |
|---|---:|---:|
| Exclude only `fc.weight` from decay | 4/10 | 3/10 |
| Fixed-scale cosine classifier | 2/10 | 6/10 |
| Classical momentum 0.9 | 2/10 | 4/10 |

**Pick**: Exclude only `fc.weight` from decay as a low-prior closure test. Preserve pooled-head matrices at `5e-4`, require 94.58%, and prohibit intermediate-decay or head-tuning rescues.
