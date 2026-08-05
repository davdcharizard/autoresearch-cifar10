# Idea Review EXP-038

Offline fallback critic:

1. Doubling only `fc.weight` decay is a valid opposite bracket, not an EXP037 rescue; `1e-3` is the symmetric one-point step around `5e-4`. Evidence is directional, not monotonic.
2. Global `7.5e-4` confounds backbone, pooled head, and classifier with an arbitrary magnitude.
3. Epsilon-0.05 smoothing stacks soft-target regularization without a calibration diagnosis.

| Candidate | Evidence | Impact |
|---|---:|---:|
| Double classifier decay | 5/10 | 3/10 |
| Global 7.5e-4 decay | 3/10 | 5/10 |
| Early label smoothing | 2/10 | 4/10 |

**Pick**: Double only terminal classifier decay to `1e-3` as a low-ceiling closure test; a normal-exposure miss closes both directions.
