# EXP-013 Blind Idea Review

## Feedback

1. EMA assumes unobserved late-iterate noise; terminal-best histories make lag the main risk. Reject only this exact window on a miss.
2. EMA BatchNorm moments are coherent but approximate. Require complete keys, finite FP32 shadows, unchanged identities/optimizer references, and normal/exception restoration.
3. Transition smoothing repeats negative soft-target directions and shortens the useful hard-label tail.
4. Zero endpoints are technically sound but lack local efficacy evidence for this shallow WRN.
5. No finalist has a fatal scope or reward-hacking issue.

## Scores

| Candidate | Evidence | Impact |
|---|---:|---:|
| Late Whole-State EMA | 7.0/10 | 6.0/10 |
| Zero-Initialized Residual Endpoints | 6.0/10 | 6.5/10 |
| Transition-Only Label Smoothing | 5.5/10 | 5.0/10 |

## Pick

**Late Whole-State EMA.** It is the best-supported remaining generalization treatment and leaves the accepted live training trajectory unchanged. Test only 65% start, 0.999 decay, whole-state averaging; no recalibration, combination, or post-result tuning.
