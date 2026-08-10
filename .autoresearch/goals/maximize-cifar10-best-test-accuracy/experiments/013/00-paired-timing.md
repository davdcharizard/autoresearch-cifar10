# EXP-013 Fresh Paired Timing Gate

**Device**: one idle NVIDIA H20
**Protocol**: five alternating fresh-process control/candidate pairs; 100 warmups plus 500 synchronized H2D/forward/hard-or-soft-CE/backward/SGD steps per trial; fresh model and optimizer state

| Trial | Control mean | Candidate mean | Control images/s | Candidate images/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.9199 ms | 18.2549 ms | 11,722 | 14,024 |
| 2 | 10.8185 ms | 18.2653 ms | 11,832 | 14,016 |
| 3 | 10.9094 ms | 18.1738 ms | 11,733 | 14,086 |
| 4 | 10.8437 ms | 18.2380 ms | 11,804 | 14,037 |
| 5 | 10.8094 ms | 18.2225 ms | 11,842 | 14,049 |

## Summary

- Median trial mean: control 10.8437 ms; candidate 18.2380 ms.
- Candidate/control step ratio: **1.68189**; required <=1.6667: **FAIL**.
- Image-throughput ratio: **1.18914x**; required >=1.20x: **FAIL**.
- Ratio-projected steps / slots: **15,992 / 4,093,952**; required >=16,139 / 4,131,000: **FAIL**.
- P95 image-throughput ratio: 1.18851x; required >=1.15x: pass.
- Trial-mean CV: control 0.474%, candidate 0.197%; required <3%: pass.
- Peak allocation: control 598.7 MB, candidate 1,120.2 MB; required candidate <1,500 MB: pass.

## Verdict

The 20% image-exposure premise misses by 1.09 percentage points with very stable trials. This is a research feasibility no-go, not infrastructure noise. Per the reviewed plan, do not run loader/wall gates or the full seed-42 accuracy experiment, and do not relax the threshold, substitute batch 512, add fusion/mixed precision, or change LR/batch as a fallback.
