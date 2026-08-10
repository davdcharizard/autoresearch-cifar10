# EXP-012 Preactivation Timing Gate

**Device**: one idle NVIDIA H20, 97,871 MiB
**Control**: accepted commit `7c1e7d8`
**Candidate**: canonical full-preactivation width-2 ResNet-20
**Protocol**: five alternating fresh-process trials; fresh model/optimizer state; 100 warmups and 500 synchronized timed steps; batch 128 with alternating hard/probability targets

## Training Results

| Metric | Control | Candidate | Gate |
| --- | ---: | ---: | ---: |
| Median trial mean | 10.9466 ms | 10.8786 ms | ratio <=1.03 |
| Ratio | - | 0.99378x | pass |
| Projected steps | 26,898 | 27,066 | >=26,091: pass |
| Trial-mean CV | 0.501% | 0.292% | <=2%: pass |
| Median p95 ratio | - | 0.98869x | informational |
| Peak allocation | 598.67 MB | 582.67 MB | informational |

## Inference Results

| Metric | Control | Candidate | Gate |
| --- | ---: | ---: | ---: |
| Median mean | 1.9858 ms | 2.0041 ms | informational |
| Ratio | - | 1.00925x | informational |
| Trial-mean CV | 0.547% | 0.536% | <=2%: pass |
| Projected total | 330.7 s | 330.73 s | <540 s: pass |

## Verdict

All pre-registered timing gates pass. The candidate is compute-neutral within measurement variation and is cleared for one full seed-42 run.
