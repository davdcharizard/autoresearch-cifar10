# EXP-012 Squeeze-Excitation Timing Gate

**Device**: NVIDIA H20, idle at launch
**Control**: accepted commit `7c1e7d8`
**Candidate**: nine identity-initialized r=16 SE gates
**Protocol**: five alternating paired trials; 100 warmups and 500 synchronized timed steps per trial; batch 128; alternating hard and probability targets

## Functional Checks

- Shared control/candidate tensors were bitwise identical before training.
- Global CPU RNG state after model construction was identical.
- Candidate had exactly nine gates with hidden widths 2/4/8 and 1,082,740 parameters.

## Results

| Metric | Control | Candidate | Gate |
| --- | ---: | ---: | ---: |
| Median trial mean | 10.8382 ms | 13.3662 ms | ratio <= 1.0526 |
| Training ratio | - | 1.23324x | **FAIL** |
| Projected steps | 26,898 | 21,810 | >= 25,553: **FAIL** |
| Trial-mean CV | 0.100% | 0.123% | < 3%: pass |
| Median p95 ratio | - | 1.23447x | <= 1.10: **FAIL** |
| Peak allocation | 598.67 MB | 630.89 MB | delta 32.21 MB: pass |
| Inference ratio | - | 1.15206x | <= 1.10: **FAIL** |

## Verdict

The candidate fails the pre-registered feasibility gate by a wide, low-variance margin. Tiny pooling, linear, sigmoid, and broadcast operations are launch-bound on this shallow CIFAR model. Do not run SE, shrink its scope, change its reduction, or add an execution optimization as a rescue. Per the external adversarial review's conditional selection, advance canonical full preactivation instead.
