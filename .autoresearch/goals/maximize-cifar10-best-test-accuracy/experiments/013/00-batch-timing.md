# EXP-013 Batch-Scaling Feasibility

**Device**: one idle NVIDIA H20
**Protocol**: accepted width-2 model, fresh state per batch size, 100 warmups plus 500 synchronized H2D/forward/CE/backward/SGD steps, alternating hard and probability targets

| Batch | Mean step | Images/s | Projected steps | Projected images | Epochs | Peak allocation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 11.733 ms | 10,909 | 25,568 | 3.273M | 65.45 | 598.7 MB |
| 256 | 18.270 ms | 14,012 | 16,420 | 4.204M | 84.07 | 1,120.2 MB |
| 512 | 34.762 ms | 14,729 | 8,630 | 4.419M | 88.37 | 2,163.2 MB |

Batch 256 is the measured throughput knee: 28.44% more image exposure than batch 128. Batch 512 adds only 5.12% more images than batch 256 while removing 47.4% of its optimizer updates. These figures establish systems feasibility only; they do not show that fewer, lower-variance updates at the accepted LR preserve accuracy.
