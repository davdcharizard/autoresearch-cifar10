# EXP-015 Paired Timing Attempt 1

**Training**: pass. Accepted/candidate median trial means were 10.9233/10.8353 ms, ratio 0.991940, projected 27,116 steps, CV 0.169%/0.290%, p95 ratio 0.987446, and identical 598.671 MiB allocation.

**Inference**: candidate/control median ratio 1.006866 and projected total 330.720 seconds passed. Candidate CV was 0.413%. Control CV failed at 6.282% because the first fresh control process measured 2.2608 ms while the remaining four measured 1.9689-1.9855 ms. The paired candidate in that first trial was 1.9897 ms, consistent with its other trials.

**Verdict**: infrastructure-stability failure before any accuracy run. Retry the exact protocol once without code, threshold, warmup, ordering, or candidate changes. If instability recurs, stop EXP-015 before accuracy.
