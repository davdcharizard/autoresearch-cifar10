# EXP-013
## Execution
- **Outcome**: failed
## Run Log
### Run 1
- 94.71% best, 87.36% final — T_max=55 too low for 87 epochs
### Run 2
- 94.79% best, 94.73% final — T_max=82, proper alignment
## Errors & Dead Ends
### VGG-style SpeedNet underperforms ResNet-k4
- Without airbench-specific optimizations (whitening, 1x1 expansion), simpler VGG is less efficient
