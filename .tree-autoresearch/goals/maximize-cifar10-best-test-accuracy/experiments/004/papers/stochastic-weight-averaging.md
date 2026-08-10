# Averaging Weights Leads to Wider Optima and Better Generalization
- **Authors**: Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
- **Venue**: UAI 2018
- **URL**: https://www.auai.org/uai2018/proceedings/papers/313.pdf

## Key Contributions
- Averages multiple SGD trajectory points into one model rather than ensembling predictions.
- Reports improved generalization across residual networks, PyramidNets, DenseNets, and Shake-Shake models on CIFAR-10/100 and ImageNet.
- Connects the averaged solution to a wider optimum and reports almost no computational overhead.

## Relevance
The current CIFAR-10 model converges with low late loss and has ample memory headroom. Online sparse parameter averaging could reduce trajectory variance without adding a second forward/backward pass, but the short decaying-LR regime and BatchNorm buffers require a design adapted from classical constant/cyclic-LR SWA.

## Key Techniques
- Average weight-space samples from the later SGD trajectory.
- Preserve a single deployable model.
- Treat learning-rate schedule and BatchNorm statistics as coupled implementation choices.
