# Balanced Mixture of Supernets for Learning the CNN Pooling Architecture
- **Authors**: Mehraveh Javan Roshtkhari, Matthew Toews, Marco Pedersoli
- **Venue**: AutoML Conference 2023
- **URL**: https://proceedings.mlr.press/v224/roshtkhari23a.html

## Key Contributions
- Studies downsampling configurations directly in ResNet20 on CIFAR-10.
- Finds that downsampling position materially affects accuracy and default configurations are not always optimal.
- Develops a balanced supernet method to search pooling placement while reducing weight-sharing interference.

## Relevance
This is direct architecture/dataset evidence that the accepted two transition points are a real representation lever. EXP016 cannot run a NAS search, but it can isolate one evidence-backed anti-aliased transition policy under the fixed-time gate.

## Key Techniques
- Treat downsampling placement and operator as accuracy-critical architecture choices.
- Avoid inferring a candidate from shared-weight proxy performance.
- Validate the exact ResNet20 configuration end to end.
