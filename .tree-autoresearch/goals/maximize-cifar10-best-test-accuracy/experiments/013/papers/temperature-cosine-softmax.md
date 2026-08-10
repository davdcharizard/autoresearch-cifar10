# Temperature in Cosine-based Softmax Loss
- **Authors**: Takumi Kobayashi
- **Venue**: ICCV 2025
- **URL**: https://openaccess.thecvf.com/content/ICCV2025/papers/Kobayashi_Temperature_in_Cosine-based_Softmax_Loss_ICCV_2025_paper.pdf

## Key Contributions
- Studies the scale/temperature parameter in cosine-normalized classifiers and proposes adapting it to task geometry.
- Shows strong scale sensitivity: fixed scales from 1 to 60 range from failed optimization to competitive accuracy.
- On ResNet-34/CIFAR-10, standard softmax reached 95.56%, while fixed scale 40 reached 95.85%; the proposed learned-scale method reached 95.49%.

## Relevance

The fixed-scale result suggests an orthogonal margin-geometry lever with negligible arithmetic cost, but the learned method did not beat standard softmax and the optimum is architecture/task specific. A candidate would need a preregistered scale and initialization-preserving implementation, with no validation search.

## Key Techniques
- L2-normalize features and classifier weights before a scaled cosine-softmax loss.
- Treat classifier scale as a high-sensitivity optimization parameter rather than a benign calibration constant.
