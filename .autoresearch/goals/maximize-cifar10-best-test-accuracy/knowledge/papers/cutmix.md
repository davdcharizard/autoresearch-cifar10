# CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
- **Authors**: Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo
- **Venue**: ICCV 2019
- **URL**: https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html

## Key Contributions

- Replaces an image rectangle with pixels from another training example and mixes class targets by the pasted area.
- Retains information in regions that Cutout or random erasing would discard while preserving regional-occlusion regularization.
- Reports CIFAR and ImageNet improvements over several contemporary augmentation strategies, with robustness and localization benefits.

## Relevance

CutMix directly addresses EXP-006's lossy fixed-square Cutout failure: donor pixels remain class-bearing and target mass reflects visible area. The installed torchvision `v2.CutMix(alpha=1.0, num_classes=10)` is batch-native and can run in worker collation, but dense-target cross-entropy remains inside synchronized training and requires a throughput gate. Composing it with accepted N1/M7 preserves broad invariances but risks excessive regularization for a 0.27M model and short horizon.
