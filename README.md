# Transfer Learning Lab

WIP

### Notes

- ResNet and InceptionV3 take a long time to load just from
calling `model = ResNet(...)`. Maybe just settle for VGG? 
- Don't think it's worth doing AlexNet:
    1. It's outdated
    2. It's probably oging to be the most difficult to package nicely into
    a lab/quiz.

### TODO

- TFSlim might be better for this lab, explore that. 
- Looks like we perform much worse with feature extraction
than training from scratch. This might be because:
    1. We need to use 224x224 images instead of 32x32 (the pretrained
    model was trained on 224x224). In general our preprocessing might
    not be up to par. TFSlim has this taken care of for us.
    2. Traffic signs might not benefit from transfer learning.

