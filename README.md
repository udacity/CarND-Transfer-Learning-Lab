# Transfer Learning Lab

WIP

## Notes

- calling `model = ResNet50(...)` takes a while to compile
- need to resize images to 224x224 in order for transfer
learning to do anything.
- VGG takes up too much memory for me to use with 224x224 images
- Transfer learning clearly is better than training from scratch
for Cifar10. However, for the traffic sign dataset it's maybe slightly better
than training from scratch on a Resnet but we can make smaller nets that do
much better and are more efficient.
