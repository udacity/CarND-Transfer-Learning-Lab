# Transfer Learning Lab

WIP

### Notes

- ResNet and InceptionV3 take a long time to load just from
calling `model = ResNet(...)`. Maybe just settle for VGG? 
- Don't think it's worth doing AlexNet:
    1. It's outdated
    2. It's going to be the most difficult to package nicely into
    a lab/quiz.

- ~~TFSlim might be better for this lab, explore that.~~
- Looks like we perform much worse with feature extraction
than training from scratch or finetuning.

I tried (all on VGG):

- feature extraction
- finetuning on imagenet
- train from scratch

Feature extraction turns out too be much worse than the rest. Probably
because only 1 or 2 out of 1000 classes are related to our dataset so 
the base features in the convolutional layers learned from imagenet don't transfer well.

Finetuning performed slightly better (~3%) than training from scratch but
this only over 10 epochs (enough time to reach 90%+ testing accuracy). Need
to run a test over 50 or 100 epochs to make sure.
