import tensorflow as tf
# import numpy as np
from tensorflow.contrib.slim.nets import alexnet
from tensorflow.contrib import slim

batch_size = 5
height, width = 224, 224
num_classes = 43

with tf.Session() as sess:
    inputs = tf.random_uniform((batch_size, height, width, 3))
    alexnet.alexnet_v2(inputs, num_classes)
    model_variables = [v.op.name for v in slim.get_model_variables()]
    for v in model_variables:
        print(v)


# # Load the Pascal VOC data
# image, label = MyPascalVocDataLoader(...)
# images, labels = tf.train.batch([image, label], batch_size=32)

# # Create the model
# predictions = vgg.vgg_16(images)

# train_op = slim.learning.create_train_op(...)

# # Specify where the Model, trained on ImageNet, was saved.
# model_path = '/path/to/pre_trained_on_imagenet.checkpoint'

# # Specify where the new model will live:
# log_dir = '/path/to/my_pascal_model_dir/'

# # Restore only the convolutional layers:
# variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
# init_fn = assign_from_checkpoint_fn(model_path, variables_to_restore)

# # Start training.
# slim.learning.train(train_op, log_dir, init_fn=init_fn)