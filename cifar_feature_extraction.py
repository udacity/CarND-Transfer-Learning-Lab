from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Input, AveragePooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
import keras.backend as K
from keras.datasets import cifar10
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import keras.backend as K

nb_classes = 10
batch_size = 32
nb_epoch = 2

h, w, ch = 299, 299, 3

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

with tf.Session() as sess:
    K.set_session(sess)

    input_tensor = Input(shape=(h, w, ch))
    # base_model = ResNet50(input_tensor=input_tensor, include_top=False)
    # base_model = InceptionV3(input_tensor=input_tensor, include_top=False)
    base_model = VGG16(input_tensor=input_tensor, include_top=False)

    x = base_model.output
    # x = AveragePooling2D((8,8), strides=(8,8))(x)
    x = AveragePooling2D((7,7))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(base_model.input, x)

    # # freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    img_placeholder = tf.placeholder("uint8", (None, 32, 32, 3))
    resize_op = tf.image.resize_images(img_placeholder, (h, w), method=0)

    def gen(data, labels, size=(h, w)):
        def _f():
            start = 0
            end = start + batch_size
            n = data.shape[0]
            while True:
                X_batch_old, y_batch = data[start:end], labels[start:end]
                X_batch = sess.run(resize_op, {img_placeholder: data[start:end]})
                X_batch = preprocess_input(X_batch)
                start += batch_size
                end += batch_size
                if start >= n:
                    start = 0
                    end = start + batch_size

                yield (X_batch, y_batch)
        return _f

    train_gen = gen(X_train, y_train)
    val_gen = gen(X_val, y_val)
    print("Starting training ... ")
    model.fit_generator(
        train_gen(),
        X_train.shape[0],
        nb_epoch,
        verbose=1,
        nb_val_samples=X_val.shape[0],
        validation_data=val_gen())