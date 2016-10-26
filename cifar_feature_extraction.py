from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Flatten, Input, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
import keras.backend as K
from keras.datasets import cifar10
from skimage.transform import resize
import numpy as np
import tensorflow as tf

nb_classes = 10
batch_size = 16
nb_epoch = 2

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

with tf.Session() as sess:
    K.set_session(sess)

    input_tensor = Input(shape=(299, 299, 3))
    # base_model = ResNet50(input_tensor=input_tensor, include_top=False)
    base_model = InceptionV3(input_tensor=input_tensor, include_top=False)

    x = base_model.output
    x = AveragePooling2D((8, 8), strides=(8, 8))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(base_model.input, x)

    # # freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # inp = base_model.input
    # out = base_model.output
    # print(inp)
    # print(out)
    #
    # top_model = Sequential()
    # top_model.add(Flatten(input_shape=(1, 1, 2048)))
    # top_model.add(Dense(512, activation='relu'))
    # top_model.add(Dense(nb_classes, activation='softmax'))

    # top_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def gen(data, labels, size=(299, 299)):
        def _f():
            start = 0
            end = start + batch_size
            n = data.shape[0]
            while True:
                X_batch_old, y_batch = data[start:end], labels[start:end]
                X_batch = []
                for i in range(X_batch_old.shape[0]):
                    img = resize(X_batch_old[i, ...], size)
                    X_batch.append(img)

                X_batch = np.array(X_batch)
                # X_batch = X_batch.astype('float32') / 255
                X_batch = preprocess_input(X_batch)
                start += batch_size
                end += batch_size
                if start >= n:
                    start = 0
                    end = start + batch_size

                # xx = sess.run(out, feed_dict={inp: X_batch, K.learning_phase(): 1})
                # print(xx)
                # yield (xx, y_batch)
                yield (X_batch, y_batch)
        return _f

    train_gen = gen(X_train, y_train)
    val_gen = gen(X_val, y_val)
    model.fit_generator(
        train_gen(),
        X_train.shape[0],
        nb_epoch,
        nb_val_samples=X_val.shape[0],
        validation_data=val_gen())