from keras.layers import Flatten, Dense, Dropout, Input
from keras.models import Sequential, Model
import tensorflow as tf
import pickle

flags = tf.app.flags
FLAGS = flags.FLAGS

nb_epoch = 50
batch_size = 256
nb_classes = 10

# flags.DEFINE_string('feature_file', '', '')

with open('inception_cifar10_bottleneck_features_train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('inception_cifar10_bottleneck_features_validation.p', 'rb') as f:
    validation_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = validation_data['features'], validation_data['labels']

print('Feature shape', X_train.shape[1:])

# model.add(Flatten(input_shape=X_train.shape[1:]))

inp = Input(shape=X_train.shape[1:])
x = Flatten()(inp)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(nb_classes, activation='softmax')(x)
model = Model(inp, x)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(X_val, y_val), shuffle=True)
