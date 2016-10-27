from keras.layers import Flatten, Dense, Dropout, Input
from keras.models import Sequential, Model
import tensorflow as tf
import pickle

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10', "Make bottleneck features this for dataset, one of 'cifar10', or 'traffic'")
flags.DEFINE_string('network', 'resnet', "The model to bottleneck, one of 'vgg', 'inception', or 'resnet'")
flags.DEFINE_integer('batch_size', 256, 'The batch size for the generator')

batch_size = FLAGS.batch_size
nb_epoch = 50
nb_classes = 43 # NOTE: change this!

train_output_file = "{}_{}_{}.p".format(FLAGS.network, FLAGS.dataset, 'bottleneck_features_train')
validation_output_file = "{}_{}_{}.p".format(FLAGS.network, FLAGS.dataset, 'bottleneck_features_validation')

with open(train_output_file, 'rb') as f:
    train_data = pickle.load(f)
with open(validation_output_file, 'rb') as f:
    validation_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = validation_data['features'], validation_data['labels']

print('Feature shape', X_train.shape[1:])

inp = Input(shape=X_train.shape[1:])
x = Flatten()(inp)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(nb_classes, activation='softmax')(x)
model = Model(inp, x)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(X_val, y_val), shuffle=True)
