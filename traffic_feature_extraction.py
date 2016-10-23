from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Input
from sklearn.model_selection import train_test_split
from keras.models import Model
from skimage.transform import resize
import numpy as np
import pickle

nb_classes = 43
batch_size = 64
nb_epoch = 10

with open('./data/train.p', mode='rb') as f:
    train = pickle.load(f)
with open('./data/test.p', mode='rb') as f:
    test = pickle.load(f)

X_train, X_val, y_train, y_val = train_test_split(train['features'], train['labels'], test_size=0.33, random_state=0)
X_test, y_test = test['features'], test['labels']

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')

# 0-255 -> 0-1
X_train /= 255
X_val /= 255
X_test /= 255

def gen(data, labels, size=(224, 224)):
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
            start += batch_size
            end += batch_size
            if start >= n:
                start = 0
                end = batch_size

            yield (X_batch, y_batch)
    return _f

input_tensor = Input(shape=(224, 224, 3))
base_model = ResNet50(input_tensor=input_tensor, include_top=False, weights=None)

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(nb_classes, activation='softmax')(x)
model = Model(base_model.input, x)

# freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_gen = gen(X_train, y_train)
val_gen = gen(X_val, y_val)
model.fit_generator(
    train_gen(),
    X_train.shape[0],
    nb_epoch,
    nb_val_samples=X_val.shape[0],
    validation_data=val_gen())

_, acc = model.evaluate(X_test, y_test, verbose=0)
print("Testing accuracy =", acc)

