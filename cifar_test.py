from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Input
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.datasets import cifar10

nb_classes = 10
batch_size = 64
nb_epoch = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255

input_tensor = Input(shape=(32, 32, 3))
base_model = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

# freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    X_train,
    y_train,
    verbose=1,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    validation_data=(X_val, y_val))

_, acc = model.evaluate(X_test, y_test, verbose=1)
print("Testing accuracy =", acc)
