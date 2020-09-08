from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
import keras
import PIL
from PIL import Image
import os
from os.path import join, isfile
import pickle

batch_size = 128
num_classes = 240
epochs = 10
data_augmentation = False
num_predictions = 20
save_dir = os.getcwd()
model_name = 'cnn_basic_model.h5'
label_name = 'cnn_basic_label.pickle'

image_width = 224
image_height = 224

imagePath = '../images/trainingSet'
pattern = '*.jpg'

datas = []
labels = []
idMapping = {}

for index, id in enumerate(os.listdir(imagePath)):
    if id.isnumeric():
        files = [f for f in os.listdir(join(imagePath, id)) if isfile(join(imagePath, id, f))]
        for file in files:
            img = Image.open(join(imagePath, id, file))
            img = img.resize((image_width, image_height), PIL.Image.LANCZOS)
            img = img.convert('L')
            data = np.asarray(img)
            datas.append(data)
            labels.append(index)
            if not index in idMapping:
                idMapping[index] = id

with open(join(save_dir, label_name), 'wb') as handle:
    pickle.dump(idMapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

X = np.array(datas)
Y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(X, Y)

x_train = x_train.reshape(x_train.shape[0], image_width, image_height, 1)
x_test = x_test.reshape(x_test.shape[0], image_width, image_height, 1)
input_shape = (image_width, image_height, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model.save(join(save_dir, model_name))

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
